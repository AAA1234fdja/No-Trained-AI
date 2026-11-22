#!/usr/bin/env python3
"""
inference_engine.py — Motor de inferencia en memoria (mmap) para el modelo 800M INT8.

Implementa un modelo LlamaForCausalLM completo con:
  - Carga de pesos .safetensors vía mmap (no carga todo en RAM de una vez).
  - Descuantización per-canal INT8 -> float32 en el vuelo.
  - RMSNorm, RoPE (interleaved), GQA (Grouped-Query Attention), SwiGLU.
  - KV-cache para decodificación autoregresiva incremental.
  - Generación con: greedy, temperatura, top-k, top-p, repetition penalty.
  - Estadísticas de generación: entropía, top-5, tiempo, tokens por segundo.
  - Soporte para múltiples prompts en lote (secuencial).
  - Guardado/carga de caché de KV entre generaciones.

Requisitos: numpy, safetensors, tokenizers.

Autor: AI Experiment — edición analítica de pesos, cero entrenamiento.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

# Importar utilidades del proyecto
from scripts.rope_utils import build_rope_cache, apply_rope_interleaved
from scripts.attention_utils import (
    stable_softmax,
    causal_mask,
    expand_kv_heads,
    KVCache,
    compute_attention_metrics,
)
from scripts.mlp_utils import rms_norm, silu, swiglu
from scripts.quantization_tools import dequantize_int8_per_row


# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH: str = "model/config.json"
DEFAULT_TOKENIZER_PATH: str = "model/tokenizer.json"
DEFAULT_WEIGHTS_PATH: str = "model/model.safetensors"


class MmapInt8LlamaModel:
    """
    Modelo Llama 800M con pesos en .safetensors accesibles vía mmap.

    Las características principales:
        - Los pesos se mantienen mmapeados en disco; solo se cargan las
          filas/columnas necesarias en cada operación.
        - Cuantización INT8 per-canal para pesos 2D, FP16 para escalas.
        - Arquitectura: Llama/GQA/RoPE/SwiGLU/RMSNorm, 21 capas.
        - KV-cache opcional para generación autoregresiva rápida.
    """

    def __init__(
        self,
        model_dir: str | Path,
        weights_filename: str = "model.safetensors",
        config_filename: str = "config.json",
        tokenizer_filename: str = "tokenizer.json",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / weights_filename
        self.config_path = self.model_dir / config_filename
        self.tokenizer_path = self.model_dir / tokenizer_filename

        # Cargar configuración
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        # Cargar tokenizador
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # Abrir archivo de pesos (mmap, solo lectura)
        self.sf = safe_open(str(self.weights_path), framework="numpy")

        # Dimensiones del modelo
        self.d = int(self.config["hidden_size"])
        self.d_ff = int(self.config["intermediate_size"])
        self.n_layers = int(self.config["num_hidden_layers"])
        self.n_heads = int(self.config["num_attention_heads"])
        self.n_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.gqa_groups = self.n_heads // self.n_kv_heads
        self.eps = float(self.config["rms_norm_eps"])
        self.vocab_size = int(self.config["vocab_size"])
        self.max_positions = int(self.config.get("max_position_embeddings", 2048))
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))

        # Caché de RoPE (precalculada)
        self.cos, self.sin = build_rope_cache(
            self.max_positions, self.head_dim, self.rope_theta
        )

        # Estadísticas de inferencia acumuladas
        self.inference_stats: List[dict] = []

    # ------------------------------------------------------------------
    # Acceso a tensores
    # ------------------------------------------------------------------

    def _get_tensor(self, name: str) -> np.ndarray:
        """Obtiene un tensor del archivo .safetensors (mmap)."""
        return self.sf.get_tensor(name)

    def _get_weight_f32(self, prefix: str) -> np.ndarray:
        """
        Obtiene un peso 2D desquantizado: w = w_q * scale (per-row).

        Retorna float32 listo para multiplicación matricial.
        """
        w_q = self._get_tensor(f"{prefix}.weight").astype(np.float32)
        scale = self._get_tensor(f"{prefix}.weight_scale").astype(np.float32)
        return w_q * scale.T  # (out_features, in_features)

    def _get_weight_f32_chunked(
        self,
        prefix: str,
        chunk_rows: int = 4096,
    ) -> np.ndarray:
        """
        Obtiene un peso 2D desquantizado por bloques de filas para evitar
        allocar toda la matriz float32 de una vez (útil para lm_head con
        32,000 filas).
        """
        w_q = self._get_tensor(f"{prefix}.weight")
        scale = self._get_tensor(f"{prefix}.weight_scale").astype(np.float32)
        n_rows = w_q.shape[0]
        n_cols = w_q.shape[1]
        out = np.empty((n_rows, n_cols), dtype=np.float32)
        for i in range(0, n_rows, chunk_rows):
            end = min(i + chunk_rows, n_rows)
            w_blk = w_q[i:end].astype(np.float32)
            out[i:end] = w_blk * scale[i:end].T
        return out

    def _get_norm_weight(self, prefix: str) -> np.ndarray:
        """Obtiene el peso de una norma RMSNorm (float16)."""
        return self._get_tensor(f"{prefix}.weight").astype(np.float32)

    def _embed(self, token_ids: List[int]) -> np.ndarray:
        """
        Obtiene embeddings de una lista de IDs de token.

        Retorna array de forma (seq_len, d) en float32.
        """
        w_q = self._get_tensor("model.embed_tokens.weight").astype(np.float32)
        scale = self._get_tensor(
            "model.embed_tokens.weight_scale"
        ).astype(np.float32)
        ids = np.array(token_ids, dtype=np.int64)
        return w_q[ids] * scale[ids]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: List[int],
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
        return_attn_metrics: bool = False,
    ) -> Tuple[np.ndarray, Optional[KVCache], Optional[List[dict]]]:
        """
        Pase forward completo por el modelo.

        Parameters
        ----------
        token_ids : lista de IDs de token (secuencia de entrada).
        kv_cache : caché KV opcional para generación incremental.
        start_pos : posición inicial en la caché (para tokens previos).
        return_attn_metrics : si calcular métricas de atención por capa.

        Returns
        -------
        logits : np.ndarray de forma (vocab_size,) (solo el último token).
        kv_cache : caché actualizada (o None si no se usó).
        attn_metrics : lista de diccionarios con métricas por capa, o None.
        """
        h = self._embed(token_ids)  # (seq_len, d)
        seq_len = h.shape[0]

        if kv_cache is None:
            kv_cache = KVCache(
                self.n_layers, self.n_kv_heads, self.head_dim,
                max_seq_len=self.max_positions,
            )

        new_cache = KVCache(
            self.n_layers, self.n_kv_heads, self.head_dim,
            max_seq_len=self.max_positions,
        )
        scale_attn = 1.0 / math.sqrt(self.head_dim)
        attn_metrics: List[dict] = [] if return_attn_metrics else None

        for layer_idx in range(self.n_layers):
            lp = f"model.layers.{layer_idx}"

            # --- RMSNorm de entrada ---
            ln1_w = self._get_norm_weight(f"{lp}.input_layernorm")
            x_norm = rms_norm(h, ln1_w, self.eps)

            # --- Proyecciones de atención ---
            q = self._linear_f32(x_norm, f"{lp}.self_attn.q_proj")
            k = self._linear_f32(x_norm, f"{lp}.self_attn.k_proj")
            v = self._linear_f32(x_norm, f"{lp}.self_attn.v_proj")

            q = q.reshape(seq_len, self.n_heads, self.head_dim)
            k = k.reshape(seq_len, self.n_kv_heads, self.head_dim)
            v = v.reshape(seq_len, self.n_kv_heads, self.head_dim)

            # RoPE
            q = apply_rope_interleaved(q, self.cos, self.sin, start_pos)
            k = apply_rope_interleaved(k, self.cos, self.sin, start_pos)

            # Caché KV
            prev = kv_cache.get(layer_idx)
            if prev is not None:
                prev_k, prev_v = prev
                k_all = np.concatenate([prev_k, k], axis=0)
                v_all = np.concatenate([prev_v, v], axis=0)
            else:
                k_all, v_all = k, v

            new_cache.append(layer_idx, k_all, v_all)

            total_len = k_all.shape[0]

            # Expandir KV heads para GQA
            k_exp, v_exp = expand_kv_heads(
                k_all, v_all, self.n_heads, self.n_kv_heads
            )

            # Atención por cabeza
            q_t = np.transpose(q, (1, 0, 2))            # (heads, seq, dim)
            k_t = np.transpose(k_exp, (1, 2, 0))        # (heads, dim, total)
            v_t = np.transpose(v_exp, (1, 0, 2))        # (heads, total, dim)

            scores = (q_t @ k_t) * scale_attn           # (heads, seq, total)

            # Máscara causal
            mask = causal_mask(seq_len, total_len)
            scores = scores + mask[None, :, :]

            probs = stable_softmax(scores, axis=-1)
            attn_out = np.matmul(probs, v_t)            # (heads, seq, dim)
            attn_out = np.transpose(attn_out, (1, 0, 2)).reshape(seq_len, self.d)

            if return_attn_metrics and attn_metrics is not None:
                attn_metrics.append(
                    compute_attention_metrics(scores, probs, attn_out)
                )

            # Proyección de salida de atención
            h = h + self._linear_f32(attn_out, f"{lp}.self_attn.o_proj")

            # --- MLP SwiGLU ---
            ln2_w = self._get_norm_weight(f"{lp}.post_attention_layernorm")
            m_norm = rms_norm(h, ln2_w, self.eps)

            gate = silu(self._linear_f32(m_norm, f"{lp}.mlp.gate_proj"))
            up = self._linear_f32(m_norm, f"{lp}.mlp.up_proj")
            down = self._linear_f32(silu(gate) * up, f"{lp}.mlp.down_proj")
            h = h + down

        # --- Norma final ---
        final_ln_w = self._get_norm_weight("model.norm")
        h_final = rms_norm(h[-1:, :], final_ln_w, self.eps)

        # --- Logits de salida ---
        logits = self._linear_f32_chunked(h_final, "lm_head")[0]  # (vocab_size,)

        return logits, new_cache, attn_metrics

    # ------------------------------------------------------------------
    # Operaciones lineales
    # ------------------------------------------------------------------

    def _linear_f32(
        self,
        x: np.ndarray,
        prefix: str,
    ) -> np.ndarray:
        """
        Calcula x @ W^T donde W está almacenado como INT8 + escala.

        x : (..., in_features)
        Retorna: (..., out_features) en float32.
        """
        w_q = self._get_tensor(f"{prefix}.weight")
        scale = self._get_tensor(f"{prefix}.weight_scale").astype(np.float32)

        n_rows = w_q.shape[0]
        if n_rows > 8192:
            # Procesar por bloques para evitar memoria excesiva
            out = np.empty(x.shape[:-1] + (n_rows,), dtype=np.float32)
            chunk = 4096
            for i in range(0, n_rows, chunk):
                end = min(i + chunk, n_rows)
                w_blk = w_q[i:end].astype(np.float32)
                out[..., i:end] = (x @ w_blk.T) * scale[i:end].T
            return out

        return (x @ w_q.astype(np.float32).T) * scale.T

    # ------------------------------------------------------------------
    # Generación
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        return_details: bool = False,
    ) -> dict:
        """
        Genera texto de forma autoregresiva a partir de un prompt.

        Parameters
        ----------
        prompt : texto de entrada.
        max_new_tokens : número máximo de tokens a generar.
        temperature : 0.0 = greedy (argmax), > 0 = muestreo.
        top_k : limitar muestreo a los k más probables (0 = desactivado).
        top_p : muestreo nucleus (acumulado de probabilidad).
        repetition_penalty : penalizar tokens repetidos (>1.0 reduce repetición).
        seed : semilla para generador aleatorio (muestreo).
        return_details : si incluir top-5, entropía, etc.

        Returns
        -------
        dict con: prompt, generated_ids, generated_text, top5_first_token,
                  first_token_entropy_nats, elapsed_sec, prompt_tokens.
        """
        t0 = time.time()
        rng = np.random.default_rng(seed if seed is not None else 42)

        # Codificar prompt
        enc = self.tokenizer.encode(prompt)
        prompt_ids = enc.ids if len(enc.ids) > 0 else [0]
        prompt_tokens = len(prompt_ids)

        # Forward del prompt
        logits, cache, _ = self.forward(
            prompt_ids, kv_cache=None, start_pos=0
        )

        # Top-5 del primer token generado
        first_probs = stable_softmax(logits)
        top5_idx = np.argsort(logits)[-5:][::-1]
        top5 = [
            (
                self.tokenizer.decode([int(i)]),
                float(logits[i]),
                float(first_probs[i]),
            )
            for i in top5_idx
        ]
        entropy = float(
            -np.sum(first_probs * np.log(np.clip(first_probs, 1e-12, 1.0)))
        )

        generated_ids: List[int] = []
        cur_pos = prompt_tokens
        seen_tokens: set = set(prompt_ids)

        for step in range(max_new_tokens):
            step_logits = logits.copy()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tid in seen_tokens:
                    if step_logits[tid] > 0:
                        step_logits[tid] /= repetition_penalty
                    else:
                        step_logits[tid] *= repetition_penalty

            # Muestreo
            if temperature <= 0.0:
                next_id = int(np.argmax(step_logits))
            else:
                scaled = step_logits / temperature
                if top_k > 0:
                    idx = np.argpartition(scaled, -top_k)[-top_k:]
                    scaled_subset = scaled[idx]
                    probs = stable_softmax(scaled_subset)
                    if top_p < 1.0:
                        # Top-p dentro del top-k
                        sorted_idx = np.argsort(probs)[::-1]
                        cum_probs = np.cumsum(probs[sorted_idx])
                        cutoff = np.searchsorted(cum_probs, top_p) + 1
                        keep = sorted_idx[:cutoff]
                        probs_keep = probs[keep]
                        probs_keep = probs_keep / probs_keep.sum()
                        next_id = int(rng.choice(keep, p=probs_keep))
                    else:
                        next_id = int(rng.choice(idx, p=probs))
                else:
                    probs = stable_softmax(scaled)
                    if top_p < 1.0:
                        sorted_idx = np.argsort(probs)[::-1]
                        cum_probs = np.cumsum(probs[sorted_idx])
                        cutoff = np.searchsorted(cum_probs, top_p) + 1
                        keep = sorted_idx[:cutoff]
                        probs_keep = probs[keep]
                        probs_keep = probs_keep / probs_keep.sum()
                        next_id = int(rng.choice(keep, p=probs_keep))
                    else:
                        next_id = int(rng.choice(len(probs), p=probs))

            generated_ids.append(next_id)
            seen_tokens.add(next_id)

            if next_id == int(self.config.get("eos_token_id", 1)):
                break

            if step + 1 < max_new_tokens:
                logits, cache, _ = self.forward(
                    [next_id], kv_cache=cache, start_pos=cur_pos
                )
                cur_pos += 1

        elapsed = time.time() - t0
        generated_text = self.tokenizer.decode(generated_ids)

        result: dict = {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "top5_first_token": top5,
            "first_token_entropy_nats": entropy,
            "elapsed_sec": round(elapsed, 4),
            "tokens_per_second": round(len(generated_ids) / elapsed, 2)
            if elapsed > 0 else 0.0,
        }

        if return_details:
            result["cache_memory_mb"] = round(cache.memory_usage_mb(), 2)

        self.inference_stats.append(result)
        return result

    def generate_multiple(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[dict]:
        """
        Genera texto para múltiples prompts de forma secuencial.

        Returns lista de diccionarios de resultado (uno por prompt).
        """
        return [self.generate(p, **kwargs) for p in prompts]

    def get_inference_stats(self) -> List[dict]:
        """Devuelve las estadísticas de todas las generaciones realizadas."""
        return self.inference_stats

    def reset_stats(self) -> None:
        """Limpia las estadísticas de inferencia acumuladas."""
        self.inference_stats = []

    def close(self) -> None:
        """Cierra el archivo de pesos."""
        try:
            self.sf.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Función de conveniencia para uso independiente
# ---------------------------------------------------------------------------

def run_inference(
    model_dir: str | Path,
    prompts: List[str],
    weights_filename: str = "model.safetensors",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
) -> List[dict]:
    """
    Ejecuta inferencia sobre una lista de prompts y devuelve los resultados.

    Útil para pruebas rápidas desde línea de comandos.
    """
    model = MmapInt8LlamaModel(model_dir, weights_filename)
    results = model.generate_multiple(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_details=True,
    )
    model.close()
    return results


if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model"
    model = MmapInt8LlamaModel(model_dir)

    test_prompts = [
        "def fibonacci(n):",
        "Analyze the time complexity of binary search:",
        "Hello! How can you help me?",
    ]

    for prompt in test_prompts:
        res = model.generate(prompt, max_new_tokens=12, temperature=0.0)
        print(f"Prompt: {prompt!r}")
        print(f"  Generated: {res['generated_text']!r}")
        print(f"  Entropy: {res['first_token_entropy_nats']:.3f} nats")
        print(f"  Time: {res['elapsed_sec']}s")
        print()

    model.close()
