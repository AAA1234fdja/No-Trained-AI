#!/usr/bin/env python3
"""
anti_loop_inference.py — Extensión del motor de inferencia con capacidades anti-loop.

Implementa:
  1. N-gram blocking: bloquea n-grams ya generados para evitar repeticiones
  2. Frequency penalty: penaliza tokens proporcional a su frecuencia en la salida
  3. Presence penalty: penaliza tokens que han aparecido al menos una vez
  4. Repetition penalty dinámica: aumenta la penalidad si se detecta loop
  5. Temperature annealing: reduce temperatura progresivamente si hay repetición
  6. Top-k dinámico: reduce k si se detecta repetición de tokens

Diseñado para integrarse con inference_engine.py sin modificar los pesos del modelo.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference_engine import MmapInt8LlamaModel, stable_softmax


# ---------------------------------------------------------------------------
# Utilidades de detección de loops
# ---------------------------------------------------------------------------

def detect_token_loop(text: str, min_repeats: int = 3) -> Tuple[bool, str, int]:
    """Detecta repetición de token completo (ej. 'forforfor')."""
    pattern = r"(.{2,30}?)\1{" + str(min_repeats - 1) + r",}"
    match = re.search(pattern, text)
    if match:
        return True, match.group(1), len(match.group(0)) // len(match.group(1))
    return False, "", 0


def detect_ngram_loop(tokens: List[int], n: int = 3, min_repeats: int = 2) -> bool:
    """Detecta n-grams repetidos."""
    if len(tokens) < n * min_repeats:
        return False
    ngrams: Dict[Tuple[int, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return any(count >= min_repeats for count in ngrams.values())


def get_repeated_ngrams(tokens: List[int], n: int = 3) -> List[Tuple[Tuple[int, ...], int]]:
    """Devuelve n-grams repetidos y sus conteos."""
    ngrams: Dict[Tuple[int, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return [(ngram, count) for ngram, count in ngrams.items() if count >= 2]


# ---------------------------------------------------------------------------
# Generador anti-loop
# ---------------------------------------------------------------------------

class AntiLoopInferenceEngine:
    """
    Envoltura sobre MmapInt8LlamaModel con capacidades anti-loop.

    No modifica los pesos del modelo. Toda la intervención ocurre en
    tiempo de inferencia sobre los logits antes del muestreo.
    """

    def __init__(
        self,
        model_dir: Path,
        weights_filename: str = "model.safetensors",
    ) -> None:
        self.model = MmapInt8LlamaModel(model_dir, weights_filename=weights_filename)
        self.tokenizer = self.model.tokenizer

    # ------------------------------------------------------------------
    # Aplicación de penalidades a los logits
    # ------------------------------------------------------------------

    def apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        penalty: float = 1.5,
    ) -> np.ndarray:
        """
        Penaliza tokens ya generados (repetición clásica).

        Si penalty > 1, divide los logits positivos por penalty y multiplica
        los negativos por penalty, haciendo menos probable la repetición.
        """
        if penalty == 1.0 or not generated_ids:
            return logits

        for tid in set(generated_ids):
            if tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= penalty
                else:
                    logits[tid] *= penalty
        return logits

    def apply_frequency_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        penalty: float = 0.5,
    ) -> np.ndarray:
        """
        Penaliza tokens proporcional a su frecuencia en la salida generada.

        logits[tid] -= penalty * count(tid en generated_ids)
        """
        if penalty == 0.0 or not generated_ids:
            return logits

        counts = Counter(generated_ids)
        for tid, count in counts.items():
            if tid < len(logits):
                logits[tid] -= penalty * count
        return logits

    def apply_presence_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        penalty: float = 0.5,
    ) -> np.ndarray:
        """
        Penaliza tokens que han aparecido al menos una vez (sin importar cuántas).

        logits[tid] -= penalty si tid ha aparecido
        """
        if penalty == 0.0 or not generated_ids:
            return logits

        seen = set(generated_ids)
        for tid in seen:
            if tid < len(logits):
                logits[tid] -= penalty
        return logits

    def apply_ngram_blocking(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        n: int = 3,
        penalty: float = 10.0,
    ) -> np.ndarray:
        """
        Bloquea tokens que completarían un n-gram ya generado.

        Si los últimos n-1 tokens + candidato forman un n-gram ya visto,
        se penaliza fuertemente ese candidato.
        """
        if n < 2 or not generated_ids or len(generated_ids) < n - 1:
            return logits

        # N-grams ya generados
        existing_ngrams: set = set()
        for i in range(len(generated_ids) - n + 1):
            existing_ngrams.add(tuple(generated_ids[i:i + n]))

        # Contexto actual (últimos n-1 tokens)
        context = tuple(generated_ids[-(n - 1):])

        # Probar cada candidato
        for candidate in range(len(logits)):
            if candidate >= len(logits):
                continue
            potential_ngram = context + (candidate,)
            if potential_ngram in existing_ngrams:
                logits[candidate] -= penalty  # Penalización fuerte

        return logits

    # ------------------------------------------------------------------
    # Generación con anti-loop
    # ------------------------------------------------------------------

    def generate_anti_loop(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.3,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        n_gram_block: int = 0,
        dynamic_rep_pen: bool = False,
        temp_anneal: bool = False,
        dynamic_top_k: bool = False,
        loop_threshold_severity: float = 0.3,
    ) -> dict:
        """
        Genera texto con capacidades anti-loop.

        Parameters
        ----------
        prompt : str
            Texto de entrada.
        max_new_tokens : int
            Máximo de tokens a generar.
        temperature : float
            Temperatura de muestreo.
        top_k : int
            Top-k para muestreo.
        top_p : float
            Top-p para muestreo.
        repetition_penalty : float
            Penalidad de repetición clásica.
        frequency_penalty : float
            Penalidad por frecuencia de token.
        presence_penalty : float
            Penalidad por presencia de token.
        n_gram_block : int
            Tamaño de n-gram para bloquear (0 = desactivado).
        dynamic_rep_pen : bool
            Si True, aumenta repetition_penalty cuando se detecta loop.
        temp_anneal : bool
            Si True, reduce temperatura cuando se detecta loop.
        dynamic_top_k : bool
            Si True, reduce top_k cuando se detecta loop.
        loop_threshold_severity : float
            Severidad mínima para activar medidas dinámicas.

        Returns
        -------
        dict con: generated_text, generated_ids, has_loop, severity,
                  entropy, elapsed_sec, loop_detected_at_step.
        """
        t0 = time.time()
        rng = np.random.default_rng(42)

        # Codificar prompt
        enc = self.tokenizer.encode(prompt)
        prompt_ids = enc.ids if len(enc.ids) > 0 else [0]
        prompt_tokens = len(prompt_ids)

        # Parámetros dinámicos
        curr_temp = temperature
        curr_top_k = top_k
        curr_rep_pen = repetition_penalty

        # Forward del prompt
        logits, cache, _ = self.model.forward(
            prompt_ids, kv_cache=None, start_pos=0
        )

        generated_ids: List[int] = []
        cur_pos = prompt_tokens
        seen_tokens = set(prompt_ids)
        loop_detected_at_step: Optional[int] = None
        has_loop = False
        severity = 0.0

        for step in range(max_new_tokens):
            step_logits = logits.copy()

            # Repetition penalty clásica
            step_logits = self.apply_repetition_penalty(
                step_logits, generated_ids, curr_rep_pen
            )

            # Frequency penalty
            step_logits = self.apply_frequency_penalty(
                step_logits, generated_ids, frequency_penalty
            )

            # Presence penalty
            step_logits = self.apply_presence_penalty(
                step_logits, generated_ids, presence_penalty
            )

            # N-gram blocking
            if n_gram_block > 0:
                step_logits = self.apply_ngram_blocking(
                    step_logits, generated_ids, n_gram_block
                )

            # Top-k filtering
            if curr_top_k > 0 and curr_top_k < len(step_logits):
                top_k_indices = np.argsort(step_logits)[-curr_top_k:][::-1]
                mask = np.full_like(step_logits, -np.inf)
                mask[top_k_indices] = step_logits[top_k_indices]
                step_logits = mask

            # Top-p filtering (nucleus)
            if top_p < 1.0:
                sorted_indices = np.argsort(step_logits)[::-1]
                sorted_logits = step_logits[sorted_indices]
                probs = stable_softmax(sorted_logits)
                cumulative_probs = np.cumsum(probs)
                cutoff = np.searchsorted(cumulative_probs, top_p)
                cutoff = max(1, cutoff)
                keep_indices = sorted_indices[:cutoff]
                mask = np.full_like(step_logits, -np.inf)
                mask[keep_indices] = step_logits[keep_indices]
                step_logits = mask

            # Muestreo
            if temperature == 0.0:
                next_id = int(np.argmax(step_logits))
            else:
                probs = stable_softmax(step_logits / curr_temp)
                next_id = int(rng.choice(len(probs), p=probs))

            generated_ids.append(next_id)
            seen_tokens.add(next_id)

            # Detectar loop en el texto generado hasta ahora
            partial_text = self.tokenizer.decode(generated_ids)
            has_loop_step, _, _ = detect_token_loop(partial_text, min_repeats=3)
            if has_loop_step and loop_detected_at_step is None:
                loop_detected_at_step = step
                has_loop = True
                severity = 1.0  # Loop detectado

                # Aplicar medidas dinámicas
                if dynamic_rep_pen:
                    curr_rep_pen = min(curr_rep_pen * 1.5, 5.0)
                if temp_anneal:
                    curr_temp = max(curr_temp * 0.7, 0.05)
                if dynamic_top_k:
                    curr_top_k = max(curr_top_k - 2, 1)

            # Forward siguiente
            logits, cache, _ = self.model.forward(
                [next_id], kv_cache=cache, start_pos=cur_pos
            )
            cur_pos += 1

        generated_text = self.tokenizer.decode(generated_ids)
        elapsed = time.time() - t0

        # Evaluar severidad final del texto completo
        _, _, _ = detect_token_loop(generated_text, min_repeats=3)
        severity = 1.0 if has_loop else 0.0

        # Entropía del primer token
        first_probs = stable_softmax(logits)
        first_entropy = float(
            -np.sum(first_probs * np.log(np.clip(first_probs, 1e-12, 1.0)))
        )

        return {
            "generated_text": generated_text,
            "generated_ids": generated_ids,
            "has_loop": has_loop,
            "severity": severity,
            "entropy": round(first_entropy, 2),
            "elapsed_sec": round(elapsed, 2),
            "loop_detected_at_step": loop_detected_at_step,
            "n_generated": len(generated_ids),
        }


# ---------------------------------------------------------------------------
# Evaluación rápida de configuraciones anti-loop
# ---------------------------------------------------------------------------

def evaluate_anti_loop_configs(
    model_dir: Path,
    prompts: Optional[List[str]] = None,
) -> dict:
    """
    Evalúa múltiples configuraciones anti-loop en prompts clave.

    Returns dict con resultados y resumen.
    """
    if prompts is None:
        prompts = ["def ", "for i in range(", "Complexity:"]

    engine = AntiLoopInferenceEngine(model_dir)

    configs = [
        {
            "name": "baseline",
            "params": {
                "temperature": 0.3, "top_k": 6, "repetition_penalty": 1.5,
            },
        },
        {
            "name": "high_rep_pen",
            "params": {
                "temperature": 0.3, "top_k": 6, "repetition_penalty": 3.0,
            },
        },
        {
            "name": "freq_pen",
            "params": {
                "temperature": 0.3, "top_k": 6, "repetition_penalty": 1.5,
                "frequency_penalty": 0.5,
            },
        },
        {
            "name": "presence_pen",
            "params": {
                "temperature": 0.3, "top_k": 6, "repetition_penalty": 1.5,
                "presence_penalty": 0.5,
            },
        },
        {
            "name": "combined",
            "params": {
                "temperature": 0.3, "top_k": 6, "repetition_penalty": 1.8,
                "frequency_penalty": 0.3, "presence_penalty": 0.3,
            },
        },
        {
            "name": "n_gram_block",
            "params": {
                "temperature": 0.3, "top_k": 6, "repetition_penalty": 1.5,
                "n_gram_block": 3,
            },
        },
        {
            "name": "low_temp",
            "params": {
                "temperature": 0.1, "top_k": 3, "repetition_penalty": 1.5,
            },
        },
        {
            "name": "dynamic",
            "params": {
                "temperature": 0.3, "top_k": 8, "repetition_penalty": 1.5,
                "dynamic_rep_pen": True, "temp_anneal": True, "dynamic_top_k": True,
            },
        },
    ]

    results: Dict[str, List[dict]] = {}

    for config in configs:
        print(f"\n--- {config['name']} ---")
        config_results: List[dict] = []

        for prompt in prompts:
            try:
                result = engine.generate_anti_loop(
                    prompt, max_new_tokens=10, **config["params"]
                )
                config_results.append(result)

                loop_icon = "🔴" if result["has_loop"] else "🟢"
                print(
                    f"  {loop_icon} [{prompt[:20]:<20}] -> "
                    f"{repr(result['generated_text'][:45])} "
                    f"(loop={result['has_loop']}, sev={result['severity']:.1f}, "
                    f"H={result['entropy']:.2f})"
                )
            except Exception as e:
                print(f"  ❌ [{prompt[:20]:<20}] ERROR: {e}")

        results[config["name"]] = config_results

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN ANTI-LOOP")
    print("=" * 60)

    summary: Dict[str, dict] = {}
    for name, cfg_results in results.items():
        total = len(cfg_results)
        loop_count = sum(1 for r in cfg_results if r["has_loop"])
        avg_severity = float(np.mean([r["severity"] for r in cfg_results]))
        avg_entropy = float(np.mean([r["entropy"] for r in cfg_results]))
        avg_elapsed = float(np.mean([r["elapsed_sec"] for r in cfg_results]))

        summary[name] = {
            "total": total,
            "loop_count": loop_count,
            "loop_rate": round(loop_count / total * 100, 1),
            "avg_severity": round(avg_severity, 3),
            "avg_entropy": round(avg_entropy, 2),
            "avg_elapsed": round(avg_elapsed, 1),
        }

    print(f"{'Config':<18} {'Loop%':<8} {'Sev':<6} {'Entropía':<9} {'Tiempo':<8}")
    print("-" * 60)
    for name, stats in sorted(
        summary.items(), key=lambda x: x[1]["loop_rate"]
    ):
        print(
            f"{name:<18} {stats['loop_rate']:<8} {stats['avg_severity']:<6} "
            f"{stats['avg_entropy']:<9} {stats['avg_elapsed']:<8}"
        )

    best = min(summary.items(), key=lambda x: x[1]["loop_rate"])
    print(f"\n>>> Mejor configuración: {best[0]} (Loop rate: {best[1]['loop_rate']}%)")

    return {"results": results, "summary": summary, "best_config": best[0]}


def main() -> None:
    model_dir = Path("/home/user/No-Trained-AI/model")
    if not (model_dir / "model.safetensors").exists():
        print(f"ERROR: No se encontró {model_dir / 'model.safetensors'}")
        sys.exit(1)

    print("=" * 60)
    print("EVALUACIÓN DE CONFIGURACIONES ANTI-LOOP")
    print("=" * 60)
    print()

    output = evaluate_anti_loop_configs(model_dir)

    out_path = Path("/home/user/ai_experiment/anti_loop_configs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()