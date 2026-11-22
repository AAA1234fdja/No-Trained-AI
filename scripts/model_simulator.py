#!/usr/bin/env python3
"""
model_simulator.py — Simulador de forward pass del modelo 800M (NumPy puro).

Este módulo implementa un forward pass completo del modelo LlamaForCausalLM
usando solo NumPy, sin depender de PyTorch o TensorFlow. Incluye:

  - Embedding lookup y posicionamiento (RoPE).
  - Atención causal con GQA (Grouped Query Attention).
  - MLP con SwiGLU activación.
  - RMSNorm.
  - Capa final de salida (lm_head).
  - Generación autoregresiva con muestreo (temperatura, top-k, top-p).
  - Caché KV para generación eficiente.
  - Cálculo de entropía y perplexidad.
  - Estadísticas de activación por capa.
  - Simulación de forward pass para validación de pesos.

Todas las operaciones son numéricas y determinísticas (sin gradientes).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


# ---------------------------------------------------------------------------
# Arquitectura del modelo (parámetros fijos para el modelo 800M)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32000
HIDDEN_SIZE = 1536
NUM_LAYERS = 21
NUM_ATTENTION_HEADS = 12
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 128
INTERMEDIATE_SIZE = 5888
MAX_SEQ_LEN = 2048
RMS_NORM_EPS = 1e-5
ROPE_BASE = 10000.0


@dataclass
class ModelWeights:
    """Contenedor de todos los pesos del modelo."""
    embed_tokens: np.ndarray  # (vocab_size, hidden_size)
    lm_head: np.ndarray  # (vocab_size, hidden_size)
    final_norm_weight: np.ndarray  # (hidden_size,)

    # Por capa
    layer_input_ln_weight: List[np.ndarray] = field(default_factory=list)
    layer_q_proj: List[np.ndarray] = field(default_factory=list)
    layer_k_proj: List[np.ndarray] = field(default_factory=list)
    layer_v_proj: List[np.ndarray] = field(default_factory=list)
    layer_o_proj: List[np.ndarray] = field(default_factory=list)
    layer_post_attn_ln_weight: List[np.ndarray] = field(default_factory=list)
    layer_gate_proj: List[np.ndarray] = field(default_factory=list)
    layer_up_proj: List[np.ndarray] = field(default_factory=list)
    layer_down_proj: List[np.ndarray] = field(default_factory=list)

    def n_params(self) -> int:
        total = self.embed_tokens.size + self.lm_head.size + self.final_norm_weight.size
        for i in range(NUM_LAYERS):
            total += (
                self.layer_input_ln_weight[i].size +
                self.layer_q_proj[i].size +
                self.layer_k_proj[i].size +
                self.layer_v_proj[i].size +
                self.layer_o_proj[i].size +
                self.layer_post_attn_ln_weight[i].size +
                self.layer_gate_proj[i].size +
                self.layer_up_proj[i].size +
                self.layer_down_proj[i].size
            )
        return total


@dataclass
class ForwardPassResult:
    """Resultado de un forward pass."""
    logits: np.ndarray  # (seq_len, vocab_size)
    hidden_states: List[np.ndarray] = field(default_factory=list)
    attention_weights: List[np.ndarray] = field(default_factory=list)
    kv_cache: Optional[Dict] = None
    n_tokens: int = 0
    elapsed_sec: float = 0.0

    def get_next_token_logits(self, position: int = -1) -> np.ndarray:
        """Obtiene los logits para el siguiente token en una posición."""
        return self.logits[position]


@dataclass
class GenerationResult:
    """Resultado de generación autoregresiva."""
    generated_tokens: List[int]
    generated_text: str
    entropies: List[float] = field(default_factory=list)
    tokens_per_second: float = 0.0
    n_tokens_generated: int = 0


# ---------------------------------------------------------------------------
# Operaciones numéricas básicas
# ---------------------------------------------------------------------------

def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = RMS_NORM_EPS,
) -> np.ndarray:
    """
    Aplica RMSNorm: x * weight / sqrt(mean(x^2) + eps).

    Parameters
    ----------
    x : np.ndarray
        Entrada de forma (..., hidden_size).
    weight : np.ndarray
        Pesos de escala de forma (hidden_size,).
    eps : float
        Epsilon para estabilidad numérica.

    Returns
    -------
    np.ndarray con la misma forma que x.
    """
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + eps)
    return x * inv_rms * weight


def silu(x: np.ndarray) -> np.ndarray:
    """
    Función de activación SiLU (Sigmoid Linear Unit): x * sigmoid(x).
    """
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


def swiglu(
    gate: np.ndarray,
    up: np.ndarray,
    down_weight: np.ndarray,
) -> np.ndarray:
    """
    Bloque SwiGLU: down(silu(gate) * up).

    Parameters
    ----------
    gate : np.ndarray
        Salida de gate_proj de forma (..., intermediate_size).
    up : np.ndarray
        Salida de up_proj de forma (..., intermediate_size).
    down_weight : np.ndarray
        Pesos de down_proj de forma (hidden_size, intermediate_size).

    Returns
    -------
    np.ndarray de forma (..., hidden_size).
    """
    activated = silu(gate) * up
    return activated @ down_weight.T


def apply_rope_interleaved(
    x: np.ndarray,
    position_ids: np.ndarray,
    base: float = ROPE_BASE,
) -> np.ndarray:
    """
    Aplica RoPE (Rotary Positional Embeddings) en formato interleaved.

    Para cada par de dimensiones (2i, 2i+1), aplica rotación por el ángulo
    correspondiente a la posición.

    Parameters
    ----------
    x : np.ndarray
        Tensor de forma (..., seq_len, head_dim).
    position_ids : np.ndarray
        IDs de posición de forma (seq_len,) o (batch, seq_len).
    base : float
        Base para las frecuencias de RoPE.

    Returns
    -------
    np.ndarray con la misma forma que x.
    """
    *dims, seq_len, head_dim = x.shape
    half_dim = head_dim // 2

    # Frecuencias: theta_i = base^(-2i / head_dim)
    i = np.arange(half_dim, dtype=np.float64)
    freqs = 1.0 / (base ** (2 * i / head_dim))  # (half_dim,)

    # Ángulos: position * freq
    if position_ids.ndim == 1:
        position_ids = position_ids[None, :]  # (1, seq_len)
    angles = position_ids.astype(np.float64)[:, :, None] * freqs[None, None, :]  # (1, seq_len, half_dim)

    # Separar x en pares
    x1 = x[..., :, 0::2]  # (..., seq_len, half_dim)
    x2 = x[..., :, 1::2]  # (..., seq_len, half_dim)

    cos = np.cos(angles)  # (1, seq_len, half_dim)
    sin = np.sin(angles)  # (1, seq_len, half_dim)

    # Rotación
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # Intercalar de vuelta
    result = np.empty_like(x)
    result[..., :, 0::2] = x1_rot
    result[..., :, 1::2] = x2_rot

    return result


def apply_rope_neox(
    x: np.ndarray,
    position_ids: np.ndarray,
    base: float = ROPE_BASE,
) -> np.ndarray:
    """
    Aplica RoPE en formato GPT-NeoX (primer mitad y segunda mitad rotadas).

    En lugar de intercalar pares, rota la primera mitad contra la segunda.

    Parameters
    ----------
    x : np.ndarray
        Tensor de forma (..., seq_len, head_dim).
    position_ids : np.ndarray
        IDs de posición.
    base : float
        Base para frecuencias de RoPE.

    Returns
    -------
    np.ndarray con la misma forma que x.
    """
    *dims, seq_len, head_dim = x.shape
    half_dim = head_dim // 2

    i = np.arange(half_dim, dtype=np.float64)
    freqs = 1.0 / (base ** (2 * i / head_dim))

    if position_ids.ndim == 1:
        position_ids = position_ids[None, :]
    angles = position_ids.astype(np.float64)[:, :, None] * freqs[None, None, :]

    cos = np.cos(angles)
    sin = np.sin(angles)

    x1 = x[..., :, :half_dim]  # primera mitad
    x2 = x[..., :, half_dim:]  # segunda mitad

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    result = np.concatenate([x1_rot, x2_rot], axis=-1)
    return result


def stable_softmax(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Softmax numéricamente estable.

    Parameters
    ----------
    x : np.ndarray
        Entrada.
    axis : int
        Eje sobre el cual aplicar softmax.

    Returns
    -------
    np.ndarray con probabilidades que suman 1.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def causal_mask(
    seq_len: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Genera una máscara causal (lower triangular) para atención.

    Returns
    -------
    np.ndarray de forma (seq_len, seq_len) con -inf en la parte superior
    derecha y 0 en la parte inferior izquierda.
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=dtype))
    mask = np.where(mask == 0, -1e9, 0.0)
    return mask


# ---------------------------------------------------------------------------
# Carga de pesos desde .safetensors
# ---------------------------------------------------------------------------

def load_weights_from_safetensors(
    safetensors_path: str | Path,
) -> ModelWeights:
    """
    Carga todos los pesos del modelo desde un archivo .safetensors.

    Returns
    -------
    ModelWeights con todos los tensores cargados.
    """
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    weights = ModelWeights(
        embed_tokens=np.empty((0, 0)),
        lm_head=np.empty((0, 0)),
        final_norm_weight=np.empty((0,)),
    )

    with safe_open(str(safetensors_path), framework="numpy") as f:
        # Embeddings
        if "embed_tokens.weight" in f.keys():
            weights.embed_tokens = f.get_tensor("embed_tokens.weight").astype(np.float32)
        if "lm_head.weight" in f.keys():
            weights.lm_head = f.get_tensor("lm_head.weight").astype(np.float32)
        if "final_norm.weight" in f.keys():
            weights.final_norm_weight = f.get_tensor("final_norm.weight").astype(np.float32)

        # Por capa
        for layer_idx in range(NUM_LAYERS):
            prefix = f"layers.{layer_idx}."

            def get_tensor(name: str) -> np.ndarray:
                full_name = prefix + name
                if full_name in f.keys():
                    return f.get_tensor(full_name).astype(np.float32)
                return np.empty((0, 0))

            weights.layer_input_ln_weight.append(get_tensor("input_layernorm.weight"))
            weights.layer_q_proj.append(get_tensor("self_attn.q_proj.weight"))
            weights.layer_k_proj.append(get_tensor("self_attn.k_proj.weight"))
            weights.layer_v_proj.append(get_tensor("self_attn.v_proj.weight"))
            weights.layer_o_proj.append(get_tensor("self_attn.o_proj.weight"))
            weights.layer_post_attn_ln_weight.append(get_tensor("post_attention_layernorm.weight"))
            weights.layer_gate_proj.append(get_tensor("mlp.gate_proj.weight"))
            weights.layer_up_proj.append(get_tensor("mlp.up_proj.weight"))
            weights.layer_down_proj.append(get_tensor("mlp.down_proj.weight"))

    return weights


# ---------------------------------------------------------------------------
# Forward pass de una sola capa Transformer
# ---------------------------------------------------------------------------

def transformer_layer_forward(
    hidden_states: np.ndarray,
    layer_idx: int,
    weights: ModelWeights,
    position_ids: np.ndarray,
    attention_mask: Optional[np.ndarray] = None,
    kv_cache: Optional[Dict] = None,
    use_cache: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Forward pass de una sola capa Transformer.

    Parameters
    ----------
    hidden_states : np.ndarray
        Entrada de forma (batch, seq_len, hidden_size).
    layer_idx : int
        Índice de la capa.
    weights : ModelWeights
        Pesos del modelo.
    position_ids : np.ndarray
        IDs de posición para RoPE.
    attention_mask : np.ndarray opcional
        Máscara de atención.
    kv_cache : dict opcional
        Caché KV para generación.
    use_cache : bool
        Si True, guarda y retorna caché KV.

    Returns
    -------
    Tupla (salida, atención_pesos_o_None).
    """
    batch, seq_len, hidden = hidden_states.shape

    # --- Self-Attention ---
    # Pre-norm
    normed = rms_norm(hidden_states, weights.layer_input_ln_weight[layer_idx])

    # Proyecciones Q, K, V
    q = normed @ weights.layer_q_proj[layer_idx].T  # (batch, seq_len, hidden)
    k = normed @ weights.layer_k_proj[layer_idx].T
    v = normed @ weights.layer_v_proj[layer_idx].T

    # Reshape para cabezas: (batch, seq_len, num_heads, head_dim)
    q = q.reshape(batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM)
    k = k.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    v = v.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)

    # RoPE
    q = apply_rope_neox(q, position_ids)
    k = apply_rope_neox(k, position_ids)

    # Transpose para atención: (batch, num_heads, seq_len, head_dim)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # GQA: expandir K, V para coincidir con num_heads
    if NUM_KV_HEADS < NUM_ATTENTION_HEADS:
        repeats = NUM_ATTENTION_HEADS // NUM_KV_HEADS
        k = np.repeat(k, repeats, axis=1)
        v = np.repeat(v, repeats, axis=1)

    # Caché KV
    if kv_cache is not None and "k" in kv_cache and "v" in kv_cache:
        past_k = kv_cache["k"]  # (batch, num_heads, past_seq, head_dim)
        past_v = kv_cache["v"]
        k = np.concatenate([past_k, k], axis=2)
        v = np.concatenate([past_v, v], axis=2)

    if use_cache:
        kv_cache = {"k": k, "v": v}

    # Attention scores
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (batch, heads, seq, seq)

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_weights = stable_softmax(scores, axis=-1)
    attn_output = attn_weights @ v  # (batch, heads, seq, head_dim)

    # Reshape de vuelta
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden)

    # Output projection
    attn_output = attn_output @ weights.layer_o_proj[layer_idx].T

    # Residual connection
    hidden_states = hidden_states + attn_output

    # --- MLP ---
    normed = rms_norm(hidden_states, weights.layer_post_attn_ln_weight[layer_idx])

    gate = normed @ weights.layer_gate_proj[layer_idx].T  # (batch, seq, intermediate)
    up = normed @ weights.layer_up_proj[layer_idx].T
    mlp_output = swiglu(gate, up, weights.layer_down_proj[layer_idx])

    # Residual
    hidden_states = hidden_states + mlp_output

    return hidden_states, attn_weights


# ---------------------------------------------------------------------------
# Forward pass completo del modelo
# ---------------------------------------------------------------------------

def forward_pass(
    input_ids: np.ndarray,
    weights: ModelWeights,
    position_ids: Optional[np.ndarray] = None,
    attention_mask: Optional[np.ndarray] = None,
    use_cache: bool = False,
    return_attention: bool = False,
) -> ForwardPassResult:
    """
    Forward pass completo del modelo.

    Parameters
    ----------
    input_ids : np.ndarray
        IDs de tokens de forma (batch, seq_len).
    weights : ModelWeights
        Pesos del modelo.
    position_ids : np.ndarray opcional
        IDs de posición. Si None, se generan secuencialmente.
    attention_mask : np.ndarray opcional
        Máscara de atención causal.
    use_cache : bool
        Si True, usa caché KV.
    return_attention : bool
        Si True, retorna los pesos de atención.

    Returns
    -------
    ForwardPassResult con logits y estados ocultos.
    """
    import time
    start = time.time()

    batch, seq_len = input_ids.shape

    if position_ids is None:
        position_ids = np.arange(seq_len, dtype=np.float64)[None, :]

    if attention_mask is None:
        attention_mask = causal_mask(seq_len)

    # Embedding lookup
    hidden_states = weights.embed_tokens[input_ids]  # (batch, seq, hidden)

    # Forward a través de todas las capas
    all_hidden = [hidden_states.copy()]
    all_attn = []
    kv_cache = {} if use_cache else None

    for layer_idx in range(NUM_LAYERS):
        hidden_states, attn_weights = transformer_layer_forward(
            hidden_states,
            layer_idx,
            weights,
            position_ids,
            attention_mask,
            kv_cache,
            use_cache,
        )
        all_hidden.append(hidden_states.copy())
        if return_attention and attn_weights is not None:
            all_attn.append(attn_weights)

    # Norma final
    hidden_states = rms_norm(hidden_states, weights.final_norm_weight)

    # lm_head
    logits = hidden_states @ weights.lm_head.T  # (batch, seq, vocab)

    elapsed = time.time() - start

    return ForwardPassResult(
        logits=logits,
        hidden_states=all_hidden,
        attention_weights=all_attn,
        kv_cache=kv_cache,
        n_tokens=seq_len * batch,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Generación autoregresiva
# ---------------------------------------------------------------------------

def sample_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    previous_tokens: Optional[List[int]] = None,
) -> int:
    """
    Muestrea un token de los logits.

    Parameters
    ----------
    logits : np.ndarray
        Logits del siguiente token (vocab_size,).
    temperature : float
        Temperatura de muestreo (0 = greedy).
    top_k : int
        Filtrar a los k tokens más probables (0 = desactivado).
    top_p : float
        Filtrar por probabilidad acumulada (nucleus sampling).
    repetition_penalty : float
        Penalización por repetición de tokens previos.
    previous_tokens : list de int opcional
        Tokens ya generados para penalización.

    Returns
    -------
    int: ID del token muestreado.
    """
    logits = logits.copy().astype(np.float64)

    # Repetition penalty
    if repetition_penalty != 1.0 and previous_tokens:
        for token_id in set(previous_tokens):
            if token_id < len(logits):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    # Temperatura
    if temperature > 0:
        logits = logits / temperature

    # Top-k
    if top_k > 0 and top_k < len(logits):
        top_k_indices = np.argsort(logits)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_indices] = 0.0
        logits = logits + mask

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = stable_softmax(sorted_logits)
        cumulative = np.cumsum(probs)
        cutoff = np.searchsorted(cumulative, top_p)
        cutoff = max(cutoff, 1)
        top_p_indices = sorted_indices[:cutoff]
        mask = np.full_like(logits, -np.inf)
        mask[top_p_indices] = 0.0
        logits = logits + mask

    # Muestreo
    probs = stable_softmax(logits)
    if temperature == 0:
        return int(np.argmax(probs))
    else:
        return int(np.random.choice(len(probs), p=probs))


def generate(
    input_ids: np.ndarray,
    weights: ModelWeights,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
) -> GenerationResult:
    """
    Generación autoregresiva de tokens.

    Parameters
    ----------
    input_ids : np.ndarray
        Tokens de entrada (batch, seq_len). Usualmente batch=1.
    weights : ModelWeights
        Pesos del modelo.
    max_new_tokens : int
        Número máximo de tokens nuevos a generar.
    temperature : float
        Temperatura de muestreo.
    top_k : int
        Filtro top-k.
    top_p : float
        Filtro top-p (nucleus).
    repetition_penalty : float
        Penalización de repetición.
    use_cache : bool
        Si True, usa caché KV.

    Returns
    -------
    GenerationResult con tokens generados y entropías.
    """
    import time
    start = time.time()

    batch, seq_len = input_ids.shape
    assert batch == 1, "Solo se soporta batch_size=1 para generación"

    generated = input_ids[0].tolist()
    entropies: List[float] = []
    kv_cache = {} if use_cache else None

    position_ids = np.arange(seq_len, dtype=np.float64)[None, :]
    attention_mask = causal_mask(seq_len)

    for _ in range(max_new_tokens):
        # Forward pass con solo el último token (si hay caché)
        if use_cache and kv_cache is not None and "k" in kv_cache:
            # Solo procesar el último token
            current_input = np.array([[generated[-1]]], dtype=np.int64)
            current_pos = np.array([[len(generated) - 1]], dtype=np.float64)
            current_mask = None  # no se necesita máscara causal para un solo token
        else:
            current_input = np.array([generated], dtype=np.int64)
            current_pos = position_ids
            current_mask = attention_mask

        result = forward_pass(
            current_input,
            weights,
            position_ids=current_pos,
            attention_mask=current_mask,
            use_cache=use_cache,
            return_attention=False,
        )

        if use_cache:
            kv_cache = result.kv_cache

        # Obtener logits del último token
        next_logits = result.logits[0, -1]  # (vocab_size,)

        # Calcular entropía
        probs = stable_softmax(next_logits)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))
        entropies.append(entropy)

        # Muestrear siguiente token
        next_token = sample_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=generated,
        )

        generated.append(next_token)

        # Actualizar position_ids para la siguiente iteración
        if not use_cache:
            position_ids = np.arange(len(generated), dtype=np.float64)[None, :]
            attention_mask = causal_mask(len(generated))

    elapsed = time.time() - start

    return GenerationResult(
        generated_tokens=generated,
        generated_text="",  # se llena con el tokenizer
        entropies=entropies,
        tokens_per_second=len(entropies) / elapsed if elapsed > 0 else 0.0,
        n_tokens_generated=len(entropies),
    )


# ---------------------------------------------------------------------------
# Utilidades de análisis
# ---------------------------------------------------------------------------

def calculate_perplexity(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Calcula la perplexidad de un forward pass.

    Parameters
    ----------
    logits : np.ndarray
        Logits del modelo (batch, seq_len, vocab_size).
    targets : np.ndarray
        Tokens objetivo (batch, seq_len).

    Returns
    -------
    float: Perplexidad.
    """
    batch, seq_len, vocab = logits.shape

    # Cross-entropy
    log_probs = stable_softmax(logits, axis=-1)
    log_probs = np.log(np.clip(log_probs, 1e-10, 1.0))

    # Obtener log-prob de los targets
    target_log_probs = np.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1,
    ).squeeze(-1)  # (batch, seq_len)

    # Perplexidad = exp(-mean(log_prob))
    nll = -np.mean(target_log_probs)
    perplexity = math.exp(nll)

    return float(perplexity)


def get_activation_statistics(
    hidden_states: List[np.ndarray],
) -> dict:
    """
    Calcula estadísticas de activación para cada capa.

    Returns
    -------
    dict con estadísticas por capa.
    """
    stats = {}
    for i, hs in enumerate(hidden_states):
        stats[f"layer_{i}"] = {
            "mean": float(np.mean(hs)),
            "std": float(np.std(hs)),
            "min": float(np.min(hs)),
            "max": float(np.max(hs)),
            "norm": float(np.linalg.norm(hs)),
        }
    return stats


def count_model_params(
    weights: ModelWeights,
) -> dict:
    """
    Cuenta parámetros por componente del modelo.

    Returns
    -------
    dict con conteo de parámetros.
    """
    embedding_params = weights.embed_tokens.size
    lm_head_params = weights.lm_head.size
    final_norm_params = weights.final_norm_weight.size

    layer_params = 0
    per_layer_breakdown = {}
    for i in range(NUM_LAYERS):
        lp = (
            weights.layer_input_ln_weight[i].size +
            weights.layer_q_proj[i].size +
            weights.layer_k_proj[i].size +
            weights.layer_v_proj[i].size +
            weights.layer_o_proj[i].size +
            weights.layer_post_attn_ln_weight[i].size +
            weights.layer_gate_proj[i].size +
            weights.layer_up_proj[i].size +
            weights.layer_down_proj[i].size
        )
        layer_params += lp
        per_layer_breakdown[f"layer_{i}"] = lp

    total = embedding_params + lm_head_params + final_norm_params + layer_params

    return {
        "embedding": embedding_params,
        "lm_head": lm_head_params,
        "final_norm": final_norm_params,
        "all_layers": layer_params,
        "per_layer": per_layer_breakdown,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Simulación completa de un modelo aleatorio (para pruebas)
# ---------------------------------------------------------------------------

def create_random_weights(
    seed: int = 42,
) -> ModelWeights:
    """
    Crea pesos aleatorios para pruebas (sin entrenar).

    Returns
    -------
    ModelWeights con pesos inicializados aleatoriamente.
    """
    rng = np.random.RandomState(seed)

    weights = ModelWeights(
        embed_tokens=rng.randn(VOCAB_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02,
        lm_head=rng.randn(VOCAB_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02,
        final_norm_weight=np.ones(HIDDEN_SIZE, dtype=np.float32),
    )

    for _ in range(NUM_LAYERS):
        weights.layer_input_ln_weight.append(np.ones(HIDDEN_SIZE, dtype=np.float32))
        weights.layer_q_proj.append(
            rng.randn(HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02
        )
        weights.layer_k_proj.append(
            rng.randn(HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02
        )
        weights.layer_v_proj.append(
            rng.randn(HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02
        )
        weights.layer_o_proj.append(
            rng.randn(HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02
        )
        weights.layer_post_attn_ln_weight.append(np.ones(HIDDEN_SIZE, dtype=np.float32))
        weights.layer_gate_proj.append(
            rng.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02
        )
        weights.layer_up_proj.append(
            rng.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.02
        )
        weights.layer_down_proj.append(
            rng.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32) * 0.02
        )

    return weights


def quick_forward_pass(
    input_ids: List[int],
    weights_path: Optional[str | Path] = None,
) -> ForwardPassResult:
    """
    Función de conveniencia: ejecuta un forward pass rápido.

    Parameters
    ----------
    input_ids : list de int
        Tokens de entrada.
    weights_path : str | Path opcional
        Ruta al archivo .safetensors. Si None, usa pesos aleatorios.

    Returns
    -------
    ForwardPassResult con el resultado.
    """
    input_array = np.array([input_ids], dtype=np.int64)

    if weights_path is not None:
        weights = load_weights_from_safetensors(weights_path)
    else:
        weights = create_random_weights()

    return forward_pass(input_array, weights, return_attention=False)


__all__ = [
    "VOCAB_SIZE",
    "HIDDEN_SIZE",
    "NUM_LAYERS",
    "NUM_ATTENTION_HEADS",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "INTERMEDIATE_SIZE",
    "ModelWeights",
    "ForwardPassResult",
    "GenerationResult",
    "rms_norm",
    "silu",
    "swiglu",
    "apply_rope_interleaved",
    "apply_rope_neox",
    "stable_softmax",
    "causal_mask",
    "load_weights_from_safetensors",
    "transformer_layer_forward",
    "forward_pass",
    "sample_token",
    "generate",
    "calculate_perplexity",
    "get_activation_statistics",
    "count_model_params",
    "create_random_weights",
    "quick_forward_pass",
]
