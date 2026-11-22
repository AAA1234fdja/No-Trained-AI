#!/usr/bin/env python3
"""
attention_utils.py — Utilidades de atención para Llama/GQA (800M).

Implementa:
  - Softmax numéricamente estable con máscara causal.
  - Expansión de cabezas KV para GQA (n_kv_heads -> n_heads).
  - Cálculo de scores de atención (q @ k^T / sqrt(head_dim)).
  - Aplicación de máscara causal (lower triangular).
  - Caché de KV (key-value cache) para decodificación autoregresiva incremental.
  - Cálculo de entropía de la distribución de atención.
  - Métricas de dispersión (sparsity) de la matriz de atención.

Todo en NumPy puro para operar sobre pesos INT8 desquantizados.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def stable_softmax(
    x: np.ndarray,
    axis: int = -1,
    clip_max: float = 60.0,
) -> np.ndarray:
    """
    Softmax numéricamente estable.

    Resta el máximo a lo largo del eje antes de exponentiar para evitar
    overflow. Los valores se clipean a [-clip_max, 0] después de restar
    el máximo para evitar underflow extremo.

    Parameters
    ----------
    x : np.ndarray
        Logits de entrada.
    axis : int
        Eje sobre el cual calcular softmax.
    clip_max : float
        Límite superior para (x - max) antes de exp.

    Returns
    -------
    np.ndarray de la misma forma que x, con valores en (0, 1) sumando 1.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = np.clip(x - x_max, -clip_max, 0.0)
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    # Evitar división por cero
    sum_exp = np.where(sum_exp == 0.0, 1.0, sum_exp)
    return exp_x / sum_exp


def causal_mask(
    seq_len: int,
    total_len: int,
    dtype: np.dtype = np.float32,
    neg_inf: float = -1e9,
) -> np.ndarray:
    """
    Construye una máscara causal (lower triangular) de forma
    (seq_len, total_len).

    Las posiciones j > i (query position i, key position j) se enmascaran
    con neg_inf para que el modelo no pueda atender al futuro.

    Parameters
    ----------
    seq_len : int
        Longitud de la secuencia de queries.
    total_len : int
        Longitud total de la secuencia de keys (incluye caché).
    dtype : np.dtype
    neg_inf : float
        Valor para posiciones enmascaradas.

    Returns
    -------
    np.ndarray de forma (seq_len, total_len).
    """
    q_pos = np.arange(seq_len)[:, None]       # (seq_len, 1)
    k_pos = np.arange(total_len)[None, :]     # (1, total_len)
    mask = np.where(k_pos <= q_pos, 0.0, neg_inf).astype(dtype)
    return mask


def attention_scores(
    q: np.ndarray,
    k: np.ndarray,
    scale: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calcula scores de atención: scores = (q @ k^T) * scale.

    Parameters
    ----------
    q : np.ndarray de forma (n_heads, seq_len, head_dim)
    k : np.ndarray de forma (n_heads, total_len, head_dim)
    scale : float
        Factor de escala, típicamente 1 / sqrt(head_dim).
    mask : np.ndarray opcional de forma (seq_len, total_len) o broadcastable.

    Returns
    -------
    np.ndarray de forma (n_heads, seq_len, total_len).
    """
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
    if mask is not None:
        scores = scores + mask[None, :, :]
    return scores


def apply_attention(
    scores: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Aplica softmax a los scores y multiplica por V.

    scores : (n_heads, seq_len, total_len)
    v      : (n_heads, total_len, head_dim)

    Returns
    -------
    np.ndarray de forma (n_heads, seq_len, head_dim).
    """
    probs = stable_softmax(scores, axis=-1)
    out = np.matmul(probs, v)
    return out


def expand_kv_heads(
    k: np.ndarray,
    v: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expande cabezas KV para GQA (Grouped-Query Attention).

    Cuando n_heads > n_kv_heads, cada cabeza KV se repite para atender
    a múltiples cabezas de query. El ratio es gqa_groups = n_heads // n_kv_heads.

    Parameters
    ----------
    k : np.ndarray de forma (total_len, n_kv_heads, head_dim)
    v : np.ndarray de forma (total_len, n_kv_heads, head_dim)
    n_heads : int
    n_kv_heads : int

    Returns
    -------
    k_exp : np.ndarray de forma (total_len, n_heads, head_dim)
    v_exp : np.ndarray de forma (total_len, n_heads, head_dim)
    """
    gqa_groups = n_heads // n_kv_heads
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads ({n_heads}) no es múltiplo de n_kv_heads ({n_kv_heads})"
        )
    k_exp = np.repeat(k, gqa_groups, axis=1)
    v_exp = np.repeat(v, gqa_groups, axis=1)
    return k_exp, v_exp


def transpose_for_attention(
    x: np.ndarray,
) -> np.ndarray:
    """
    Transpone un tensor de forma (seq_len, n_heads, head_dim) a
    (n_heads, seq_len, head_dim) para operaciones de atención por cabeza.

    Returns
    -------
    np.ndarray de forma (n_heads, seq_len, head_dim).
    """
    return np.transpose(x, (1, 0, 2))


def transpose_back(
    x: np.ndarray,
) -> np.ndarray:
    """
    Invierte transpose_for_attention: (n_heads, seq_len, head_dim) ->
    (seq_len, n_heads, head_dim), luego reshaping a (seq_len, n_heads * head_dim).
    """
    seq_len, n_heads, head_dim = x.shape
    x_t = np.transpose(x, (1, 0, 2))  # (seq_len, n_heads, head_dim)
    return x_t.reshape(seq_len, n_heads * head_dim)


class KVCache:
    """
    Caché de clave-valor para decodificación autoregresiva incremental.

    Almacena las claves y valores de todos los pasos previos para evitar
    recalcularlos en cada nuevo token generado.

    Attributes
    ----------
    n_layers : int
        Número de capas del modelo.
    n_kv_heads : int
        Número de cabezas KV por capa.
    head_dim : int
        Dimensión por cabeza.
    max_seq_len : int
        Longitud máxima esperada.
    cache : list de tuples (k_cache, v_cache) o None
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.cache: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [
            None
        ] * n_layers
        self.current_length: List[int] = [0] * n_layers

    def get(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Devuelve la caché de la capa layer_idx."""
        return self.cache[layer_idx]

    def set(
        self,
        layer_idx: int,
        k_all: np.ndarray,
        v_all: np.ndarray,
    ) -> None:
        """Almacena la caché completa de la capa layer_idx."""
        self.cache[layer_idx] = (k_all, v_all)
        self.current_length[layer_idx] = k_all.shape[0]

    def append(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Añade nuevas claves y valores a la caché de la capa y devuelve
        la caché actualizada completa.

        k_new, v_new: forma (1, n_kv_heads, head_dim) para un solo token.
        """
        prev = self.cache[layer_idx]
        if prev is None:
            k_all, v_all = k_new, v_new
        else:
            k_prev, v_prev = prev
            k_all = np.concatenate([k_prev, k_new], axis=0)
            v_all = np.concatenate([v_prev, v_new], axis=0)
        self.set(layer_idx, k_all, v_all)
        return k_all, v_all

    def length(
        self,
        layer_idx: int,
    ) -> int:
        """Longitud actual de la caché para la capa layer_idx."""
        return self.current_length[layer_idx]

    def reset(self) -> None:
        """Limpia toda la caché."""
        self.cache = [None] * self.n_layers
        self.current_length = [0] * self.n_layers

    def memory_usage_mb(self) -> float:
        """
        Estima el uso de memoria de la caché en MB.
        """
        total_bytes = 0
        for layer_idx in range(self.n_layers):
            entry = self.cache[layer_idx]
            if entry is not None:
                k, v = entry
                total_bytes += k.nbytes + v.nbytes
        return total_bytes / (1024 * 1024)


def attention_entropy(
    probs: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calcula la entropía de Shannon de las distribuciones de atención.

    H = -sum(p * log(p)) por cada query y cabeza.

    Parameters
    ----------
    probs : np.ndarray de forma (..., total_len)
    axis : int
        Eje sobre el cual calcular entropía.
    eps : float
        Para evitar log(0).

    Returns
    -------
    np.ndarray de forma (...,) con entropía en nats.
    """
    log_probs = np.log(np.clip(probs, eps, 1.0))
    return -np.sum(probs * log_probs, axis=axis)


def attention_sparsity(
    probs: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """
    Calcula la fracción de pesos de atención por debajo de un umbral.

    Útil para medir qué tan "dispersa" es la atención (muchos pesos cerca
    de cero = atención focalizada en pocos tokens).

    Returns
    -------
    float en [0, 1].
    """
    return float(np.mean(probs < threshold))


def top_k_attention(
    probs: np.ndarray,
    k: int,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve los valores y índices de los k mayores pesos de atención.

    Returns
    -------
    top_values : np.ndarray
    top_indices : np.ndarray
    """
    idx = np.argpartition(probs, -k, axis=axis)[..., -k:]
    # Ordenar los top-k de mayor a menor
    sorted_idx = np.take_along_axis(
        probs, idx, axis=axis
    )
    order = np.argsort(-sorted_idx, axis=axis)
    top_values = np.take_along_axis(sorted_idx, order, axis=axis)
    top_indices = np.take_along_axis(idx, order, axis=axis)
    return top_values, top_indices


def attention_head_diversity(
    attn_probs: np.ndarray,
) -> float:
    """
    Mide la diversidad entre cabezas de atención.

    Calcula la distancia promedio par a par (KL divergence aproximada)
    entre las distribuciones de atención de cada par de cabezas.

    Returns valor en [0, inf): mayor = cabezas más diversas.
    """
    n_heads = attn_probs.shape[0]
    if n_heads < 2:
        return 0.0

    total_div = 0.0
    count = 0
    eps = 1e-10

    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            p = attn_probs[i] + eps
            q = attn_probs[j] + eps
            p = p / p.sum(axis=-1, keepdims=True)
            q = q / q.sum(axis=-1, keepdims=True)
            kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
            total_div += float(np.mean(kl))
            count += 1

    return total_div / count if count > 0 else 0.0


def compute_attention_metrics(
    scores: np.ndarray,
    probs: np.ndarray,
    v_out: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula un conjunto completo de métricas de atención para una capa.

    Returns
    -------
    dict con claves:
        - max_score, min_score, mean_score
        - entropy_mean, entropy_std
        - sparsity_0.01, sparsity_0.05
        - head_diversity
        - output_norm_mean
    """
    return {
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "mean_score": float(np.mean(scores)),
        "entropy_mean": float(np.mean(attention_entropy(probs))),
        "entropy_std": float(np.std(attention_entropy(probs))),
        "sparsity_0.01": attention_sparsity(probs, 0.01),
        "sparsity_0.05": attention_sparsity(probs, 0.05),
        "head_diversity": attention_head_diversity(probs),
        "output_norm_mean": float(np.mean(np.linalg.norm(v_out, axis=-1))),
    }


__all__ = [
    "stable_softmax",
    "causal_mask",
    "attention_scores",
    "apply_attention",
    "expand_kv_heads",
    "transpose_for_attention",
    "transpose_back",
    "KVCache",
    "attention_entropy",
    "attention_sparsity",
    "top_k_attention",
    "attention_head_diversity",
    "compute_attention_metrics",
]
