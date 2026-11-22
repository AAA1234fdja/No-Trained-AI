#!/usr/bin/env python3
"""
rope_utils.py — Utilidades de Rotary Position Embedding (RoPE) para el modelo 800M.

Este módulo implementa toda la matemática necesaria para RoPE (Rotary Position
Embeddings) tal como se usa en arquitecturas tipo Llama/Qwen2. Incluye:
  - Construcción de cachés de frecuencias cos/sin.
  - Aplicación de RoPE a tensores de query y key.
  - Variantes interleaved y NeoX (half-split).
  - Cálculo de inversas de frecuencia para cabezas de atención.
  - Utilidades para depuración y verificación de propiedades ortogonales.

Todas las funciones están escritas en NumPy puro para poder operar sobre
pesos en formato INT8 desquantizados sin depender de PyTorch/TensorFlow.

Autor: AI Experiment — edición analítica de pesos, cero entrenamiento.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constantes por defecto del modelo 800M
# ---------------------------------------------------------------------------

DEFAULT_ROPE_THETA: float = 10000.0
DEFAULT_HEAD_DIM: int = 128
DEFAULT_MAX_POSITIONS: int = 2048


def build_inv_freq(
    head_dim: int,
    theta: float = DEFAULT_ROPE_THETA,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Construye el vector de frecuencias inversas inv_freq para RoPE.

    Para una dimensión de cabeza head_dim (debe ser par), las frecuencias son:

        inv_freq[j] = theta ** (-2j / head_dim),  j = 0, 1, ..., head_dim/2 - 1

    Esto produce frecuencias que decaen exponencialmente, permitiendo que
    las dimensiones de baja frecuencia capturen dependencias de largo alcance
    y las de alta frecuencia dependencias locales.

    Parameters
    ----------
    head_dim : int
        Dimensión por cabeza de atención. Debe ser par.
    theta : float, opcional
        Base de la progresión geométrica (default 10000.0).
    dtype : np.dtype, opcional
        Tipo de datos del array devuelto.

    Returns
    -------
    np.ndarray de forma (head_dim // 2,)
        Vector de frecuencias inversas.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim debe ser par, se recibió {head_dim}")
    j = np.arange(0, head_dim, 2, dtype=dtype)
    inv_freq = 1.0 / (theta ** (j / head_dim))
    return inv_freq.astype(dtype)


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = DEFAULT_ROPE_THETA,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precalcula las tablas cos/sin de RoPE para todas las posiciones.

    Para cada posición t y cada par de dimensiones (2j, 2j+1), se calcula:

        freq[t, j] = t * inv_freq[j]
        cos[t, j]  = cos(freq[t, j])
        sin[t, j]  = sin(freq[t, j])

    Estas tablas se usan después para rotar vectores de query y key sin
    tener que recomputar trigonometría en cada paso de inferencia.

    Parameters
    ----------
    max_seq_len : int
        Longitud máxima de secuencia soportada.
    head_dim : int
        Dimensión por cabeza (par).
    theta : float, opcional
        Base de frecuencias (default 10000.0).
    dtype : np.dtype, opcional
        Tipo de datos de las tablas.

    Returns
    -------
    cos : np.ndarray de forma (max_seq_len, head_dim // 2)
    sin : np.ndarray de forma (max_seq_len, head_dim // 2)
    """
    inv_freq = build_inv_freq(head_dim, theta, dtype)
    positions = np.arange(max_seq_len, dtype=dtype)
    # (max_seq_len, head_dim // 2): cada posición multiplica cada frecuencia
    freqs = np.outer(positions, inv_freq)
    cos = np.cos(freqs).astype(dtype)
    sin = np.sin(freqs).astype(dtype)
    return cos, sin


def apply_rope_interleaved(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    start_pos: int = 0,
) -> np.ndarray:
    """
    Aplica RoPE con emparejamiento interleaved (Llama style).

    Para cada par de dimensiones (2j, 2j+1) en la última dimensión de x,
    la rotación es:

        out[..., 2j]   = x[..., 2j] * cos[pos, j] - x[..., 2j+1] * sin[pos, j]
        out[..., 2j+1] = x[..., 2j] * sin[pos, j] + x[..., 2j+1] * cos[pos, j]

    Esto equivale a rotar cada par 2D por el ángulo freq[pos, j] en el plano.

    Parameters
    ----------
    x : np.ndarray de forma (seq_len, n_heads, head_dim)
        Tensor de entrada (queries o keys).
    cos : np.ndarray de forma (max_seq_len, head_dim // 2)
    sin : np.ndarray de forma (max_seq_len, head_dim // 2)
    start_pos : int, opcional
        Posición inicial en la caché (para KV-cache incremental).

    Returns
    -------
    np.ndarray de la misma forma que x, con RoPE aplicado.
    """
    seq_len, n_heads, head_dim = x.shape
    half_dim = head_dim // 2

    # Tomar el segmento relevante de las tablas cos/sin
    c = cos[start_pos : start_pos + seq_len][:, None, :]  # (seq, 1, half_dim)
    s = sin[start_pos : start_pos + seq_len][:, None, :]

    x_even = x[..., 0::2]  # (seq, heads, half_dim)
    x_odd = x[..., 1::2]   # (seq, heads, half_dim)

    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c

    out = np.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def apply_rope_neox(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    start_pos: int = 0,
) -> np.ndarray:
    """
    Aplica RoPE con esquema NeoX / GPT-NeoX (half-split).

    En lugar de emparejar (2j, 2j+1), NeoX divide la dimensión en dos mitades
    y rota la primera mitad contra la segunda:

        out[..., :half]   = x[..., :half] * cos - x[..., half:] * sin
        out[..., half:]   = x[..., :half] * sin + x[..., half:] * cos

    Parameters
    ----------
    x : np.ndarray de forma (seq_len, n_heads, head_dim)
    cos, sin : tablas de RoPE
    start_pos : int

    Returns
    -------
    np.ndarray de la misma forma que x.
    """
    seq_len, n_heads, head_dim = x.shape
    half_dim = head_dim // 2

    c = cos[start_pos : start_pos + seq_len][:, None, :]
    s = sin[start_pos : start_pos + seq_len][:, None, :]

    x_first = x[..., :half_dim]
    x_second = x[..., half_dim:]

    out_first = x_first * c - x_second * s
    out_second = x_first * s + x_second * c

    out = np.empty_like(x)
    out[..., :half_dim] = out_first
    out[..., half_dim:] = out_second
    return out


def rotate_half(x: np.ndarray) -> np.ndarray:
    """
    Operación auxiliar: rota la mitad del último eje.

        rotate_half(x)[..., :half] = -x[..., half:]
        rotate_half(x)[..., half:] =  x[..., :half]

    Usada internamente por algunas implementaciones de RoPE.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope_with_rotate_half(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    start_pos: int = 0,
) -> np.ndarray:
    """
    Aplica RoPE usando la identidad:

        rope(x) = x * cos + rotate_half(x) * sin

    Esta es una forma equivalente y a menudo más eficiente de aplicar RoPE
    cuando se precalculan cos y sin por posición.
    """
    seq_len, n_heads, head_dim = x.shape
    c = cos[start_pos : start_pos + seq_len][:, None, :]
    s = sin[start_pos : start_pos + seq_len][:, None, :]

    return x * c + rotate_half(x) * s


def compute_rope_angles(
    positions: np.ndarray,
    inv_freq: np.ndarray,
) -> np.ndarray:
    """
    Calcula los ángulos de rotación freq[pos, j] = pos * inv_freq[j].

    Útil para inspección visual y depuración de qué frecuencias se activan
    en cada posición de la secuencia.
    """
    return np.outer(positions, inv_freq)


def verify_rope_orthogonality(
    head_dim: int,
    theta: float = DEFAULT_ROPE_THETA,
    n_positions: int = 100,
) -> Dict[str, float]:
    """
    Verifica que las rotaciones de RoPE preservan la norma L2 de los vectores.

    Para cada posición t y cada vector unitario e_j en el subespacio de par
    (2j, 2j+1), la norma después de RoPE debe ser 1.0 (dentro de error
    numérico).

    Returns un diccionario con métricas de error máximo y medio.
    """
    inv_freq = build_inv_freq(head_dim, theta)
    cos, sin = build_rope_cache(n_positions, head_dim, theta)

    max_err = 0.0
    mean_err = 0.0
    count = 0

    for t in range(n_positions):
        for j in range(head_dim // 2):
            # Vector de prueba: solo el par (2j, 2j+1) activo
            x = np.zeros((1, 1, head_dim), dtype=np.float32)
            x[0, 0, 2 * j] = 1.0
            x_rot = apply_rope_interleaved(x, cos, sin, start_pos=t)
            norm_orig = np.linalg.norm(x)
            norm_rot = np.linalg.norm(x_rot)
            err = abs(norm_rot - norm_orig)
            max_err = max(max_err, err)
            mean_err += err
            count += 1

    mean_err /= count if count > 0 else 1
    return {
        "max_norm_error": float(max_err),
        "mean_norm_error": float(mean_err),
        "positions_tested": n_positions,
        "pairs_per_position": head_dim // 2,
    }


def get_rope_frequency_bands(
    head_dim: int,
    theta: float = DEFAULT_ROPE_THETA,
) -> List[Tuple[int, float, float]]:
    """
    Devuelve una lista de bandas de frecuencia con su rango de posiciones
    efectivas.

    Cada entrada es (índice_j, frecuencia, longitud_de_onda_en_tokens).
    Útil para entender qué dimensiones capturan dependencias de corto vs
    largo alcance.
    """
    inv_freq = build_inv_freq(head_dim, theta)
    bands = []
    for j, freq in enumerate(inv_freq):
        wavelength = 2.0 * math.pi / freq if freq > 0 else float("inf")
        bands.append((j, float(freq), float(wavelength)))
    return bands


def save_rope_cache(
    cos: np.ndarray,
    sin: np.ndarray,
    path: str | Path,
) -> None:
    """
    Guarda las tablas cos/sin de RoPE en un archivo NPZ para reuso.
    """
    np.savez(str(path), cos=cos, sin=sin)


def load_rope_cache(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga tablas cos/sin de RoPE desde un archivo NPZ.
    """
    data = np.load(str(path))
    return data["cos"], data["sin"]


def rope_positional_shift(
    head_dim: int,
    theta: float = DEFAULT_ROPE_THETA,
) -> np.ndarray:
    """
    Calcula el desplazamiento posicional efectivo que introduce RoPE entre
    dos posiciones consecutivas.

    Devuelve un vector de tamaño head_dim//2 con el ángulo incremental
    por par de dimensiones. Útil para analizar la "memoria posicional"
    del modelo.
    """
    positions = np.array([0, 1], dtype=np.float32)
    freqs = compute_rope_angles(positions, build_inv_freq(head_dim, theta))
    return freqs[1] - freqs[0]


__all__ = [
    "build_inv_freq",
    "build_rope_cache",
    "apply_rope_interleaved",
    "apply_rope_neox",
    "rotate_half",
    "apply_rope_with_rotate_half",
    "compute_rope_angles",
    "verify_rope_orthogonality",
    "get_rope_frequency_bands",
    "save_rope_cache",
    "load_rope_cache",
    "rope_positional_shift",
    "DEFAULT_ROPE_THETA",
    "DEFAULT_HEAD_DIM",
    "DEFAULT_MAX_POSITIONS",
]
