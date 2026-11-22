#!/usr/bin/env python3
"""
mlp_utils.py — Utilidades de MLP (SwiGLU) y RMSNorm para el modelo 800M.

Incluye:
  - SiLU (Sigmoid Linear Unit) numéricamente estable.
  - SwiGLU: SwiGLU(x) = SiLU(W_gate @ x) * (W_up @ x).
  - RMSNorm con peso aprendible y epsilon configurable.
  - Inicialización de pesos SwiGLU con escalas controladas.
  - Compuertas lógicas suaves implementadas como neuronas SwiGLU editadas.
  - Análisis de activaciones: qué neuronas se activan para qué entradas.
  - Edición de filas específicas de gate_proj, up_proj, down_proj.

Todo en NumPy puro.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Sigmoid Linear Unit): x * sigmoid(x).

    Implementación numéricamente estable que evita overflow en exp(-x)
    para valores muy negativos de x.
    """
    x_clip = np.clip(x, -60.0, 60.0)
    return x * (1.0 / (1.0 + np.exp(-x_clip)))


def swiglu(
    gate: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """
    SwiGLU activación: SiLU(gate) * up.

    gate y up deben tener la misma forma. Cada elemento de salida es:
        out[i] = SiLU(gate[i]) * up[i]

    Parameters
    ----------
    gate : np.ndarray
        Salida de gate_proj.
    up : np.ndarray
        Salida de up_proj.

    Returns
    -------
    np.ndarray de la misma forma.
    """
    return silu(gate) * up


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Root Mean Square Layer Normalization.

    Calcula:
        rms = sqrt(mean(x^2) + eps)
        out = (x / rms) * weight

    A diferencia de LayerNorm, RMSNorm no centra (no resta la media),
    solo escala por la norma cuadrática media.

    Parameters
    ----------
    x : np.ndarray de forma (..., d)
    weight : np.ndarray de forma (d,)
    eps : float
        Constante de estabilidad numérica.

    Returns
    -------
    np.ndarray de la misma forma que x.
    """
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(var + eps)
    return x * inv_rms * weight


def swiglu_forward(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    """
    Pase completo por una capa MLP SwiGLU.

        hidden = SiLU(x @ W_gate^T) * (x @ W_up^T)
        out  = hidden @ W_down^T

    Parameters
    ----------
    x : np.ndarray de forma (seq_len, d)
    w_gate : np.ndarray de forma (d_ff, d)
    w_up : np.ndarray de forma (d_ff, d)
    w_down : np.ndarray de forma (d, d_ff)

    Returns
    -------
    np.ndarray de forma (seq_len, d).
    """
    gate = x @ w_gate.T     # (seq_len, d_ff)
    up = x @ w_up.T         # (seq_len, d_ff)
    hidden = swiglu(gate, up)  # (seq_len, d_ff)
    out = hidden @ w_down.T   # (seq_len, d)
    return out


def swiglu_forward_int8(
    x: np.ndarray,
    w_gate_q: np.ndarray,
    s_gate: np.ndarray,
    w_up_q: np.ndarray,
    s_up: np.ndarray,
    w_down_q: np.ndarray,
    s_down: np.ndarray,
) -> np.ndarray:
    """
    Pase SwiGLU con pesos INT8 desquantizados por canal.

    Los pesos se desquantizan como: w = w_q * s (donde s es el scale por fila).
    """
    gate = (x @ w_gate_q.T) * s_gate.T
    up = (x @ w_up_q.T) * s_up.T
    hidden = swiglu(gate, up)
    out = (hidden @ w_down_q.T) * s_down.T
    return out


def initialize_swiglu_weights(
    d: int,
    d_ff: int,
    seed: int = 42,
    scale_gate: float = 0.02,
    scale_up: float = 0.02,
    scale_down: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inicializa pesos de una capa SwiGLU con distribución normal escalada.

    Returns
    -------
    w_gate, w_up, w_down : arrays de formas (d_ff, d), (d_ff, d), (d, d_ff).
    """
    rng = np.random.default_rng(seed)
    w_gate = rng.standard_normal((d_ff, d), dtype=np.float32) * scale_gate
    w_up = rng.standard_normal((d_ff, d), dtype=np.float32) * scale_up
    w_down = rng.standard_normal((d, d_ff), dtype=np.float32) * scale_down
    return w_gate, w_up, w_down


def quantize_swiglu_weights(
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cuantiza pesos SwiGLU a INT8 simétrico por canal.

    Devuelve (w_q, scale) para cada una de las tres matrices.
    """
    def quantize(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_abs = np.max(np.abs(w), axis=1, keepdims=True)
        scale = np.where(max_abs > 1e-8, max_abs / 127.0, np.float32(1.0 / 127.0)).astype(np.float16)
        scale_f32 = scale.astype(np.float32)
        w_q = np.rint(np.clip(w / scale_f32, -127.0, 127.0)).astype(np.int8)
        return w_q, scale

    w_gate_q, s_gate = quantize(w_gate)
    w_up_q, s_up = quantize(w_up)
    w_down_q, s_down = quantize(w_down)

    return w_gate_q, s_gate, w_up_q, s_up, w_down_q, s_down


def logical_gate_swiglu(
    x: np.ndarray,
    gate_weight_row: np.ndarray,
    up_weight_row: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Implementa una compuerta lógica suave como una sola neurona SwiGLU.

    Dada una entrada x (vector de dimensión d), y filas específicas de
    gate_proj y up_proj, calcula:
        out = SiLU(gate_row @ x) * (up_row @ x)

    Se puede interpretar como una compuerta que se "abre" cuando la entrada
    activa simultáneamente tanto la puerta como el valor.

    Returns el valor escalar de salida.
    """
    gate_val = float(np.dot(gate_weight_row, x))
    up_val = float(np.dot(up_weight_row, x))
    return float(silu(np.array([gate_val]))[0] * up_val)


def build_analytical_swiglu_block(
    d: int,
    d_ff: int,
    feature_indices: Dict[str, int],
    gate_scale: float = 1.5,
    up_scale: float = 1.2,
    noise_scale: float = 0.01,
    seed: int = 20261122,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye una capa SwiGLU con pesos analíticos para implementar
    compuertas lógicas específicas sin entrenamiento.

    feature_indices mapea nombres de características a índices en el
    espacio de dimensión d. Se configuran neuronas específicas para
    activarse cuando ciertas combinaciones de características estén
    presentes.

    Returns
    -------
    w_gate, w_up, w_down : matrices inicializadas con reglas analíticas.
    """
    rng = np.random.default_rng(seed)
    w_gate = np.zeros((d_ff, d), dtype=np.float32)
    w_up = np.zeros((d_ff, d), dtype=np.float32)
    w_down = np.zeros((d, d_ff), dtype=np.float32)

    # Llenar el resto con ruido determinista de capacidad (orthogonal blocks)
    cap_std = 0.01 / np.sqrt(d_ff)
    for i in range(d_ff):
        w_gate[i] = rng.standard_normal(d, dtype=np.float32) * cap_std
        w_up[i] = rng.standard_normal(d, dtype=np.float32) * cap_std
        w_down[:, i] = rng.standard_normal(d, dtype=np.float32) * cap_std

    # Configurar las primeras neuronas como compuertas lógicas específicas
    # Neurona 0: detecta "def " + keyword de clase -> PY_MODE
    if "F_DEF_CLASS" in feature_indices and "PY_MODE" in feature_indices:
        w_gate[0, feature_indices["F_DEF_CLASS"]] = gate_scale
        w_gate[0, feature_indices["PY_MODE"]] = gate_scale * 0.5
        w_up[0, feature_indices["PY_MODE"]] = up_scale

    # Neurona 1: detecta "(" + identificador -> PY_EXP_PAREN
    if "F_OPEN_PAREN" in feature_indices and "PY_EXP_PAREN" in feature_indices:
        w_gate[1, feature_indices["F_OPEN_PAREN"]] = gate_scale
        w_gate[1, feature_indices["PY_EXP_PAREN"]] = gate_scale * 0.5
        w_up[1, feature_indices["PY_EXP_PAREN"]] = up_scale

    # Neurona 2: detecta ":" + indentación -> PY_EXP_INDENT
    if "F_COLON" in feature_indices and "PY_EXP_INDENT" in feature_indices:
        w_gate[2, feature_indices["F_COLON"]] = gate_scale
        w_gate[2, feature_indices["PY_EXP_INDENT"]] = gate_scale * 0.5
        w_up[2, feature_indices["PY_EXP_INDENT"]] = up_scale

    # Neurona 3: detecta "return" -> PY_EXP_RETURN
    if "F_RETURN" in feature_indices and "PY_EXP_RETURN" in feature_indices:
        w_gate[3, feature_indices["F_RETURN"]] = gate_scale
        w_up[3, feature_indices["PY_EXP_RETURN"]] = up_scale

    # Neurona 4: paso lógico 1
    if "LG_STEP1" in feature_indices and "LG_MODE" in feature_indices:
        w_gate[4, feature_indices["LG_STEP1"]] = gate_scale
        w_up[4, feature_indices["LG_MODE"]] = up_scale

    # Neurona 5: paso lógico 2
    if "LG_STEP2" in feature_indices and "LG_COMPLEXITY" in feature_indices:
        w_gate[5, feature_indices["LG_STEP2"]] = gate_scale
        w_up[5, feature_indices["LG_COMPLEXITY"]] = up_scale

    # Neurona 6: paso lógico 3
    if "LG_STEP3" in feature_indices and "F_LOGIC_CONN" in feature_indices:
        w_gate[6, feature_indices["LG_STEP3"]] = gate_scale
        w_up[6, feature_indices["F_LOGIC_CONN"]] = up_scale

    # Neurona 7: saludo conversacional
    if "F_GREETING" in feature_indices and "CV_GREETING" in feature_indices:
        w_gate[7, feature_indices["F_GREETING"]] = gate_scale
        w_up[7, feature_indices["CV_GREETING"]] = up_scale

    # Neurona 8: pregunta conversacional
    if "F_QUESTION" in feature_indices and "CV_QUESTION" in feature_indices:
        w_gate[8, feature_indices["F_QUESTION"]] = gate_scale
        w_up[8, feature_indices["CV_QUESTION"]] = up_scale

    # Neurona 9: identificación Python
    if "F_IDENT" in feature_indices and "PY_EXP_IDENT" in feature_indices:
        w_gate[9, feature_indices["F_IDENT"]] = gate_scale
        w_up[9, feature_indices["PY_EXP_IDENT"]] = up_scale

    # Neurona 10: loop "in"
    if "F_CTRL_FLOW" in feature_indices and "PY_EXP_LOOP_IN" in feature_indices:
        w_gate[10, feature_indices["F_CTRL_FLOW"]] = gate_scale
        w_up[10, feature_indices["PY_EXP_LOOP_IN"]] = up_scale

    return w_gate, w_up, w_down


def analyze_swiglu_activations(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    top_k: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Analiza qué neuronas SwiGLU se activan más para una entrada dada.

    Returns
    -------
    dict con:
        - gate_values : (d_ff,) valores pre-activación de gate
        - up_values   : (d_ff,) valores pre-activación de up
        - hidden      : (d_ff,) valores post-SiLU * up
        - top_neurons : índices de las top_k neuronas más activas
        - top_values  : valores de las top_k neuronas
    """
    gate = x @ w_gate.T   # (d_ff,)
    up = x @ w_up.T       # (d_ff,)
    hidden = swiglu(gate, up)

    top_indices = np.argsort(hidden)[-top_k:][::-1]
    top_values = hidden[top_indices]

    return {
        "gate_values": gate,
        "up_values": up,
        "hidden": hidden,
        "top_neurons": top_indices,
        "top_values": top_values,
    }


def edit_swiglu_row(
    w_down: np.ndarray,
    scale_down: np.ndarray,
    dst_row: int,
    src_col: int,
    value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Edita una entrada específica de down_proj para enrutar una neurona
    SwiGLU hacia un estado de salida particular.

    w_down[dst_row, src_col] = value

    Returns copias actualizadas de w_down y scale_down.
    """
    w_down = w_down.copy()
    scale_down = scale_down.copy()
    w_down[dst_row, src_col] = value
    # Recalcular scale para la fila afectada si es necesario
    row_max = np.max(np.abs(w_down[dst_row]))
    if row_max > 0:
        scale_down[dst_row, 0] = np.float16(row_max / 127.0)
    return w_down, scale_down


def compute_mlp_capacity(
    d: int,
    d_ff: int,
    n_active_neurons: int,
    target_std: float = 0.01,
) -> float:
    """
    Calcula la desviación estándar apropiada para los pesos de capacidad
    (no editados) de una capa MLP, preservando el flujo residual.

    La fórmula asegura que la varianza de salida sea aproximadamente 1/d.
    """
    return target_std / np.sqrt(d_ff)


__all__ = [
    "silu",
    "swiglu",
    "rms_norm",
    "swiglu_forward",
    "swiglu_forward_int8",
    "initialize_swiglu_weights",
    "quantize_swiglu_weights",
    "logical_gate_swiglu",
    "build_analytical_swiglu_block",
    "analyze_swiglu_activations",
    "edit_swiglu_row",
    "compute_mlp_capacity",
]
