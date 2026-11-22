#!/usr/bin/env python3
"""
quantization_tools.py — Herramientas de cuantización INT8 per-canal para .safetensors.

Implementa:
  - Cuantización simétrica por canal (fila) a INT8 con escala FP16.
  - Descuantización (reconstrucción aproximada).
  - Validación de tensores quantizados (rango, NaN, Inf).
  - Cálculo de error de cuantización (MSE relativo).
  - Herramientas para editar directamente pesos quantizados sin
    desquantizar todo el modelo.
  - Conversión entre formatos FP32 <-> INT8+scale.
  - Estadísticas de uso del rango dinámico INT8.

Diseñado para trabajar con archivos .safetensors de gran tamaño
usando mapeo de memoria (mmap) cuando sea posible.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Cuantización / descuantización básica
# ---------------------------------------------------------------------------

def quantize_int8_per_row(
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cuantiza una matriz 2D a INT8 simétrico por fila (per-output-channel).

    Para cada fila r:
        s_r = max_c |W[r, c]| / 127.0
        Q[r, c] = round(clip(W[r, c] / s_r, -127, 127))

    Parameters
    ----------
    w : np.ndarray de forma (R, C), dtype float32.

    Returns
    -------
    w_q : np.ndarray de forma (R, C), dtype int8
    scale : np.ndarray de forma (R, 1), dtype float16
    """
    if w.ndim != 2:
        raise ValueError(f"Se espera matriz 2D, se recibió forma {w.shape}")
    max_abs = np.max(np.abs(w), axis=1, keepdims=True).astype(np.float32)
    # Evitar división por cero en filas nulas
    safe_max = np.where(max_abs > 1e-8, max_abs, np.float32(1.0))
    scale_f32 = (safe_max / 127.0).astype(np.float32)
    scale = scale_f32.astype(np.float16)
    w_q = np.rint(np.clip(w / scale_f32, -127.0, 127.0)).astype(np.int8)
    return w_q, scale


def dequantize_int8_per_row(
    w_q: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Reconstruye una aproximación de la matriz original desde INT8 + scale.

        W_approx[r, c] = w_q[r, c] * scale[r]
    """
    scale_f32 = scale.astype(np.float32)
    return w_q.astype(np.float32) * scale_f32.T


def quantize_tensor(
    w: np.ndarray,
    dtype: str = "I8",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Cuantiza un tensor según el dtype indicado.

    "I8"  -> INT8 per-row + escala FP16.
    "F16" -> devuelve tal cual (float16).
    "F32" -> devuelve tal cual (float32).

    Returns (data, scale_or_None).
    """
    if dtype == "I8":
        if w.dtype != np.float32:
            w = w.astype(np.float32)
        w_q, scale = quantize_int8_per_row(w)
        return w_q, scale
    elif dtype in ("F16", "F32"):
        return w, None
    else:
        raise ValueError(f"dtype no soportado: {dtype}")


def quantize_model_weights(
    weights: Dict[str, np.ndarray],
    tensor_spec: Dict[str, str],
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Cuantiza un diccionario completo de pesos según una especificación.

    tensor_spec mapea nombre de tensor -> "I8", "F16" o "F32".

    Returns
    -------
    dict con misma clave, valor (data, scale).
    """
    result = {}
    for name, arr in weights.items():
        dtype = tensor_spec.get(name, "F16")
        data, scale = quantize_tensor(arr, dtype)
        result[name] = (data, scale)
    return result


# ---------------------------------------------------------------------------
# Errores de cuantización
# ---------------------------------------------------------------------------

def quantization_error(
    w_orig: np.ndarray,
    w_q: np.ndarray,
    scale: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula métricas de error entre la matriz original y su versión
    cuantizada + descuantizada.

    Returns
    -------
    dict con:
        - mse: error cuadrático medio
        - rmse: raíz del error cuadrático medio
        - max_abs_error: error absoluto máximo
        - relative_error: rmse / std(w_orig)
        - dynamic_range_used: [min_int8, max_int8]
    """
    w_recon = dequantize_int8_per_row(w_q, scale)
    diff = w_orig - w_recon
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_abs_error = float(np.max(np.abs(diff)))
    std_orig = float(np.std(w_orig))
    relative_error = rmse / std_orig if std_orig > 0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
        "dynamic_range_used": [int(w_q.min()), int(w_q.max())],
    }


def validate_quantized_tensor(
    w_q: np.ndarray,
    scale: np.ndarray,
    name: str = "tensor",
) -> Dict[str, object]:
    """
    Valida que un tensor cuantizado esté dentro de rangos esperados.

    Checks:
        - dtype int8
        - valores en [-127, 127]
        - escala positiva y finita
        - no hay NaN o Inf en escala

    Returns
    -------
    dict con resultado de validación y métricas.
    """
    issues: List[str] = []

    if w_q.dtype != np.int8:
        issues.append(f"dtype es {w_q.dtype}, se esperaba int8")

    if w_q.min() < -127 or w_q.max() > 127:
        issues.append(f"valores fuera de rango int8: [{w_q.min()}, {w_q.max()}]")

    scale_f32 = scale.astype(np.float32)
    if np.any(scale_f32 <= 0):
        issues.append("escala contiene valores <= 0")
    if np.any(~np.isfinite(scale_f32)):
        issues.append("escala contiene NaN o Inf")
    if np.any(scale_f32 > 1024.0):
        issues.append("escala contiene valores muy grandes (>1024)")

    return {
        "name": name,
        "valid": len(issues) == 0,
        "issues": issues,
        "shape": list(w_q.shape),
        "dtype": str(w_q.dtype),
        "int8_min": int(w_q.min()),
        "int8_max": int(w_q.max()),
        "scale_min": float(scale_f32.min()),
        "scale_max": float(scale_f32.max()),
        "scale_mean": float(scale_f32.mean()),
    }


def validate_safetensors_file(
    path: str | Path,
) -> Dict[str, object]:
    """
    Valida un archivo .safetensors completo: verifica todos los tensores,
    reporta NaN/Inf, rango INT8 usado, y conteo de parámetros.

    Returns
    -------
    dict con resultado de validación.
    """
    from safetensors import safe_open

    issues: List[str] = []
    tensor_count = 0
    total_core_params = 0
    total_scale_params = 0
    nan_count = 0
    inf_count = 0
    i8_min = 127
    i8_max = -128
    tensor_details: List[Dict[str, object]] = []

    with safe_open(str(path), framework="numpy") as f:
        keys = list(f.keys())
        tensor_count = len(keys)

        for k in keys:
            t = f.get_tensor(k)
            detail: Dict[str, object] = {
                "name": k,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "size": int(t.size),
            }

            if k.endswith(".weight_scale"):
                total_scale_params += t.size
                t_f32 = t.astype(np.float32)
                if np.any(~np.isfinite(t_f32)):
                    nan_count += int(np.sum(~np.isfinite(t_f32)))
                    issues.append(f"{k}: escala contiene NaN/Inf")
                detail["scale_min"] = float(t_f32.min())
                detail["scale_max"] = float(t_f32.max())
            else:
                total_core_params += t.size
                if t.dtype == np.int8:
                    i8_min = min(i8_min, int(t.min()))
                    i8_max = max(i8_max, int(t.max()))
                    detail["int8_min"] = int(t.min())
                    detail["int8_max"] = int(t.max())
                if np.issubdtype(t.dtype, np.floating):
                    t_f32 = t.astype(np.float32)
                    if np.any(np.isnan(t_f32)):
                        nan_count += int(np.sum(np.isnan(t_f32)))
                        issues.append(f"{k}: contiene NaN")
                    if np.any(np.isinf(t_f32)):
                        inf_count += int(np.sum(np.isinf(t_f32)))
                        issues.append(f"{k}: contiene Inf")

            tensor_details.append(detail)

    return {
        "path": str(path),
        "valid": len(issues) == 0,
        "issues": issues,
        "tensor_count": tensor_count,
        "total_core_params": total_core_params,
        "total_scale_params": total_scale_params,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "int8_range_used": [i8_min, i8_max],
        "tensor_details": tensor_details,
    }


# ---------------------------------------------------------------------------
# Edición directa de pesos quantizados
# ---------------------------------------------------------------------------

def apply_row_edit_int8(
    w_q: np.ndarray,
    scale: np.ndarray,
    edits: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica ediciones por fila a una matriz quantizada sin desquantizar
    completamente.

    edits mapea índice de fila -> nuevo vector float32 (desquantizado).
    La función reconvierte el vector editado a INT8 + escala.

    Parameters
    ----------
    w_q : np.ndarray (R, C) int8
    scale : np.ndarray (R, 1) float16
    edits : dict fila -> vector float32 de longitud C

    Returns
    -------
    w_q_editado, scale_editado
    """
    w_q = w_q.copy()
    scale = scale.copy()

    for row_idx, new_row in edits.items():
        new_row = np.asarray(new_row, dtype=np.float32)
        if new_row.shape[0] != w_q.shape[1]:
            raise ValueError(
                f"Fila {row_idx}: longitud {new_row.shape[0]} no coincide con {w_q.shape[1]}"
            )
        # Re-cuantizar la fila editada
        row_q, row_scale = quantize_int8_per_row(new_row.reshape(1, -1))
        w_q[row_idx] = row_q[0]
        scale[row_idx, 0] = row_scale[0, 0]

    return w_q, scale


def apply_scalar_edit_int8(
    w_q: np.ndarray,
    scale: np.ndarray,
    row: int,
    col: int,
    value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Edita un solo elemento de una matriz quantizada.

    Desquantiza la fila, modifica el elemento, y re-cuantiza.
    """
    w_f32 = dequantize_int8_per_row(w_q, scale)
    w_f32[row, col] = value
    w_q_new, scale_new = quantize_int8_per_row(w_f32)
    return w_q_new, scale_new


# ---------------------------------------------------------------------------
# Estadísticas de rango dinámico
# ---------------------------------------------------------------------------

def dynamic_range_stats(
    w_q: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula estadísticas del uso del rango dinámico INT8 de una matriz.

    Returns
    -------
    dict con min, max, mean abs, std, percentil 1, 50, 99.
    """
    w_f = w_q.astype(np.float32)
    return {
        "min": float(w_f.min()),
        "max": float(w_f.max()),
        "mean_abs": float(np.mean(np.abs(w_f))),
        "std": float(np.std(w_f)),
        "p1": float(np.percentile(w_f, 1)),
        "p50": float(np.percentile(w_f, 50)),
        "p99": float(np.percentile(w_f, 99)),
        "utilization_pct": float(
            (np.max(np.abs(w_f)) / 127.0) * 100.0
        ),
    }


def model_dynamic_range_report(
    path: str | Path,
) -> Dict[str, object]:
    """
    Genera un reporte completo de uso del rango dinámico INT8 para todo
    un modelo .safetensors.
    """
    from safetensors import safe_open

    report: Dict[str, object] = {
        "path": str(path),
        "tensor_reports": [],
        "overall": {
            "i8_min": 127,
            "i8_max": -128,
            "total_int8_tensors": 0,
            "total_f16_tensors": 0,
            "total_scale_tensors": 0,
        },
    }

    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            entry: Dict[str, object] = {"name": k, "shape": list(t.shape), "dtype": str(t.dtype)}

            if t.dtype == np.int8:
                stats = dynamic_range_stats(t)
                entry["stats"] = stats
                report["overall"]["total_int8_tensors"] += 1
                report["overall"]["i8_min"] = min(
                    report["overall"]["i8_min"], int(t.min())
                )
                report["overall"]["i8_max"] = max(
                    report["overall"]["i8_max"], int(t.max())
                )
            elif t.dtype == np.float16:
                if k.endswith(".weight_scale"):
                    report["overall"]["total_scale_tensors"] += 1
                else:
                    report["overall"]["total_f16_tensors"] += 1
                entry["min"] = float(t.min())
                entry["max"] = float(t.max())

            report["tensor_reports"].append(entry)

    return report


# ---------------------------------------------------------------------------
# Utilidades de archivo .safetensors
# ---------------------------------------------------------------------------

def estimate_safetensors_size(
    tensor_spec: List[Tuple[str, Tuple[int, ...], str]],
) -> int:
    """
    Estima el tamaño en bytes de un archivo .safetensors dado un tensor_spec.

    tensor_spec: lista de (nombre, forma, dtype) donde dtype es "I8", "F16" o "F32".

    Returns tamaño total en bytes (incluye header JSON estimado).
    """
    total_bytes = 0
    for _name, shape, dtype in tensor_spec:
        n_elems = 1
        for s in shape:
            n_elems *= s
        if dtype == "I8":
            total_bytes += n_elems * 1
        elif dtype in ("F16",):
            total_bytes += n_elems * 2
        elif dtype in ("F32",):
            total_bytes += n_elems * 4
        else:
            raise ValueError(f"dtype desconocido: {dtype}")

    # Aproximación del header JSON (8 bytes longitud + JSON + padding)
    header_est = 2048  # bytes aproximados
    total_bytes += 8 + header_est
    # Padding a múltiplo de 8
    total_bytes = (total_bytes + 7) & ~7
    return total_bytes


__all__ = [
    "quantize_int8_per_row",
    "dequantize_int8_per_row",
    "quantize_tensor",
    "quantize_model_weights",
    "quantization_error",
    "validate_quantized_tensor",
    "validate_safetensors_file",
    "apply_row_edit_int8",
    "apply_scalar_edit_int8",
    "dynamic_range_stats",
    "model_dynamic_range_report",
    "estimate_safetensors_size",
]
