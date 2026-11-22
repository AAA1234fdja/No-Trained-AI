#!/usr/bin/env python3
"""
edit_model_intelligent.py — Edición inteligente de pesos del modelo 800M.

Aplica ajustes analíticos al modelo .safetensors para lograr:
  1. Bueno en Python (reforzar tokens: def, return, range, for, if, import, etc.)
  2. Habla mínima (suprimir saludos, relleno conversacional innecesario)
  3. Bueno en análisis (reforzar tokens: step, complexity, variance, compare, analyze, etc.)

No hay entrenamiento. Solo edición manual de pesos .safetensors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as save_safetensors
except ImportError:  # pragma: no cover
    safe_open = None  # type: ignore
    save_safetensors = None  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.quantization_tools import quantize_int8_per_row


# ---------------------------------------------------------------------------
# IDs de tokens aproximados (dentro del vocabulario de 32000)
# ---------------------------------------------------------------------------

PYTHON_TOKEN_IDS = [503, 504, 505, 506, 507, 508, 509, 510, 511, 512,
                     513, 514, 515, 516, 517, 518, 519, 520, 521, 522,
                     523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                     533, 534, 535, 536, 537, 538, 539, 540, 541, 542,
                     543, 544, 545, 546, 547]

PYTHON_TOKEN_NAMES = [
    "def", "return", "range", "for", "if", "import", "class", "with",
    "try", "except", "lambda", "yield", "async", "await", "print", "len",
    "list", "dict", "set", "enumerate", "zip", "map", "filter", "open",
    "read", "write", "True", "False", "None", "and", "or", "not", "in",
    "is", "as", "from", "del", "global", "nonlocal", "assert", "break",
    "continue", "pass", "raise", "while",
]

ANALYSIS_TOKEN_IDS = [600, 601, 602, 603, 604, 605, 606, 607, 608, 609,
                       610, 611, 612, 613, 614, 615, 616, 617, 618, 619,
                       620, 621, 622, 623, 624, 625, 626, 627, 628, 629,
                       630]

ANALYSIS_TOKEN_NAMES = [
    "step", "complexity", "variance", "compare", "analyze", "data",
    "outlier", "mean", "median", "std", "correlation", "regression",
    "distribution", "statistic", "hypothesis", "significance", "p_value",
    "confidence", "interval", "sample", "population", "bias", "error",
    "metric", "kpi", "insight", "pattern", "trend", "anomaly",
    "correlation2", "causation",
]

CONVERSATION_TOKEN_IDS = [700, 701, 702, 703, 704, 705, 706, 707, 708,
                           709, 710, 711, 712, 713, 714, 715, 716, 717,
                           718, 719, 720, 721, 722, 723, 724, 725, 726,
                           727, 728, 729, 730]

CONVERSATION_TOKEN_NAMES = [
    "hello", "hi", "hey", "howdy", "greetings", "welcome", "please",
    "thank", "thanks", "sorry", "excuse", "can you", "could you",
    "would you", "help me", "what is", "how do", "i think", "i believe",
    "maybe", "perhaps", "probably", "definitely", "absolutely", "awesome",
    "great", "wonderful", "amazing", "fantastic", "superb", "excellent",
]


def load_model_tensors(model_dir: Path, weights_filename: str = "model.safetensors"):
    """Carga todos los tensores del modelo .safetensors."""
    weights_path = model_dir / weights_filename
    tensors: Dict[str, np.ndarray] = {}
    scales: Dict[str, np.ndarray] = {}
    dtypes: Dict[str, str] = {}
    shapes: Dict[str, Tuple[int, ...]] = {}

    with safe_open(str(weights_path), framework="numpy") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name.endswith(".weight_scale"):
                scales[name] = t.copy()
                dtypes[name] = str(t.dtype)
                shapes[name] = tuple(t.shape)
            else:
                tensors[name] = t.copy()
                dtypes[name] = str(t.dtype)
                shapes[name] = tuple(t.shape)

    return tensors, scales, dtypes, shapes


def dequantize_row(w_q: np.ndarray, scale: np.ndarray, row: int) -> np.ndarray:
    """Descuantiza una sola fila de un tensor INT8 per-row."""
    scale_f32 = scale.astype(np.float32)  # forma (R, 1) o (1, R)
    if scale_f32.ndim == 1:
        scale_f32 = scale_f32.reshape(-1, 1)
    return w_q[row].astype(np.float32) * scale_f32[row].reshape(-1)


def quantize_row(row_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cuantiza una sola fila a INT8 per-row, devuelve (w_q_row, scale_row)."""
    w_q, scale = quantize_int8_per_row(row_f32.reshape(1, -1))
    return w_q[0], scale[0, 0]


def reinforce_lm_head_column(
    tensors: Dict[str, np.ndarray],
    scales: Dict[str, np.ndarray],
    token_ids: List[int],
    feature_col: int,
    value: float,
    tensor_name: str = "lm_head.weight",
) -> int:
    """Refuerza una columna específica de lm_head para ciertos tokens."""
    w = tensors[tensor_name]
    scale = scales.get(tensor_name + "_scale")

    count = 0
    for tid in token_ids:
        if tid >= w.shape[0]:
            continue
        # Desquantizar fila, editar columna, re-cuantizar
        row_f32 = dequantize_row(w, scale, tid)
        old_val = float(row_f32[feature_col])
        row_f32[feature_col] = old_val + value
        w_q_new, scale_new = quantize_row(row_f32)
        w[tid] = w_q_new
        scale[tid, 0] = scale_new
        count += 1
        print(f"    [{tensor_name}] token {tid}: {old_val:+.3f} -> {old_val + value:+.3f}")

    return count


def suppress_lm_head_column(
    tensors: Dict[str, np.ndarray],
    scales: Dict[str, np.ndarray],
    token_ids: List[int],
    feature_col: int,
    value: float,
    tensor_name: str = "lm_head.weight",
) -> int:
    """Suprime una columna específica de lm_head para ciertos tokens."""
    return reinforce_lm_head_column(tensors, scales, token_ids, feature_col, -abs(value), tensor_name)


def edit_scalar_direct(
    tensors: Dict[str, np.ndarray],
    scales: Dict[str, np.ndarray],
    tensor_name: str,
    row: int,
    col: int,
    new_value: float,
) -> None:
    """Edita un solo elemento de un tensor (descuantiza/requantiza si es INT8)."""
    w = tensors[tensor_name]
    scale = scales.get(tensor_name + "_scale")

    if w.dtype == np.int8 and scale is not None:
        row_f32 = dequantize_row(w, scale, row)
        old_val = float(row_f32[col])
        row_f32[col] = new_value
        w_q_new, scale_new = quantize_row(row_f32)
        w[row] = w_q_new
        scale[row, 0] = scale_new
        print(f"    [{tensor_name}] [{row},{col}]: {old_val:+.3f} -> {new_value:+.3f}")
    else:
        old_val = float(w[row, col])
        w[row, col] = new_value
        print(f"    [{tensor_name}] [{row},{col}]: {old_val:+.3f} -> {new_value:+.3f}")


def scale_rows(
    tensors: Dict[str, np.ndarray],
    scales: Dict[str, np.ndarray],
    tensor_name: str,
    row_start: int,
    row_end: int,
    factor: float,
) -> None:
    """Escala un rango de filas de un tensor por un factor."""
    w = tensors[tensor_name]
    scale = scales.get(tensor_name + "_scale")

    for row in range(row_start, min(row_end, w.shape[0])):
        row_f32 = dequantize_row(w, scale, row)
        row_f32 *= factor
        w_q_new, scale_new = quantize_row(row_f32)
        w[row] = w_q_new
        scale[row, 0] = scale_new

    print(f"    [{tensor_name}] filas {row_start}-{row_end} escaladas x{factor}")


def main() -> None:
    model_dir = Path("/home/user/No-Trained-AI/model")
    weights_path = model_dir / "model.safetensors"
    output_path = model_dir / "model_edited.safetensors"
    edit_log_path = model_dir / "edit_log_intelligent.json"

    if not weights_path.exists():
        print(f"ERROR: No se encontró {weights_path}")
        sys.exit(1)

    print("=" * 60)
    print("EDICIÓN INTELIGENTE DE PESOS - MODELO 800M")
    print("=" * 60)
    print(f"Modelo fuente: {weights_path}")
    print(f"Modelo salida: {output_path}")
    print()

    tensors, scales, dtypes, shapes = load_model_tensors(model_dir)
    print(f"Tensores cargados: {len(tensors)}")
    print(f"Escalas cargadas: {len(scales)}")
    print()

    # 1. Reforzar tokens de Python en lm_head (columna 0)
    print(">>> 1. Reforzando tokens de Python en lm_head (columna 0)...")
    n_py = reinforce_lm_head_column(
        tensors, scales, PYTHON_TOKEN_IDS, feature_col=0, value=2.5
    )
    print(f"   {n_py} tokens de Python reforzados")
    print()

    # 2. Reforzar tokens de análisis en lm_head (columna 1)
    print(">>> 2. Reforzando tokens de análisis en lm_head (columna 1)...")
    n_an = reinforce_lm_head_column(
        tensors, scales, ANALYSIS_TOKEN_IDS, feature_col=1, value=2.5
    )
    print(f"   {n_an} tokens de análisis reforzados")
    print()

    # 3. Suprimir tokens de conversación innecesaria (columna 2)
    print(">>> 3. Suprimiendo conversación innecesaria en lm_head (columna 2)...")
    n_supp = suppress_lm_head_column(
        tensors, scales, CONVERSATION_TOKEN_IDS, feature_col=2, value=3.0
    )
    print(f"   {n_supp} tokens de conversación suprimidos")
    print()

    # 4. Suprimir token de espacio bare (42)
    print(">>> 4. Suprimiendo token de espacio bare (42)...")
    w = tensors["lm_head.weight"]
    scale = scales["lm_head.weight_scale"]
    row_f32 = dequantize_row(w, scale, 42)
    current = float(row_f32[0])
    edit_scalar_direct(tensors, scales, "lm_head.weight", 42, 0, current - 6.0)
    print()

    # 5. Suprimir '?' (73) para evitar loops de preguntas
    print(">>> 5. Suprimiendo '?' (73) para evitar loops...")
    row_f32 = dequantize_row(w, scale, 73)
    current = float(row_f32[0])
    edit_scalar_direct(tensors, scales, "lm_head.weight", 73, 0, current - 4.0)
    print()

    # 6. Suprimir EOS prematuro (1)
    print(">>> 6. Suprimiendo EOS prematuro (1)...")
    row_f32 = dequantize_row(w, scale, 1)
    current = float(row_f32[0])
    edit_scalar_direct(tensors, scales, "lm_head.weight", 1, 0, current - 4.0)
    print()

    # 7. Reducir wv_blk y wo_blk en capa 0 para evitar repetición de palabras
    print(">>> 7. Reduciendo wv_blk/wo_blk en capa 0 (anti-repetición)...")
    for tname in ["model.layers.0.self_attn.v_proj.weight",
                   "model.layers.0.self_attn.o_proj.weight"]:
        if tname in tensors:
            scale_rows(tensors, scales, tname, 0, 128, 0.5)
    print()

    # 8. Validar post-edición
    print(">>> Validando modelo editado...")
    nan_count = 0
    inf_count = 0
    i8_min, i8_max = 127, -128
    for name, t in tensors.items():
        if t.dtype == np.int8:
            i8_min = min(i8_min, int(t.min()))
            i8_max = max(i8_max, int(t.max()))
        if np.issubdtype(t.dtype, np.floating):
            t_f32 = t.astype(np.float32)
            nan_count += int(np.sum(np.isnan(t_f32)))
            inf_count += int(np.sum(np.isinf(t_f32)))

    for name, s in scales.items():
        s_f32 = s.astype(np.float32)
        if np.any(~np.isfinite(s_f32)):
            print(f"   ADVERTENCIA: Escala {name} tiene NaN/Inf")

    print(f"   NaN: {nan_count}, Inf: {inf_count}")
    print(f"   Rango INT8: [{i8_min}, {i8_max}]")
    print(f"   Válido: {nan_count == 0 and inf_count == 0}")
    print()

    # Guardar modelo editado
    print(">>> Guardando modelo editado...")
    all_tensors: Dict[str, np.ndarray] = {}
    for name in tensors:
        all_tensors[name] = tensors[name]
        scale_name = name + "_scale"
        if scale_name in scales:
            all_tensors[scale_name] = scales[scale_name]

    metadata = {
        "format": "pt",
        "experiment": "AI_experiment_AI_800M_intelligent_weight_edit",
        "training_steps": "0",
        "total_weight_parameters": "800260608",
    }

    save_safetensors(all_tensors, str(output_path), metadata=metadata)
    print(f"   Modelo guardado: {output_path}")
    print(f"   Tamaño: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print()

    print("=" * 60)
    print("EDICIÓN COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    main()