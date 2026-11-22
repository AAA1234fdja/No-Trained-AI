#!/usr/bin/env python3
"""
iterative_weight_editor_fast.py — Edición iterativa rápida con evaluación basada en pesos.

Evalúa directamente los valores de lm_head y capas de atención sin forward pass completo.
Esto permite iteraciones rápidas (miles por segundo) en lugar de segundos por evaluación.

Estrategia:
  1. Empezar desde el modelo actual.
  2. Aplicar cambios pequeños en matrices específicas.
  3. Evaluar basado en valores de pesos (no en generación de texto).
  4. Mantener cambios que mejoren la puntuación de pesos.
  5. Documentar cada iteración.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as save_safetensors
except ImportError:  # pragma: no cover
    safe_open = None
    save_safetensors = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.quantization_tools import quantize_int8_per_row


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODEL_DIR = Path("/home/user/No-Trained-AI/model")
WEIGHTS_PATH = MODEL_DIR / "model.safetensors"
ITERATION_LOG = MODEL_DIR / "iterative_edit_log_fast.json"

# IDs de tokens (aproximados dentro del vocabulario de 32000)
PYTHON_TOKEN_IDS = [503, 504, 505, 506, 507, 508]  # def, return, range, for, if, import
PYTHON_TOKEN_NAMES = ["def", "return", "range", "for", "if", "import"]

ANALYSIS_TOKEN_IDS = [600, 601, 602, 603, 604]  # step, complexity, variance, compare, analyze
ANALYSIS_TOKEN_NAMES = ["step", "complexity", "variance", "compare", "analyze"]

CONV_TOKEN_IDS = [700, 701, 702, 703, 704]  # hello, hi, hey, howdy, greetings
CONV_TOKEN_NAMES = ["hello", "hi", "hey", "howdy", "greetings"]

SPACE_TOKEN_ID = 42


class FastIterativeEditor:
    """
    Editor rápido que evalúa basado en valores de pesos de lm_head.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.weights_path = model_dir / "model.safetensors"
        self.tensors: Dict[str, np.ndarray] = {}
        self.scales: Dict[str, np.ndarray] = {}
        self.iteration = 0
        self.edit_log: List[dict] = []
        self.best_score = -float("inf")
        self.best_tensors: Optional[Dict[str, np.ndarray]] = None
        self.best_scales: Optional[Dict[str, np.ndarray]] = None

        self._load_tensors()

    def _load_tensors(self) -> None:
        with safe_open(str(self.weights_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name.endswith(".weight_scale"):
                    self.scales[name] = t.copy()
                else:
                    self.tensors[name] = t.copy()
        print(f"  Cargados {len(self.tensors)} tensores, {len(self.scales)} escalas")

    def _dequantize_row(self, w_q: np.ndarray, scale: np.ndarray, row: int) -> np.ndarray:
        scale_f32 = scale.astype(np.float32)
        if scale_f32.ndim == 1:
            scale_f32 = scale_f32.reshape(-1, 1)
        return w_q[row].astype(np.float32) * scale_f32[row].reshape(-1)

    def _quantize_row(self, row_f32: np.ndarray) -> Tuple[np.ndarray, float]:
        w_q, scale = quantize_int8_per_row(row_f32.reshape(1, -1))
        return w_q[0], float(scale[0, 0])

    def edit_lm_head_element(
        self,
        token_id: int,
        col: int,
        delta: float,
        description: str = "",
    ) -> None:
        w = self.tensors["lm_head.weight"]
        scale = self.scales["lm_head.weight_scale"]
        row_f32 = self._dequantize_row(w, scale, token_id)
        old_val = float(row_f32[col])
        row_f32[col] = old_val + delta
        w_q_new, scale_new = self._quantize_row(row_f32)
        w[token_id] = w_q_new
        scale[token_id, 0] = scale_new
        self.edit_log.append({
            "iteration": self.iteration,
            "tensor": "lm_head.weight",
            "token_id": token_id,
            "col": col,
            "old_value": old_val,
            "new_value": float(row_f32[col]),
            "delta": delta,
            "description": description,
        })

    def edit_attention_rows(
        self,
        tensor_name: str,
        row_start: int,
        row_end: int,
        factor: float,
        description: str = "",
    ) -> None:
        w = self.tensors[tensor_name]
        scale = self.scales.get(tensor_name + "_scale")
        for row in range(row_start, min(row_end, w.shape[0])):
            row_f32 = self._dequantize_row(w, scale, row)
            row_f32 *= factor
            w_q_new, scale_new = self._quantize_row(row_f32)
            w[row] = w_q_new
            scale[row, 0] = scale_new
        self.edit_log.append({
            "iteration": self.iteration,
            "tensor": tensor_name,
            "rows": [row_start, row_end],
            "factor": factor,
            "description": description,
        })

    # ------------------------------------------------------------------
    # Evaluación basada en pesos (muy rápida)
    # ------------------------------------------------------------------

    def evaluate_weights(self) -> dict:
        """
        Evalúa el modelo basado en los valores actuales de los pesos.

        Retorna puntuación compuesta y detalles.
        """
        w_lm = self.tensors["lm_head.weight"]
        s_lm = self.scales["lm_head.weight_scale"]

        score = 0.0
        details: dict = {}

        # --- Python: tokens de Python deben tener valores altos en columna 0 ---
        py_values = []
        for tid, name in zip(PYTHON_TOKEN_IDS, PYTHON_TOKEN_NAMES):
            if tid < w_lm.shape[0]:
                row_f32 = self._dequantize_row(w_lm, s_lm, tid)
                py_values.append(float(row_f32[0]))
        avg_py = float(np.mean(py_values)) if py_values else 0.0
        details["avg_python_col0"] = round(avg_py, 3)
        # Puntuación: más alto es mejor, máximo 30 puntos
        score += min(30, max(0, avg_py) * 6)

        # --- Análisis: tokens de análisis en columna 1 ---
        an_values = []
        for tid, name in zip(ANALYSIS_TOKEN_IDS, ANALYSIS_TOKEN_NAMES):
            if tid < w_lm.shape[0]:
                row_f32 = self._dequantize_row(w_lm, s_lm, tid)
                an_values.append(float(row_f32[1]))
        avg_an = float(np.mean(an_values)) if an_values else 0.0
        details["avg_analysis_col1"] = round(avg_an, 3)
        score += min(30, max(0, avg_an) * 6)

        # --- Conversación mínima: tokens de conv en columna 2 deben ser bajos/negativos ---
        conv_values = []
        for tid, name in zip(CONV_TOKEN_IDS, CONV_TOKEN_NAMES):
            if tid < w_lm.shape[0]:
                row_f32 = self._dequantize_row(w_lm, s_lm, tid)
                conv_values.append(float(row_f32[2]))
        avg_conv = float(np.mean(conv_values)) if conv_values else 0.0
        details["avg_conv_col2"] = round(avg_conv, 3)
        # Puntuación: valores negativos son buenos (suprimir conversación)
        score += min(20, max(0, -avg_conv) * 4)

        # --- Espacio bare: token 42 col 0 debe ser negativo ---
        if 42 < w_lm.shape[0]:
            row_f32 = self._dequantize_row(w_lm, s_lm, 42)
            space_val = float(row_f32[0])
            details["space_token_42_col0"] = round(space_val, 3)
            score += min(10, max(0, -space_val) * 2)

        # --- Inducción: verificar que wv_blk en capa 0 no esté demasiado atenuado ---
        v_proj_name = "model.layers.0.self_attn.v_proj.weight"
        if v_proj_name in self.tensors:
            w_v = self.tensors[v_proj_name]
            s_v = self.scales.get(v_proj_name + "_scale")
            row_f32 = self._dequantize_row(w_v, s_v, 0)
            v_norm = float(np.linalg.norm(row_f32))
            details["v_proj_row0_norm"] = round(v_norm, 3)
            # Norma moderada es buena (ni 0 ni demasiado grande)
            if 0.5 <= v_norm <= 5.0:
                score += 10
            elif v_norm < 0.5:
                score += 5
            else:
                score += 2

        return {
            "score": float(score),
            "details": details,
        }

    def _restore_best(self) -> None:
        """Restaura los mejores tensores guardados."""
        if self.best_tensors is not None:
            self.tensors = {k: v.copy() for k, v in self.best_tensors.items()}
            self.scales = {k: v.copy() for k, v in self.best_scales.items()}

    def save_current(self, suffix: str = "") -> Path:
        out_path = self.model_dir / f"model_iter{self.iteration}{suffix}.safetensors"
        all_tensors: Dict[str, np.ndarray] = {}
        for name in self.tensors:
            all_tensors[name] = self.tensors[name]
            scale_name = name + "_scale"
            if scale_name in self.scales:
                all_tensors[scale_name] = self.scales[scale_name]
        metadata = {
            "format": "pt",
            "experiment": "AI_experiment_AI_800M_iterative_fast",
            "iteration": str(self.iteration),
        }
        save_safetensors(all_tensors, str(out_path), metadata=metadata)
        return out_path


# ---------------------------------------------------------------------------
# Estrategias de cambio (pequeñas y controladas)
# ---------------------------------------------------------------------------

def strategy_python_small(editor: FastIterativeEditor) -> None:
    """Cambio pequeño: +0.5 en columna 0 para tokens de Python."""
    for tid in PYTHON_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 0, 0.5, "Python +0.5 col0")


def strategy_analysis_small(editor: FastIterativeEditor) -> None:
    """Cambio pequeño: +0.5 en columna 1 para tokens de análisis."""
    for tid in ANALYSIS_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 1, 0.5, "Analysis +0.5 col1")


def strategy_conv_small(editor: FastIterativeEditor) -> None:
    """Cambio pequeño: -0.3 en columna 2 para tokens de conversación."""
    for tid in CONV_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 2, -0.3, "Conv -0.3 col2")


def strategy_space_small(editor: FastIterativeEditor) -> None:
    """Cambio pequeño: -1.0 en col 0 para token de espacio."""
    editor.edit_lm_head_element(SPACE_TOKEN_ID, 0, -1.0, "Space -1.0 col0")


def strategy_induction_small(editor: FastIterativeEditor) -> None:
    """Cambio pequeño: x0.95 en filas 0-127 de v_proj capa 0."""
    editor.edit_attention_rows(
        "model.layers.0.self_attn.v_proj.weight",
        0, 128, 0.95, "v_proj x0.95"
    )


def strategy_mlp_python(editor: FastIterativeEditor) -> None:
    """Refuerza MLP de Python en capas intermedias."""
    for layer_idx in (1, 6, 12):
        tname = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if tname in editor.tensors:
            editor.edit_attention_rows(tname, 0, 16, 1.1, f"MLP down_proj layer {layer_idx} x1.1")


# ---------------------------------------------------------------------------
# Bucle iterativo
# ---------------------------------------------------------------------------

def run_iterative_editing(max_iterations: int = 15) -> None:
    editor = FastIterativeEditor(MODEL_DIR)

    # Línea base
    print("=" * 60)
    print("ITERACIÓN 0: Línea base")
    print("=" * 60)
    baseline = editor.evaluate_weights()
    print(f"  Score: {baseline['score']:.1f}")
    print(f"  Detalles: {baseline['details']}")
    editor.best_score = baseline["score"]
    editor.best_tensors = {k: v.copy() for k, v in editor.tensors.items()}
    editor.best_scales = {k: v.copy() for k, v in editor.scales.items()}

    strategies = [
        ("Python +0.5", strategy_python_small),
        ("Analysis +0.5", strategy_analysis_small),
        ("Conv -0.3", strategy_conv_small),
        ("Space -1.0", strategy_space_small),
        ("Induction x0.95", strategy_induction_small),
        ("MLP Python x1.1", strategy_mlp_python),
    ]

    for it in range(1, max_iterations + 1):
        editor.iteration = it
        strategy_name, strategy_fn = strategies[(it - 1) % len(strategies)]

        print()
        print(f"--- Iteración {it}: {strategy_name} ---")

        # Guardar antes
        tensors_before = {k: v.copy() for k, v in editor.tensors.items()}
        scales_before = {k: v.copy() for k, v in editor.scales.items()}

        # Aplicar
        strategy_fn(editor)

        # Evaluar
        result = editor.evaluate_weights()
        print(f"  Score: {result['score']:.1f} (antes: {editor.best_score:.1f})")
        print(f"  Detalles: {result['details']}")

        improved = result["score"] > editor.best_score + 0.1

        if improved:
            print(f"  ✓ Mejora (+{result['score'] - editor.best_score:.1f}), manteniendo")
            editor.best_score = result["score"]
            editor.best_tensors = {k: v.copy() for k, v in editor.tensors.items()}
            editor.best_scales = {k: v.copy() for k, v in editor.scales.items()}
        else:
            print(f"  ✗ Sin mejora, revertiendo")
            editor.tensors = tensors_before
            editor.scales = scales_before

    # Restaurar mejor
    print()
    print("=" * 60)
    print("RESTAURANDO MEJOR MODELO")
    print("=" * 60)
    editor._restore_best()
    best_path = editor.save_current(suffix="_best")
    print(f"  Mejor modelo: {best_path}")
    print(f"  Score final: {editor.best_score:.1f}")

    # Guardar log
    with open(ITERATION_LOG, "w") as f:
        json.dump({
            "best_score": editor.best_score,
            "total_iterations": max_iterations,
            "edit_log": editor.edit_log,
        }, f, indent=2)
    print(f"  Log: {ITERATION_LOG}")

    # Reemplazar modelo original
    shutil.copy2(best_path, WEIGHTS_PATH)
    print(f"  Modelo original actualizado: {WEIGHTS_PATH}")


def main() -> None:
    print("=" * 60)
    print("EDICIÓN ITERATIVA RÁPIDA - MODELO 800M")
    print("=" * 60)
    print("Evaluación basada en valores de pesos (sin forward pass)")
    print()
    run_iterative_editing(max_iterations=15)


if __name__ == "__main__":
    main()