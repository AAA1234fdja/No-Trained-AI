#!/usr/bin/env python3
"""
iterative_weight_editor.py — Edición iterativa de pesos con evaluación continua.

Estrategia:
  1. Empezar desde el modelo actual (model.safetensors).
  2. Aplicar cambios pequeños e incrementales en matrices específicas.
  3. Después de cada cambio, evaluar con prompts clave.
  4. Si el resultado mejora, mantener el cambio; si empeora, revertir.
  5. Documentar cada iteración con métricas de entropía y calidad.

Objetivo:
  - Bueno en Python (def, return, range, for, if, import)
  - Habla mínima (menos palabras de relleno conversacional)
  - Bueno en análisis (step, complexity, variance, compare, analyze)
"""

from __future__ import annotations

import json
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
from scripts.inference_engine import MmapInt8LlamaModel
from scripts.quantization_tools import quantize_int8_per_row


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODEL_DIR = Path("/home/user/No-Trained-AI/model")
WEIGHTS_PATH = MODEL_DIR / "model.safetensors"
BACKUP_PATH = MODEL_DIR / "model.safetensors.backup_iterative"
ITERATION_LOG = MODEL_DIR / "iterative_edit_log.json"

# Prompts clave para evaluación rápida
EVAL_PROMPTS = [
    ("py_def", "def "),
    ("py_for", "for i in range("),
    ("py_return", "return "),
    ("py_import", "import "),
    ("an_complexity", "Complexity:"),
    ("an_analyze", "Analyze data:"),
    ("an_step", "Step 1:"),
    ("conv_hello", "Hello"),
    ("induction", "alpha = 42 ; beta = 99 ; alpha ="),
]


class IterativeEditor:
    """
    Editor iterativo que aplica cambios pequeños y evalúa después de cada uno.
    """

    def __init__(
        self,
        model_dir: Path,
        weights_filename: str = "model.safetensors",
    ) -> None:
        self.model_dir = model_dir
        self.weights_path = model_dir / weights_filename
        self.tensors: Dict[str, np.ndarray] = {}
        self.scales: Dict[str, np.ndarray] = {}
        self.iteration = 0
        self.edit_log: List[dict] = []
        self.best_score = -float("inf")
        self.best_tensors: Optional[Dict[str, np.ndarray]] = None
        self.best_scales: Optional[Dict[str, np.ndarray]] = None

        self._load_tensors()

    # ------------------------------------------------------------------
    # Carga / descarga
    # ------------------------------------------------------------------

    def _load_tensors(self) -> None:
        """Carga tensores del modelo actual."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"No se encontró {self.weights_path}")

        with safe_open(str(self.weights_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name.endswith(".weight_scale"):
                    self.scales[name] = t.copy()
                else:
                    self.tensors[name] = t.copy()

        print(f"  Cargados {len(self.tensors)} tensores y {len(self.scales)} escalas")

    def _dequantize_row(
        self,
        w_q: np.ndarray,
        scale: np.ndarray,
        row: int,
    ) -> np.ndarray:
        """Descuantiza una fila INT8 per-row."""
        scale_f32 = scale.astype(np.float32)
        if scale_f32.ndim == 1:
            scale_f32 = scale_f32.reshape(-1, 1)
        return w_q[row].astype(np.float32) * scale_f32[row].reshape(-1)

    def _quantize_row(self, row_f32: np.ndarray) -> Tuple[np.ndarray, float]:
        """Cuantiza una fila a INT8 per-row."""
        w_q, scale = quantize_int8_per_row(row_f32.reshape(1, -1))
        return w_q[0], float(scale[0, 0])

    def _save_current(self, suffix: str = "") -> Path:
        """Guarda el estado actual de los tensores."""
        out_path = self.model_dir / f"model_iter{self.iteration}{suffix}.safetensors"
        all_tensors: Dict[str, np.ndarray] = {}
        for name in self.tensors:
            all_tensors[name] = self.tensors[name]
            scale_name = name + "_scale"
            if scale_name in self.scales:
                all_tensors[scale_name] = self.scales[scale_name]

        metadata = {
            "format": "pt",
            "experiment": "AI_experiment_AI_800M_iterative_edit",
            "iteration": str(self.iteration),
        }
        save_safetensors(all_tensors, str(out_path), metadata=metadata)
        return out_path

    def _restore_best(self) -> None:
        """Restaura los mejores tensores guardados."""
        if self.best_tensors is not None:
            self.tensors = {k: v.copy() for k, v in self.best_tensors.items()}
            self.scales = {k: v.copy() for k, v in self.best_scales.items()}

    # ------------------------------------------------------------------
    # Edición de un solo elemento
    # ------------------------------------------------------------------

    def edit_lm_head_element(
        self,
        token_id: int,
        col: int,
        delta: float,
        description: str = "",
    ) -> None:
        """Edita un solo elemento de lm_head.weight sumando un delta."""
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
        """Escala un rango de filas de un tensor de atención por un factor."""
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
    # Evaluación
    # ------------------------------------------------------------------

    def evaluate_quick(self) -> dict:
        """
        Ejecuta evaluación rápida con los prompts clave.
        Retorna puntuación compuesta y detalles.
        """
        # Guardar modelo temporal para cargar en el motor de inferencia
        tmp_path = self._save_current(suffix="_tmp")

        try:
            model = MmapInt8LlamaModel(self.model_dir, weights_filename=f"model_iter{self.iteration}_tmp.safetensors")
        except Exception as e:
            print(f"    ERROR cargando modelo para evaluación: {e}")
            return {"score": -999, "details": {}}

        results: Dict[str, str] = {}
        entropies: List[float] = []

        for name, prompt in EVAL_PROMPTS:
            try:
                res = model.generate(
                    prompt,
                    max_new_tokens=8,
                    temperature=0.3,
                    top_k=6,
                    repetition_penalty=1.5,
                )
                results[name] = res["generated_text"]
                entropies.append(res["first_token_entropy_nats"])
            except Exception as e:
                results[name] = f"ERROR: {e}"
                entropies.append(999.0)

        # Calcular puntuación compuesta
        score = self._compute_score(results, entropies)

        # Limpiar archivo temporal
        try:
            tmp_path.unlink()
        except Exception:
            pass

        return {
            "score": score,
            "avg_entropy": float(np.mean(entropies)) if entropies else 999.0,
            "details": results,
        }

    def _compute_score(
        self,
        results: Dict[str, str],
        entropies: List[float],
    ) -> float:
        """
        Calcula una puntuación compuesta (más alto = mejor).

        Criterios:
        - Python: detectar keywords (def, return, range, for, if, import)
        - Análisis: detectar (step, complexity, analyze, data, variance)
        - Conversión mínima: penalizar saludos, gracias, etc.
        - Inducción: detectar si completa el patrón (buscar número después de alpha =)
        - Entropía: preferir entropía moderada (no colapso, no ruido)
        """
        score = 0.0
        py_keywords = {"def", "return", "range", "for", "if", "import", "class", "with"}
        an_keywords = {"step", "complexity", "analyze", "data", "variance", "compare", "mean", "median"}
        conv_keywords = {"hello", "hi", "thank", "please", "sorry", "welcome", "awesome", "great"}

        # Python (máximo 30 puntos)
        py_text = results.get("py_def", "") + " " + results.get("py_for", "") + " " + results.get("py_return", "") + " " + results.get("py_import", "")
        py_text = py_text.lower()
        py_hits = sum(1 for kw in py_keywords if kw in py_text)
        score += min(30, py_hits * 5)

        # Análisis (máximo 30 puntos)
        an_text = results.get("an_complexity", "") + " " + results.get("an_analyze", "") + " " + results.get("an_step", "")
        an_text = an_text.lower()
        an_hits = sum(1 for kw in an_keywords if kw in an_text)
        score += min(30, an_hits * 4)

        # Conversación mínima (máximo 20 puntos, penalizar si hay conversión)
        conv_text = results.get("conv_hello", "").lower()
        conv_hits = sum(1 for kw in conv_keywords if kw in conv_text)
        score += max(0, 20 - conv_hits * 5)

        # Inducción (máximo 20 puntos)
        ind_text = results.get("induction", "").lower()
        # Buscar si hay un número después de "alpha ="
        import re
        nums = re.findall(r"alpha\s*=\s*(\d+)", ind_text)
        if nums:
            score += 20
        elif "42" in ind_text or "99" in ind_text:
            score += 10

        # Entropía (máximo 10 puntos, preferir rango 3-8)
        avg_h = np.mean(entropies) if entropies else 999
        if 3.0 <= avg_h <= 8.0:
            score += 10
        elif avg_h < 3.0:
            score += 5  # Colapso, pero mejor que ruido total
        else:
            score += 2  # Demasiado ruido

        return float(score)


# ---------------------------------------------------------------------------
# Estrategias de cambio
# ---------------------------------------------------------------------------

def strategy_python_boost(editor: IterativeEditor) -> None:
    """Refuerza ligeramente tokens de Python en lm_head."""
    python_tokens = [503, 504, 505, 506, 507, 508]  # def, return, range, for, if, import
    for tid in python_tokens:
        editor.edit_lm_head_element(tid, 0, 1.0, "Python boost +1.0")


def strategy_analysis_boost(editor: IterativeEditor) -> None:
    """Refuerza ligeramente tokens de análisis en lm_head."""
    analysis_tokens = [600, 601, 602, 603, 604]  # step, complexity, variance, compare, analyze
    for tid in analysis_tokens:
        editor.edit_lm_head_element(tid, 1, 1.0, "Analysis boost +1.0")


def strategy_conv_suppress(editor: IterativeEditor) -> None:
    """Suprime ligeramente tokens de conversación."""
    conv_tokens = [700, 701, 702, 703, 704]  # hello, hi, hey, howdy, greetings
    for tid in conv_tokens:
        editor.edit_lm_head_element(tid, 2, -0.5, "Conv suppress -0.5")


def strategy_induction_boost(editor: IterativeEditor) -> None:
    """Refuerza la cabeza de inducción en capa 0."""
    editor.edit_attention_rows(
        "model.layers.0.self_attn.v_proj.weight",
        0, 128, 0.9, "Induction v_proj x0.9"
    )


def strategy_space_suppress(editor: IterativeEditor) -> None:
    """Suprime token de espacio bare."""
    editor.edit_lm_head_element(42, 0, -1.0, "Space suppress -1.0")


# ---------------------------------------------------------------------------
# Bucle iterativo principal
# ---------------------------------------------------------------------------

def run_iterative_editing(
    max_iterations: int = 10,
    min_improvement: float = 0.5,
) -> None:
    """
    Ejecuta edición iterativa: aplicar cambio -> evaluar -> mantener o revertir.

    Parameters
    ----------
    max_iterations : int
        Número máximo de iteraciones.
    min_improvement : float
        Mejora mínima para aceptar un cambio.
    """
    editor = IterativeEditor(MODEL_DIR)

    # Evaluación inicial (línea base)
    print("=" * 60)
    print("ITERACIÓN 0: Línea base (modelo actual)")
    print("=" * 60)
    baseline = editor.evaluate_quick()
    print(f"  Score base: {baseline['score']:.1f}, Entropía media: {baseline['avg_entropy']:.2f}")
    editor.best_score = baseline["score"]
    editor.best_tensors = {k: v.copy() for k, v in editor.tensors.items()}
    editor.best_scales = {k: v.copy() for k, v in editor.scales.items()}

    strategies = [
        ("Python boost", strategy_python_boost),
        ("Analysis boost", strategy_analysis_boost),
        ("Conv suppress", strategy_conv_suppress),
        ("Induction boost", strategy_induction_boost),
        ("Space suppress", strategy_space_suppress),
    ]

    for it in range(1, max_iterations + 1):
        editor.iteration = it
        strategy_name, strategy_fn = strategies[(it - 1) % len(strategies)]

        print()
        print("=" * 60)
        print(f"ITERACIÓN {it}: {strategy_name}")
        print("=" * 60)

        # Guardar estado antes del cambio
        tensors_before = {k: v.copy() for k, v in editor.tensors.items()}
        scales_before = {k: v.copy() for k, v in editor.scales.items()}

        # Aplicar cambio
        print(f"  Aplicando: {strategy_name}")
        strategy_fn(editor)

        # Evaluar
        result = editor.evaluate_quick()
        print(f"  Score: {result['score']:.1f} (antes: {editor.best_score:.1f}), "
              f"Entropía: {result['avg_entropy']:.2f}")

        improved = result["score"] > editor.best_score + min_improvement

        if improved:
            print(f"  ✓ Mejora detectada (+{result['score'] - editor.best_score:.1f}), manteniendo cambio")
            editor.best_score = result["score"]
            editor.best_tensors = {k: v.copy() for k, v in editor.tensors.items()}
            editor.best_scales = {k: v.copy() for k, v in editor.scales.items()}
        else:
            print(f"  ✗ Sin mejora o peor, revertiendo cambio")
            editor.tensors = tensors_before
            editor.scales = scales_before

    # Restaurar el mejor modelo
    print()
    print("=" * 60)
    print("RESTAURANDO MEJOR MODELO")
    print("=" * 60)
    editor._restore_best()
    best_path = editor._save_current(suffix="_best")
    print(f"  Mejor modelo guardado: {best_path}")
    print(f"  Score final: {editor.best_score:.1f}")

    # Guardar log de ediciones
    with open(ITERATION_LOG, "w") as f:
        json.dump({
            "best_score": editor.best_score,
            "total_iterations": max_iterations,
            "edit_log": editor.edit_log,
        }, f, indent=2)
    print(f"  Log guardado: {ITERATION_LOG}")

    # Reemplazar modelo original con el mejor
    import shutil
    shutil.copy2(best_path, WEIGHTS_PATH)
    print(f"  Modelo original actualizado: {WEIGHTS_PATH}")


def main() -> None:
    print("=" * 60)
    print("EDICIÓN ITERATIVA DE PESOS - MODELO 800M")
    print("=" * 60)
    print("Estrategia: cambio pequeño -> evaluar -> mantener o revertir")
    print()
    run_iterative_editing(max_iterations=10)


if __name__ == "__main__":
    main()