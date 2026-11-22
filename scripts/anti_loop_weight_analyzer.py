#!/usr/bin/env python3
"""
anti_loop_weight_analyzer.py — Analizador rápido de loops basado en pesos.

Analiza directamente los pesos del modelo para identificar:
  1. Cabezas de atención con alto auto-enfoque (propensas a loops)
  2. Filas de lm_head con alta auto-similitud (propensas a repetir tokens)
  3. Patrón de inducción que causa repetición de palabras completas

Luego prueba ediciones de pesos en esas zonas y puntúa la mejora
sin ejecutar forward pass completo (muy rápido).
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
ANALYSIS_LOG = MODEL_DIR / "anti_loop_analysis.json"

# IDs de tokens para Python y análisis
PYTHON_TOKEN_IDS = [503, 504, 505, 506, 507, 508]
ANALYSIS_TOKEN_IDS = [600, 601, 602, 603, 604]
CONV_TOKEN_IDS = [700, 701, 702, 703, 704]


class FastAntiLoopAnalyzer:
    """
    Analizador rápido que trabaja directamente sobre pesos.
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

    def _full_dequantize(self, w_q: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Descuantiza una matriz completa INT8 per-row."""
        scale_f32 = scale.astype(np.float32)
        if scale_f32.ndim == 1:
            scale_f32 = scale_f32.reshape(-1, 1)
        return w_q.astype(np.float32) * scale_f32

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
    # Análisis de atención para detectar cabezas propensas a loops
    # ------------------------------------------------------------------

    def analyze_attention_self_similarity(
        self,
        layer_idx: int,
        head_start: int,
        head_end: int,
    ) -> dict:
        """
        Analiza la auto-similitud de las cabezas de atención en una capa.
        Alta auto-similitud = la cabeza tiende a atender a sí misma = loops.

        Returns métricas de auto-similitud por cabeza.
        """
        q_name = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_name = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_name = f"model.layers.{layer_idx}.self_attn.v_proj.weight"

        results = {}

        for name in [q_name, k_name, v_name]:
            if name not in self.tensors:
                continue
            w = self.tensors[name]
            scale = self.scales.get(name + "_scale")
            w_f32 = self._full_dequantize(w, scale)

            # Para cada cabeza (filas head_start a head_end)
            head_sims = []
            for row in range(head_start, min(head_end, w_f32.shape[0])):
                row_vec = w_f32[row]
                norm = np.linalg.norm(row_vec)
                if norm > 0:
                    row_vec = row_vec / norm
                # Auto-similitud: coseno del vector consigo mismo desplazado
                # (detecta patrones repetitivos en los pesos)
                if len(row_vec) > 1:
                    shifted = np.roll(row_vec, 1)
                    sim = float(np.dot(row_vec, shifted))
                    head_sims.append(sim)
                else:
                    head_sims.append(0.0)

            results[name] = {
                "mean_self_sim": float(np.mean(head_sims)) if head_sims else 0.0,
                "max_self_sim": float(np.max(head_sims)) if head_sims else 0.0,
                "std_self_sim": float(np.std(head_sims)) if head_sims else 0.0,
            }

        return results

    def analyze_all_attention_heads(self) -> dict:
        """Analiza todas las capas de atención para encontrar cabezas propensas a loops."""
        layer_results: Dict[int, dict] = {}

        for layer_idx in range(21):  # 21 capas
            layer_sim = self.analyze_attention_self_similarity(layer_idx, 0, 128)
            layer_results[layer_idx] = layer_sim

        return layer_results

    def find_repetition_prone_heads(
        self,
        threshold: float = 0.3,
    ) -> List[dict]:
        """
        Encuentra cabezas de atención con alta auto-similitud
        (propensas a generar loops de repetición).
        """
        all_results = self.analyze_all_attention_heads()
        prone_heads: List[dict] = []

        for layer_idx, layer_sim in all_results.items():
            for tensor_name, stats in layer_sim.items():
                if stats["mean_self_sim"] > threshold:
                    prone_heads.append({
                        "layer": layer_idx,
                        "tensor": tensor_name,
                        "mean_self_sim": stats["mean_self_sim"],
                        "max_self_sim": stats["max_self_sim"],
                        "std_self_sim": stats["std_self_sim"],
                    })

        # Ordenar por mayor auto-similitud
        prone_heads.sort(key=lambda x: -x["mean_self_sim"])
        return prone_heads

    # ------------------------------------------------------------------
    # Análisis de lm_head para detectar tokens propensos a repetición
    # ------------------------------------------------------------------

    def analyze_lm_head_repetition_risk(
        self,
        token_ids: List[int],
        col: int = 0,
    ) -> dict:
        """Analiza el riesgo de repetición para tokens específicos en lm_head."""
        w = self.tensors["lm_head.weight"]
        scale = self.scales["lm_head.weight_scale"]

        risks: Dict[int, float] = {}
        for tid in token_ids:
            if tid >= w.shape[0]:
                continue
            row_f32 = self._dequantize_row(w, scale, tid)
            val = float(row_f32[col])
            # Valor positivo alto = más probable que se repita
            risks[tid] = val

        return {
            "token_risks": {str(tid): round(val, 3) for tid, val in risks.items()},
            "avg_risk": float(np.mean(list(risks.values()))) if risks else 0.0,
            "max_risk": float(np.max(list(risks.values()))) if risks else 0.0,
            "min_risk": float(np.min(list(risks.values()))) if risks else 0.0,
        }

    # ------------------------------------------------------------------
    # Evaluación rápida basada en pesos (sin forward pass)
    # ------------------------------------------------------------------

    def evaluate_weights_anti_loop(self) -> dict:
        """
        Evalúa el modelo basado en propiedades de los pesos relacionadas con loops.

        Métricas:
        1. Auto-similitud de cabezas de atención (menor = menos loops)
        2. Riesgo de repetición en lm_head para tokens Python/análisis
        3. Supresión de tokens de conversación
        4. Norma de v_proj en capa 0 (induction head)
        """
        score = 0.0
        details: dict = {}

        # --- 1. Auto-similitud de atención (menor es mejor) ---
        all_sim = self.analyze_all_attention_heads()
        total_self_sim = 0.0
        head_count = 0
        max_self_sim = 0.0

        for layer_idx, layer_sim in all_sim.items():
            for tensor_name, stats in layer_sim.items():
                total_self_sim += stats["mean_self_sim"]
                max_self_sim = max(max_self_sim, stats["max_self_sim"])
                head_count += 1

        avg_self_sim = total_self_sim / head_count if head_count > 0 else 0.0
        details["avg_attention_self_sim"] = round(avg_self_sim, 4)
        details["max_attention_self_sim"] = round(max_self_sim, 4)
        # Puntuación: menor auto-similitud es mejor (máx 25 pts)
        score += max(0, 25 - avg_self_sim * 50)

        # --- 2. Riesgo de repetición Python (debe ser moderado, no cero) ---
        py_risk = self.analyze_lm_head_repetition_risk(PYTHON_TOKEN_IDS, col=0)
        details["python_repetition_risk"] = py_risk["avg_risk"]
        # Valor ideal: entre 1.0 y 4.0 (suficiente para generar Python, no tanto para loop)
        py_val = py_risk["avg_risk"]
        if 1.0 <= py_val <= 4.0:
            score += 25
        elif py_val > 0:
            score += 10
        else:
            score += 0

        # --- 3. Riesgo de repetición Análisis ---
        an_risk = self.analyze_lm_head_repetition_risk(ANALYSIS_TOKEN_IDS, col=1)
        details["analysis_repetition_risk"] = an_risk["avg_risk"]
        an_val = an_risk["avg_risk"]
        if 1.0 <= an_val <= 4.0:
            score += 25
        elif an_val > 0:
            score += 10
        else:
            score += 0

        # --- 4. Supresión de conversación (debe ser negativo) ---
        conv_risk = self.analyze_lm_head_repetition_risk(CONV_TOKEN_IDS, col=2)
        details["conversation_repetition_risk"] = conv_risk["avg_risk"]
        conv_val = conv_risk["avg_risk"]
        if conv_val < -1.0:
            score += 15
        elif conv_val < 0:
            score += 8
        else:
            score += 0

        # --- 5. Espacio bare suprimido ---
        if 42 < self.tensors["lm_head.weight"].shape[0]:
            scale = self.scales["lm_head.weight_scale"]
            row_f32 = self._dequantize_row(
                self.tensors["lm_head.weight"], scale, 42
            )
            space_val = float(row_f32[0])
            details["space_token_val"] = round(space_val, 3)
            if space_val < -3.0:
                score += 10
            elif space_val < 0:
                score += 5

        # --- 6. Norma v_proj capa 0 (induction head, moderada es mejor) ---
        v_proj_name = "model.layers.0.self_attn.v_proj.weight"
        if v_proj_name in self.tensors:
            w_v = self.tensors[v_proj_name]
            s_v = self.scales.get(v_proj_name + "_scale")
            row_f32 = self._dequantize_row(w_v, s_v, 0)
            v_norm = float(np.linalg.norm(row_f32))
            details["v_proj_row0_norm"] = round(v_norm, 3)
            if 0.3 <= v_norm <= 1.5:
                score += 10
            elif v_norm > 0:
                score += 5

        return {
            "score": float(score),
            "details": details,
        }

    def save_current(self, suffix: str = "") -> Path:
        out_path = self.model_dir / f"model_anti_loop{self.iteration}{suffix}.safetensors"
        all_tensors: Dict[str, np.ndarray] = {}
        for name in self.tensors:
            all_tensors[name] = self.tensors[name]
            scale_name = name + "_scale"
            if scale_name in self.scales:
                all_tensors[scale_name] = self.scales[scale_name]
        metadata = {
            "format": "pt",
            "experiment": "AI_experiment_AI_800M_anti_loop",
            "iteration": str(self.iteration),
        }
        save_safetensors(all_tensors, str(out_path), metadata=metadata)
        return out_path

    def _restore_best(self) -> None:
        if self.best_tensors is not None:
            self.tensors = {k: v.copy() for k, v in self.best_tensors.items()}
            self.scales = {k: v.copy() for k, v in self.best_scales.items()}


# ---------------------------------------------------------------------------
# Estrategias de edición anti-loop
# ---------------------------------------------------------------------------

def strategy_suppress_repetition_tokens(editor: FastAntiLoopAnalyzer) -> None:
    """Suprime tokens que tienden a repetirse en lm_head."""
    # Tokens de Python: reducir col 0 si es demasiado alto
    for tid in PYTHON_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 0, -0.3, "Reduce Python repetition risk")

    # Tokens de análisis: reducir col 1
    for tid in ANALYSIS_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 1, -0.3, "Reduce Analysis repetition risk")


def strategy_boost_generation_diversity(editor: FastAntiLoopAnalyzer) -> None:
    """Aumenta la diversidad reduciendo auto-similitud en atención."""
    # Reducir ligeramente v_proj en capas con alta auto-similitud
    for layer_idx in range(0, 21, 3):  # Cada 3 capas
        tname = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        if tname in editor.tensors:
            editor.edit_attention_rows(tname, 0, 128, 0.98, f"Reduce v_proj layer {layer_idx}")


def strategy_strengthen_induction_selectivity(editor: FastAntiLoopAnalyzer) -> None:
    """Refuerza la selectividad de la cabeza de inducción para evitar repetición de palabras."""
    # La cabeza de inducción en capa 0 debe ser selectiva, no repetitiva
    editor.edit_attention_rows(
        "model.layers.0.self_attn.v_proj.weight",
        0, 64, 0.9, "Reduce induction head repetition"
    )
    editor.edit_attention_rows(
        "model.layers.0.self_attn.v_proj.weight",
        64, 128, 0.95, "Slightly reduce rest of induction head"
    )


def strategy_suppress_conv_tokens(editor: FastAntiLoopAnalyzer) -> None:
    """Suprime tokens de conversación para reducir relleno."""
    for tid in CONV_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 2, -0.4, "Suppress conversation tokens")


def strategy_balance_python_analysis(editor: FastAntiLoopAnalyzer) -> None:
    """Equilibra Python y análisis para evitar que uno domine y cause loops."""
    # Reducir Python ligeramente para dar espacio a análisis
    for tid in PYTHON_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 0, -0.2, "Balance Python vs Analysis")
    # Aumentar análisis ligeramente
    for tid in ANALYSIS_TOKEN_IDS:
        editor.edit_lm_head_element(tid, 1, +0.2, "Boost Analysis")


# ---------------------------------------------------------------------------
# Bucle iterativo anti-loop
# ---------------------------------------------------------------------------

def run_anti_loop_editing(max_iterations: int = 12) -> None:
    editor = FastAntiLoopAnalyzer(MODEL_DIR)

    # Línea base
    print("=" * 60)
    print("ANTI-LOOP: Línea base (modelo actual)")
    print("=" * 60)
    baseline = editor.evaluate_weights_anti_loop()
    print(f"  Score: {baseline['score']:.1f}")
    print(f"  Detalles: {baseline['details']}")
    editor.best_score = baseline["score"]
    editor.best_tensors = {k: v.copy() for k, v in editor.tensors.items()}
    editor.best_scales = {k: v.copy() for k, v in editor.scales.items()}

    # Encontrar cabezas propensas a loops
    print("\n>>> Buscando cabezas de atención propensas a loops...")
    prone_heads = editor.find_repetition_prone_heads(threshold=0.1)
    print(f"  Cabezas con auto-similitud > 0.1: {len(prone_heads)}")
    for h in prone_heads[:10]:
        print(f"    Capa {h['layer']}: {h['tensor']} (sim={h['mean_self_sim']:.4f})")

    strategies = [
        ("Suppress repetition tokens", strategy_suppress_repetition_tokens),
        ("Boost generation diversity", strategy_boost_generation_diversity),
        ("Strengthen induction selectivity", strategy_strengthen_induction_selectivity),
        ("Suppress conv tokens", strategy_suppress_conv_tokens),
        ("Balance Python/Analysis", strategy_balance_python_analysis),
    ]

    for it in range(1, max_iterations + 1):
        editor.iteration = it
        strategy_name, strategy_fn = strategies[(it - 1) % len(strategies)]

        print(f"\n--- Iteración {it}: {strategy_name} ---")

        # Guardar antes
        tensors_before = {k: v.copy() for k, v in editor.tensors.items()}
        scales_before = {k: v.copy() for k, v in editor.scales.items()}

        # Aplicar
        strategy_fn(editor)

        # Evaluar
        result = editor.evaluate_weights_anti_loop()
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
    print("\n" + "=" * 60)
    print("RESTAURANDO MEJOR MODELO ANTI-LOOP")
    print("=" * 60)
    editor._restore_best()
    best_path = editor.save_current(suffix="_best")
    print(f"  Mejor modelo: {best_path}")
    print(f"  Score final: {editor.best_score:.1f}")

    # Guardar log
    with open(ANALYSIS_LOG, "w") as f:
        json.dump({
            "best_score": editor.best_score,
            "total_iterations": max_iterations,
            "edit_log": editor.edit_log,
        }, f, indent=2)
    print(f"  Log: {ANALYSIS_LOG}")

    # Reemplazar modelo original
    shutil.copy2(best_path, WEIGHTS_PATH)
    print(f"  Modelo original actualizado: {WEIGHTS_PATH}")


def main() -> None:
    print("=" * 60)
    print("ANÁLISIS Y EDICIÓN ANTI-LOOP BASADO EN PESOS")
    print("=" * 60)
    print()
    run_anti_loop_editing(max_iterations=12)


if __name__ == "__main__":
    main()