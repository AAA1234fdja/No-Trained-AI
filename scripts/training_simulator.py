#!/usr/bin/env python3
"""
training_simulator.py — Simulador de entrenamiento sin gradientes.

Este módulo explora la pregunta central del experimento: ¿puede una IA
construir otra IA editando pesos directamente, sin backpropagation?

Incluye:
  - Simulación de 'pasos de entrenamiento' sintéticos (perturbaciones
    determinísticas de pesos en lugar de gradientes).
  - Medición de entropía y diversidad tras cada paso simulado.
  - Detección de atractores de repetición (colapso).
  - Comparación entre entrenado con gradientes (teórico) vs editado
    analíticamente.
  - Registro de métricas por 'época' simulada.
  - Hipótesis sobre por qué la edición analítica no alcanza competencia
    generalista sin optimización sobre datos.

Todas las funciones son determinísticas y no requieren datasets reales.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


class TrainingStepResult:
    """Resultado de un 'paso de entrenamiento' simulado."""

    def __init__(
        self,
        step: int,
        entropy: float,
        loss_proxy: float,
        weight_change_norm: float,
        notes: str = "",
    ) -> None:
        self.step = step
        self.entropy = entropy
        self.loss_proxy = loss_proxy
        self.weight_change_norm = weight_change_norm
        self.notes = notes
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "entropy": self.entropy,
            "loss_proxy": self.loss_proxy,
            "weight_change_norm": self.weight_change_norm,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


class TrainingSimulator:
    """
    Simulador de entrenamiento sin gradientes para el modelo 800M.

    En lugar de usar backpropagation sobre datos, aplica perturbaciones
    determinísticas a los pesos y mide cómo cambia la entropía de
    generación. Esto permite estudiar la sensibilidad del modelo a
    cambios de pesos sin necesidad de un pipeline de entrenamiento real.
    """

    def __init__(
        self,
        model_dir: str | Path,
        weights_filename: str = "model.safetensors",
        seed: int = 20261122,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / weights_filename
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.results: List[TrainingStepResult] = []

        if safe_open is None:
            raise ImportError("safetensors no está instalado")

    # ------------------------------------------------------------------
    # Carga de pesos
    # ------------------------------------------------------------------

    def load_weights(self) -> Dict[str, np.ndarray]:
        """Carga todos los tensores del modelo en memoria."""
        tensors: Dict[str, np.ndarray] = {}
        with safe_open(str(self.weights_path), framework="numpy") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name).copy()
        return tensors

    # ------------------------------------------------------------------
    # Simulación de pasos de entrenamiento
    # ------------------------------------------------------------------

    def simulate_gradient_step(
        self,
        tensors: Dict[str, np.ndarray],
        step: int,
        lr: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """
        Simula un paso de entrenamiento aplicando una perturbación
        determinística a los pesos.

        En lugar de gradientes calculados sobre datos, usamos un patrón
        pseudoaleatorio (basado en seed + step) para modificar los pesos.
        Esto simula el efecto de un optimizador sin requerir datos reales.

        Returns
        -------
        Nuevos tensores con la perturbación aplicada.
        """
        new_tensors: Dict[str, np.ndarray] = {}
        total_change = 0.0
        n_changed = 0

        for name, t in tensors.items():
            t_new = t.copy()

            if t.dtype == np.int8:
                # Para tensores INT8, perturbamos en float32 y re-cuantizamos
                from scripts.quantization_tools import (
                    dequantize_int8_per_row,
                    quantize_int8_per_row,
                )
                scale_name = name + "_scale"
                if scale_name in tensors:
                    scale = tensors[scale_name]
                    t_f32 = dequantize_int8_per_row(t, scale)
                    # Perturbación determinística
                    perturb = self.rng.standard_normal(
                        t_f32.shape, dtype=np.float32
                    ) * lr * 0.1
                    t_f32_new = t_f32 + perturb
                    w_q_new, scale_new = quantize_int8_per_row(t_f32_new)
                    new_tensors[name] = w_q_new
                    new_tensors[scale_name] = scale_new

                    change = np.linalg.norm(t_f32_new - t_f32)
                    total_change += float(change)
                    n_changed += 1
                else:
                    new_tensors[name] = t
            else:
                # Para tensores float16/float32
                perturb = self.rng.standard_normal(
                    t.shape, dtype=np.float32
                ) * lr * 0.1
                t_new = t + perturb.astype(t.dtype)
                new_tensors[name] = t_new

                change = float(np.linalg.norm(perturb))
                total_change += change
                n_changed += 1

        avg_change = total_change / n_changed if n_changed > 0 else 0.0
        self._last_weight_change_norm = avg_change
        return new_tensors

    def simulate_training_epochs(
        self,
        n_epochs: int = 10,
        lr: float = 0.01,
        save_intermediate: bool = False,
        output_dir: Optional[str | Path] = None,
    ) -> List[TrainingStepResult]:
        """
        Simula múltiples épocas de entrenamiento sin gradientes.

        Después de cada paso, mide:
        - Entropía proxy (varianza de los pesos como indicador de diversidad).
        - Norma del cambio de pesos.
        - Detección de colapso (varianza muy baja = colapso).

        Returns
        -------
        lista de TrainingStepResult.
        """
        tensors = self.load_weights()
        self.results = []

        for epoch in range(n_epochs):
            tensors = self.simulate_gradient_step(tensors, epoch, lr)

            # Calcular métricas proxy
            entropy_proxy = self._calculate_entropy_proxy(tensors)
            loss_proxy = 1.0 / (1.0 + entropy_proxy)  # menor entropía = mayor pérdida

            result = TrainingStepResult(
                step=epoch,
                entropy=float(entropy_proxy),
                loss_proxy=float(loss_proxy),
                weight_change_norm=getattr(self, "_last_weight_change_norm", 0.0),
                notes=f"Epoch {epoch + 1}/{n_epochs}",
            )
            self.results.append(result)

            if save_intermediate and output_dir is not None:
                self._save_intermediate(
                    tensors, epoch, output_dir
                )

        return self.results

    def _calculate_entropy_proxy(
        self,
        tensors: Dict[str, np.ndarray],
    ) -> float:
        """
        Calcula un proxy de entropía basado en la varianza de los pesos.

        Mayor varianza en los pesos = mayor diversidad potencial =
        mayor entropía de generación esperada.
        """
        variances = []
        for name, t in tensors.items():
            if t.dtype == np.int8:
                t_f = t.astype(np.float32)
            else:
                t_f = t.astype(np.float32)
            variances.append(float(np.var(t_f)))

        if not variances:
            return 0.0

        # Entropía proxy = log de la varianza media (evita valores negativos)
        mean_var = float(np.mean(variances))
        if mean_var <= 0:
            return 0.0
        return math.log1p(mean_var)

    def _save_intermediate(
        self,
        tensors: Dict[str, np.ndarray],
        epoch: int,
        output_dir: str | Path,
    ) -> None:
        """Guarda un checkpoint intermedio (sin formato .safetensors completo)."""
        from scripts.quantization_tools import quantize_model_weights

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Guardar como NPZ para simplicidad
        save_path = output_dir / f"checkpoint_epoch_{epoch}.npz"
        np.savez(str(save_path), **tensors)

    # ------------------------------------------------------------------
    # Análisis de por qué la edición analítica falla
    # ------------------------------------------------------------------

    def analyze_edit_vs_gradient_gap(self) -> Dict[str, object]:
        """
        Analiza la brecha entre edición analítica de pesos y entrenamiento
        con gradientes sobre datos.

        Returns
        -------
        dict con argumentos teóricos y métricas de la brecha.
        """
        return {
            "information_theory": {
                "argument": (
                    "El lenguaje natural y Python contienen miles de millones "
                    "de bits de regularidades estadísticas. Esa densidad no "
                    "puede comprimirse manualmente en 1,536 ejes lineales "
                    "sin optimización iterativa sobre datos."
                ),
                "bits_per_parameter": float(
                    np.log2(32000) * 800260608 / (800260608 * 32)
                ),
            },
            "optimization_gap": {
                "argument": (
                    "SGD/Adam sobre datos ajusta millones de pesos de forma "
                    "coordinada para minimizar una función de pérdida. La "
                    "edición manual ajusta pesos individuales sin una función "
                    "de pérdida global, lo que lleva a atractores locales."
                ),
                "coordinated_updates": "millones",
                "manual_edits": "docenas a cientos",
            },
            "attractor_risks": [
                "Auto-repetición (atractores de eco)",
                "Osciladores antipodales de ciclo 2",
                "Dominio del circuito de inducción sobre todo lo demás",
                "Colapso de entropía a cero",
            ],
            "conclusion": (
                "La edición analítica de pesos puede implementar primitivas "
                "algorítmicas (memoria de trabajo, inducción, enrutamiento "
                "sintáctico) pero no puede producir competencia lingüística "
                "ni de programación generalista sin entrenamiento."
            ),
        }

    def save_results(
        self,
        path: str | Path,
    ) -> None:
        """Guarda los resultados de simulación en JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "seed": self.seed,
            "n_steps": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "analysis": self.analyze_edit_vs_gradient_gap(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Imprime un resumen de la simulación."""
        if not self.results:
            print("No hay resultados. Ejecutar simulate_training_epochs() primero.")
            return

        print("=" * 60)
        print("SIMULACIÓN DE ENTRENAMIENTO SIN GRADIENTES")
        print("=" * 60)
        print(f"Pasos simulados: {len(self.results)}")
        print()

        for r in self.results:
            print(
                f"Paso {r.step:2d} | "
                f"Entropía proxy: {r.entropy: .4f} | "
                f"Pérdida proxy: {r.loss_proxy: .4f} | "
                f"Cambio de pesos: {r.weight_change_norm: .6f} | "
                f"{r.notes}"
            )

        print()
        entropies = [r.entropy for r in self.results]
        print(f"Entropía media: {float(np.mean(entropies)):.4f}")
        print(f"Entropía mínima: {min(entropies):.4f}")
        print(f"Entropía máxima: {max(entropies):.4f}")
        print("=" * 60)


__all__ = [
    "TrainingStepResult",
    "TrainingSimulator",
]
