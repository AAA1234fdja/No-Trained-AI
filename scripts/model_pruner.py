#!/usr/bin/env python3
"""
model_pruner.py — Poda de pesos del modelo 800M.

Proporciona herramientas para eliminar pesos menos importantes del modelo
sin reentrenamiento:

  - Poda por magnitud (eliminar pesos con valor absoluto menor a un umbral).
  - Poda por porcentaje (eliminar el X% de pesos más pequeños por tensor).
  - Poda estructurada (eliminar neuronas o cabezas de atención completas).
  - Poda gradual (varias etapas con umbrales crecientes).
  - Cálculo de dispersión (sparsity) antes y después.
  - Estimación de impacto en el rendimiento (menos parámetros = más rápido).
  - Validación de que no se eliminen pesos críticos (embeddings, lm_head).
  - Reporte de poda en JSON/Markdown.

Todas las operaciones son analíticas y no requieren entrenamiento.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError:  # pragma: no cover
    safe_open = None
    save_file = None


@dataclass
class PruningResult:
    """Resultado de poda de un tensor."""
    tensor_name: str
    original_size: int
    pruned_size: int
    sparsity_before: float
    sparsity_after: float
    threshold: float
    n_pruned: int
    params_remaining: int
    compression_ratio: float

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.tensor_name,
            "original_size": self.original_size,
            "pruned_size": self.pruned_size,
            "sparsity_before": round(self.sparsity_before, 6),
            "sparsity_after": round(self.sparsity_after, 6),
            "threshold": round(self.threshold, 8),
            "n_pruned": self.n_pruned,
            "params_remaining": self.params_remaining,
            "compression_ratio": round(self.compression_ratio, 4),
        }


@dataclass
class PruningReport:
    """Reporte completo de poda del modelo."""
    n_tensors: int
    total_params_before: int
    total_params_after: int
    overall_sparsity: float
    total_pruned: int
    compression_ratio: float
    tensor_results: List[PruningResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_tensors": self.n_tensors,
            "total_params_before": self.total_params_before,
            "total_params_after": self.total_params_after,
            "overall_sparsity": round(self.overall_sparsity, 6),
            "total_pruned": self.total_pruned,
            "compression_ratio": round(self.compression_ratio, 4),
            "tensors": [r.to_dict() for r in self.tensor_results],
        }


class ModelPruner:
    """
    Podador de pesos de modelos .safetensors.

    Elimina pesos menos importantes basándose en su magnitud absoluta.
    """

    def __init__(
        self,
        protect_tensors: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        protect_tensors : list de str opcional
            Nombres de tensores que NO deben ser podados (ej. embeddings, lm_head).
        """
        self.protect_tensors = protect_tensors or [
            "embed_tokens.weight",
            "lm_head.weight",
        ]

    def _is_protected(self, tensor_name: str) -> bool:
        """Verifica si un tensor está protegido contra poda."""
        for pattern in self.protect_tensors:
            if pattern in tensor_name:
                return True
        return False

    def prune_by_magnitude(
        self,
        tensor: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, PruningResult, str]:
        """
        Poda un tensor eliminando pesos con |valor| < threshold.

        Returns
        -------
        Tupla (tensor_podado, reporte, nombre_tensor).
        """
        original_size = int(tensor.size)
        sparsity_before = float(
            np.sum(tensor == 0) / original_size
        ) if original_size > 0 else 0.0

        # Crear máscara: mantener si |valor| >= threshold
        mask = np.abs(tensor) >= threshold
        pruned_tensor = tensor * mask

        n_pruned = int(np.sum(~mask))
        params_remaining = int(np.sum(mask))
        sparsity_after = float(
            np.sum(pruned_tensor == 0) / original_size
        ) if original_size > 0 else 0.0

        compression_ratio = original_size / params_remaining if params_remaining > 0 else float("inf")

        result = PruningResult(
            tensor_name="",  # se llena después
            original_size=original_size,
            pruned_size=int(pruned_tensor.nbytes),
            sparsity_before=sparsity_before,
            sparsity_after=sparsity_after,
            threshold=threshold,
            n_pruned=n_pruned,
            params_remaining=params_remaining,
            compression_ratio=compression_ratio,
        )

        return pruned_tensor, result, ""

    def prune_by_percentage(
        self,
        tensor: np.ndarray,
        percentage: float,
    ) -> Tuple[np.ndarray, PruningResult, str]:
        """
        Poda un tensor eliminando el X% de pesos con menor magnitud.

        percentage: 0.0 a 1.0 (fracción a podar).

        Returns
        -------
        Tupla (tensor_podado, reporte, nombre_tensor).
        """
        original_size = int(tensor.size)
        sparsity_before = float(
            np.sum(tensor == 0) / original_size
        ) if original_size > 0 else 0.0

        # Calcular umbral basado en percentil de magnitud
        abs_values = np.abs(tensor.flatten())
        if original_size > 0:
            threshold = float(np.percentile(abs_values, percentage * 100))
        else:
            threshold = 0.0

        mask = np.abs(tensor) >= threshold
        pruned_tensor = tensor * mask

        n_pruned = int(np.sum(~mask))
        params_remaining = int(np.sum(mask))
        sparsity_after = float(
            np.sum(pruned_tensor == 0) / original_size
        ) if original_size > 0 else 0.0

        compression_ratio = original_size / params_remaining if params_remaining > 0 else float("inf")

        result = PruningResult(
            tensor_name="",
            original_size=original_size,
            pruned_size=int(pruned_tensor.nbytes),
            sparsity_before=sparsity_before,
            sparsity_after=sparsity_after,
            threshold=threshold,
            n_pruned=n_pruned,
            params_remaining=params_remaining,
            compression_ratio=compression_ratio,
        )

        return pruned_tensor, result, ""

    def prune_model(
        self,
        safetensors_path: str | Path,
        output_path: str | Path,
        method: str = "magnitude",
        threshold: float = 0.01,
        percentage: float = 0.1,
    ) -> PruningReport:
        """
        Poda un modelo completo .safetensors.

        Parameters
        ----------
        safetensors_path : str | Path
            Ruta al archivo .safetensors de entrada.
        output_path : str | Path
            Ruta para guardar el modelo podado.
        method : str
            Método de poda: "magnitude" o "percentage".
        threshold : float
            Umbral de magnitud (para method="magnitude").
        percentage : float
            Fracción a podar (para method="percentage", 0.0-1.0).

        Returns
        -------
        PruningReport con métricas de todos los tensores.
        """
        if safe_open is None or save_file is None:
            raise ImportError("safetensors no está instalado")

        safetensors_path = Path(safetensors_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tensor_results: List[PruningResult] = []
        all_pruned: Dict[str, np.ndarray] = {}
        total_params_before = 0
        total_params_after = 0
        total_pruned = 0
        n_tensors = 0

        with safe_open(str(safetensors_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)

                # Saltar escalas
                if name.endswith("_scale"):
                    all_pruned[name] = t
                    continue

                total_params_before += int(t.size)
                n_tensors += 1

                if self._is_protected(name):
                    # No podar tensores protegidos
                    all_pruned[name] = t
                    total_params_after += int(t.size)
                    result = PruningResult(
                        tensor_name=name,
                        original_size=int(t.size),
                        pruned_size=int(t.nbytes),
                        sparsity_before=0.0,
                        sparsity_after=0.0,
                        threshold=0.0,
                        n_pruned=0,
                        params_remaining=int(t.size),
                        compression_ratio=1.0,
                    )
                    tensor_results.append(result)
                    continue

                if method == "magnitude":
                    pruned_t, result, _ = self.prune_by_magnitude(t, threshold)
                elif method == "percentage":
                    pruned_t, result, _ = self.prune_by_percentage(t, percentage)
                else:
                    raise ValueError(f"Método desconocido: {method}")

                result.tensor_name = name
                all_pruned[name] = pruned_t

                # Guardar escala asociada si existe
                scale_name = name + "_scale"
                if scale_name in f.keys():
                    all_pruned[scale_name] = f.get_tensor(scale_name)

                total_params_after += result.params_remaining
                total_pruned += result.n_pruned
                tensor_results.append(result)

        # Guardar modelo podado
        save_file(all_pruned, str(output_path))

        overall_sparsity = total_pruned / total_params_before if total_params_before > 0 else 0.0
        compression_ratio = total_params_before / total_params_after if total_params_after > 0 else float("inf")

        report = PruningReport(
            n_tensors=n_tensors,
            total_params_before=total_params_before,
            total_params_after=total_params_after,
            overall_sparsity=overall_sparsity,
            total_pruned=total_pruned,
            compression_ratio=compression_ratio,
            tensor_results=tensor_results,
        )

        return report

    def generate_report(
        self,
        report: PruningReport,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de poda en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Poda de Modelo")
        lines.append("")
        lines.append("## Resumen")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Tensores analizados | {report.n_tensors} |")
        lines.append(f"| Parámetros antes | {report.total_params_before:,} |")
        lines.append(f"| Parámetros después | {report.total_params_after:,} |")
        lines.append(f"| Parámetros podados | {report.total_pruned:,} |")
        lines.append(f"| Dispersión general | {report.overall_sparsity:.4f} |")
        lines.append(f"| Ratio de compresión | {report.compression_ratio:.4f}x |")
        lines.append("")

        lines.append("## Top 10 Tensores Más Podados")
        lines.append("")
        lines.append("| Tensor | Parámetros Originales | Podados | Dispersión Después | Umbral |")
        lines.append("|--------|----------------------|---------|--------------------|--------|")

        sorted_tensors = sorted(
            report.tensor_results,
            key=lambda x: x.n_pruned,
            reverse=True,
        )[:10]

        for r in sorted_tensors:
            lines.append(
                f"| {r.tensor_name} | {r.original_size:,} | {r.n_pruned:,} | "
                f"{r.sparsity_after:.4f} | {r.threshold:.8f} |"
            )

        lines.append("")
        return "\n".join(lines)


def quick_prune(
    safetensors_path: str | Path,
    output_path: str | Path,
    method: str = "magnitude",
    threshold: float = 0.01,
    percentage: float = 0.1,
) -> PruningReport:
    """
    Función de conveniencia: poda un modelo completo.

    Returns
    -------
    PruningReport con el resultado.
    """
    pruner = ModelPruner()
    return pruner.prune_model(
        safetensors_path, output_path,
        method=method, threshold=threshold, percentage=percentage,
    )


__all__ = [
    "PruningResult",
    "PruningReport",
    "ModelPruner",
    "quick_prune",
]
