#!/usr/bin/env python3
"""
experiment_tracker.py — Tracker de experimentos para el modelo 800M.

Proporciona herramientas para rastrear, organizar y comparar experimentos:

  - Registro de experimentos con hiperparámetros y resultados.
  - Búsqueda y filtrado de experimentos por métricas.
  - Comparación entre experimentos (métricas, configuración, resultados).
  - Guardado y carga de experimentos en JSON.
  - Generación de tablas comparativas en Markdown.
  - Detección del mejor experimento por métrica.
  - Historial de cambios de configuración.
  - Etiquetado y categorización de experimentos.
  - Exportación de resultados para análisis externo.

Diseñado para llevar un registro de todas las iteraciones de síntesis
y edición de pesos del modelo.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExperimentRun:
    """Registro de un experimento individual."""
    experiment_id: str
    name: str
    description: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed, cancelled
    hyperparameters: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "notes": self.notes,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description", ""),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            status=data.get("status", "running"),
            hyperparameters=data.get("hyperparameters", {}),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", []),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            error_message=data.get("error_message"),
        )

    def duration(self) -> Optional[float]:
        """Devuelve la duración del experimento en segundos."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class ExperimentComparison:
    """Resultado de comparar múltiples experimentos."""
    experiment_ids: List[str]
    metric_name: str
    best_experiment_id: str
    best_value: float
    comparison_table: List[dict]

    def to_dict(self) -> dict:
        return {
            "experiment_ids": self.experiment_ids,
            "metric_name": self.metric_name,
            "best_experiment_id": self.best_experiment_id,
            "best_value": round(self.best_value, 6),
            "comparison_table": self.comparison_table,
        }


class ExperimentTracker:
    """
    Tracker de experimentos para el modelo AI 800M.

    Lleva un registro de todos los experimentos de síntesis, edición
    y evaluación del modelo, permitiendo compararlos y encontrar el mejor.
    """

    def __init__(
        self,
        storage_dir: str | Path = "/home/user/ai_experiment/experiments",
    ) -> None:
        """
        Parameters
        ----------
        storage_dir : str | Path
            Directorio para guardar los registros de experimentos.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentRun] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Carga experimentos existentes desde el almacenamiento."""
        if not self.storage_dir.exists():
            return

        for exp_file in self.storage_dir.glob("*.json"):
            try:
                with open(exp_file, "r") as f:
                    data = json.load(f)
                exp = ExperimentRun.from_dict(data)
                self.experiments[exp.experiment_id] = exp
            except Exception:
                continue  # Saltar archivos corruptos

    def create_experiment(
        self,
        name: str,
        description: str = "",
        hyperparameters: Optional[Dict[str, object]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> str:
        """
        Crea un nuevo experimento.

        Returns
        -------
        str: ID del experimento creado.
        """
        experiment_id = str(uuid.uuid4())[:8]

        exp = ExperimentRun(
            experiment_id=experiment_id,
            name=name,
            description=description,
            start_time=time.time(),
            hyperparameters=hyperparameters or {},
            tags=tags or [],
            notes=notes,
        )

        self.experiments[experiment_id] = exp
        self._save_experiment(exp)

        return experiment_id

    def update_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
    ) -> bool:
        """
        Actualiza las métricas de un experimento.

        Returns
        -------
        bool: True si se actualizó correctamente.
        """
        if experiment_id not in self.experiments:
            return False

        exp = self.experiments[experiment_id]
        exp.metrics.update(metrics)
        self._save_experiment(exp)
        return True

    def update_hyperparameters(
        self,
        experiment_id: str,
        hyperparameters: Dict[str, object],
    ) -> bool:
        """
        Actualiza los hiperparámetros de un experimento.

        Returns
        -------
        bool: True si se actualizó correctamente.
        """
        if experiment_id not in self.experiments:
            return False

        exp = self.experiments[experiment_id]
        exp.hyperparameters.update(hyperparameters)
        self._save_experiment(exp)
        return True

    def add_artifact(
        self,
        experiment_id: str,
        artifact_path: str,
    ) -> bool:
        """
        Añade un artefacto (archivo generado) a un experimento.

        Returns
        -------
        bool: True si se añadió correctamente.
        """
        if experiment_id not in self.experiments:
            return False

        exp = self.experiments[experiment_id]
        if artifact_path not in exp.artifacts:
            exp.artifacts.append(artifact_path)
            self._save_experiment(exp)
        return True

    def add_tag(
        self,
        experiment_id: str,
        tag: str,
    ) -> bool:
        """
        Añade una etiqueta a un experimento.

        Returns
        -------
        bool: True si se añadió correctamente.
        """
        if experiment_id not in self.experiments:
            return False

        exp = self.experiments[experiment_id]
        if tag not in exp.tags:
            exp.tags.append(tag)
            self._save_experiment(exp)
        return True

    def add_note(
        self,
        experiment_id: str,
        note: str,
    ) -> bool:
        """
        Añade una nota a un experimento.

        Returns
        -------
        bool: True si se añadió correctamente.
        """
        if experiment_id not in self.experiments:
            return False

        exp = self.experiments[experiment_id]
        if exp.notes:
            exp.notes += f"\n{note}"
        else:
            exp.notes = note
        self._save_experiment(exp)
        return True

    def complete_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Marca un experimento como completado.

        Returns
        -------
        bool: True si se completó correctamente.
        """
        if experiment_id not in self.experiments:
            return False

        exp = self.experiments[experiment_id]
        exp.end_time = time.time()
        exp.status = status
        exp.error_message = error_message
        self._save_experiment(exp)
        return True

    def get_experiment(
        self,
        experiment_id: str,
    ) -> Optional[ExperimentRun]:
        """Obtiene un experimento por ID."""
        return self.experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[ExperimentRun]:
        """
        Lista experimentos con filtros opcionales.

        Parameters
        ----------
        status : str opcional
            Filtrar por estado (running, completed, failed, cancelled).
        tag : str opcional
            Filtrar por etiqueta.

        Returns
        -------
        Lista de ExperimentRun.
        """
        results: List[ExperimentRun] = []
        for exp in self.experiments.values():
            if status is not None and exp.status != status:
                continue
            if tag is not None and tag not in exp.tags:
                continue
            results.append(exp)
        return sorted(results, key=lambda e: e.start_time, reverse=True)

    def search_experiments(
        self,
        query: str,
    ) -> List[ExperimentRun]:
        """
        Busca experimentos por nombre, descripción o etiquetas.

        Returns
        -------
        Lista de ExperimentRun que coinciden.
        """
        query_lower = query.lower()
        results: List[ExperimentRun] = []
        for exp in self.experiments.values():
            if (
                query_lower in exp.name.lower()
                or query_lower in exp.description.lower()
                or any(query_lower in tag.lower() for tag in exp.tags)
                or query_lower in exp.notes.lower()
            ):
                results.append(exp)
        return sorted(results, key=lambda e: e.start_time, reverse=True)

    def find_best(
        self,
        metric_name: str,
        higher_is_better: bool = True,
        status: str = "completed",
    ) -> Optional[tuple]:
        """
        Encuentra el mejor experimento por una métrica específica.

        Parameters
        ----------
        metric_name : str
            Nombre de la métrica a optimizar.
        higher_is_better : bool
            Si True, mayor valor es mejor. Si False, menor es mejor.
        status : str
            Filtrar por estado de experimento.

        Returns
        -------
        Tupla (experiment_id, valor_métrica) o None si no hay resultados.
        """
        best_exp: Optional[ExperimentRun] = None
        best_value: Optional[float] = None

        for exp in self.experiments.values():
            if exp.status != status:
                continue
            if metric_name not in exp.metrics:
                continue

            value = exp.metrics[metric_name]
            if best_value is None:
                best_exp = exp
                best_value = value
            elif higher_is_better and value > best_value:
                best_exp = exp
                best_value = value
            elif not higher_is_better and value < best_value:
                best_exp = exp
                best_value = value

        if best_exp is None:
            return None

        return (best_exp.experiment_id, best_value)

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_name: str,
    ) -> ExperimentComparison:
        """
        Compara múltiples experimentos por una métrica.

        Returns
        -------
        ExperimentComparison con la tabla comparativa.
        """
        table: List[dict] = []
        best_id = ""
        best_value = float("-inf")

        for exp_id in experiment_ids:
            exp = self.experiments.get(exp_id)
            if exp is None:
                continue

            value = exp.metrics.get(metric_name, 0.0)
            table.append({
                "experiment_id": exp_id,
                "name": exp.name,
                "metric_value": round(value, 6),
                "status": exp.status,
                "duration_sec": round(exp.duration() or 0, 2),
            })

            if value > best_value:
                best_value = value
                best_id = exp_id

        return ExperimentComparison(
            experiment_ids=experiment_ids,
            metric_name=metric_name,
            best_experiment_id=best_id,
            best_value=best_value,
            comparison_table=table,
        )

    def delete_experiment(
        self,
        experiment_id: str,
    ) -> bool:
        """
        Elimina un experimento del tracker y del almacenamiento.

        Returns
        -------
        bool: True si se eliminó correctamente.
        """
        if experiment_id not in self.experiments:
            return False

        exp_file = self.storage_dir / f"{experiment_id}.json"
        if exp_file.exists():
            exp_file.unlink()

        del self.experiments[experiment_id]
        return True

    def generate_report(
        self,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de todos los experimentos en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Experimentos")
        lines.append("")
        lines.append(f"Total de experimentos: {len(self.experiments)}")
        lines.append("")

        # Resumen por estado
        status_counts: Dict[str, int] = {}
        for exp in self.experiments.values():
            status_counts[exp.status] = status_counts.get(exp.status, 0) + 1

        lines.append("## Resumen por Estado")
        lines.append("")
        lines.append("| Estado | Cantidad |")
        lines.append("|--------|----------|")
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {count} |")
        lines.append("")

        # Tabla de experimentos
        lines.append("## Experimentos")
        lines.append("")
        lines.append("| ID | Nombre | Estado | Métricas | Etiquetas | Duración |")
        lines.append("|----|--------|--------|----------|-----------|----------|")

        for exp in sorted(
            self.experiments.values(),
            key=lambda e: e.start_time,
            reverse=True,
        )[:50]:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" for k, v in exp.metrics.items()
            ) if exp.metrics else "—"
            tags_str = ", ".join(exp.tags) if exp.tags else "—"
            duration = exp.duration()
            duration_str = f"{duration:.1f}s" if duration else "—"

            lines.append(
                f"| {exp.experiment_id} | {exp.name} | {exp.status} | "
                f"{metrics_str} | {tags_str} | {duration_str} |"
            )

        if len(self.experiments) > 50:
            lines.append(f"| ... ({len(self.experiments) - 50} más) | | | | | |")

        lines.append("")

        # Mejores experimentos por métrica común
        common_metrics = ["entropy", "loss", "pass_rate", "tokens_per_second"]
        for metric in common_metrics:
            best = self.find_best(metric, higher_is_better=True)
            if best:
                exp_id, value = best
                exp = self.experiments[exp_id]
                lines.append(f"## Mejor por '{metric}'")
                lines.append("")
                lines.append(f"- Experimento: `{exp_id}` ({exp.name})")
                lines.append(f"- Valor: {value:.6f}")
                lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text

    def export_to_csv(
        self,
        output_path: str | Path,
    ) -> str:
        """
        Exporta todos los experimentos a formato CSV.

        Returns
        -------
        str con la ruta del archivo CSV.
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "experiment_id", "name", "description", "status",
                "start_time", "end_time", "duration_sec",
                "hyperparameters", "metrics", "tags", "notes",
            ])

            for exp in self.experiments.values():
                writer.writerow([
                    exp.experiment_id,
                    exp.name,
                    exp.description,
                    exp.status,
                    exp.start_time,
                    exp.end_time or "",
                    exp.duration() or "",
                    json.dumps(exp.hyperparameters),
                    json.dumps(exp.metrics),
                    ",".join(exp.tags),
                    exp.notes,
                ])

        return str(output_path)

    def _save_experiment(
        self,
        exp: ExperimentRun,
    ) -> None:
        """Guarda un experimento en el almacenamiento."""
        exp_file = self.storage_dir / f"{exp.experiment_id}.json"
        with open(exp_file, "w") as f:
            json.dump(exp.to_dict(), f, indent=2)


def quick_tracker_demo() -> ExperimentTracker:
    """
    Función de demostración: crea un tracker con experimentos de ejemplo.

    Returns
    -------
    ExperimentTracker con experimentos simulados.
    """
    tracker = ExperimentTracker(storage_dir="/tmp/demo_experiments")

    # Crear algunos experimentos de ejemplo
    exp1_id = tracker.create_experiment(
        name="Síntesis base 800M",
        description="Primera síntesis de pesos sin entrenamiento",
        hyperparameters={"layers": 21, "hidden": 1536, "heads": 12},
        tags=["síntesis", "base", "800M"],
        notes="Modelo base sin modificaciones analíticas",
    )
    tracker.update_metrics(exp1_id, {"entropy": 5.2, "loss": 8.1, "pass_rate": 0.15})
    tracker.complete_experiment(exp1_id, "completed")

    exp2_id = tracker.create_experiment(
        name="Etapa 5 ajustada",
        description="run_stage5.py con circuito de inducción reducido",
        hyperparameters={"layers": 21, "hidden": 1536, "heads": 12, "induction": 0.06},
        tags=["síntesis", "etapa5", "ajustado"],
        notes="Reducción de circuito de inducción de 0.55 a 0.06",
    )
    tracker.update_metrics(exp2_id, {"entropy": 9.0, "loss": 4.2, "pass_rate": 0.28})
    tracker.complete_experiment(exp2_id, "completed")

    exp3_id = tracker.create_experiment(
        name="Con space suppression",
        description="Añadida suppression de espacios -6.0",
        hyperparameters={"layers": 21, "hidden": 1536, "heads": 12, "space_suppression": -6.0},
        tags=["síntesis", "etapa5", "suppression"],
        notes="Suppresión de espacios mejoró la legibilidad",
    )
    tracker.update_metrics(exp3_id, {"entropy": 8.7, "loss": 4.5, "pass_rate": 0.31})
    tracker.complete_experiment(exp3_id, "completed")

    return tracker


__all__ = [
    "ExperimentRun",
    "ExperimentComparison",
    "ExperimentTracker",
    "quick_tracker_demo",
]
