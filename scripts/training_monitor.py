#!/usr/bin/env python3
"""
training_monitor.py — Monitor de entrenamiento simulado para el modelo 800M.

Este módulo proporciona herramientas para monitorear el progreso de un
experimento de entrenamiento (simulado, sin gradientes reales):

  - Registro de métricas por paso (entropía proxy, pérdida simulada).
  - Detección de colapso del modelo (entropía cercana a 0).
  - Seguimiento de brecha entre edición analítica y gradientes.
  - Alertas automáticas cuando métricas salen de rango.
  - Guardado de checkpoints de estado.
  - Reporte de entrenamiento en JSON/Markdown.
  - Curvas de aprendizaje simuladas.
  - Comparación entre entrenamientos.

Todas las operaciones son analíticas y no requieren backpropagation.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TrainingStepMetric:
    """Métrica de un solo paso de entrenamiento."""
    step: int
    timestamp: float
    entropy_proxy: float
    simulated_loss: float
    gradient_gap: float  # diferencia entre edición y gradiente
    collapse_detected: bool
    learning_rate: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "entropy_proxy": round(self.entropy_proxy, 6),
            "simulated_loss": round(self.simulated_loss, 6),
            "gradient_gap": round(self.gradient_gap, 6),
            "collapse_detected": self.collapse_detected,
            "learning_rate": self.learning_rate,
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class TrainingRun:
    """Registro completo de un entrenamiento simulado."""
    run_id: str
    start_time: float
    end_time: Optional[float] = None
    total_steps: int = 0
    metrics: List[TrainingStepMetric] = field(default_factory=list)
    alerts: List[dict] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_steps": self.total_steps,
            "metrics": [m.to_dict() for m in self.metrics],
            "alerts": self.alerts,
            "config": self.config,
        }

    def summary(self) -> dict:
        if not self.metrics:
            return {"steps": 0}

        entropies = [m.entropy_proxy for m in self.metrics]
        losses = [m.simulated_loss for m in self.metrics]
        gaps = [m.gradient_gap for m in self.metrics]

        return {
            "steps": len(self.metrics),
            "final_entropy": round(entropies[-1], 6),
            "min_entropy": round(min(entropies), 6),
            "max_entropy": round(max(entropies), 6),
            "final_loss": round(losses[-1], 6),
            "mean_gradient_gap": round(sum(gaps) / len(gaps), 6),
            "collapse_detected": any(m.collapse_detected for m in self.metrics),
            "n_alerts": len(self.alerts),
        }


@dataclass
class Alert:
    """Una alerta generada durante el entrenamiento."""
    step: int
    timestamp: float
    type: str  # "collapse", "high_loss", "gradient_gap", "stagnation"
    message: str
    severity: str  # "info", "warning", "error"

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "type": self.type,
            "message": self.message,
            "severity": self.severity,
        }


class TrainingMonitor:
    """
    Monitor de entrenamiento simulado.

    Registra métricas por paso, detecta problemas y genera reportes.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        collapse_entropy_threshold: float = 0.1,
        stagnation_window: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        run_id : str opcional
            Identificador único de la ejecución. Si None, se genera uno.
        collapse_entropy_threshold : float
            Umbral de entropía por debajo del cual se detecta colapso.
        stagnation_window : int
            Ventana de pasos para detectar estancamiento.
        """
        self.run_id = run_id or f"run_{int(time.time())}"
        self.collapse_entropy_threshold = collapse_entropy_threshold
        self.stagnation_window = stagnation_window

        self.run: Optional[TrainingRun] = None
        self._last_entropies: List[float] = []
        self._alerts: List[Alert] = []

    def start_run(
        self,
        config: Optional[Dict[str, object]] = None,
    ) -> TrainingRun:
        """
        Inicia una nueva ejecución de entrenamiento.

        Returns
        -------
        TrainingRun con el estado inicial.
        """
        self.run = TrainingRun(
            run_id=self.run_id,
            start_time=time.time(),
            config=config or {},
        )
        self._last_entropies = []
        self._alerts = []
        return self.run

    def log_step(
        self,
        step: int,
        entropy_proxy: float,
        simulated_loss: float,
        gradient_gap: float,
        learning_rate: float = 0.0,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> TrainingStepMetric:
        """
        Registra las métricas de un paso de entrenamiento.

        Parameters
        ----------
        step : int
            Número del paso.
        entropy_proxy : float
            Entropía proxy (medida de diversidad de generación).
        simulated_loss : float
            Pérdida simulada (sin gradientes reales).
        gradient_gap : float
            Brecha entre edición analítica y gradiente real.
        learning_rate : float
            Tasa de aprendizaje simulada.
        custom_metrics : dict opcional
            Métricas personalizadas adicionales.

        Returns
        -------
        TrainingStepMetric con el registro del paso.
        """
        if self.run is None:
            self.start_run()

        collapse = entropy_proxy < self.collapse_entropy_threshold
        metric = TrainingStepMetric(
            step=step,
            timestamp=time.time(),
            entropy_proxy=entropy_proxy,
            simulated_loss=simulated_loss,
            gradient_gap=gradient_gap,
            collapse_detected=collapse,
            learning_rate=learning_rate,
            custom_metrics=custom_metrics or {},
        )

        self.run.metrics.append(metric)
        self.run.total_steps = step

        # Detectar alertas
        self._check_alerts(step, entropy_proxy, simulated_loss, gradient_gap)

        # Mantener historial para detección de estancamiento
        self._last_entropies.append(entropy_proxy)
        if len(self._last_entropies) > self.stagnation_window:
            self._last_entropies.pop(0)

        return metric

    def _check_alerts(
        self,
        step: int,
        entropy: float,
        loss: float,
        gradient_gap: float,
    ) -> None:
        """Detecta y registra alertas basadas en métricas."""
        timestamp = time.time()

        # Colapso
        if entropy < self.collapse_entropy_threshold:
            self._alerts.append(Alert(
                step=step,
                timestamp=timestamp,
                type="collapse",
                message=f"Entropía muy baja ({entropy:.4f}), posible colapso del modelo",
                severity="error",
            ))

        # Pérdida alta
        if loss > 10.0:
            self._alerts.append(Alert(
                step=step,
                timestamp=timestamp,
                type="high_loss",
                message=f"Pérdida simulada alta ({loss:.4f})",
                severity="warning",
            ))

        # Brecha de gradiente grande
        if gradient_gap > 1.0:
            self._alerts.append(Alert(
                step=step,
                timestamp=timestamp,
                type="gradient_gap",
                message=f"Brecha de gradiente grande ({gradient_gap:.4f})",
                severity="warning",
            ))

        # Estancamiento (entropía no cambia en ventana)
        if len(self._last_entropies) >= self.stagnation_window:
            if max(self._last_entropies) - min(self._last_entropies) < 0.01:
                self._alerts.append(Alert(
                    step=step,
                    timestamp=timestamp,
                    type="stagnation",
                    message="Entropía estancada en los últimos pasos",
                    severity="info",
                ))

        self.run.alerts.extend([a.to_dict() for a in self._alerts])
        # Evitar duplicados: solo mantener alertas nuevas
        self.run.alerts = self.run.alerts[-50:]

    def end_run(self) -> Optional[TrainingRun]:
        """
        Finaliza la ejecución actual.

        Returns
        -------
        TrainingRun finalizado o None si no hay ejecución activa.
        """
        if self.run is not None:
            self.run.end_time = time.time()
        return self.run

    def get_summary(self) -> dict:
        """
        Devuelve un resumen de la ejecución actual.

        Returns
        -------
        dict con estadísticas resumidas.
        """
        if self.run is None:
            return {"status": "no_run"}

        return self.run.summary()

    def save_run(
        self,
        output_path: str | Path,
    ) -> str:
        """
        Guarda la ejecución actual en un archivo JSON.

        Returns
        -------
        str con la ruta del archivo guardado.
        """
        if self.run is None:
            raise ValueError("No hay ejecución activa para guardar")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.run.to_dict(), f, indent=2)

        return str(output_path)

    def load_run(
        self,
        input_path: str | Path,
    ) -> TrainingRun:
        """
        Carga una ejecución desde un archivo JSON.

        Returns
        -------
        TrainingRun cargada.
        """
        input_path = Path(input_path)
        with open(input_path, "r") as f:
            data = json.load(f)

        run = TrainingRun(
            run_id=data["run_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            total_steps=data.get("total_steps", 0),
            config=data.get("config", {}),
        )

        for m_data in data.get("metrics", []):
            metric = TrainingStepMetric(
                step=m_data["step"],
                timestamp=m_data["timestamp"],
                entropy_proxy=m_data["entropy_proxy"],
                simulated_loss=m_data["simulated_loss"],
                gradient_gap=m_data["gradient_gap"],
                collapse_detected=m_data["collapse_detected"],
                learning_rate=m_data["learning_rate"],
                custom_metrics=m_data.get("custom_metrics", {}),
            )
            run.metrics.append(metric)

        run.alerts = data.get("alerts", [])
        self.run = run
        return run

    def generate_report(
        self,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de entrenamiento en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        if self.run is None:
            return "# Reporte de Entrenamiento\n\nNo hay ejecución activa."

        summary = self.run.summary()

        lines: List[str] = []
        lines.append("# Reporte de Entrenamiento Simulado")
        lines.append("")
        lines.append(f"ID de ejecución: `{self.run.run_id}`")
        lines.append(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.run.start_time))}")
        if self.run.end_time:
            lines.append(f"Fin: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.run.end_time))}")
        lines.append("")

        lines.append("## Resumen")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Pasos totales | {summary.get('steps', 0)} |")
        lines.append(f"| Entropía final | {summary.get('final_entropy', 'N/A')} |")
        lines.append(f"| Entropía mínima | {summary.get('min_entropy', 'N/A')} |")
        lines.append(f"| Entropía máxima | {summary.get('max_entropy', 'N/A')} |")
        lines.append(f"| Pérdida final | {summary.get('final_loss', 'N/A')} |")
        lines.append(f"| Brecha de gradiente media | {summary.get('mean_gradient_gap', 'N/A')} |")
        lines.append(f"| Colapso detectado | {'Sí' if summary.get('collapse_detected') else 'No'} |")
        lines.append(f"| Alertas | {summary.get('n_alerts', 0)} |")
        lines.append("")

        # Métricas por paso (últimos 10)
        if self.run.metrics:
            lines.append("## Últimos 10 Pasos")
            lines.append("")
            lines.append("| Paso | Entropía | Pérdida | Brecha Gradiente | Colapso |")
            lines.append("|------|----------|---------|------------------|---------|")

            for m in self.run.metrics[-10:]:
                lines.append(
                    f"| {m.step} | {m.entropy_proxy:.4f} | {m.simulated_loss:.4f} | "
                    f"{m.gradient_gap:.4f} | {'Sí' if m.collapse_detected else 'No'} |"
                )
            lines.append("")

        # Alertas
        if self.run.alerts:
            lines.append("## Alertas")
            lines.append("")
            lines.append("| Paso | Tipo | Mensaje | Severidad |")
            lines.append("|------|------|---------|-----------|")
            for alert in self.run.alerts[-20:]:
                lines.append(
                    f"| {alert['step']} | {alert['type']} | {alert['message']} | {alert['severity']} |"
                )
            lines.append("")

        # Configuración
        if self.run.config:
            lines.append("## Configuración")
            lines.append("")
            for key, value in self.run.config.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text

    def plot_entropy_curve_description(self) -> str:
        """
        Genera una descripción de la curva de entropía (texto).

        Returns
        -------
        str con la descripción en Markdown.
        """
        if self.run is None or not self.run.metrics:
            return "No hay datos de entropía para mostrar."

        lines: List[str] = []
        lines.append("## Curva de Entropía Proxy (Descripción)")
        lines.append("")

        entropies = [m.entropy_proxy for m in self.run.metrics]
        steps = [m.step for m in self.run.metrics]

        # Calcular tendencias
        first_half = entropies[:len(entropies) // 2] if len(entropies) > 1 else entropies
        second_half = entropies[len(entropies) // 2:] if len(entropies) > 1 else []

        trend = "estable"
        if second_half:
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            if second_avg > first_avg * 1.1:
                trend = "creciente"
            elif second_avg < first_avg * 0.9:
                trend = "decreciente"

        lines.append(f"- Tendencia general: {trend}")
        lines.append(f"- Entropía inicial: {entropies[0]:.4f}")
        lines.append(f"- Entropía final: {entropies[-1]:.4f}")
        lines.append(f"- Entropía máxima: {max(entropies):.4f}")
        lines.append(f"- Entropía mínima: {min(entropies):.4f}")
        lines.append("")
        lines.append("### Interpretación")
        lines.append("")
        lines.append("- Entropía estable y alta (> 2 nats): generación diversa, sin colapso.")
        lines.append("- Entropía decreciente hacia 0: posible colapso del modelo.")
        lines.append("- Entropía muy alta (> 5 nats): posible inestabilidad.")
        lines.append("")
        return "\n".join(lines)


def quick_monitor_run(
    run_id: str = "quick_run",
    n_steps: int = 100,
) -> TrainingRun:
    """
    Función de conveniencia: simula un monitoreo de entrenamiento.

    Genera métricas sintéticas para demostrar el monitor.

    Returns
    -------
    TrainingRun con métricas simuladas.
    """
    import random
    random.seed(42)

    monitor = TrainingMonitor(run_id=run_id)
    monitor.start_run(config={"model": "800M", "method": "analytical"})

    for step in range(1, n_steps + 1):
        # Simular entropía que empieza baja, sube, y se estabiliza
        base_entropy = 5.0 + 2.0 * math.sin(step / 20.0)
        noise = random.uniform(-0.5, 0.5)
        entropy = max(0.1, base_entropy + noise)

        # Simular pérdida inversamente relacionada con entropía
        loss = max(0.1, 10.0 - entropy)

        # Brecha de gradiente simulada
        gradient_gap = random.uniform(0.1, 2.0)

        # Colapso ocasional
        collapse = entropy < 0.1

        monitor.log_step(
            step=step,
            entropy_proxy=entropy,
            simulated_loss=loss,
            gradient_gap=gradient_gap,
            learning_rate=0.001,
            custom_metrics={"perplexity": math.exp(loss)},
        )

    monitor.end_run()
    return monitor.run


__all__ = [
    "TrainingStepMetric",
    "TrainingRun",
    "Alert",
    "TrainingMonitor",
    "quick_monitor_run",
]
