#!/usr/bin/env python3
"""
model_distiller.py — Simulador de destilación de conocimientos.

Este módulo simula el proceso de destilación de conocimientos (knowledge
distillation) sin entrenamiento real con gradientes. Proporciona:

  - Transferencia de logits simulada entre un modelo "teacher" y "student".
  - Ajuste de temperatura para suavizar distribuciones de probabilidad.
  - Cálculo de pérdida de destilación simulada (KL divergence aproximado).
  - Comparación de salidas entre modelos teacher y student.
  - Métricas de calidad de la destilación (similitud de distribuciones).
  - Reporte de destilación en JSON/Markdown.

Nota: Esto es un simulador analítico, no un entrenamiento real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DistillationResult:
    """Resultado de una operación de destilación."""
    teacher_output: List[float]
    student_output: List[float]
    temperature: float
    kl_divergence: float
    cosine_similarity: float
    mse: float
    quality_score: float

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "kl_divergence": round(self.kl_divergence, 8),
            "cosine_similarity": round(self.cosine_similarity, 6),
            "mse": round(self.mse, 10),
            "quality_score": round(self.quality_score, 4),
        }


@dataclass
class DistillationReport:
    """Reporte completo de destilación."""
    teacher_model: str
    student_model: str
    n_samples: int
    temperature: float
    mean_kl_divergence: float
    mean_cosine_similarity: float
    mean_mse: float
    mean_quality_score: float
    sample_results: List[DistillationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "mean_kl_divergence": round(self.mean_kl_divergence, 8),
            "mean_cosine_similarity": round(self.mean_cosine_similarity, 6),
            "mean_mse": round(self.mean_mse, 10),
            "mean_quality_score": round(self.mean_quality_score, 4),
            "samples": [r.to_dict() for r in self.sample_results],
        }


class ModelDistiller:
    """
    Simulador de destilación de conocimientos.

    Analiza la similitud entre salidas de un modelo teacher y student
    para evaluar qué tan bien el student está imitando al teacher.
    """

    def __init__(
        self,
        temperature: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        temperature : float
            Temperatura para suavizar distribuciones (mayor = más suave).
        """
        self.temperature = temperature

    def _softmax_with_temperature(
        self,
        logits: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """
        Aplica softmax con temperatura a los logits.

        softmax(z / T) = exp(z_i / T) / sum(exp(z_j / T))
        """
        scaled = logits / temperature
        scaled = scaled - np.max(scaled)  # estabilidad numérica
        exp_logits = np.exp(scaled)
        return exp_logits / np.sum(exp_logits)

    def _kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """
        Calcula la divergencia KL de P || Q.

        KL(P||Q) = sum(p_i * log(p_i / q_i))
        """
        # Evitar log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """Calcula la similitud de coseno entre dos vectores."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def distill_sample(
        self,
        teacher_logits: np.ndarray,
        student_logits: np.ndarray,
        temperature: Optional[float] = None,
    ) -> DistillationResult:
        """
        Destila una sola muestra comparando teacher y student.

        Parameters
        ----------
        teacher_logits : np.ndarray
            Logits del modelo teacher.
        student_logits : np.ndarray
            Logits del modelo student.
        temperature : float opcional
            Temperatura (usa self.temperature si no se especifica).

        Returns
        -------
        DistillationResult con métricas de la muestra.
        """
        temp = temperature if temperature is not None else self.temperature

        # Asegurar mismo shape
        min_len = min(len(teacher_logits), len(student_logits))
        t = teacher_logits[:min_len].astype(np.float64)
        s = student_logits[:min_len].astype(np.float64)

        # Softmax con temperatura
        teacher_probs = self._softmax_with_temperature(t, temp)
        student_probs = self._softmax_with_temperature(s, temp)

        # Métricas
        kl_div = self._kl_divergence(teacher_probs, student_probs)
        cosine_sim = self._cosine_similarity(t, s)
        mse = float(np.mean((t - s) ** 2))

        # Puntuación de calidad (0-1, mayor es mejor)
        # Combinar KL (menor es mejor) y cosine (mayor es mejor)
        kl_score = max(0.0, 1.0 - kl_div / 10.0)  # normalizar KL
        quality_score = (kl_score + cosine_sim) / 2.0

        return DistillationResult(
            teacher_output=t.tolist(),
            student_output=s.tolist(),
            temperature=temp,
            kl_divergence=kl_div,
            cosine_similarity=cosine_sim,
            mse=mse,
            quality_score=quality_score,
        )

    def distill_model(
        self,
        teacher_outputs: List[np.ndarray],
        student_outputs: List[np.ndarray],
        temperature: Optional[float] = None,
    ) -> DistillationReport:
        """
        Destila múltiples muestras comparando teacher y student.

        Parameters
        ----------
        teacher_outputs : list de np.ndarray
            Salidas del modelo teacher.
        student_outputs : list de np.ndarray
            Salidas del modelo student.
        temperature : float opcional

        Returns
        -------
        DistillationReport con métricas agregadas.
        """
        temp = temperature if temperature is not None else self.temperature

        if len(teacher_outputs) != len(student_outputs):
            raise ValueError(
                f"teacher_outputs ({len(teacher_outputs)}) y "
                f"student_outputs ({len(student_outputs)}) deben tener la misma longitud"
            )

        sample_results: List[DistillationResult] = []
        total_kl = 0.0
        total_cosine = 0.0
        total_mse = 0.0
        total_quality = 0.0

        for t_out, s_out in zip(teacher_outputs, student_outputs):
            result = self.distill_sample(t_out, s_out, temp)
            sample_results.append(result)
            total_kl += result.kl_divergence
            total_cosine += result.cosine_similarity
            total_mse += result.mse
            total_quality += result.quality_score

        n = len(sample_results)

        report = DistillationReport(
            teacher_model="teacher_model",
            student_model="student_model",
            n_samples=n,
            temperature=temp,
            mean_kl_divergence=total_kl / n if n > 0 else 0.0,
            mean_cosine_similarity=total_cosine / n if n > 0 else 0.0,
            mean_mse=total_mse / n if n > 0 else 0.0,
            mean_quality_score=total_quality / n if n > 0 else 0.0,
            sample_results=sample_results,
        )

        return report

    def generate_report(
        self,
        report: DistillationReport,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de destilación en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Destilación de Conocimientos")
        lines.append("")
        lines.append("## Resumen")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Modelo teacher | {report.teacher_model} |")
        lines.append(f"| Modelo student | {report.student_model} |")
        lines.append(f"| Muestras | {report.n_samples} |")
        lines.append(f"| Temperatura | {report.temperature} |")
        lines.append(f"| KL Divergencia media | {report.mean_kl_divergence:.8f} |")
        lines.append(f"| Similitud de coseno media | {report.mean_cosine_similarity:.6f} |")
        lines.append(f"| MSE medio | {report.mean_mse:.10f} |")
        lines.append(f"| Puntuación de calidad media | {report.mean_quality_score:.4f} |")
        lines.append("")

        lines.append("## Interpretación")
        lines.append("")
        lines.append("- **KL Divergencia baja (< 0.1):** El student imita bien al teacher.")
        lines.append("- **KL Divergencia media (0.1-1.0):** Aceptable, hay espacio para mejorar.")
        lines.append("- **KL Divergencia alta (> 1.0):** El student no está capturando la distribución del teacher.")
        lines.append("- **Similitud de coseno alta (> 0.9):** Los logits están alineados.")
        lines.append("- **Puntuación de calidad alta (> 0.8):** Buena destilación general.")
        lines.append("")

        lines.append("## Top 5 Muestras por Calidad")
        lines.append("")
        lines.append("| Muestra | KL Div | Cosine Sim | MSE | Calidad |")
        lines.append("|---------|--------|------------|-----|---------|")

        sorted_samples = sorted(
            report.sample_results,
            key=lambda x: x.quality_score,
            reverse=True,
        )[:5]

        for i, s in enumerate(sorted_samples):
            lines.append(
                f"| {i+1} | {s.kl_divergence:.8f} | "
                f"{s.cosine_similarity:.6f} | {s.mse:.10f} | {s.quality_score:.4f} |"
            )

        lines.append("")
        return "\n".join(lines)


def quick_distill(
    teacher_outputs: List[np.ndarray],
    student_outputs: List[np.ndarray],
    temperature: float = 1.0,
) -> DistillationReport:
    """
    Función de conveniencia: destila comparando teacher y student.

    Returns
    -------
    DistillationReport con el resultado.
    """
    distiller = ModelDistiller(temperature=temperature)
    return distiller.distill_model(teacher_outputs, student_outputs, temperature)


__all__ = [
    "DistillationResult",
    "DistillationReport",
    "ModelDistiller",
    "quick_distill",
]
