#!/usr/bin/env python3
"""
inference_optimizer.py — Optimizador de inferencia para el modelo 800M.

Proporciona herramientas para optimizar el rendimiento de inferencia:

  - Batching de solicitudes (agrupamiento de múltiples prompts).
  - Gestión de caché KV (cálculo de memoria requerida, optimización).
  - Estimación de tokens/segundo basado en hardware.
  - Optimización de parámetros de generación (temperatura, top-k, top-p).
  - Análisis de latencia por etapa (carga, prefill, decode).
  - Recomendaciones de configuración para diferentes escenarios.
  - Simulación de rendimiento (sin ejecutar el modelo real).
  - Reporte de optimización en Markdown/JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class InferenceConfig:
    """Configuración de inferencia optimizada."""
    batch_size: int
    max_sequence_length: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    kv_cache_enabled: bool
    use_mmap: bool
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "kv_cache_enabled": self.kv_cache_enabled,
            "use_mmap": self.use_mmap,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
        }


@dataclass
class InferencePerformanceEstimate:
    """Estimación de rendimiento de inferencia."""
    tokens_per_second: float
    latency_per_token_ms: float
    memory_usage_mb: float
    kv_cache_memory_mb: float
    prefill_time_ms: float
    decode_time_per_token_ms: float

    def to_dict(self) -> dict:
        return {
            "tokens_per_second": round(self.tokens_per_second, 2),
            "latency_per_token_ms": round(self.latency_per_token_ms, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "kv_cache_memory_mb": round(self.kv_cache_memory_mb, 2),
            "prefill_time_ms": round(self.prefill_time_ms, 2),
            "decode_time_per_token_ms": round(self.decode_time_per_token_ms, 2),
        }


@dataclass
class OptimizationRecommendation:
    """Una recomendación de optimización."""
    category: str  # "memory", "speed", "quality", "batch"
    priority: str  # "high", "medium", "low"
    message: str
    action: str


class InferenceOptimizer:
    """
    Optimizador de inferencia para el modelo AI 800M.

    Calcula estimaciones de rendimiento, recomienda configuraciones
    y analiza el uso de recursos sin ejecutar el modelo real.
    """

    def __init__(
        self,
        num_layers: int = 21,
        hidden_size: int = 1536,
        num_heads: int = 12,
        num_kv_heads: int = 4,
        vocab_size: int = 32000,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.vocab_size = vocab_size
        self.head_dim = hidden_size // num_heads

    def estimate_memory_usage(
        self,
        batch_size: int = 1,
        max_sequence_length: int = 512,
        use_kv_cache: bool = True,
    ) -> dict:
        """
        Estima el uso de memoria para una configuración de inferencia.

        Returns
        -------
        dict con desglose de memoria en MB.
        """
        # Pesos del modelo (INT8: 1 byte por param, FP16 scales: 2 bytes)
        total_params = 800_260_608  # parámetros del modelo 800M
        weights_mb = total_params / (1024 * 1024)  # INT8 approx

        # Activaciones (estimación aproximada)
        activations_mb = (
            batch_size * max_sequence_length * hidden_size * 4  # float32
        ) / (1024 * 1024) * self.num_layers * 0.1  # factor de ocupación

        # Caché KV
        kv_cache_mb = 0.0
        if use_kv_cache:
            # Cada capa: 2 * batch * seq_len * num_kv_heads * head_dim * 2 bytes (FP16)
            kv_cache_mb = (
                2 * batch_size * max_sequence_length * self.num_kv_heads *
                self.head_dim * 2 * self.num_layers
            ) / (1024 * 1024)

        # Overhead del sistema
        overhead_mb = 50.0

        total_mb = weights_mb + activations_mb + kv_cache_mb + overhead_mb

        return {
            "weights_mb": round(weights_mb, 2),
            "activations_mb": round(activations_mb, 2),
            "kv_cache_mb": round(kv_cache_mb, 2),
            "overhead_mb": overhead_mb,
            "total_mb": round(total_mb, 2),
        }

    def estimate_performance(
        self,
        batch_size: int = 1,
        max_sequence_length: int = 512,
        num_tokens_to_generate: int = 100,
        use_kv_cache: bool = True,
        use_mmap: bool = True,
    ) -> InferencePerformanceEstimate:
        """
        Estima el rendimiento de inferencia.

        Model simplificado basado en operaciones de matriz y memoria.

        Returns
        -------
        InferencePerformanceEstimate con estimaciones.
        """
        # FLOPs aproximados por token (forward pass de transformer)
        # Attention: 4 * batch * seq_len * hidden^2 * layers
        # MLP: 8 * batch * seq_len * hidden * intermediate * layers (SwiGLU)
        intermediate = 5888  # tamaño intermedio del modelo

        flops_per_token = (
            4 * self.num_layers * self.hidden_size ** 2 +  # atención
            8 * self.num_layers * self.hidden_size * intermediate  # MLP
        )

        # Asumir 100 GFLOPs efectivos (CPU con optimizaciones)
        effective_gflops = 100.0
        flops_per_second = effective_gflops * 1e9

        # Tiempo de prefill (procesar el prompt completo)
        prefill_flops = flops_per_token * max_sequence_length * batch_size
        prefill_time_ms = (prefill_flops / flops_per_second) * 1000

        # Tiempo de decode (token por token, con KV-cache)
        if use_kv_cache:
            # Con KV-cache, cada token nuevo es más barato
            decode_flops_per_token = flops_per_token * 0.3  # factor de reducción
        else:
            decode_flops_per_token = flops_per_token

        decode_time_per_token_ms = (decode_flops_per_token / flops_per_second) * 1000

        # Tiempo total
        total_decode_time_ms = decode_time_per_token_ms * num_tokens_to_generate
        total_time_ms = prefill_time_ms + total_decode_time_ms
        total_tokens = max_sequence_length + num_tokens_to_generate

        tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0.0
        latency_per_token_ms = total_time_ms / total_tokens if total_tokens > 0 else 0.0

        # Memoria
        mem = self.estimate_memory_usage(batch_size, max_sequence_length, use_kv_cache)

        return InferencePerformanceEstimate(
            tokens_per_second=tokens_per_second,
            latency_per_token_ms=latency_per_token_ms,
            memory_usage_mb=mem["total_mb"],
            kv_cache_memory_mb=mem["kv_cache_mb"],
            prefill_time_ms=prefill_time_ms,
            decode_time_per_token_ms=decode_time_per_token_ms,
        )

    def optimize_config(
        self,
        target_tokens_per_second: float = 10.0,
        max_memory_mb: float = 2000.0,
        batch_size: int = 1,
    ) -> InferenceConfig:
        """
        Genera una configuración optimizada para un objetivo de rendimiento.

        Returns
        -------
        InferenceConfig con parámetros recomendados.
        """
        # Estimar con configuración base
        base = self.estimate_performance(batch_size=batch_size)

        # Ajustar temperatura para balanceo calidad/velocidad
        temperature = 0.0 if target_tokens_per_second > 20 else 0.7

        # Top-k y top-p para calidad
        top_k = 50 if target_tokens_per_second > 10 else 100
        top_p = 0.9 if target_tokens_per_second > 10 else 0.95

        # KV-cache siempre habilitado para velocidad
        kv_cache = True

        # MMAP para cargar pesos eficientemente
        use_mmap = True

        # Longitud máxima de secuencia basada en memoria
        mem_estimate = self.estimate_memory_usage(
            batch_size=batch_size,
            max_sequence_length=512,
            use_kv_cache=kv_cache,
        )

        max_seq = 512
        if mem_estimate["total_mb"] > max_memory_mb:
            max_seq = int(512 * max_memory_mb / mem_estimate["total_mb"])
            max_seq = max(max_seq, 64)

        return InferenceConfig(
            batch_size=batch_size,
            max_sequence_length=max_seq,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.1,
            kv_cache_enabled=kv_cache,
            use_mmap=use_mmap,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def generate_recommendations(
        self,
        config: InferenceConfig,
        performance: InferencePerformanceEstimate,
    ) -> List[OptimizationRecommendation]:
        """
        Genera recomendaciones de optimización basadas en configuración y rendimiento.

        Returns
        -------
        Lista de OptimizationRecommendation.
        """
        recommendations: List[OptimizationRecommendation] = []

        # Memoria
        if performance.memory_usage_mb > 1500:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="high",
                message=f"Uso de memoria alto: {performance.memory_usage_mb:.1f} MB",
                action="Reducir batch_size o max_sequence_length, usar cuantización INT8",
            ))

        if performance.kv_cache_memory_mb > 500:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="medium",
                message=f"Caché KV consume {performance.kv_cache_memory_mb:.1f} MB",
                action="Reducir max_sequence_length o deshabilitar KV-cache para secuencias cortas",
            ))

        # Velocidad
        if performance.tokens_per_second < 5:
            recommendations.append(OptimizationRecommendation(
                category="speed",
                priority="high",
                message=f"Tokens/segundo bajo: {performance.tokens_per_second:.1f}",
                action="Habilitar KV-cache, usar MMAP, reducir max_sequence_length",
            ))

        if performance.prefill_time_ms > 1000:
            recommendations.append(OptimizationRecommendation(
                category="speed",
                priority="medium",
                message=f"Prefill lento: {performance.prefill_time_ms:.1f} ms",
                action="Reducir longitud máxima del prompt o usar batching",
            ))

        # Calidad
        if config.temperature == 0.0:
            recommendations.append(OptimizationRecommendation(
                category="quality",
                priority="low",
                message="Temperatura en 0 (muestreo determinista)",
                action="Considerar temperatura 0.5-0.7 para más diversidad",
            ))

        if config.top_k > 100:
            recommendations.append(OptimizationRecommendation(
                category="quality",
                priority="low",
                message=f"Top-k muy alto: {config.top_k}",
                action="Reducir a 20-50 para mejor calidad de generación",
            ))

        # Batching
        if config.batch_size == 1:
            recommendations.append(OptimizationRecommendation(
                category="batch",
                priority="medium",
                message="Batch size de 1",
                action="Considerar batch size mayor para mayor throughput",
            ))

        return recommendations

    def generate_report(
        self,
        config: InferenceConfig,
        performance: InferencePerformanceEstimate,
        recommendations: List[OptimizationRecommendation],
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de optimización de inferencia en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Optimización de Inferencia")
        lines.append("")
        lines.append("## Configuración")
        lines.append("")
        lines.append("| Parámetro | Valor |")
        lines.append("|-----------|-------|")
        for key, value in config.to_dict().items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

        lines.append("## Rendimiento Estimado")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Tokens/segundo | {performance.tokens_per_second:.2f} |")
        lines.append(f"| Latencia por token | {performance.latency_per_token_ms:.2f} ms |")
        lines.append(f"| Uso de memoria | {performance.memory_usage_mb:.2f} MB |")
        lines.append(f"| Memoria caché KV | {performance.kv_cache_memory_mb:.2f} MB |")
        lines.append(f"| Tiempo prefill | {performance.prefill_time_ms:.2f} ms |")
        lines.append(f"| Tiempo decode por token | {performance.decode_time_per_token_ms:.2f} ms |")
        lines.append("")

        lines.append("## Recomendaciones")
        lines.append("")
        if not recommendations:
            lines.append("No hay recomendaciones críticas.")
        else:
            lines.append("| Categoría | Prioridad | Mensaje | Acción |")
            lines.append("|-----------|-----------|---------|--------|")
            for r in recommendations:
                lines.append(
                    f"| {r.category} | {r.priority} | {r.message} | {r.action} |"
                )

        lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text


def quick_optimize_inference(
    target_tps: float = 10.0,
    max_memory_mb: float = 2000.0,
    batch_size: int = 1,
) -> dict:
    """
    Función de conveniencia: optimiza configuración de inferencia.

    Returns
    -------
    dict con configuración, estimación de rendimiento y recomendaciones.
    """
    optimizer = InferenceOptimizer()
    config = optimizer.optimize_config(target_tps, max_memory_mb, batch_size)
    performance = optimizer.estimate_performance(
        batch_size=config.batch_size,
        max_sequence_length=config.max_sequence_length,
    )
    recommendations = optimizer.generate_recommendations(config, performance)

    return {
        "config": config.to_dict(),
        "performance": performance.to_dict(),
        "recommendations": [
            {
                "category": r.category,
                "priority": r.priority,
                "message": r.message,
                "action": r.action,
            }
            for r in recommendations
        ],
    }


__all__ = [
    "InferenceConfig",
    "InferencePerformanceEstimate",
    "OptimizationRecommendation",
    "InferenceOptimizer",
    "quick_optimize_inference",
]
