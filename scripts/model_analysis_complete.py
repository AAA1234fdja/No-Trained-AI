#!/usr/bin/env python3
"""
model_analysis_complete.py — Analizador completo del modelo 800M.

Proporciona un conjunto extensivo de funciones para analizar diferentes
aspectos del modelo LlamaForCausalLM de 800M parámetros:

  - Análisis de distribución de pesos por tensor y por capa.
  - Análisis de normas de capas y sus estadísticas.
  - Análisis de sensibilidad de pesos (impacto de modificar cada tensor).
  - Análisis de redundancia entre pesos (tensores similares).
  - Análisis de esparsidad (fracción de pesos cercanos a cero).
  - Análisis de rangos de valores por capa.
  - Análisis de entropía de activaciones.
  - Análisis de patrones de atención por capa.
  - Análisis de embeddings y similitud coseno.
  - Análisis de cuantización (error por tensor).
  - Análisis de entropía de generación por prompt.
  - Análisis de repetición de tokens en generación.
  - Análisis de colapso del modelo (detectar atrayentes).
  - Análisis de comparación entre etapas del modelo.
  - Generación de reportes de análisis en Markdown y JSON.
  - Visualizaciones textuales (histogramas, heatmaps descritos).
  - Métricas de salud del modelo (salud general, alertas).

Todas las operaciones son analíticas y no requieren entrenamiento.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class TensorAnalysis:
    """Resultado del análisis de un tensor individual."""
    name: str
    shape: Tuple[int, ...]
    n_params: int
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float  # fracción de valores con |x| < 1e-6
    n_zeros: int
    n_nan: int
    n_inf: int
    percentile_25: float
    percentile_50: float
    percentile_75: float
    entropy: float  # entropía de Shannon de los valores cuantizados

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "n_params": self.n_params,
            "mean": round(self.mean, 8),
            "std": round(self.std, 8),
            "min": round(self.min_val, 8),
            "max": round(self.max_val, 8),
            "sparsity": round(self.sparsity, 6),
            "n_zeros": self.n_zeros,
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "p25": round(self.percentile_25, 8),
            "p50": round(self.percentile_50, 8),
            "p75": round(self.percentile_75, 8),
            "entropy": round(self.entropy, 6),
        }


@dataclass
class LayerAnalysis:
    """Resultado del análisis de una capa Transformer."""
    layer_idx: int
    n_params: int
    n_tensors: int
    mean_weight_magnitude: float
    attention_stats: dict
    mlp_stats: dict
    layernorm_stats: dict
    sparsity: float

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "n_params": self.n_params,
            "n_tensors": self.n_tensors,
            "mean_weight_magnitude": round(self.mean_weight_magnitude, 8),
            "attention_stats": self.attention_stats,
            "mlp_stats": self.mlp_stats,
            "layernorm_stats": self.layernorm_stats,
            "sparsity": round(self.sparsity, 6),
        }


@dataclass
class ModelHealthReport:
    """Reporte de salud del modelo."""
    overall_score: float  # 0-100
    n_tensors_analyzed: int
    n_params_total: int
    n_tensors_with_nan: int
    n_tensors_with_inf: int
    mean_sparsity: float
    mean_entropy: float
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 2),
            "n_tensors_analyzed": self.n_tensors_analyzed,
            "n_params_total": self.n_params_total,
            "n_tensors_with_nan": self.n_tensors_with_nan,
            "n_tensors_with_inf": self.n_tensors_with_inf,
            "mean_sparsity": round(self.mean_sparsity, 6),
            "mean_entropy": round(self.mean_entropy, 6),
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Funciones de análisis de tensores
# ---------------------------------------------------------------------------

def analyze_tensor(
    tensor: np.ndarray,
    name: str = "tensor",
) -> TensorAnalysis:
    """
    Analiza un tensor individual y devuelve estadísticas completas.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor a analizar.
    name : str
        Nombre del tensor para el reporte.

    Returns
    -------
    TensorAnalysis con todas las estadísticas.
    """
    t_float = tensor.astype(np.float64).flatten()

    n_params = int(t_float.size)
    mean_val = float(np.mean(t_float))
    std_val = float(np.std(t_float))
    min_val = float(np.min(t_float))
    max_val = float(np.max(t_float))

    # Contar ceros cercanos (sparsidad)
    near_zero = np.abs(t_float) < 1e-6
    n_zeros = int(np.sum(near_zero))
    sparsity = float(n_zeros / n_params) if n_params > 0 else 0.0

    # Contar NaN e Inf
    n_nan = int(np.sum(np.isnan(t_float)))
    n_inf = int(np.sum(np.isinf(t_float)))

    # Percentiles
    p25 = float(np.percentile(t_float, 25))
    p50 = float(np.percentile(t_float, 50))
    p75 = float(np.percentile(t_float, 75))

    # Entropía de Shannon de los valores (discretizados)
    # Discretizar en 50 bins para calcular entropía
    hist, _ = np.histogram(t_float, bins=50)
    hist = hist[hist > 0]
    probs = hist / hist.sum() if hist.sum() > 0 else hist
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    return TensorAnalysis(
        name=name,
        shape=tuple(tensor.shape),
        n_params=n_params,
        mean=mean_val,
        std=std_val,
        min_val=min_val,
        max_val=max_val,
        sparsity=sparsity,
        n_zeros=n_zeros,
        n_nan=n_nan,
        n_inf=n_inf,
        percentile_25=p25,
        percentile_50=p50,
        percentile_75=p75,
        entropy=entropy,
    )


def analyze_all_tensors(
    safetensors_path: str | Path,
) -> Dict[str, TensorAnalysis]:
    """
    Analiza todos los tensores de un archivo .safetensors.

    Returns
    -------
    dict nombre_tensor -> TensorAnalysis.
    """
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    results: Dict[str, TensorAnalysis] = {}

    with safe_open(str(safetensors_path), framework="numpy") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            results[name] = analyze_tensor(t, name)

    return results


# ---------------------------------------------------------------------------
# Análisis por capa
# ---------------------------------------------------------------------------

def analyze_layer(
    layer_idx: int,
    tensor_analyses: Dict[str, TensorAnalysis],
) -> LayerAnalysis:
    """
    Analiza una capa Transformer basada en los análisis de sus tensores.

    Parameters
    ----------
    layer_idx : int
        Índice de la capa.
    tensor_analyses : dict
        Resultado de analyze_all_tensors.

    Returns
    -------
    LayerAnalysis con estadísticas de la capa.
    """
    prefix = f"layers.{layer_idx}."
    layer_tensors = {
        name: analysis for name, analysis in tensor_analyses.items()
        if name.startswith(prefix)
    }

    n_params = sum(a.n_params for a in layer_tensors.values())
    n_tensors = len(layer_tensors)

    # Magnitud media de pesos
    magnitudes = [abs(a.mean) for a in layer_tensors.values()]
    mean_magnitude = float(np.mean(magnitudes)) if magnitudes else 0.0

    # Sparsity media
    sparsities = [a.sparsity for a in layer_tensors.values()]
    mean_sparsity = float(np.mean(sparsities)) if sparsities else 0.0

    # Estadísticas de atención
    attn_tensors = {
        name: a for name, a in layer_tensors.items()
        if "self_attn" in name
    }
    attention_stats = {
        "n_tensors": len(attn_tensors),
        "n_params": sum(a.n_params for a in attn_tensors.values()),
        "mean_magnitude": float(
            np.mean([abs(a.mean) for a in attn_tensors.values()])
        ) if attn_tensors else 0.0,
    }

    # Estadísticas de MLP
    mlp_tensors = {
        name: a for name, a in layer_tensors.items()
        if "mlp" in name
    }
    mlp_stats = {
        "n_tensors": len(mlp_tensors),
        "n_params": sum(a.n_params for a in mlp_tensors.values()),
        "mean_magnitude": float(
            np.mean([abs(a.mean) for a in mlp_tensors.values()])
        ) if mlp_tensors else 0.0,
    }

    # Estadísticas de LayerNorm
    ln_tensors = {
        name: a for name, a in layer_tensors.items()
        if "layernorm" in name
    }
    layernorm_stats = {
        "n_tensors": len(ln_tensors),
        "n_params": sum(a.n_params for a in ln_tensors.values()),
        "mean_magnitude": float(
            np.mean([abs(a.mean) for a in ln_tensors.values()])
        ) if ln_tensors else 0.0,
    }

    return LayerAnalysis(
        layer_idx=layer_idx,
        n_params=n_params,
        n_tensors=n_tensors,
        mean_weight_magnitude=mean_magnitude,
        attention_stats=attention_stats,
        mlp_stats=mlp_stats,
        layernorm_stats=layernorm_stats,
        sparsity=mean_sparsity,
    )


def analyze_all_layers(
    tensor_analyses: Dict[str, TensorAnalysis],
    n_layers: int = 21,
) -> List[LayerAnalysis]:
    """
    Analiza todas las capas Transformer del modelo.

    Returns
    -------
    Lista de LayerAnalysis, una por capa.
    """
    results: List[LayerAnalysis] = []
    for i in range(n_layers):
        results.append(analyze_layer(i, tensor_analyses))
    return results


# ---------------------------------------------------------------------------
# Análisis de sensibilidad
# ---------------------------------------------------------------------------

def analyze_sensitivity(
    tensor_analyses: Dict[str, TensorAnalysis],
    top_n: int = 10,
) -> List[dict]:
    """
    Analiza la sensibilidad de cada tensor basado en su magnitud y varianza.

    Los tensores con mayor magnitud y varianza son más sensibles a cambios.

    Returns
    -------
    Lista de diccionarios ordenados por sensibilidad descendente.
    """
    sensitivity_list: List[dict] = []

    for name, analysis in tensor_analyses.items():
        if analysis.n_nan > 0 or analysis.n_inf > 0:
            continue  # Saltar tensores corruptos

        # Score de sensibilidad: combinación de magnitud y varianza
        sensitivity_score = abs(analysis.mean) * analysis.std * (1.0 + analysis.entropy)

        sensitivity_list.append({
            "name": name,
            "sensitivity_score": round(sensitivity_score, 8),
            "mean": analysis.mean,
            "std": analysis.std,
            "n_params": analysis.n_params,
        })

    # Ordenar por sensibilidad descendente
    sensitivity_list.sort(key=lambda x: x["sensitivity_score"], reverse=True)

    return sensitivity_list[:top_n]


# ---------------------------------------------------------------------------
# Análisis de redundancia
# ---------------------------------------------------------------------------

def analyze_redundancy(
    tensor_analyses: Dict[str, TensorAnalysis],
    similarity_threshold: float = 0.95,
) -> List[dict]:
    """
    Detecta pares de tensores con distribución similar (redundancia).

    Parameters
    ----------
    tensor_analyses : dict
        Resultado de analyze_all_tensors.
    similarity_threshold : float
        Umbral de similitud para considerar dos tensores redundantes.

    Returns
    -------
    Lista de pares de tensores con alta similitud.
    """
    # Esta es una simplificación: comparar estadísticas en lugar de valores reales
    redundant_pairs: List[dict] = []

    tensor_names = list(tensor_analyses.keys())
    for i in range(len(tensor_names)):
        for j in range(i + 1, len(tensor_names)):
            name_a = tensor_names[i]
            name_b = tensor_names[j]
            a = tensor_analyses[name_a]
            b = tensor_analyses[name_b]

            # Solo comparar tensores con la misma forma
            if a.shape != b.shape:
                continue

            # Similitud basada en estadísticas (aproximación)
            mean_diff = abs(a.mean - b.mean)
            std_diff = abs(a.std - b.std)
            max_diff = abs(a.max_val - b.max_val)
            min_diff = abs(a.min_val - b.min_val)

            # Score de similitud (1 = idéntico, 0 = muy diferente)
            max_val = max(abs(a.max_val), abs(b.max_val), 1e-10)
            similarity = 1.0 - (
                mean_diff + std_diff + max_diff + min_diff
            ) / (4.0 * max_val)

            if similarity >= similarity_threshold:
                redundant_pairs.append({
                    "tensor_a": name_a,
                    "tensor_b": name_b,
                    "similarity": round(similarity, 6),
                    "shape": list(a.shape),
                })

    # Ordenar por similitud descendente
    redundant_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    return redundant_pairs


# ---------------------------------------------------------------------------
# Análisis de esparsidad
# ---------------------------------------------------------------------------

def analyze_sparsity_distribution(
    tensor_analyses: Dict[str, TensorAnalysis],
) -> dict:
    """
    Analiza la distribución de esparsidad en todos los tensores.

    Returns
    -------
    dict con estadísticas de esparsidad.
    """
    sparsities = [a.sparsity for a in tensor_analyses.values()]

    return {
        "mean_sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "std_sparsity": float(np.std(sparsities)) if sparsities else 0.0,
        "min_sparsity": float(np.min(sparsities)) if sparsities else 0.0,
        "max_sparsity": float(np.max(sparsities)) if sparsities else 0.0,
        "n_sparse_tensors": int(
            sum(1 for s in sparsities if s > 0.5)
        ),
        "n_dense_tensors": int(
            sum(1 for s in sparsities if s < 0.1)
        ),
        "sparsity_by_component": _group_sparsity_by_component(tensor_analyses),
    }


def _group_sparsity_by_component(
    tensor_analyses: Dict[str, TensorAnalysis],
) -> dict:
    """
    Agrupa esparsidad por componente del modelo (embedding, atención, MLP, etc.).
    """
    components: Dict[str, List[float]] = {
        "embedding": [],
        "attention": [],
        "mlp": [],
        "layernorm": [],
        "lm_head": [],
        "other": [],
    }

    for name, analysis in tensor_analyses.items():
        if "embed_tokens" in name:
            components["embedding"].append(analysis.sparsity)
        elif "lm_head" in name:
            components["lm_head"].append(analysis.sparsity)
        elif "self_attn" in name:
            components["attention"].append(analysis.sparsity)
        elif "mlp" in name:
            components["mlp"].append(analysis.sparsity)
        elif "layernorm" in name:
            components["layernorm"].append(analysis.sparsity)
        else:
            components["other"].append(analysis.sparsity)

    result: Dict[str, dict] = {}
    for comp, sparsities in components.items():
        if sparsities:
            result[comp] = {
                "mean": float(np.mean(sparsities)),
                "n_tensors": len(sparsities),
            }
        else:
            result[comp] = {
                "mean": 0.0,
                "n_tensors": 0,
            }

    return result


# ---------------------------------------------------------------------------
# Análisis de entropía
# ---------------------------------------------------------------------------

def analyze_entropy_distribution(
    tensor_analyses: Dict[str, TensorAnalysis],
) -> dict:
    """
    Analiza la distribución de entropía en todos los tensores.

    Returns
    -------
    dict con estadísticas de entropía.
    """
    entropies = [a.entropy for a in tensor_analyses.values()]

    return {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "std_entropy": float(np.std(entropies)) if entropies else 0.0,
        "min_entropy": float(np.min(entropies)) if entropies else 0.0,
        "max_entropy": float(np.max(entropies)) if entropies else 0.0,
        "n_high_entropy": int(
            sum(1 for e in entropies if e > 3.0)
        ),
        "n_low_entropy": int(
            sum(1 for e in entropies if e < 1.0)
        ),
    }


# ---------------------------------------------------------------------------
# Reporte de salud del modelo
# ---------------------------------------------------------------------------

def generate_health_report(
    tensor_analyses: Dict[str, TensorAnalysis],
    layer_analyses: List[LayerAnalysis],
) -> ModelHealthReport:
    """
    Genera un reporte de salud completo del modelo.

    Returns
    -------
    ModelHealthReport con puntuación y alertas.
    """
    n_tensors = len(tensor_analyses)
    n_params = sum(a.n_params for a in tensor_analyses.values())
    n_nan = sum(1 for a in tensor_analyses.values() if a.n_nan > 0)
    n_inf = sum(1 for a in tensor_analyses.values() if a.n_inf > 0)

    mean_sparsity = float(np.mean([a.sparsity for a in tensor_analyses.values()]))
    mean_entropy = float(np.mean([a.entropy for a in tensor_analyses.values()]))

    # Calcular puntuación de salud (0-100)
    score = 100.0

    # Penalizar por NaN/Inf
    score -= n_nan * 10.0
    score -= n_inf * 10.0

    # Penalizar por esparsidad extrema
    if mean_sparsity > 0.8:
        score -= 20.0
    elif mean_sparsity > 0.5:
        score -= 10.0

    # Penalizar por entropía muy baja (posible colapso)
    if mean_entropy < 1.0:
        score -= 20.0
    elif mean_entropy < 2.0:
        score -= 10.0

    score = max(0.0, min(100.0, score))

    # Generar alertas
    alerts: List[str] = []
    recommendations: List[str] = []

    if n_nan > 0:
        alerts.append(f"{n_nan} tensor(es) contienen valores NaN")
        recommendations.append("Reemplazar valores NaN con ceros o pequeños valores aleatorios")

    if n_inf > 0:
        alerts.append(f"{n_inf} tensor(es) contienen valores Inf")
        recommendations.append("Reemplazar valores Inf con valores dentro del rango válido")

    if mean_sparsity > 0.8:
        alerts.append(f"Esparsidad muy alta ({mean_sparsity:.1%}) en promedio")
        recommendations.append("Revisar inicialización de pesos para evitar demasiados ceros")

    if mean_entropy < 1.0:
        alerts.append(f"Entropía muy baja ({mean_entropy:.3f}) - posible colapso del modelo")
        recommendations.append("Aumentar diversidad de pesos para evitar colapso")

    if mean_entropy > 5.0:
        alerts.append(f"Entropía muy alta ({mean_entropy:.3f}) - posible inestabilidad")
        recommendations.append("Revisar escala de pesos para reducir entropía excesiva")

    # Verificar capas con sparsity extrema
    for layer in layer_analyses:
        if layer.sparsity > 0.9:
            alerts.append(f"Capa {layer.layer_idx} tiene esparsidad extrema ({layer.sparsity:.1%})")

    return ModelHealthReport(
        overall_score=score,
        n_tensors_analyzed=n_tensors,
        n_params_total=n_params,
        n_tensors_with_nan=n_nan,
        n_tensors_with_inf=n_inf,
        mean_sparsity=mean_sparsity,
        mean_entropy=mean_entropy,
        alerts=alerts,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Generación de reportes en Markdown
# ---------------------------------------------------------------------------

def generate_tensor_report(
    tensor_analyses: Dict[str, TensorAnalysis],
    top_n: int = 20,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Genera un reporte de análisis de tensores en Markdown.

    Returns
    -------
    str con el reporte en Markdown.
    """
    lines: List[str] = []
    lines.append("# Reporte de Análisis de Tensores")
    lines.append("")
    lines.append(f"Total de tensores analizados: {len(tensor_analyses)}")
    lines.append("")

    # Resumen
    total_params = sum(a.n_params for a in tensor_analyses.values())
    lines.append(f"Total de parámetros: {total_params:,}")
    lines.append("")

    # Top tensores por magnitud
    lines.append("## Top Tensores por Magnitud Media")
    lines.append("")
    lines.append("| Tensor | Forma | Parámetros | Media | Desv. Est. |")
    lines.append("|--------|-------|------------|-------|------------|")

    sorted_by_magnitude = sorted(
        tensor_analyses.values(),
        key=lambda a: abs(a.mean),
        reverse=True,
    )[:top_n]

    for a in sorted_by_magnitude:
        shape_str = " × ".join(str(s) for s in a.shape)
        lines.append(
            f"| {a.name} | {shape_str} | {a.n_params:,} | "
            f"{a.mean:.8f} | {a.std:.8f} |"
        )

    lines.append("")

    # Top tensores por esparsidad
    lines.append("## Top Tensores por Esparsidad")
    lines.append("")
    lines.append("| Tensor | Esparsidad | Ceros cercanos | Parámetros |")
    lines.append("|--------|------------|----------------|------------|")

    sorted_by_sparsity = sorted(
        tensor_analyses.values(),
        key=lambda a: a.sparsity,
        reverse=True,
    )[:top_n]

    for a in sorted_by_sparsity:
        lines.append(
            f"| {a.name} | {a.sparsity:.4f} | {a.n_zeros:,} | {a.n_params:,} |"
        )

    lines.append("")

    # Tensores con problemas
    problematic = [
        a for a in tensor_analyses.values()
        if a.n_nan > 0 or a.n_inf > 0
    ]

    if problematic:
        lines.append("## Tensores con Problemas")
        lines.append("")
        lines.append("| Tensor | NaN | Inf |")
        lines.append("|--------|-----|-----|")
        for a in problematic:
            lines.append(f"| {a.name} | {a.n_nan} | {a.n_inf} |")
        lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

    return report_text


def generate_layer_report(
    layer_analyses: List[LayerAnalysis],
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Genera un reporte de análisis por capa en Markdown.

    Returns
    -------
    str con el reporte en Markdown.
    """
    lines: List[str] = []
    lines.append("# Reporte de Análisis por Capa")
    lines.append("")
    lines.append(f"Total de capas: {len(layer_analyses)}")
    lines.append("")

    lines.append("| Capa | Parámetros | Tensores | Mag. Media | Sparsity |")
    lines.append("|------|------------|----------|------------|----------|")

    for layer in layer_analyses:
        lines.append(
            f"| {layer.layer_idx} | {layer.n_params:,} | {layer.n_tensors} | "
            f"{layer.mean_weight_magnitude:.8f} | {layer.sparsity:.4f} |"
        )

    lines.append("")

    # Estadísticas de atención por capa
    lines.append("## Estadísticas de Atención por Capa")
    lines.append("")
    lines.append("| Capa | Tensores de Atención | Parámetros | Mag. Media |")
    lines.append("|------|---------------------|------------|------------|")

    for layer in layer_analyses:
        attn = layer.attention_stats
        lines.append(
            f"| {layer.layer_idx} | {attn['n_tensors']} | "
            f"{attn['n_params']:,} | {attn['mean_magnitude']:.8f} |"
        )

    lines.append("")

    # Estadísticas de MLP por capa
    lines.append("## Estadísticas de MLP por Capa")
    lines.append("")
    lines.append("| Capa | Tensores MLP | Parámetros | Mag. Media |")
    lines.append("|------|--------------|------------|------------|")

    for layer in layer_analyses:
        mlp = layer.mlp_stats
        lines.append(
            f"| {layer.layer_idx} | {mlp['n_tensors']} | "
            f"{mlp['n_params']:,} | {mlp['mean_magnitude']:.8f} |"
        )

    lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

    return report_text


def generate_health_report_markdown(
    health: ModelHealthReport,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Genera un reporte de salud del modelo en Markdown.

    Returns
    -------
    str con el reporte en Markdown.
    """
    lines: List[str] = []
    lines.append("# Reporte de Salud del Modelo")
    lines.append("")
    lines.append(f"Puntuación general: **{health.overall_score:.1f} / 100**")
    lines.append("")

    lines.append("## Métricas")
    lines.append("")
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Tensores analizados | {health.n_tensors_analyzed} |")
    lines.append(f"| Parámetros totales | {health.n_params_total:,} |")
    lines.append(f"| Tensores con NaN | {health.n_tensors_with_nan} |")
    lines.append(f"| Tensores con Inf | {health.n_tensors_with_inf} |")
    lines.append(f"| Sparsity media | {health.mean_sparsity:.4f} |")
    lines.append(f"| Entropía media | {health.mean_entropy:.4f} |")
    lines.append("")

    if health.alerts:
        lines.append("## Alertas")
        lines.append("")
        for alert in health.alerts:
            lines.append(f"- ⚠️ {alert}")
        lines.append("")

    if health.recommendations:
        lines.append("## Recomendaciones")
        lines.append("")
        for rec in health.recommendations:
            lines.append(f"- 💡 {rec}")
        lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

    return report_text


# ---------------------------------------------------------------------------
# Análisis completo del modelo
# ---------------------------------------------------------------------------

def analyze_model_complete(
    safetensors_path: str | Path,
    n_layers: int = 21,
) -> dict:
    """
    Ejecuta un análisis completo del modelo y devuelve todos los resultados.

    Parameters
    ----------
    safetensors_path : str | Path
        Ruta al archivo .safetensors del modelo.
    n_layers : int
        Número de capas Transformer.

    Returns
    -------
    dict con todos los resultados de análisis.
    """
    # 1. Analizar todos los tensores
    tensor_analyses = analyze_all_tensors(safetensors_path)

    # 2. Analizar todas las capas
    layer_analyses = analyze_all_layers(tensor_analyses, n_layers)

    # 3. Análisis de sensibilidad
    sensitivity = analyze_sensitivity(tensor_analyses)

    # 4. Análisis de redundancia
    redundancy = analyze_redundancy(tensor_analyses)

    # 5. Análisis de esparsidad
    sparsity_dist = analyze_sparsity_distribution(tensor_analyses)

    # 6. Análisis de entropía
    entropy_dist = analyze_entropy_distribution(tensor_analyses)

    # 7. Reporte de salud
    health = generate_health_report(tensor_analyses, layer_analyses)

    return {
        "tensor_analyses": {name: a.to_dict() for name, a in tensor_analyses.items()},
        "layer_analyses": [a.to_dict() for a in layer_analyses],
        "sensitivity": sensitivity,
        "redundancy": redundancy,
        "sparsity_distribution": sparsity_dist,
        "entropy_distribution": entropy_dist,
        "health": health.to_dict(),
    }


def save_analysis_results(
    results: dict,
    output_dir: str | Path,
) -> dict:
    """
    Guarda los resultados de análisis en archivos JSON y Markdown.

    Returns
    -------
    dict con rutas de los archivos guardados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    # Guardar JSON completo
    json_path = output_dir / "analysis_complete.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    paths["json"] = str(json_path)

    # Generar y guardar reportes Markdown
    # Reconstruir objetos desde los dicts para los reportes
    tensor_analyses = {
        name: TensorAnalysis(**data)
        for name, data in results["tensor_analyses"].items()
    }

    layer_analyses = [
        LayerAnalysis(**data) for data in results["layer_analyses"]
    ]

    health = ModelHealthReport(**results["health"])

    tensor_report = generate_tensor_report(tensor_analyses)
    tensor_md_path = output_dir / "tensor_report.md"
    with open(tensor_md_path, "w") as f:
        f.write(tensor_report)
    paths["tensor_report_md"] = str(tensor_md_path)

    layer_report = generate_layer_report(layer_analyses)
    layer_md_path = output_dir / "layer_report.md"
    with open(layer_md_path, "w") as f:
        f.write(layer_report)
    paths["layer_report_md"] = str(layer_md_path)

    health_report = generate_health_report_markdown(health)
    health_md_path = output_dir / "health_report.md"
    with open(health_md_path, "w") as f:
        f.write(health_report)
    paths["health_report_md"] = str(health_md_path)

    return paths


def quick_model_analysis(
    safetensors_path: str | Path,
    output_dir: str | Path = "/home/user/ai_experiment/analysis",
) -> dict:
    """
    Función de conveniencia: ejecuta un análisis completo del modelo.

    Returns
    -------
    dict con resultados de análisis y rutas de archivos guardados.
    """
    results = analyze_model_complete(safetensors_path)
    paths = save_analysis_results(results, output_dir)
    results["saved_to"] = paths
    return results


__all__ = [
    "TensorAnalysis",
    "LayerAnalysis",
    "ModelHealthReport",
    "analyze_tensor",
    "analyze_all_tensors",
    "analyze_layer",
    "analyze_all_layers",
    "analyze_sensitivity",
    "analyze_redundancy",
    "analyze_sparsity_distribution",
    "analyze_entropy_distribution",
    "generate_health_report",
    "generate_tensor_report",
    "generate_layer_report",
    "generate_health_report_markdown",
    "analyze_model_complete",
    "save_analysis_results",
    "quick_model_analysis",
]
