#!/usr/bin/env python3
"""
data_visualizer.py — Generador de descripciones de visualizaciones para el modelo 800M.

Este módulo no renderiza gráficos (no depende de matplotlib), sino que
genera descripciones textuales de visualizaciones que podrían crearse
a partir de los datos del modelo:

  - Histogramas de pesos por capa.
  - Mapas de calor de matrices de atención.
  - Distribuciones de entropía por etapa.
  - Comparación de pesos entre etapas.
  - Gráficos de barras de métricas de evaluación.
  - Diagramas de arquitectura del modelo.

Cada función devuelve una descripción en formato Markdown o texto que
puede usarse en documentación o reportes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


def describe_weight_histogram(
    tensor: np.ndarray,
    tensor_name: str,
    n_bins: int = 20,
) -> str:
    """
    Genera una descripción de un histograma de pesos para un tensor.

    Returns
    -------
    str en formato Markdown con estadísticas y descripción visual.
    """
    if tensor.dtype == np.int8:
        tensor_f = tensor.astype(np.float32)
    else:
        tensor_f = tensor.astype(np.float32)

    counts, edges = np.histogram(tensor_f, bins=n_bins)
    total = tensor_f.size

    lines: List[str] = []
    lines.append(f"## Histograma de Pesos: `{tensor_name}`")
    lines.append("")
    lines.append(f"- Forma: {tensor.shape}")
    lines.append(f"- Total de elementos: {total:,}")
    lines.append(f"- Media: {float(np.mean(tensor_f)):.6f}")
    lines.append(f"- Desviación estándar: {float(np.std(tensor_f)):.6f}")
    lines.append(f"- Mínimo: {float(np.min(tensor_f)):.6f}")
    lines.append(f"- Máximo: {float(np.max(tensor_f)):.6f}")
    lines.append("")

    lines.append("### Distribución por Bins")
    lines.append("")
    lines.append("| Rango | Frecuencia | Porcentaje |")
    lines.append("|-------|------------|------------|")

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        count = int(counts[i])
        pct = count / total * 100.0 if total > 0 else 0.0
        bar = "█" * int(pct / 2)  # barra visual simple
        lines.append(f"| [{lo:.4f}, {hi:.4f}) | {count:,} | {pct:.2f}% {bar} |")

    lines.append("")
    return "\n".join(lines)


def describe_attention_heatmap(
    attn_probs: np.ndarray,
    layer_idx: int,
    head_idx: int,
    tokens: Optional[List[str]] = None,
) -> str:
    """
    Genera una descripción de un mapa de calor de atención para una
    cabeza y capa específicas.

    attn_probs: matriz de probabilidades de atención (seq_len, seq_len).
    """
    seq_len = attn_probs.shape[0]

    lines: List[str] = []
    lines.append(f"## Mapa de Calor de Atención — Capa {layer_idx}, Cabeza {head_idx}")
    lines.append("")
    lines.append(f"- Secuencia: {seq_len} tokens")
    lines.append(f"- Máxima atención: {float(np.max(attn_probs)):.4f}")
    lines.append(f"- Media: {float(np.mean(attn_probs)):.4f}")
    lines.append(f"- Desviación estándar: {float(np.std(attn_probs)):.4f}")
    lines.append("")

    # Encontrar las pares query-key con mayor atención
    flat_idx = np.argsort(attn_probs.flatten())[-5:][::-1]
    top_pairs = []
    for idx in flat_idx:
        q = idx // seq_len
        k = idx % seq_len
        val = float(attn_probs[q, k])
        top_pairs.append((q, k, val))

    lines.append("### Top 5 Pares Query-Key con Mayor Atención")
    lines.append("")
    lines.append("| Query Pos | Key Pos | Peso de Atención |")
    lines.append("|-----------|---------|------------------|")
    for q, k, val in top_pairs:
        tq = tokens[q] if tokens and q < len(tokens) else f"t{q}"
        tk = tokens[k] if tokens and k < len(tokens) else f"t{k}"
        lines.append(f"| {q} ({tq!r}) | {k} ({tk!r}) | {val:.4f} |")

    lines.append("")
    return "\n".join(lines)


def describe_entropy_evolution(
    entropies_by_stage: Dict[str, List[float]],
) -> str:
    """
    Genera una descripción de la evolución de la entropía a través
    de las etapas de síntesis.

    entropies_by_stage: dict nombre_etapa -> lista de entropías por prompt.
    """
    lines: List[str] = []
    lines.append("## Evolución de la Entropía por Etapa")
    lines.append("")
    lines.append("| Etapa | Entropía Media | Mínima | Máxima | Desv. Est. |")
    lines.append("|-------|---------------|--------|--------|------------|")

    for stage, entropies in entropies_by_stage.items():
        if not entropies:
            continue
        mean_e = float(np.mean(entropies))
        min_e = float(np.min(entropies))
        max_e = float(np.max(entropies))
        std_e = float(np.std(entropies))
        lines.append(
            f"| {stage} | {mean_e:.3f} | {min_e:.3f} | {max_e:.3f} | {std_e:.3f} |"
        )

    lines.append("")
    lines.append("### Interpretación")
    lines.append("")
    lines.append("- Entropía alta (> 3 nats): generación diversa, sin colapso.")
    lines.append("- Entropía media (1-3 nats): generación moderadamente diversa.")
    lines.append("- Entropía baja (< 1 nat): posible colapso o repetición.")
    lines.append("- Entropía cercana a 0: atractor de repetición total.")
    lines.append("")
    return "\n".join(lines)


def describe_weight_comparison(
    stage_a: Dict[str, np.ndarray],
    stage_b: Dict[str, np.ndarray],
    stage_name_a: str = "Etapa A",
    stage_name_b: str = "Etapa B",
) -> str:
    """
    Genera una descripción comparativa de dos etapas del modelo.

    Calcula diferencias absolutas medias por tensor común.
    """
    lines: List[str] = []
    lines.append(f"## Comparación de Pesos: {stage_name_a} vs {stage_name_b}")
    lines.append("")

    common_keys = set(stage_a.keys()) & set(stage_b.keys())
    total_diff = 0.0
    n_compared = 0

    lines.append("| Tensor | Diferencia Absoluta Media | Máxima Diferencia |")
    lines.append("|--------|---------------------------|-------------------|")

    for name in sorted(common_keys):
        a = stage_a[name].astype(np.float64)
        b = stage_b[name].astype(np.float64)

        if a.shape != b.shape:
            continue

        diff = np.abs(a - b)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))

        total_diff += mean_diff
        n_compared += 1

        shape_str = " × ".join(str(s) for s in a.shape)
        lines.append(
            f"| {name} ({shape_str}) | {mean_diff:.6f} | {max_diff:.6f} |"
        )

    lines.append("")
    if n_compared > 0:
        lines.append(f"- Diferencia absoluta media global: {total_diff / n_compared:.6f}")
        lines.append(f"- Tensores comparados: {n_compared}")
    else:
        lines.append("- No hay tensores comunes para comparar.")
    lines.append("")
    return "\n".join(lines)


def describe_evaluation_bar_chart(
    results: Dict[str, Dict[str, object]],
) -> str:
    """
    Genera una descripción de un gráfico de barras de resultados
    de evaluación por suite.

    results: dict nombre_suite -> {total_tests, passed, pass_rate, ...}
    """
    lines: List[str] = []
    lines.append("## Resultados de Evaluación por Suite")
    lines.append("")

    lines.append("| Suite | Pruebas | Pasadas | Falladas | Tasa de Paso |")
    lines.append("|-------|---------|---------|----------|--------------|")

    for sname, sdata in results.items():
        total = sdata.get("total_tests", 0)
        passed = sdata.get("passed", 0)
        failed = sdata.get("failed", 0)
        rate = sdata.get("pass_rate", 0.0)

        bar_len = int(rate * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)

        lines.append(
            f"| {sname} | {total} | {passed} | {failed} | {rate:.1%} {bar} |"
        )

    lines.append("")
    return "\n".join(lines)


def describe_model_architecture_diagram(
    config: Dict[str, object],
) -> str:
    """
    Genera un diagrama de arquitectura del modelo en formato Mermaid
    (texto que puede renderizarse en visores de Mermaid).
    """
    vocab = config.get("vocab_size", 32000)
    hidden = config.get("hidden_size", 1536)
    layers = config.get("num_hidden_layers", 21)
    heads = config.get("num_attention_heads", 12)
    kv_heads = config.get("num_key_value_heads", 4)
    dff = config.get("intermediate_size", 5888)

    lines: List[str] = []
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("    A[Entrada: Tokens] --> B[Embeddings]")
    lines.append(f"    B --> C[{layers}× Capas Transformer]")
    lines.append("    C --> D[Norma Final]")
    lines.append("    D --> E[lm_head]")
    lines.append("    E --> F[Salida: Próximo Token]")
    lines.append("")
    lines.append("    subgraph Capa Transformer")
    lines.append("        C1[RMSNorm] --> C2[Atención GQA]")
    lines.append("        C2 --> C3[RMSNorm]")
    lines.append("        C3 --> C4[MLP SwiGLU]")
    lines.append("        C4 --> C5[Residual]")
    lines.append("    end")
    lines.append("")
    lines.append("    subgraph Atención GQA")
    lines.append(f"        Q[Query: {heads} cabezas]")
    lines.append(f"        K[Key: {kv_heads} cabezas KV]")
    lines.append(f"        V[Value: {kv_heads} cabezas KV]")
    lines.append("        Q --> AT[Productos Punto]")
    lines.append("        K --> AT")
    lines.append("        V --> AT")
    lines.append("    end")
    lines.append("")
    lines.append("    subgraph MLP SwiGLU")
    lines.append("        Gate[Gate Proj] --> SiLU[SiLU]")
    lines.append("        Up[Up Proj] --> Mul[Multiplicación]")
    lines.append("        SiLU --> Mul")
    lines.append("        Mul --> Down[Down Proj]")
    lines.append("    end")
    lines.append("```")
    lines.append("")
    lines.append(f"### Dimensiones")
    lines.append(f"- Vocabulario: {vocab:,} tokens")
    lines.append(f"- Hidden size: {hidden}")
    lines.append(f"- Capas: {layers}")
    lines.append(f"- Cabezas de atención: {heads}")
    lines.append(f"- Cabezas KV (GQA): {kv_heads}")
    lines.append(f"- Tamaño intermedio (SwiGLU): {dff}")
    lines.append("")
    return "\n".join(lines)


def describe_parameter_distribution(
    model_dir: str | Path,
) -> str:
    """
    Genera una descripción de la distribución de parámetros por componente
    del modelo (embedings, atención, MLP, normas).
    """
    from scripts.model_builder import (
        count_embedding_params, count_layer_params,
        count_final_norm_params, count_total_params, L,
    )

    emb = count_embedding_params()
    per_layer = count_layer_params()
    all_layers = per_layer * L()
    final_norm = count_final_norm_params()
    total = count_total_params()

    lines: List[str] = []
    lines.append("## Distribución de Parámetros por Componente")
    lines.append("")
    lines.append("| Componente | Parámetros | Porcentaje |")
    lines.append("|------------|------------|------------|")

    components = [
        ("Embeddings de entrada", emb // 2),
        ("Cabeza de salida (lm_head)", emb // 2),
        ("Todas las capas Transformer", all_layers),
        ("Norma final", final_norm),
    ]

    for name, count in components:
        pct = count / total * 100.0 if total > 0 else 0.0
        bar = "█" * int(pct / 2)
        lines.append(f"| {name} | {count:,} | {pct:.1f}% {bar} |")

    lines.append("")
    lines.append(f"| **Total** | **{total:,}** | **100.0%** |")
    lines.append("")
    lines.append("### Observaciones")
    lines.append("- Las capas Transformer concentran la mayor parte de los parámetros.")
    lines.append("- Los embeddings y la cabeza de salida son simétricos (sin atar pesos).")
    lines.append("- Las normas son parámetros 1D (solo escala), muy pocos en comparación.")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "describe_weight_histogram",
    "describe_attention_heatmap",
    "describe_entropy_evolution",
    "describe_weight_comparison",
    "describe_evaluation_bar_chart",
    "describe_model_architecture_diagram",
    "describe_parameter_distribution",
]
