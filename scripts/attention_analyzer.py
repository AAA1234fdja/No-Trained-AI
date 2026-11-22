#!/usr/bin/env python3
"""
attention_analyzer.py — Analizador de patrones de atención del modelo.

Proporciona herramientas para analizar matrices de atención:

  - Extracción de pesos de atención por capa y cabeza.
  - Mapa de calor textual de atención (query-key pairs más relevantes).
  - Detección de patrones de atención (local, global, disperso).
  - Medición de entropía de atención por cabeza.
  - Comparación de atención entre capas.
  - Identificación de cabezas de atención "muertas" (baja entropía).
  - Reporte de análisis de atención en JSON/Markdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AttentionHeadStats:
    """Estadísticas de una cabeza de atención."""
    layer: int
    head: int
    mean_attention: float
    entropy: float
    max_attention: float
    sparsity: float  # fracción de pesos cerca de cero
    is_dead: bool  # cabeza con muy baja entropía

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "mean_attention": round(self.mean_attention, 6),
            "entropy": round(self.entropy, 6),
            "max_attention": round(self.max_attention, 6),
            "sparsity": round(self.sparsity, 6),
            "is_dead": self.is_dead,
        }


@dataclass
class AttentionAnalysis:
    """Resultado del análisis de atención del modelo."""
    n_layers: int
    n_heads_per_layer: int
    head_stats: List[AttentionHeadStats] = field(default_factory=list)
    dead_heads: List[tuple] = field(default_factory=list)
    high_entropy_heads: List[tuple] = field(default_factory=list)
    layer_entropies: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "n_heads_per_layer": self.n_heads_per_layer,
            "n_dead_heads": len(self.dead_heads),
            "n_high_entropy_heads": len(self.high_entropy_heads),
            "layer_entropies": [round(e, 4) for e in self.layer_entropies],
            "heads": [h.to_dict() for h in self.head_stats],
        }


class AttentionAnalyzer:
    """
    Analizador de patrones de atención.

    Calcula entropía, detecta cabezas muertas, encuentra pares
    query-key con mayor atención y genera reportes.
    """

    def __init__(
        self,
        dead_head_entropy_threshold: float = 0.5,
        high_entropy_threshold: float = 3.0,
    ) -> None:
        """
        Parameters
        ----------
        dead_head_entropy_threshold : float
            Umbral de entropía para considerar una cabeza "muerta".
        high_entropy_threshold : float
            Umbral de entropía para considerar una cabeza de alta entropía.
        """
        self.dead_head_entropy_threshold = dead_head_entropy_threshold
        self.high_entropy_threshold = high_entropy_threshold

    def _entropy(
        self,
        probs: np.ndarray,
    ) -> float:
        """Calcula la entropía de Shannon de una distribución."""
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def analyze_attention_matrix(
        self,
        attention_probs: np.ndarray,
        layer_idx: int,
        head_idx: int,
        tokens: Optional[List[str]] = None,
    ) -> AttentionHeadStats:
        """
        Analiza una matriz de atención para una capa y cabeza.

        Parameters
        ----------
        attention_probs : np.ndarray
            Matriz (seq_len, seq_len) de probabilidades de atención.
        layer_idx : int
            Índice de la capa.
        head_idx : int
            Índice de la cabeza.
        tokens : list de str opcional
            Nombres de tokens para las posiciones.

        Returns
        -------
        AttentionHeadStats con métricas de la cabeza.
        """
        seq_len = attention_probs.shape[0]

        # Entropía por fila (cada query) y luego promedio
        row_entropies = []
        for i in range(seq_len):
            row_entropies.append(self._entropy(attention_probs[i]))
        mean_entropy = float(np.mean(row_entropies))

        # Estadísticas de atención
        mean_attn = float(np.mean(attention_probs))
        max_attn = float(np.max(attention_probs))

        # Sparsity: fracción de pesos muy pequeños (< 0.01)
        sparsity = float(np.sum(attention_probs < 0.01) / attention_probs.size)

        # Detectar cabeza muerta
        is_dead = mean_entropy < self.dead_head_entropy_threshold

        return AttentionHeadStats(
            layer=layer_idx,
            head=head_idx,
            mean_attention=mean_attn,
            entropy=mean_entropy,
            max_attention=max_attn,
            sparsity=sparsity,
            is_dead=is_dead,
        )

    def analyze_model_attention(
        self,
        attention_by_layer_head: Dict[tuple, np.ndarray],
    ) -> AttentionAnalysis:
        """
        Analiza la atención de todo el modelo.

        Parameters
        ----------
        attention_by_layer_head : dict
            {(layer, head): matriz_atención} para todas las cabezas.

        Returns
        -------
        AttentionAnalysis con estadísticas de todas las cabezas.
        """
        head_stats: List[AttentionHeadStats] = []
        dead_heads: List[tuple] = []
        high_entropy_heads: List[tuple] = []
        layer_entropies: Dict[int, List[float]] = {}

        for (layer, head), probs in attention_by_layer_head.items():
            stats = self.analyze_attention_matrix(probs, layer, head)
            head_stats.append(stats)

            if stats.is_dead:
                dead_heads.append((layer, head))

            if stats.entropy > self.high_entropy_threshold:
                high_entropy_heads.append((layer, head))

            if layer not in layer_entropies:
                layer_entropies[layer] = []
            layer_entropies[layer].append(stats.entropy)

        # Entropía media por capa
        layer_entropy_means = []
        for layer in sorted(layer_entropies.keys()):
            layer_entropy_means.append(float(np.mean(layer_entropies[layer])))

        n_layers = max(k[0] for k in attention_by_layer_head.keys()) + 1
        n_heads = max(k[1] for k in attention_by_layer_head.keys()) + 1

        return AttentionAnalysis(
            n_layers=n_layers,
            n_heads_per_layer=n_heads,
            head_stats=head_stats,
            dead_heads=dead_heads,
            high_entropy_heads=high_entropy_heads,
            layer_entropies=layer_entropy_means,
        )

    def find_top_attention_pairs(
        self,
        attention_probs: np.ndarray,
        tokens: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[dict]:
        """
        Encuentra los pares query-key con mayor atención.

        Returns
        -------
        Lista de dicts con posición query, key, valor y nombres de tokens.
        """
        seq_len = attention_probs.shape[0]
        flat_indices = np.argsort(attention_probs.flatten())[-top_k:][::-1]

        pairs: List[dict] = []
        for idx in flat_indices:
            q = idx // seq_len
            k = idx % seq_len
            val = float(attention_probs[q, k])
            tq = tokens[q] if tokens and q < len(tokens) else f"q{q}"
            tk = tokens[k] if tokens and k < len(tokens) else f"k{k}"
            pairs.append({
                "query_pos": int(q),
                "query_token": tq,
                "key_pos": int(k),
                "key_token": tk,
                "attention": round(val, 6),
            })

        return pairs

    def describe_attention_heatmap(
        self,
        attention_probs: np.ndarray,
        layer_idx: int,
        head_idx: int,
        tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Genera una descripción de un mapa de calor de atención.

        Returns
        -------
        str con la descripción en Markdown.
        """
        seq_len = attention_probs.shape[0]
        top_pairs = self.find_top_attention_pairs(
            attention_probs, tokens, top_k=5
        )

        lines: List[str] = []
        lines.append(f"## Mapa de Calor de Atención — Capa {layer_idx}, Cabeza {head_idx}")
        lines.append("")
        lines.append(f"- Longitud de secuencia: {seq_len} tokens")
        lines.append(f"- Atención máxima: {float(np.max(attention_probs)):.4f}")
        lines.append(f"- Atención media: {float(np.mean(attention_probs)):.4f}")
        lines.append(f"- Entropía: {self._entropy(attention_probs.flatten()):.4f} nats")
        lines.append("")

        lines.append("### Top 5 Pares Query-Key")
        lines.append("")
        lines.append("| Query | Key | Atención |")
        lines.append("|-------|-----|----------|")
        for p in top_pairs:
            lines.append(
                f"| {p['query_token']} (pos {p['query_pos']}) | "
                f"{p['key_token']} (pos {p['key_pos']}) | {p['attention']:.4f} |"
            )
        lines.append("")

        return "\n".join(lines)

    def generate_report(
        self,
        analysis: AttentionAnalysis,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de análisis de atención en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Análisis de Atención")
        lines.append("")
        lines.append("## Resumen")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Capas | {analysis.n_layers} |")
        lines.append(f"| Cabezas por capa | {analysis.n_heads_per_layer} |")
        lines.append(f"| Total de cabezas | {analysis.n_layers * analysis.n_heads_per_layer} |")
        lines.append(f"| Cabezas muertas | {len(analysis.dead_heads)} |")
        lines.append(f"| Cabezas de alta entropía | {len(analysis.high_entropy_heads)} |")
        lines.append("")

        lines.append("## Entropía por Capa")
        lines.append("")
        lines.append("| Capa | Entropía Media |")
        lines.append("|------|----------------|")
        for layer_idx, entropy in enumerate(analysis.layer_entropies):
            bar = "█" * int(entropy * 10) + "░" * (30 - int(entropy * 10))
            lines.append(f"| {layer_idx} | {entropy:.4f} {bar} |")
        lines.append("")

        if analysis.dead_heads:
            lines.append("## Cabezas Muertas (Baja Entropía)")
            lines.append("")
            lines.append("| Capa | Cabeza |")
            lines.append("|------|--------|")
            for layer, head in analysis.dead_heads[:20]:
                lines.append(f"| {layer} | {head} |")
            if len(analysis.dead_heads) > 20:
                lines.append(f"| ... ({len(analysis.dead_heads) - 20} más) | |")
            lines.append("")

        if analysis.high_entropy_heads:
            lines.append("## Cabezas de Alta Entropía")
            lines.append("")
            lines.append("| Capa | Cabeza |")
            lines.append("|------|--------|")
            for layer, head in analysis.high_entropy_heads[:20]:
                lines.append(f"| {layer} | {head} |")
            if len(analysis.high_entropy_heads) > 20:
                lines.append(f"| ... ({len(analysis.high_entropy_heads) - 20} más) | |")
            lines.append("")

        lines.append("## Interpretación")
        lines.append("")
        lines.append("- **Cabeza muerta:** Entropía muy baja, la atención se concentra en")
        lines.append("  muy pocos tokens. Puede indicar falta de aprendizaje.")
        lines.append("- **Cabeza de alta entropía:** Atención distribuida uniformemente,")
        lines.append("  puede indicar que no está enfocando en tokens relevantes.")
        lines.append("- **Entropía equilibrada:** Entre 1.0 y 3.0 nats indica atención")
        lines.append("  selectiva pero no excesivamente concentrada.")
        lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text


def quick_analyze_attention(
    attention_by_layer_head: Dict[tuple, np.ndarray],
) -> AttentionAnalysis:
    """
    Función de conveniencia: analiza atención de un modelo.

    Returns
    -------
    AttentionAnalysis con el resultado.
    """
    analyzer = AttentionAnalyzer()
    return analyzer.analyze_model_attention(attention_by_layer_head)


__all__ = [
    "AttentionHeadStats",
    "AttentionAnalysis",
    "AttentionAnalyzer",
    "quick_analyze_attention",
]
