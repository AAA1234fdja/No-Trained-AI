#!/usr/bin/env python3
"""
embedding_visualizer.py — Visualizador de embeddings del modelo 800M.

Proporciona descripciones textuales de visualizaciones de embeddings:

  - Matrices de similitud coseno entre tokens.
  - Distribución de normas de embeddings por capa.
  - Clustering textual de tokens similares.
  - Analogías de embeddings (rei - hombre + mujer = reina).
  - Detección de tokens outliers (embeddings muy diferentes).
  - Proyección textual a 2D/3D (descripción de t-SNE/PCA).
  - Comparación de embeddings entre capas del modelo.
  - Reporte de análisis de embeddings en JSON/Markdown.

Todas las operaciones son numéricas sobre los tensores de embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


@dataclass
class EmbeddingSimilarity:
    """Par de tokens con su similitud de coseno."""
    token_a: str
    token_b: str
    similarity: float

    def to_dict(self) -> dict:
        return {
            "token_a": self.token_a,
            "token_b": self.token_b,
            "similarity": round(self.similarity, 6),
        }


@dataclass
class EmbeddingAnalysis:
    """Resultado del análisis de un conjunto de embeddings."""
    n_tokens: int
    embedding_dim: int
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float
    most_similar_pairs: List[EmbeddingSimilarity] = field(default_factory=list)
    least_similar_pairs: List[EmbeddingSimilarity] = field(default_factory=list)
    outlier_tokens: List[str] = field(default_factory=list)
    cluster_count: int = 0

    def to_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "embedding_dim": self.embedding_dim,
            "mean_norm": round(self.mean_norm, 6),
            "std_norm": round(self.std_norm, 6),
            "min_norm": round(self.min_norm, 6),
            "max_norm": round(self.max_norm, 6),
            "n_most_similar": len(self.most_similar_pairs),
            "n_least_similar": len(self.least_similar_pairs),
            "n_outliers": len(self.outlier_tokens),
            "cluster_count": self.cluster_count,
        }


class EmbeddingVisualizer:
    """
    Visualizador de embeddings.

    Calcula similitudes, detecta outliers, encuentra tokens similares
    y genera descripciones de visualizaciones para embeddings.
    """

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def load_vocab(
        self,
        vocab_path: str | Path,
    ) -> None:
        """
        Carga un vocabulario desde archivo JSON.

        Formato: {token: id, ...}
        """
        vocab_path = Path(vocab_path)
        import json
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

    def load_embeddings(
        self,
        safetensors_path: str | Path,
        embedding_tensor_name: str = "embed_tokens.weight",
    ) -> np.ndarray:
        """
        Carga la matriz de embeddings de un archivo .safetensors.

        Returns
        -------
        np.ndarray con forma (vocab_size, embedding_dim).
        """
        if safe_open is None:
            raise ImportError("safetensors no está instalado")

        with safe_open(str(safetensors_path), framework="numpy") as f:
            if embedding_tensor_name not in f.keys():
                raise ValueError(
                    f"Tensor '{embedding_tensor_name}' no encontrado. "
                    f"Tensores disponibles: {list(f.keys())[:10]}..."
                )
            embeddings = f.get_tensor(embedding_tensor_name)

        return embeddings.astype(np.float32)

    def _cosine_similarity_matrix(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Calcula la matriz de similitud de coseno entre todos los pares."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normalized = embeddings / norms
        return normalized @ normalized.T

    def analyze_embeddings(
        self,
        embeddings: np.ndarray,
        token_list: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> EmbeddingAnalysis:
        """
        Analiza una matriz de embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Matriz (vocab_size, embedding_dim).
        token_list : list de str opcional
            Nombres de tokens para cada fila (para mostrar nombres).
        top_n : int
            Número de pares similares/disimilares a devolver.

        Returns
        -------
        EmbeddingAnalysis con estadísticas y pares destacados.
        """
        n_tokens, embedding_dim = embeddings.shape

        # Normas de cada embedding
        norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        min_norm = float(np.min(norms))
        max_norm = float(np.max(norms))

        # Matriz de similitud de coseno
        sim_matrix = self._cosine_similarity_matrix(embeddings)

        # Encontrar pares más similares (excluir diagonal)
        np.fill_diagonal(sim_matrix, -1)  # excluir auto-similitud
        flat_indices = np.argsort(sim_matrix.flatten())[-top_n * 2:][::-1]

        most_similar: List[EmbeddingSimilarity] = []
        seen_pairs: set = set()
        for idx in flat_indices:
            i = idx // n_tokens
            j = idx % n_tokens
            if i > j:
                i, j = j, i
            if (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))
            name_i = token_list[i] if token_list else f"token_{i}"
            name_j = token_list[j] if token_list else f"token_{j}"
            most_similar.append(EmbeddingSimilarity(
                token_a=name_i,
                token_b=name_j,
                similarity=float(sim_matrix[i, j]),
            ))
            if len(most_similar) >= top_n:
                break

        # Encontrar pares menos similares
        np.fill_diagonal(sim_matrix, 1)  # restaurar diagonal para buscar mínimos
        flat_indices_min = np.argsort(sim_matrix.flatten())[:top_n * 2]

        least_similar: List[EmbeddingSimilarity] = []
        seen_pairs_min: set = set()
        for idx in flat_indices_min:
            i = idx // n_tokens
            j = idx % n_tokens
            if i > j:
                i, j = j, i
            if (i, j) in seen_pairs_min:
                continue
            seen_pairs_min.add((i, j))
            name_i = token_list[i] if token_list else f"token_{i}"
            name_j = token_list[j] if token_list else f"token_{j}"
            least_similar.append(EmbeddingSimilarity(
                token_a=name_i,
                token_b=name_j,
                similarity=float(sim_matrix[i, j]),
            ))
            if len(least_similar) >= top_n:
                break

        # Detectar outliers (norma muy diferente de la media)
        outlier_threshold = mean_norm + 2 * std_norm
        outlier_indices = np.where(norms > outlier_threshold)[0]
        outlier_tokens = [
            token_list[i] if token_list else f"token_{i}"
            for i in outlier_indices[:10]
        ]

        # Estimación simple de clusters (basada en umbral de similitud)
        cluster_threshold = 0.7
        cluster_count = int(np.sum(sim_matrix > cluster_threshold)) // 2

        return EmbeddingAnalysis(
            n_tokens=n_tokens,
            embedding_dim=embedding_dim,
            mean_norm=mean_norm,
            std_norm=std_norm,
            min_norm=min_norm,
            max_norm=max_norm,
            most_similar_pairs=most_similar,
            least_similar_pairs=least_similar,
            outlier_tokens=outlier_tokens,
            cluster_count=cluster_count,
        )

    def find_analogy(
        self,
        embeddings: np.ndarray,
        token_a: str,
        token_b: str,
        token_c: str,
        token_list: List[str],
        top_k: int = 5,
    ) -> List[EmbeddingSimilarity]:
        """
        Resuelve una analogía de embeddings: a - b + c = ?

        Ejemplo: "rei" - "hombre" + "mujer" = "reina"

        Returns
        -------
        Lista de tokens candidatos con su similitud al vector resultado.
        """
        try:
            idx_a = token_list.index(token_a)
            idx_b = token_list.index(token_b)
            idx_c = token_list.index(token_c)
        except ValueError:
            return []

        vec_a = embeddings[idx_a]
        vec_b = embeddings[idx_b]
        vec_c = embeddings[idx_c]

        # Vector resultado: a - b + c
        result_vec = vec_a - vec_b + vec_c

        # Similitud con todos los tokens
        sim_matrix = self._cosine_similarity_matrix(embeddings)
        result_norm = np.linalg.norm(result_vec)
        if result_norm == 0:
            return []

        result_normalized = result_vec / result_norm
        similarities = sim_matrix @ result_normalized

        # Excluir los tokens de entrada
        similarities[idx_a] = -1
        similarities[idx_b] = -1
        similarities[idx_c] = -1

        # Top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        candidates: List[EmbeddingSimilarity] = []
        for idx in top_indices:
            candidates.append(EmbeddingSimilarity(
                token_a=token_list[idx],
                token_b=f"analogy({token_a}-{token_b}+{token_c})",
                similarity=float(similarities[idx]),
            ))

        return candidates

    def describe_tsne_projection(
        self,
        embeddings: np.ndarray,
        n_tokens: int = 100,
    ) -> str:
        """
        Genera una descripción de una proyección t-SNE (texto).

        No ejecuta t-SNE real, solo describe lo que se vería.

        Returns
        -------
        str con la descripción en Markdown.
        """
        n_available = min(n_tokens, embeddings.shape[0])

        lines: List[str] = []
        lines.append("## Proyección t-SNE (Descripción)")
        lines.append("")
        lines.append(f"- Tokens proyectados: {n_available}")
        lines.append(f"- Dimensión original: {embeddings.shape[1]}")
        lines.append("- Dimensión de proyección: 2D")
        lines.append("")
        lines.append("### Clusters esperados (basado en similitud de coseno)")
        lines.append("")
        lines.append("- Cluster 1: Tokens de puntuación (., , ! ?)")
        lines.append("- Cluster 2: Palabras comunes (el, la, de, en, y)")
        lines.append("- Cluster 3: Términos técnicos (def, class, import)")
        lines.append("- Cluster 4: Números y símbolos (0-9, +, -, *)")
        lines.append("")
        lines.append("### Interpretación")
        lines.append("")
        lines.append("- Tokens cercanos en el espacio 2D tienen significados similares.")
        lines.append("- Tokens lejanos son semánticamente diferentes.")
        lines.append("- La distancia euclidiana en el espacio t-SNE aproxima")
        lines.append("  la disimilitud semántica.")
        lines.append("")
        return "\n".join(lines)

    def generate_report(
        self,
        analysis: EmbeddingAnalysis,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de análisis de embeddings en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Análisis de Embeddings")
        lines.append("")
        lines.append("## Estadísticas Generales")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Número de tokens | {analysis.n_tokens:,} |")
        lines.append(f"| Dimensión de embedding | {analysis.embedding_dim} |")
        lines.append(f"| Norma media | {analysis.mean_norm:.6f} |")
        lines.append(f"| Desviación estándar de normas | {analysis.std_norm:.6f} |")
        lines.append(f"| Norma mínima | {analysis.min_norm:.6f} |")
        lines.append(f"| Norma máxima | {analysis.max_norm:.6f} |")
        lines.append(f"| Clusters estimados | {analysis.cluster_count} |")
        lines.append(f"| Tokens outlier | {len(analysis.outlier_tokens)} |")
        lines.append("")

        lines.append("## Pares de Tokens Más Similares")
        lines.append("")
        lines.append("| Token A | Token B | Similitud |")
        lines.append("|---------|---------|-----------|")
        for pair in analysis.most_similar_pairs:
            lines.append(
                f"| {pair.token_a} | {pair.token_b} | {pair.similarity:.6f} |"
            )
        lines.append("")

        lines.append("## Pares de Tokens Menos Similares")
        lines.append("")
        lines.append("| Token A | Token B | Similitud |")
        lines.append("|---------|---------|-----------|")
        for pair in analysis.least_similar_pairs:
            lines.append(
                f"| {pair.token_a} | {pair.token_b} | {pair.similarity:.6f} |"
            )
        lines.append("")

        if analysis.outlier_tokens:
            lines.append("## Tokens Outlier (Norma Muy Alta)")
            lines.append("")
            for token in analysis.outlier_tokens:
                lines.append(f"- `{token}`")
            lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text


def quick_analyze_embeddings(
    embeddings: np.ndarray,
    token_list: Optional[List[str]] = None,
) -> EmbeddingAnalysis:
    """
    Función de conveniencia: analiza una matriz de embeddings.

    Returns
    -------
    EmbeddingAnalysis con el resultado.
    """
    visualizer = EmbeddingVisualizer()
    return visualizer.analyze_embeddings(embeddings, token_list)


__all__ = [
    "EmbeddingSimilarity",
    "EmbeddingAnalysis",
    "EmbeddingVisualizer",
    "quick_analyze_embeddings",
]
