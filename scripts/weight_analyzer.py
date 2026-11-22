#!/usr/bin/env python3
"""
weight_analyzer.py — Analizador de pesos del modelo 800M.

Proporciona herramientas para inspeccionar y entender los pesos de un
modelo Llama en formato .safetensors, sin necesidad de entrenar:

  - Estadísticas por tensor (media, std, min, max, percentiles).
  - Detección de filas/columnas anómalas (valores extremos).
  - Análisis de la matriz de embeddings: clustering básico, tokens más
    y menos similares.
  - Análisis de lm_head: qué tokens tienen mayor conexión con cada
    característica de salida.
  - Análisis de atención: qué cabezas tienen patrones estructurados.
  - Análisis de MLP: qué neuronas SwiGLU tienen pesos dominantes.
  - Comparación entre dos modelos (diff de pesos).
  - Reportes en JSON y texto legible.

Todo funciona sobre pesos cargados en memoria (NumPy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None  # type: ignore


class TensorStats:
    """Estadísticas descriptivas de un tensor."""

    def __init__(self, name: str, arr: np.ndarray) -> None:
        self.name = name
        self.shape = tuple(arr.shape)
        self.dtype = str(arr.dtype)
        self.size = int(arr.size)
        self.mean = float(np.mean(arr.astype(np.float64)))
        self.std = float(np.std(arr.astype(np.float64)))
        self.min = float(np.min(arr.astype(np.float64)))
        self.max = float(np.max(arr.astype(np.float64)))
        self.median = float(np.median(arr.astype(np.float64)))
        self.p1 = float(np.percentile(arr.astype(np.float64), 1))
        self.p99 = float(np.percentile(arr.astype(np.float64), 99))
        self.nonzero_pct = float(
            np.count_nonzero(arr) / arr.size * 100.0
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "size": self.size,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "p1": self.p1,
            "p99": self.p99,
            "nonzero_pct": self.nonzero_pct,
        }


class WeightAnalyzer:
    """
    Analizador principal de pesos del modelo.
    """

    def __init__(
        self,
        model_dir: str | Path,
        weights_filename: str = "model.safetensors",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / weights_filename
        self._tensors: Dict[str, np.ndarray] = {}
        self._loaded = False

        if safe_open is None:
            raise ImportError("safetensors no está instalado")

    def load(self) -> "WeightAnalyzer":
        """Carga todos los tensores en memoria."""
        self._tensors = {}
        with safe_open(str(self.weights_path), framework="numpy") as f:
            for name in f.keys():
                self._tensors[name] = f.get_tensor(name)
        self._loaded = True
        return self

    def is_loaded(self) -> bool:
        return self._loaded

    def tensor_names(self) -> List[str]:
        return sorted(self._tensors.keys())

    def get_tensor(self, name: str) -> np.ndarray:
        if not self._loaded:
            self.load()
        if name not in self._tensors:
            raise KeyError(f"Tensor no encontrado: {name}")
        return self._tensors[name]

    def stats_for_tensor(self, name: str) -> TensorStats:
        return TensorStats(name, self.get_tensor(name))

    def all_stats(self) -> List[TensorStats]:
        return [self.stats_for_tensor(n) for n in self.tensor_names()]

    def print_summary(self) -> None:
        """Imprime un resumen estadístico de todos los tensores."""
        print("=" * 80)
        print("RESUMEN DE PESOS DEL MODELO")
        print("=" * 80)
        total_params = 0
        for stats in self.all_stats():
            total_params += stats.size
            print(
                f"{stats.name:50s} | "
                f"shape={str(stats.shape):25s} | "
                f"mean={stats.mean: .6f} | "
                f"std={stats.std: .6f} | "
                f"min={stats.min: .6f} | "
                f"max={stats.max: .6f} | "
                f"nnz={stats.nonzero_pct:5.1f}%"
            )
        print("=" * 80)
        print(f"Total parámetros: {total_params:,}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Análisis de embeddings
    # ------------------------------------------------------------------

    def embedding_similarity(
        self,
        token_id_a: int,
        token_id_b: int,
        tensor_name: str = "model.embed_tokens.weight",
    ) -> float:
        """
        Calcula la similitud de coseno entre dos embeddings de token.
        """
        emb = self.get_tensor(tensor_name)
        if emb.dtype == np.int8:
            from scripts.quantization_tools import dequantize_int8_per_row
            scale_name = tensor_name + "_scale"
            if scale_name in self._tensors:
                # Nota: aquí asumimos que las escalas están cargadas por separado
                pass
        vec_a = emb[token_id_a].astype(np.float32)
        vec_b = emb[token_id_b].astype(np.float32)
        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def most_similar_tokens(
        self,
        token_id: int,
        top_k: int = 10,
        tensor_name: str = "model.embed_tokens.weight",
    ) -> List[Tuple[int, float]]:
        """
        Devuelve los top_k tokens más similares (coseno) a un token dado.
        """
        emb = self.get_tensor(tensor_name).astype(np.float32)
        vec = emb[token_id]
        norms = np.linalg.norm(emb, axis=1)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return []
        sims = emb @ vec / (norms * vec_norm)
        sims[token_id] = -1.0  # excluir el propio token
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_idx]

    def embedding_norm_distribution(
        self,
        tensor_name: str = "model.embed_tokens.weight",
    ) -> Dict[str, float]:
        """
        Estadísticas de normas de los vectores de embedding.
        """
        emb = self.get_tensor(tensor_name).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1)
        return {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "p1_norm": float(np.percentile(norms, 1)),
            "p99_norm": float(np.percentile(norms, 99)),
        }

    # ------------------------------------------------------------------
    # Análisis de lm_head
    # ------------------------------------------------------------------

    def lm_head_top_connections(
        self,
        feature_col: int,
        top_k: int = 10,
        tensor_name: str = "lm_head.weight",
    ) -> List[Tuple[int, float]]:
        """
        Devuelve los top_k tokens con mayor peso en una columna de lm_head.
        """
        head = self.get_tensor(tensor_name).astype(np.float32)
        col = head[:, feature_col]
        top_idx = np.argsort(col)[-top_k:][::-1]
        return [(int(i), float(col[i])) for i in top_idx]

    def lm_head_column_stats(
        self,
        tensor_name: str = "lm_head.weight",
    ) -> Dict[str, object]:
        """
        Estadísticas por columna de lm_head (qué tan concentrados son los pesos).
        """
        head = self.get_tensor(tensor_name).astype(np.float32)
        col_means = np.mean(np.abs(head), axis=0)
        return {
            "mean_abs_per_column": float(np.mean(col_means)),
            "std_abs_per_column": float(np.std(col_means)),
            "max_column_mean": float(np.max(col_means)),
            "min_column_mean": float(np.min(col_means)),
            "num_columns": int(head.shape[1]),
        }

    # ------------------------------------------------------------------
    # Análisis de atención
    # ------------------------------------------------------------------

    def attention_layer_stats(
        self,
        layer_idx: int,
    ) -> Dict[str, object]:
        """
        Estadísticas de las matrices de atención de una capa (q, k, v, o).
        """
        prefix = f"model.layers.{layer_idx}.self_attn"
        result: Dict[str, object] = {}
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"{prefix}.{proj}.weight"
            if name in self._tensors:
                stats = self.stats_for_tensor(name)
                result[proj] = stats.to_dict()
        return result

    def find_dominant_rows(
        self,
        tensor_name: str,
        threshold_std: float = 3.0,
        max_rows: int = 20,
    ) -> List[Tuple[int, float]]:
        """
        Encuentra filas con valores absolutamente dominantes (más de
        threshold_std desviaciones estándar por encima de la media).

        Útil para detectar pesos que han sido editados manualmente con
        valores anómalos.
        """
        arr = self.get_tensor(tensor_name)
        if arr.dtype == np.int8:
            arr_f = arr.astype(np.float32)
        else:
            arr_f = arr.astype(np.float32)

        row_abs_mean = np.mean(np.abs(arr_f), axis=1)
        mean = float(np.mean(row_abs_mean))
        std = float(np.std(row_abs_mean))
        threshold = mean + threshold_std * std

        dominant = []
        for i in range(arr.shape[0]):
            if row_abs_mean[i] > threshold:
                dominant.append((i, float(row_abs_mean[i])))

        dominant.sort(key=lambda x: -x[1])
        return dominant[:max_rows]

    # ------------------------------------------------------------------
    # Análisis de MLP
    # ------------------------------------------------------------------

    def mlp_layer_stats(
        self,
        layer_idx: int,
    ) -> Dict[str, object]:
        """
        Estadísticas de las matrices MLP de una capa (gate, up, down).
        """
        prefix = f"model.layers.{layer_idx}.mlp"
        result: Dict[str, object] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.{proj}.weight"
            if name in self._tensors:
                stats = self.stats_for_tensor(name)
                result[proj] = stats.to_dict()
        return result

    def dominant_mlp_neurons(
        self,
        layer_idx: int,
        proj: str = "gate_proj",
        top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        """
        Encuentra las neuronas con pesos de mayor magnitud en una
        proyección MLP de una capa.
        """
        prefix = f"model.layers.{layer_idx}.mlp"
        name = f"{prefix}.{proj}.weight"
        if name not in self._tensors:
            return []
        arr = self.get_tensor(name).astype(np.float32)
        row_magnitude = np.linalg.norm(arr, axis=1)
        top_idx = np.argsort(row_magnitude)[-top_k:][::-1]
        return [(int(i), float(row_magnitude[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Comparación entre modelos
    # ------------------------------------------------------------------

    def compare_with(
        self,
        other: "WeightAnalyzer",
        tensor_name: str,
        metric: str = "abs_diff_mean",
    ) -> float:
        """
        Compara un tensor entre este analizador y otro.

        metric: "abs_diff_mean", "max_abs_diff", "cosine_sim"
        """
        a = self.get_tensor(tensor_name).astype(np.float32)
        b = other.get_tensor(tensor_name).astype(np.float32)

        if a.shape != b.shape:
            raise ValueError(
                f"Formas diferentes: {a.shape} vs {b.shape}"
            )

        diff = a - b
        if metric == "abs_diff_mean":
            return float(np.mean(np.abs(diff)))
        elif metric == "max_abs_diff":
            return float(np.max(np.abs(diff)))
        elif metric == "cosine_sim":
            dot = float(np.sum(a * b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            return dot / (na * nb) if na > 0 and nb > 0 else 0.0
        else:
            raise ValueError(f"Métrica desconocida: {metric}")

    def full_model_diff(
        self,
        other: "WeightAnalyzer",
    ) -> Dict[str, float]:
        """
        Calcula diferencias entre dos modelos para todos los tensores
        comunes.
        """
        result: Dict[str, float] = {}
        common = set(self.tensor_names()) & set(other.tensor_names())
        for name in sorted(common):
            result[name] = self.compare_with(other, name, "abs_diff_mean")
        return result

    # ------------------------------------------------------------------
    # Reporte completo
    # ------------------------------------------------------------------

    def full_report(self) -> dict:
        """
        Genera un reporte completo del modelo como diccionario.
        """
        report: dict = {
            "model_path": str(self.weights_path),
            "tensor_count": len(self._tensors),
            "total_parameters": sum(t.size for t in self._tensors.values()),
            "tensor_stats": [s.to_dict() for s in self.all_stats()],
        }

        # Estadísticas de lm_head
        if "lm_head.weight" in self._tensors:
            report["lm_head"] = {
                "column_stats": self.lm_head_column_stats(),
                "top_connections_col_640": self.lm_head_top_connections(640, 5),
                "top_connections_col_1280": self.lm_head_top_connections(1280, 5),
            }

        # Estadísticas de embeddings
        if "model.embed_tokens.weight" in self._tensors:
            report["embeddings"] = {
                "norm_distribution": self.embedding_norm_distribution(),
            }

        # Estadísticas por capa
        report["layers"] = {}
        for i in range(21):  # modelo tiene 21 capas
            prefix = f"model.layers.{i}"
            if any(k.startswith(prefix) for k in self._tensors):
                report["layers"][str(i)] = {
                    "attention": self.attention_layer_stats(i),
                    "mlp": self.mlp_layer_stats(i),
                }

        return report

    def save_report(self, path: str | Path) -> None:
        """Guarda el reporte completo en JSON."""
        with open(path, "w") as f:
            json.dump(self.full_report(), f, indent=2)


def quick_analyze(
    model_dir: str | Path,
    output_json: Optional[str | Path] = None,
) -> dict:
    """
    Función de conveniencia: carga, analiza e imprime un reporte del modelo.
    """
    analyzer = WeightAnalyzer(model_dir)
    analyzer.load()
    analyzer.print_summary()

    if output_json:
        analyzer.save_report(output_json)

    return analyzer.full_report()


__all__ = [
    "TensorStats",
    "WeightAnalyzer",
    "quick_analyze",
]
