#!/usr/bin/env python3
"""
data_analyzer.py — Analizador estadístico de datos para el modelo 800M.

Proporciona funciones para calcular estadísticas descriptivas, inferenciales
y exploratorias sobre conjuntos de datos numéricos y categóricos. Está
diseñado para usarse como referencia de lo que un modelo de análisis de
datos debería ser capaz de hacer:

  - Medidas de tendencia central (media, mediana, moda).
  - Medidas de dispersión (varianza, desviación estándar, rango, IQR).
  - Medidas de forma (asimetría, curtosis).
  - Detección de valores atípicos (outliers) por IQR y Z-score.
  - Correlación y covarianza entre variables.
  - Regresión lineal simple.
  - Pruebas de hipótesis básicas (t-test de una muestra).
  - Intervallo de confianza.
  - Generación de reportes en texto y JSON.

Todo implementado en NumPy puro, sin pandas ni scipy, para mantener
dependencias mínimas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Medidas de tendencia central
# ---------------------------------------------------------------------------

def mean(data: List[float]) -> float:
    """Media aritmética."""
    if not data:
        raise ValueError("data no puede estar vacío")
    return float(np.mean(data))


def median(data: List[float]) -> float:
    """Mediana."""
    if not data:
        raise ValueError("data no puede estar vacío")
    return float(np.median(data))


def mode(data: List[float]) -> Tuple[float, int]:
    """
    Moda (valor más frecuente) y su conteo.

    Returns
    -------
    (valor_moda, conteo)
    """
    if not data:
        raise ValueError("data no puede estar vacío")
    values, counts = np.unique(data, return_counts=True)
    max_idx = int(np.argmax(counts))
    return float(values[max_idx]), int(counts[max_idx])


def geometric_mean(data: List[float]) -> float:
    """Media geométrica (solo para datos positivos)."""
    if not data:
        raise ValueError("data no puede estar vacío")
    arr = np.array(data, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("geometric_mean requiere todos los valores > 0")
    return float(np.exp(np.mean(np.log(arr))))


def harmonic_mean(data: List[float]) -> float:
    """Media armónica (solo para datos no nulos)."""
    if not data:
        raise ValueError("data no puede estar vacío")
    arr = np.array(data, dtype=np.float64)
    if np.any(arr == 0):
        raise ValueError("harmonic_mean no permite ceros")
    return float(len(arr) / np.sum(1.0 / arr))


# ---------------------------------------------------------------------------
# Medidas de dispersión
# ---------------------------------------------------------------------------

def variance(data: List[float], sample: bool = True) -> float:
    """Varianza (muestral por defecto, ddof=1)."""
    if len(data) < 2:
        raise ValueError("se necesitan al menos 2 valores para varianza")
    ddof = 1 if sample else 0
    return float(np.var(data, ddof=ddof))


def std_dev(data: List[float], sample: bool = True) -> float:
    """Desviación estándar."""
    return math.sqrt(variance(data, sample=sample))


def range_stat(data: List[float]) -> Tuple[float, float]:
    """Rango (mínimo, máximo)."""
    if not data:
        raise ValueError("data no puede estar vacío")
    return float(np.min(data)), float(np.max(data))


def iqr(data: List[float]) -> float:
    """Rango intercuartílico (Q3 - Q1)."""
    if not data:
        raise ValueError("data no puede estar vacío")
    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    return q3 - q1


def mad(data: List[float]) -> float:
    """Desviación absoluta mediana desde la mediana."""
    if not data:
        raise ValueError("data no puede estar vacío")
    med = np.median(data)
    return float(np.median(np.abs(data - med)))


# ---------------------------------------------------------------------------
# Medidas de forma
# ---------------------------------------------------------------------------

def skewness(data: List[float]) -> float:
    """
    Asimetría de Pearson (tercer momento estandarizado).

    Positiva = cola a la derecha, negativa = cola a la izquierda.
    """
    if len(data) < 3:
        raise ValueError("se necesitan al menos 3 valores para asimetría")
    arr = np.array(data, dtype=np.float64)
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    n = len(arr)
    return float(np.sum((arr - m) ** 3) / n / s ** 3)


def kurtosis(data: List[float]) -> float:
    """
    Curtosis excesiva (cuarto momento estandarizado - 3).

    Positiva = cola pesada (leptocúrtica), negativa = cola ligera (platicúrtica).
    """
    if len(data) < 4:
        raise ValueError("se necesitan al menos 4 valores para curtosis")
    arr = np.array(data, dtype=np.float64)
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    n = len(arr)
    return float(np.sum((arr - m) ** 4) / n / s ** 4 - 3.0)


# ---------------------------------------------------------------------------
# Detección de outliers
# ---------------------------------------------------------------------------

def detect_outliers_iqr(
    data: List[float],
    k: float = 1.5,
) -> List[int]:
    """
    Detecta índices de valores atípicos usando el criterio de IQR.

    Un valor es outlier si está por debajo de Q1 - k*IQR o por encima de
    Q3 + k*IQR.

    Returns
    -------
    lista de índices (0-based) de outliers.
    """
    if not data:
        return []
    arr = np.array(data, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr_val = q3 - q1
    lower = q1 - k * iqr_val
    upper = q3 + k * iqr_val
    outlier_indices = [i for i, v in enumerate(arr) if v < lower or v > upper]
    return outlier_indices


def detect_outliers_zscore(
    data: List[float],
    threshold: float = 3.0,
) -> List[int]:
    """
    Detecta índices de valores atípicos usando Z-score.

    Un valor es outlier si |z| > threshold.

    Returns
    -------
    lista de índices (0-based) de outliers.
    """
    if len(data) < 2:
        return []
    arr = np.array(data, dtype=np.float64)
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return []
    z_scores = (arr - m) / s
    outlier_indices = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    return outlier_indices


# ---------------------------------------------------------------------------
# Correlación y covarianza
# ---------------------------------------------------------------------------

def covariance(
    x: List[float],
    y: List[float],
    sample: bool = True,
) -> float:
    """Covarianza entre dos variables."""
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud")
    if len(x) < 2:
        raise ValueError("se necesitan al menos 2 pares")
    ddof = 1 if sample else 0
    return float(np.cov(x, y, ddof=ddof)[0, 1])


def correlation(
    x: List[float],
    y: List[float],
    sample: bool = True,
) -> float:
    """
    Correlación de Pearson entre dos variables.

    Returns valor en [-1, 1].
    """
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud")
    if len(x) < 2:
        raise ValueError("se necesitan al menos 2 pares")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_matrix(
    data: List[List[float]],
) -> np.ndarray:
    """
    Matriz de correlación para múltiples variables (columnas).

    data: lista de listas, cada lista interna es una variable.
    """
    arr = np.array(data, dtype=np.float64)
    return np.corrcoef(arr)


# ---------------------------------------------------------------------------
# Regresión lineal simple
# ---------------------------------------------------------------------------

class LinearRegression:
    """
    Regresión lineal simple y múltiple implementada con NumPy.

    y = X @ beta + epsilon
    """

    def __init__(self) -> None:
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.r_squared: float = 0.0

    def fit(
        self,
        X: List[List[float]],
        y: List[float],
    ) -> "LinearRegression":
        """
        Ajusta el modelo usando mínimos cuadrados ordinarios.

        X: matriz de características (n_samples, n_features)
        y: vector objetivo (n_samples,)
        """
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)

        # Agregar columna de unos para el intercepto
        X_with_intercept = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        # Solución normal: beta = (X^T X)^{-1} X^T y
        XtX = X_with_intercept.T @ X_with_intercept
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ X_with_intercept.T @ y_arr

        self.intercept = float(beta[0])
        self.coefficients = beta[1:]

        # R^2
        y_pred = self.predict(X)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        self.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return self

    def predict(
        self,
        X: List[List[float]],
    ) -> np.ndarray:
        """Predice valores para X."""
        if self.coefficients is None:
            raise ValueError("Modelo no ajustado. Llamar fit() primero.")
        X_arr = np.array(X, dtype=np.float64)
        return self.intercept + X_arr @ self.coefficients

    def summary(self) -> Dict[str, object]:
        """Devuelve un resumen del modelo ajustado."""
        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist()
            if self.coefficients is not None else None,
            "r_squared": self.r_squared,
            "n_features": len(self.coefficients)
            if self.coefficients is not None else 0,
        }


def simple_linear_regression(
    x: List[float],
    y: List[float],
) -> Dict[str, float]:
    """
    Regresión lineal simple (una variable independiente).

    Returns
    -------
    dict con pendiente, intercepto, r_squared.
    """
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud")
    if len(x) < 2:
        raise ValueError("se necesitan al menos 2 puntos")

    model = LinearRegression()
    model.fit([[xi] for xi in x], y)
    return {
        "slope": float(model.coefficients[0]),
        "intercept": model.intercept,
        "r_squared": model.r_squared,
    }


# ---------------------------------------------------------------------------
# Pruebas de hipótesis
# ---------------------------------------------------------------------------

def one_sample_t_test(
    data: List[float],
    mu: float = 0.0,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Prueba t de una muestra: H0: media = mu vs H1: media != mu.

    Returns
    -------
    dict con t_statistico, p_valor_aproximado, rechazar_h0, intervalo_confianza.
    """
    if len(data) < 2:
        raise ValueError("se necesitan al menos 2 valores")
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    se = s / math.sqrt(n)
    t_stat = (m - mu) / se if se > 0 else 0.0

    # Valor crítico aproximado (distribución t, dos colas, alpha=0.05)
    # Para n > 30, t_critico ≈ 1.96. Para n pequeño, usar tabla simplificada.
    t_critical_table = {
        2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45,
        8: 2.37, 9: 2.31, 10: 2.26, 15: 2.13, 20: 2.09, 30: 2.04,
    }
    t_crit = t_critical_table.get(min(n, 30), 1.96)
    if n > 30:
        t_crit = 1.96

    reject = abs(t_stat) > t_crit

    # Intervalo de confianza aproximado
    margin = t_crit * se
    ci_lower = m - margin
    ci_upper = m + margin

    return {
        "t_statistic": float(t_stat),
        "t_critical": float(t_crit),
        "sample_mean": float(m),
        "sample_std": float(s),
        "n": n,
        "alpha": alpha,
        "reject_null": bool(reject),
        "confidence_interval": [float(ci_lower), float(ci_upper)],
    }


def confidence_interval_mean(
    data: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Intervalo de confianza para la media poblacional.

    Usa distribución t (aproximada con valores críticos simplificados).
    """
    if len(data) < 2:
        raise ValueError("se necesitan al menos 2 valores")
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    se = s / math.sqrt(n)

    # Valor crítico aproximado
    t_critical_table = {
        0.90: {10: 1.81, 20: 1.72, 30: 1.70},
        0.95: {10: 2.23, 20: 2.09, 30: 2.04},
        0.99: {10: 3.17, 20: 2.85, 30: 2.75},
    }
    table = t_critical_table.get(confidence, t_critical_table[0.95])
    t_crit = table.get(min(n, 30), 1.96)
    if n > 30:
        if confidence == 0.90:
            t_crit = 1.645
        elif confidence == 0.95:
            t_crit = 1.96
        else:
            t_crit = 2.576

    margin = t_crit * se
    return float(m - margin), float(m + margin)


# ---------------------------------------------------------------------------
# Estadísticas descriptivas completas
# ---------------------------------------------------------------------------

def descriptive_statistics(
    data: List[float],
) -> Dict[str, float]:
    """
    Calcula todas las estadísticas descriptivas principales de un dataset.

    Returns
    -------
    dict con: n, mean, median, mode, mode_count, variance, std_dev,
              min, max, range, iqr, mad, skewness, kurtosis,
              geometric_mean, harmonic_mean.
    """
    if not data:
        raise ValueError("data no puede estar vacío")
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    m = float(np.mean(arr))
    med = float(np.median(arr))
    mod_val, mod_count = mode(data)
    var = variance(data, sample=True)
    sd = std_dev(data, sample=True)
    min_val, max_val = range_stat(data)
    rng = max_val - min_val
    iqr_val = iqr(data)
    mad_val = mad(data)
    skew = skewness(data)
    kurt = kurtosis(data)

    try:
        gm = geometric_mean(data)
    except ValueError:
        gm = float("nan")
    try:
        hm = harmonic_mean(data)
    except ValueError:
        hm = float("nan")

    return {
        "n": float(n),
        "mean": m,
        "median": med,
        "mode": mod_val,
        "mode_count": float(mod_count),
        "variance": var,
        "std_dev": sd,
        "min": min_val,
        "max": max_val,
        "range": rng,
        "iqr": iqr_val,
        "mad": mad_val,
        "skewness": skew,
        "kurtosis": kurt,
        "geometric_mean": gm,
        "harmonic_mean": hm,
    }


def frequency_table(
    data: List[float],
    n_bins: int = 10,
) -> Dict[str, object]:
    """
    Tabla de frecuencias con histogramas.

    Returns
    -------
    dict con bin_edges, bin_counts, bin_frequencies.
    """
    if not data:
        return {"bin_edges": [], "bin_counts": [], "bin_frequencies": []}
    arr = np.array(data, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=n_bins)
    total = len(arr)
    freqs = counts / total if total > 0 else counts
    return {
        "bin_edges": edges.tolist(),
        "bin_counts": counts.tolist(),
        "bin_frequencies": freqs.tolist(),
    }


# ---------------------------------------------------------------------------
# Reporte de análisis
# ---------------------------------------------------------------------------

def analyze_dataset(
    data: List[float],
    name: str = "dataset",
) -> Dict[str, object]:
    """
    Ejecuta un análisis completo de un dataset numérico.

    Returns
    -------
    dict con estadísticas descriptivas, outliers (IQR y Z-score),
    tabla de frecuencias, e interpretación básica.
    """
    if not data:
        raise ValueError("data no puede estar vacío")

    stats = descriptive_statistics(data)
    outliers_iqr = detect_outliers_iqr(data)
    outliers_z = detect_outliers_zscore(data)
    freq = frequency_table(data)

    # Interpretación de asimetría
    skew = stats["skewness"]
    if skew > 1:
        skew_interp = "fuertemente sesgada a la derecha"
    elif skew > 0.5:
        skew_interp = "moderadamente sesgada a la derecha"
    elif skew < -1:
        skew_interp = "fuertemente sesgada a la izquierda"
    elif skew < -0.5:
        skew_interp = "moderadamente sesgada a la izquierda"
    else:
        skew_interp = "aproximadamente simétrica"

    # Interpretación de curtosis
    kurt = stats["kurtosis"]
    if kurt > 1:
        kurt_interp = "cola pesada (leptocúrtica)"
    elif kurt < -1:
        kurt_interp = "cola ligera (platicúrtica)"
    else:
        kurt_interp = "curtosis normal (mesocúrtica)"

    return {
        "name": name,
        "n": int(stats["n"]),
        "descriptive_statistics": stats,
        "outliers_iqr": outliers_iqr,
        "outliers_iqr_count": len(outliers_iqr),
        "outliers_zscore": outliers_z,
        "outliers_zscore_count": len(outliers_z),
        "frequency_table": freq,
        "interpretation": {
            "skewness": skew_interp,
            "kurtosis": kurt_interp,
            "has_outliers": len(outliers_iqr) > 0 or len(outliers_z) > 0,
        },
    }


def compare_datasets(
    data_a: List[float],
    data_b: List[float],
    name_a: str = "A",
    name_b: str = "B",
) -> Dict[str, object]:
    """
    Compara dos conjuntos de datos numéricos.

    Returns
    -------
    dict con estadísticas de cada uno, diferencia de medias, y prueba t
    aproximada para diferencia de medias (si tamaños lo permiten).
    """
    stats_a = descriptive_statistics(data_a)
    stats_b = descriptive_statistics(data_b)

    diff_mean = stats_a["mean"] - stats_b["mean"]
    diff_median = stats_a["median"] - stats_b["median"]

    return {
        "name_a": name_a,
        "name_b": name_b,
        "stats_a": stats_a,
        "stats_b": stats_b,
        "difference": {
            "mean": diff_mean,
            "median": diff_median,
            "std_dev": stats_a["std_dev"] - stats_b["std_dev"],
        },
        "relative_difference_pct": (
            (diff_mean / stats_b["mean"] * 100.0)
            if stats_b["mean"] != 0 else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------

def save_analysis(
    analysis: Dict[str, object],
    path: str | Path,
) -> None:
    """Guarda un análisis en archivo JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)


__all__ = [
    "mean", "median", "mode", "geometric_mean", "harmonic_mean",
    "variance", "std_dev", "range_stat", "iqr", "mad",
    "skewness", "kurtosis",
    "detect_outliers_iqr", "detect_outliers_zscore",
    "covariance", "correlation", "correlation_matrix",
    "LinearRegression", "simple_linear_regression",
    "one_sample_t_test", "confidence_interval_mean",
    "descriptive_statistics", "frequency_table",
    "analyze_dataset", "compare_datasets",
    "save_analysis",
]
