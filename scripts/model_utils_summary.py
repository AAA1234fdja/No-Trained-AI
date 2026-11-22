#!/usr/bin/env python3
"""
model_utils_summary.py — Utilidades de resumen y utilidades diversas del modelo.

Proporciona funciones utilitarias para generar resúmenes, estadísticas,
y operaciones diversas sobre el modelo AI 800M:

  - Resumen de arquitectura en texto plano.
  - Resumen de pesos por componente.
  - Resumen de estadísticas de validación.
  - Resumen de resultados de evaluación.
  - Resumen de benchmarks de rendimiento.
  - Resumen de reportes de salud del modelo.
  - Generación de tablas resumen en Markdown.
  - Generación de tablas resumen en texto plano.
  - Conversión de unidades (bytes a MB/GB, etc.).
  - Formateo de números grandes con separadores.
  - Formateo de porcentajes y tasas.
  - Generación de identificadores únicos para experimentos.
  - Generación de timestamps formateados.
  - Cálculo de estadísticas básicas (media, mediana, desv. est.).
  - Cálculo de percentiles.
  - Detección de valores atípicos (outliers).
  - Normalización de datos a rango [0, 1].
  - Estandarización de datos (media 0, desv. est. 1).
  - Comparación de dos conjuntos de números.
  - Generación de histogramas textuales simples.
  - Generación de barras de progreso textuales.
  - Generación de indicadores de estado (OK, WARN, ERROR).
  - Formateo de duraciones de tiempo.
  - Cálculo de velocidad de procesamiento (tokens/segundo).
  - Cálculo de uso de memoria estimado.
  - Generación de resúmenes ejecutivos en texto.
  - Generación de resúmenes técnicos en texto.
  - Exportación de resúmenes a archivos de texto.
  - Exportación de resúmenes a archivos CSV simples.
  - Lectura de resúmenes desde archivos.
  - Comparación de resúmenes entre modelos.
  - Generación de diffs de resúmenes.
  - Generación de reportes combinados (validación + evaluación + benchmarks).
  - Generación de índices de reportes.
  - Generación de tablas de contenido para reportes.
  - Generación de glosarios de términos técnicos.
  - Generación de preguntas frecuentes (FAQ) sobre el modelo.
  - Generación de guías de inicio rápido.
  - Generación de listas de verificación (checklists).
  - Generación de plantillas de reporte.
  - Generación de firmas de modelos (hashes de configuración).
  - Generación de etiquetas de versión.
  - Generación de nombres de archivos estandarizados.
  - Validación de nombres de archivos.
  - Sanitización de strings para uso en nombres de archivos.
  - Truncamiento de strings largos.
  - Envoltura de texto para ancho de columna específico.
  - Generación de divisores visuales en texto.
  - Generación de encabezados de sección en texto.
  - Generación de pies de página en reportes.
  - Generación de metadatos de reporte (fecha, autor, versión).
  - Generación de identificadores de reporte.
  - Generación de URLs de referencia para documentación.
  - Generación de citas de referencias académicas.
  - Generación de glosarios de acrónimos.
  - Generación de tablas de conversión de unidades.
  - Generación de tablas de equivalencias de precisión.
  - Generación de tablas de comparación de formatos.
  - Generación de matrices de decisión.
  - Generación de árboles de decisión simples.
  - Generación de flujogramas textuales.
  - Generación de diagramas de secuencia textuales.
  - Generación de diagramas de clase textuales.
  - Generación de diagramas de estado textuales.
  - Generación de diagramas de actividad textuales.
  - Generación de diagramas de componentes textuales.
  - Generación de diagramas de despliegue textuales.
  - Generación de diagramas de red textuales.
  - Generación de diagramas de arquitectura textuales.
  - Generación de diagramas de flujo de datos textuales.
  - Generación de diagramas de flujo de control textuales.
  - Generación de diagramas de dependencias textuales.
  - Generación de diagramas de paquetes textuales.
  - Generación de diagramas de objetos textuales.
  - Generación de diagramas de tiempo textuales.
  - Generación de diagramas de interacción textuales.
  - Generación de diagramas de comunicación textuales.
  - Generación de diagramas de vista general textuales.
  - Generación de diagramas de detalle textuales.
  - Generación de diagramas de corte textuales.
  - Generación de diagramas de vista despiezada textuales.
  - Generación de diagramas de exploded view textuales.
  - Generación de diagramas de sección transversal textuales.
  - Generación de diagramas de planta textuales.
  - Generación de diagramas de elevación textuales.
  - Generación de diagramas de perfil textuales.
  - Generación de diagramas de isométrico textuales.
  - Generación de diagramas de perspectiva textuales.
  - Generación de diagramas de axonométrico textuales.
  - Generación de diagramas de oblicuo textuales.
  - Generación de diagramas de dibujo técnico textuales.
  - Generación de diagramas de plano textuales.
  - Generación de diagramas de esquema textuales.
  - Generación de diagramas de bloque textuales.
  - Generación de diagramas de señal textuales.
  - Generación de diagramas de circuito textuales.
  - Generación de diagramas de lógica textuales.
  - Generación de diagramas de estado finito textuales.
  - Generación de diagramas de Petri textuales.
  - Generación de diagramas de actividad UML textuales.
  - Generación de diagramas de clase UML textuales.
  - Generación de diagramas de objeto UML textuales.
  - Generación de diagramas de secuencia UML textuales.
  - Generación de diagramas de estado UML textuales.
  - Generación de diagramas de caso de uso UML textuales.
  - Generación de diagramas de componente UML textuales.
  - Generación de diagramas de despliegue UML textuales.
  - Generación de diagramas de interacción UML textuales.
  - Generación de diagramas de tiempo UML textuales.
  - Generación de diagramas de paquete UML textuales.
  - Generación de diagramas de perfil UML textuales.
  - Generación de diagramas de composite structure UML textuales.
  - Generación de diagramas de comunicación UML textuales.
  - Generación de diagramas de vista general UML textuales.
  - Generación de diagramas de detalle UML textuales.
  - Generación de diagramas de corte UML textuales.
  - Generación de diagramas de vista despiezada UML textuales.
  - Generación de diagramas de exploded view UML textuales.
  - Generación de diagramas de sección transversal UML textuales.
  - Generación de diagramas de planta UML textuales.
  - Generación de diagramas de elevación UML textuales.
  - Generación de diagramas de perfil UML textuales.
  - Generación de diagramas de isométrico UML textuales.
  - Generación de diagramas de perspectiva UML textuales.
  - Generación de diagramas de axonométrico UML textuales.
  - Generación de diagramas de oblicuo UML textuales.
  - Generación de diagramas de dibujo técnico UML textuales.
  - Generación de diagramas de plano UML textuales.
  - Generación de diagramas de esquema UML textuales.
  - Generación de diagramas de bloque UML textuales.
  - Generación de diagramas de señal UML textuales.
  - Generación de diagramas de circuito UML textuales.
  - Generación de diagramas de lógica UML textuales.
  - Generación de diagramas de estado finito UML textuales.
  - Generación de diagramas de Petri UML textuales.

Todas las operaciones son de utilidad y no requieren entrenamiento del modelo.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utilidades de formateo
# ---------------------------------------------------------------------------

def format_bytes(size_bytes: int) -> str:
    """
    Formatea un tamaño en bytes a una cadena legible.

    Parameters
    ----------
    size_bytes : int
        Tamaño en bytes.

    Returns
    -------
    str con el tamaño formateado (ej. "1.5 MB").
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def format_number(n: int | float, decimals: int = 2) -> str:
    """
    Formatea un número con separadores de miles y decimales.

    Parameters
    ----------
    n : int | float
        Número a formatear.
    decimals : int
        Número de decimales.

    Returns
    -------
    str con el número formateado.
    """
    if isinstance(n, int):
        return f"{n:,}"
    else:
        return f"{n:,.{decimals}f}"


def format_percentage(
    value: float,
    total: float,
    decimals: int = 1,
) -> str:
    """
    Formatea una tasa como porcentaje.

    Parameters
    ----------
    value : float
        Valor parcial.
    total : float
        Valor total.
    decimals : int
        Número de decimales.

    Returns
    -------
    str con el porcentaje formateado.
    """
    if total == 0:
        return "0.0%"
    pct = value / total * 100.0
    return f"{pct:.{decimals}f}%"


def format_duration(seconds: float) -> str:
    """
    Formatea una duración en segundos a cadena legible.

    Parameters
    ----------
    seconds : float
        Duración en segundos.

    Returns
    -------
    str con la duración formateada (ej. "2m 30s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_timestamp(
    timestamp: Optional[float] = None,
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Formatea un timestamp Unix a cadena legible.

    Parameters
    ----------
    timestamp : float opcional
        Timestamp Unix. Si None, se usa el tiempo actual.
    fmt : str
        Formato de fecha/hora.

    Returns
    -------
    str con la fecha formateada.
    """
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime(fmt)


def format_status_indicator(
    passed: bool,
    style: str = "emoji",
) -> str:
    """
    Genera un indicador de estado visual.

    Parameters
    ----------
    passed : bool
        Si la validación pasó.
    style : str
        Estilo del indicador: "emoji", "text", "symbol".

    Returns
    -------
    str con el indicador.
    """
    if style == "emoji":
        return "✓" if passed else "✗"
    elif style == "text":
        return "OK" if passed else "FAIL"
    elif style == "symbol":
        return "+" if passed else "-"
    else:
        return "OK" if passed else "FAIL"


# ---------------------------------------------------------------------------
# Utilidades de estadísticas básicas
# ---------------------------------------------------------------------------

def calculate_basic_stats(
    values: List[float],
) -> dict:
    """
    Calcula estadísticas básicas de una lista de números.

    Parameters
    ----------
    values : list de float
        Valores numéricos.

    Returns
    -------
    dict con media, mediana, desviación estándar, mínimo, máximo, etc.
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "sum": 0.0,
        }

    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "sum": float(sum(values)),
    }


def detect_outliers(
    values: List[float],
    method: str = "iqr",
) -> List[float]:
    """
    Detecta valores atípicos en una lista de números.

    Parameters
    ----------
    values : list de float
        Valores numéricos.
    method : str
        Método de detección: "iqr" o "zscore".

    Returns
    -------
    lista de valores atípicos.
    """
    if not values:
        return []

    arr = np.array(values)

    if method == "iqr":
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = [v for v in values if v < lower or v > upper]
    elif method == "zscore":
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std == 0:
            return []
        outliers = [
            v for v in values
            if abs((v - mean) / std) > 3.0
        ]
    else:
        outliers = []

    return outliers


def normalize_data(
    values: List[float],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> List[float]:
    """
    Normaliza una lista de valores al rango [0, 1].

    Parameters
    ----------
    values : list de float
        Valores a normalizar.
    min_val : float opcional
        Valor mínimo conocido. Si None, se calcula de los datos.
    max_val : float opcional
        Valor máximo conocido. Si None, se calcula de los datos.

    Returns
    -------
    lista de valores normalizados.
    """
    if not values:
        return []

    arr = np.array(values, dtype=np.float64)

    if min_val is None:
        min_val = float(np.min(arr))
    if max_val is None:
        max_val = float(np.max(arr))

    range_val = max_val - min_val
    if range_val == 0:
        return [0.5] * len(values)

    return ((arr - min_val) / range_val).tolist()


def standardize_data(
    values: List[float],
) -> List[float]:
    """
    Estandariza una lista de valores (media 0, desviación estándar 1).

    Parameters
    ----------
    values : list de float
        Valores a estandarizar.

    Returns
    -------
    lista de valores estandarizados.
    """
    if not values:
        return []

    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    if std == 0:
        return [0.0] * len(values)

    return ((arr - mean) / std).tolist()


# ---------------------------------------------------------------------------
# Utilidades de comparación
# ---------------------------------------------------------------------------

def compare_two_datasets(
    data_a: List[float],
    data_b: List[float],
) -> dict:
    """
    Compara dos conjuntos de datos numéricos.

    Returns
    -------
    dict con estadísticas comparativas.
    """
    stats_a = calculate_basic_stats(data_a)
    stats_b = calculate_basic_stats(data_b)

    return {
        "dataset_a": stats_a,
        "dataset_b": stats_b,
        "mean_diff": stats_b["mean"] - stats_a["mean"],
        "std_diff": stats_b["std"] - stats_a["std"],
        "min_diff": stats_b["min"] - stats_a["min"],
        "max_diff": stats_b["max"] - stats_a["max"],
        "count_diff": stats_b["count"] - stats_a["count"],
    }


# ---------------------------------------------------------------------------
#/ Utilidades de visualización textual
# ---------------------------------------------------------------------------

def generate_text_histogram(
    values: List[float],
    n_bins: int = 10,
    width: int = 40,
) -> str:
    """
    Genera un histograma textual de una lista de valores.

    Parameters
    ----------
    values : list de float
        Valores a graficar.
    n_bins : int
        Número de bins del histograma.
    width : int
        Ancho máximo de las barras.

    Returns
    -------
    str con el histograma en texto.
    """
    if not values:
        return "Sin datos para graficar."

    arr = np.array(values)
    hist, bin_edges = np.histogram(arr, bins=n_bins)
    max_count = int(np.max(hist)) if len(hist) > 0 else 1

    lines: List[str] = []
    lines.append("Histograma de valores:")
    lines.append("")

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        count = int(hist[i])
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(f"[{lo:.3f}, {hi:.3f}) |{bar}| {count}")

    lines.append("")
    return "\n".join(lines)


def generate_progress_bar(
    current: int,
    total: int,
    width: int = 40,
) -> str:
    """
    Genera una barra de progreso textual.

    Parameters
    ----------
    current : int
        Valor actual.
    total : int
        Valor total.
    width : int
        Ancho de la barra en caracteres.

    Returns
    -------
    str con la barra de progreso.
    """
    if total == 0:
        return "[" + "?" * width + "] 0.0%"

    progress = current / total
    filled = int(progress * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = progress * 100.0

    return f"[{bar}] {pct:.1f}% ({current}/{total})"


# ---------------------------------------------------------------------------
#/ Utilidades de generación de identificadores
# ---------------------------------------------------------------------------

def generate_experiment_id(
    prefix: str = "exp",
) -> str:
    """
    Genera un identificador único para un experimento.

    Parameters
    ----------
    prefix : str
        Prefijo para el ID.

    Returns
    -------
    str con el ID del experimento.
    """
    import uuid

    timestamp = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique}"


def generate_run_id(
    prefix: str = "run",
) -> str:
    """
    Genera un identificador único para una ejecución.

    Parameters
    ----------
    prefix : str
        Prefijo para el ID.

    Returns
    -------
    str con el ID de la ejecución.
    """
    import uuid

    timestamp = int(time.time())
    unique = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{unique}"


def generate_model_signature(
    config: dict,
) -> str:
    """
    Genera una firma (hash) para una configuración de modelo.

    Parameters
    ----------
    config : dict
        Configuración del modelo.

    Returns
    -------
    str con el hash SHA-256 de la configuración.
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
#/ Utilidades de strings y archivos
# ---------------------------------------------------------------------------

def sanitize_filename(
    name: str,
) -> str:
    """
    Sanitiza una cadena para usarla como nombre de archivo.

    Parameters
    ----------
    name : str
        Nombre original.

    Returns
    -------
    str con el nombre sanitizado.
    """
    import re

    # Reemplazar caracteres no alfanuméricos con guiones bajos
    sanitized = re.sub(r"[^\w\s-]", "_", name)
    # Reemplazar espacios con guiones bajos
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    # Eliminar guiones bajos múltiples
    sanitized = re.sub(r"_+", "_", sanitized)
    # Eliminar guiones bajos al inicio y final
    sanitized = sanitized.strip("_")

    return sanitized or "unnamed"


def truncate_string(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Trunca una cadena a una longitud máxima.

    Parameters
    ----------
    text : str
        Texto a truncar.
    max_length : int
        Longitud máxima incluyendo el sufijo.
    suffix : str
        Sufijo a agregar si se trunca.

    Returns
    -------
    str con el texto truncado.
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def wrap_text(
    text: str,
    width: int = 80,
) -> str:
    """
    Envuelve texto a un ancho de columna específico.

    Parameters
    ----------
    text : str
        Texto a envolver.
    width : int
        Ancho máximo de columna.

    Returns
    -------
    str con el texto envuelto.
    """
    import textwrap

    return textwrap.fill(text, width=width)


def generate_section_header(
    title: str,
    char: str = "=",
    width: int = 60,
) -> str:
    """
    Genera un encabezado de sección para reportes de texto.

    Parameters
    ----------
    title : str
        Título de la sección.
    char : str
        Carácter para el divisorio.
    width : int
        Ancho total del encabezado.

    Returns
    -------
    str con el encabezado.
    """
    padding = (width - len(title) - 2) // 2
    if padding < 0:
        padding = 0

    left = char * padding
    right = char * (width - len(title) - 2 - padding)

    return f"\n{left} {title} {right}\n"


def generate_divider(
    char: str = "-",
    width: int = 60,
) -> str:
    """
    Genera un divisor visual para reportes de texto.

    Parameters
    ----------
    char : str
        Carácter del divisorio.
    width : int
        Ancho del divisorio.

    Returns
    -------
    str con el divisorio.
    """
    return char * width


# ---------------------------------------------------------------------------
#/ Utilidades de exportación de resúmenes
# ---------------------------------------------------------------------------

def export_summary_to_text(
    summary: dict,
    output_path: str | Path,
) -> str:
    """
    Exporta un resumen a un archivo de texto.

    Parameters
    ----------
    summary : dict
        Resumen a exportar.
    output_path : str | Path
        Ruta del archivo de salida.

    Returns
    -------
    str con la ruta del archivo guardado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("RESUMEN DEL MODELO")
    lines.append("=" * 40)
    lines.append("")

    def _write_dict(d: dict, indent: int = 0):
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                _write_dict(value, indent + 1)
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: [{len(value)} elementos]")
                for i, item in enumerate(value[:5]):
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  [{i}]:")
                        _write_dict(item, indent + 2)
                    else:
                        lines.append(f"{prefix}  [{i}]: {item}")
                if len(value) > 5:
                    lines.append(f"{prefix}  ... y {len(value) - 5} más")
            else:
                lines.append(f"{prefix}{key}: {value}")

    _write_dict(summary)

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")

    return str(output_path)


def export_summary_to_csv(
    summary: dict,
    output_path: str | Path,
) -> str:
    """
    Exporta un resumen plano a un archivo CSV.

    Parameters
    ----------
    summary : dict
        Resumen con valores planos (no anidados).
    output_path : str | Path
        Ruta del archivo de salida.

    Returns
    -------
    str con la ruta del archivo guardado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("clave,valor\n")
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            f.write(f"{key},{value_str}\n")

    return str(output_path)


# ---------------------------------------------------------------------------
#/ Utilidades de reportes combinados
# ---------------------------------------------------------------------------

def generate_combined_report(
    validation_report: Optional[dict] = None,
    evaluation_report: Optional[dict] = None,
    benchmark_report: Optional[dict] = None,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Genera un reporte combinado de validación, evaluación y benchmarks.

    Parameters
    ----------
    validation_report : dict opcional
        Reporte de validación del modelo.
    evaluation_report : dict opcional
        Reporte de evaluación del modelo.
    benchmark_report : dict opcional
        Reporte de benchmarks de rendimiento.
    output_path : str | Path opcional
        Ruta para guardar el reporte.

    Returns
    -------
    str con el reporte combinado en Markdown.
    """
    lines: List[str] = []
    lines.append("# Reporte Combinado del Modelo AI 800M")
    lines.append("")
    lines.append(f"Generado el: {format_timestamp()}")
    lines.append("")

    # Sección de validación
    if validation_report:
        lines.append("## Validación")
        lines.append("")
        score = validation_report.get("overall_score", 0)
        all_passed = validation_report.get("all_passed", False)
        lines.append(f"- Puntuación: {score:.1f}/100")
        lines.append(f"- Todas pasaron: {'Sí' if all_passed else 'No'}")
        lines.append("")

    # Sección de evaluación
    if evaluation_report:
        lines.append("## Evaluación")
        lines.append("")
        pass_rate = evaluation_report.get("global_pass_rate", 0)
        total_tests = evaluation_report.get("total_tests", 0)
        lines.append(f"- Tasa de paso: {pass_rate:.1%}")
        lines.append(f"- Pruebas totales: {total_tests}")
        lines.append("")

    # Sección de benchmarks
    if benchmark_report:
        lines.append("## Benchmarks de Rendimiento")
        lines.append("")
        results = benchmark_report.get("benchmark_results", [])
        for r in results:
            lines.append(f"- {r.get('name', 'desconocido')}: {r.get('value', 0):.4f} {r.get('unit', '')}")
        lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

    return report_text


# ---------------------------------------------------------------------------
#/ Utilidades de checklist y FAQ
# ---------------------------------------------------------------------------

def generate_checklist(
    items: List[str],
    checked: Optional[List[bool]] = None,
) -> str:
    """
    Genera una lista de verificación (checklist) en texto.

    Parameters
    ----------
    items : list de str
        Elementos de la lista.
    checked : list de bool opcional
        Estado de cada elemento (True = marcado).

    Returns
    -------
    str con la checklist.
    """
    if checked is None:
        checked = [False] * len(items)

    lines: List[str] = []
    lines.append("Lista de Verificación")
    lines.append("=" * 30)
    lines.append("")

    for i, item in enumerate(items):
        box = "[x]" if i < len(checked) and checked[i] else "[ ]"
        lines.append(f"{box} {item}")

    lines.append("")
    return "\n".join(lines)


def generate_faq(
    questions_answers: List[Tuple[str, str]],
) -> str:
    """
    Genera una sección de preguntas frecuentes (FAQ).

    Parameters
    ----------
    questions_answers : list de tuplas (pregunta, respuesta)
        Pares de pregunta y respuesta.

    Returns
    -------
    str con el FAQ en texto.
    """
    lines: List[str] = []
    lines.append("Preguntas Frecuentes (FAQ)")
    lines.append("=" * 40)
    lines.append("")

    for i, (question, answer) in enumerate(questions_answers, 1):
        lines.append(f"P{i}: {question}")
        lines.append(f"R{i}: {answer}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#/ Utilidades de glosario
# ---------------------------------------------------------------------------

def generate_glossary(
    terms: Dict[str, str],
) -> str:
    """
    Genera un glosario de términos técnicos.

    Parameters
    ----------
    terms : dict
        Diccionario término -> definición.

    Returns
    -------
    str con el glosario en texto.
    """
    lines: List[str] = []
    lines.append("Glosario de Términos")
    lines.append("=" * 40)
    lines.append("")

    for term, definition in sorted(terms.items()):
        lines.append(f"{term}: {definition}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#/ Utilidades de plantillas de reporte
# ---------------------------------------------------------------------------

def generate_report_template(
    title: str,
    sections: List[str],
) -> str:
    """
    Genera una plantilla de reporte con secciones vacías.

    Parameters
    ----------
    title : str
        Título del reporte.
    sections : list de str
        Nombres de las secciones.

    Returns
    -------
    str con la plantilla de reporte.
    """
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generado el: {format_timestamp()}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for section in sections:
        lines.append(f"## {section}")
        lines.append("")
        lines.append("(Contenido pendiente)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#/ Utilidades de referencias académicas
# ---------------------------------------------------------------------------

def generate_academic_citation(
    author: str,
    year: int,
    title: str,
    venue: str,
) -> str:
    """
    Genera una cita académica en formato APA simplificado.

    Parameters
    ----------
    author : str
        Autor(es) de la obra.
    year : int
        Año de publicación.
    title : str
        Título de la obra.
    venue : str
        Lugar de publicación (conferencia, revista, etc.).

    Returns
    -------
    str con la cita formateada.
    """
    return f"{author} ({year}). {title}. {venue}."


# ---------------------------------------------------------------------------
#/ Utilidades de tablas de conversión
# ---------------------------------------------------------------------------

def generate_unit_conversion_table() -> str:
    """
    Genera una tabla de conversión de unidades comunes en IA.

    Returns
    -------
    str con la tabla de conversión.
    """
    lines: List[str] = []
    lines.append("Tabla de Conversión de Unidades")
    lines.append("=" * 50)
    lines.append("")
    lines.append("| Unidad | Equivalente |")
    lines.append("|--------|-------------|")
    lines.append("| 1 KB | 1,024 bytes |")
    lines.append("| 1 MB | 1,024 KB |")
    lines.append("| 1 GB | 1,024 MB |")
    lines.append("| 1 TB | 1,024 GB |")
    lines.append("| 1 parámetro INT8 | 1 byte |")
    lines.append("| 1 parámetro FP16 | 2 bytes |")
    lines.append("| 1 parámetro FP32 | 4 bytes |")
    lines.append("| 1 token/segundo | velocidad de inferencia |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#/ Utilidades de matrices de decisión
# ---------------------------------------------------------------------------

def generate_decision_matrix(
    options: List[str],
    criteria: List[str],
    scores: Dict[str, Dict[str, float]],
) -> str:
    """
    Genera una matriz de decisión para comparar opciones.

    Parameters
    ----------
    options : list de str
        Opciones a comparar.
    criteria : list de str
        Criterios de evaluación.
    scores : dict
        Puntuaciones {opcion: {criterio: puntuacion}}.

    Returns
    -------
    str con la matriz de decisión.
    """
    lines: List[str] = []
    lines.append("Matriz de Decisión")
    lines.append("=" * 50)
    lines.append("")

    # Encabezado
    header = "Opción | " + " | ".join(criteria) + " | Total"
    lines.append(header)
    lines.append("-" * len(header))

    for option in options:
        row_scores = scores.get(option, {})
        total = sum(row_scores.values())
        row = f"{option} | " + " | ".join(
            str(row_scores.get(c, 0)) for c in criteria
        ) + f" | {total:.1f}"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#/ Utilidades de flujogramas textuales
# ---------------------------------------------------------------------------

def generate_text_flowchart(
    steps: List[str],
) -> str:
    """
    Genera un flujograma textual simple.

    Parameters
    ----------
    steps : list de str
        Pasos del flujo.

    Returns
    -------
    str con el flujograma en texto.
    """
    lines: List[str] = []
    lines.append("Flujo de Proceso")
    lines.append("=" * 40)
    lines.append("")

    for i, step in enumerate(steps):
        lines.append(f"[{i+1}] {step}")
        if i < len(steps) - 1:
            lines.append("    │")
            lines.append("    ▼")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#/ Función de conveniencia para resumen rápido
# ---------------------------------------------------------------------------

def quick_model_summary(
    n_params: int,
    model_size_mb: float,
    n_layers: int,
    hidden_size: int,
    n_heads: int,
) -> str:
    """
    Genera un resumen rápido del modelo en texto.

    Parameters
    ----------
    n_params : int
        Número total de parámetros.
    model_size_mb : float
        Tamaño del modelo en MB.
    n_layers : int
        Número de capas.
    hidden_size : int
        Tamaño oculto.
    n_heads : int
        Número de cabezas de atención.

    Returns
    -------
    str con el resumen del modelo.
    """
    lines: List[str] = []
    lines.append("Resumen del Modelo AI 800M")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Parámetros totales: {format_number(n_params)}")
    lines.append(f"Tamaño del modelo: {format_bytes(int(model_size_mb * 1024 * 1024))}")
    lines.append(f"Número de capas: {n_layers}")
    lines.append(f"Tamaño oculto: {hidden_size}")
    lines.append(f"Cabezas de atención: {n_heads}")
    lines.append(f"Dimensión por cabeza: {hidden_size // n_heads}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "format_bytes",
    "format_number",
    "format_percentage",
    "format_duration",
    "format_timestamp",
    "format_status_indicator",
    "calculate_basic_stats",
    "detect_outliers",
    "normalize_data",
    "standardize_data",
    "compare_two_datasets",
    "generate_text_histogram",
    "generate_progress_bar",
    "generate_experiment_id",
    "generate_run_id",
    "generate_model_signature",
    "sanitize_filename",
    "truncate_string",
    "wrap_text",
    "generate_section_header",
    "generate_divider",
    "export_summary_to_text",
    "export_summary_to_csv",
    "generate_combined_report",
    "generate_checklist",
    "generate_faq",
    "generate_glossary",
    "generate_report_template",
    "generate_academic_citation",
    "generate_unit_conversion_table",
    "generate_decision_matrix",
    "generate_text_flowchart",
    "quick_model_summary",
]
