#!/usr/bin/env python3
"""
model_merger.py — Fusionador de modelos 800M.

Permite combinar pesos de dos modelos .safetensors en uno nuevo usando
diferentes estrategias de fusión:

  - Promedio simple (arithmético).
  - Promedio ponderado (por porcentaje de cada modelo).
  - Fusión por capa (usar capas de un modelo y el resto del otro).
  - Fusión interpolada (interpolación lineal entre pesos).
  - Fusión por umbral (usar el modelo con mayor magnitud de peso por tensor).
  - Fusión diferencial (aplicar la diferencia de un modelo sobre otro).

Incluye validación de que ambos modelos tengan la misma arquitectura
y reporte detallado de qué tensores se fusionaron y con qué estrategia.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError:  # pragma: no cover
    safe_open = None
    save_file = None


def _load_tensors(
    safetensors_path: str | Path,
) -> Dict[str, np.ndarray]:
    """Carga todos los tensores de un archivo .safetensors."""
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    tensors: Dict[str, np.ndarray] = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    return tensors


def _load_scales(
    safetensors_path: str | Path,
) -> Dict[str, np.ndarray]:
    """Carga las escalas de cuantización si existen."""
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    scales: Dict[str, np.ndarray] = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        for name in f.keys():
            if name.endswith("_scale"):
                scales[name] = f.get_tensor(name)
    return scales


def _save_tensors(
    tensors: Dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Guarda tensores en un archivo .safetensors."""
    if save_file is None:
        raise ImportError("safetensors no está instalado")

    # Convertir todo a float32 para almacenamiento intermedio
    # (el guardado real en INT8 se haría con quantización previa)
    save_file(tensors, str(output_path))


def merge_models(
    model_a_dir: str | Path,
    model_b_dir: str | Path,
    output_dir: str | Path,
    strategy: str = "average",
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    layers_from_a: Optional[List[int]] = None,
    threshold_strategy: str = "max_magnitude",
) -> dict:
    """
    Fusiona dos modelos en uno nuevo.

    Parameters
    ----------
    model_a_dir : str | Path
        Directorio del primer modelo.
    model_b_dir : str | Path
        Directorio del segundo modelo.
    output_dir : str | Path
        Directorio de salida para el modelo fusionado.
    strategy : str
        Estrategia de fusión: "average", "weighted", "layer_select",
        "interpolate", "threshold", "differential".
    weight_a : float
        Peso del modelo A en estrategias ponderadas (0-1).
    weight_b : float
        Peso del modelo B en estrategias ponderadas (0-1).
    layers_from_a : list de int opcional
        Índices de capas a tomar del modelo A (estrategia layer_select).
    threshold_strategy : str
        Para estrategia "threshold": "max_magnitude" o "min_magnitude".

    Returns
    -------
    dict con resultados de la fusión.
    """
    if safe_open is None or save_file is None:
        raise ImportError("safetensors no está instalado")

    model_a_dir = Path(model_a_dir)
    model_b_dir = Path(model_b_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_a = model_a_dir / "model.safetensors"
    path_b = model_b_dir / "model.safetensors"

    # Cargar tensores y escalas
    tensors_a = _load_tensors(path_a)
    tensors_b = _load_tensors(path_b)
    scales_a = _load_scales(path_a)
    scales_b = _load_scales(path_b)

    # Verificar que tengan las mismas claves
    keys_a = set(tensors_a.keys())
    keys_b = set(tensors_b.keys())
    common_keys = keys_a & keys_b

    if keys_a != keys_b:
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        raise ValueError(
            f"Modelos tienen claves diferentes. Solo en A: {only_a}, "
            f"Solo en B: {only_b}"
        )

    # Fusionar tensores
    merged_tensors: Dict[str, np.ndarray] = {}
    merged_scales: Dict[str, np.ndarray] = {}
    merge_details: List[dict] = []

    for name in sorted(common_keys):
        a = tensors_a[name].astype(np.float64)
        b = tensors_b[name].astype(np.float64)

        if a.shape != b.shape:
            raise ValueError(
                f"Tensor {name} tiene formas diferentes: {a.shape} vs {b.shape}"
            )

        # Aplicar estrategia
        if strategy == "average":
            merged = (a + b) / 2.0
            detail = f"Promedio simple (50/50)"

        elif strategy == "weighted":
            merged = weight_a * a + weight_b * b
            detail = f"Promedio ponderado ({weight_a:.2f}/{weight_b:.2f})"

        elif strategy == "interpolate":
            # Lo mismo que weighted pero con nombre semántico diferente
            merged = weight_a * a + weight_b * b
            detail = f"Interpolación lineal (t={weight_b:.2f})"

        elif strategy == "threshold":
            if threshold_strategy == "max_magnitude":
                mask = np.abs(a) >= np.abs(b)
            else:
                mask = np.abs(a) <= np.abs(b)
            merged = np.where(mask, a, b)
            detail = f"Umbral ({threshold_strategy})"

        elif strategy == "differential":
            # A = base, B = delta a aplicar
            merged = a + (b - a) * weight_b
            detail = f"Diferencial (A + {weight_b:.2f} * (B - A))"

        elif strategy == "layer_select":
            # Determinar si este tensor pertenece a una capa de A
            # Buscar patrón "layers.X." en el nombre
            import re
            layer_match = re.search(r"layers\.(\d+)\.", name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                if layers_from_a and layer_idx in layers_from_a:
                    merged = a
                    detail = f"Capa {layer_idx} del modelo A"
                else:
                    merged = b
                    detail = f"Capa {layer_idx} del modelo B"
            else:
                merged = (a + b) / 2.0
                detail = "Promedio (no es tensor de capa)"

        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")

        # Guardar tensor fusionado (float32)
        merged_tensors[name] = merged.astype(np.float32)

        # Escalas: promedio de las escalas
        scale_a = scales_a.get(name + "_scale")
        scale_b = scales_b.get(name + "_scale")
        if scale_a is not None and scale_b is not None:
            merged_scale = (scale_a.astype(np.float64) + scale_b.astype(np.float64)) / 2.0
            merged_scales[name + "_scale"] = merged_scale.astype(np.float32)
        elif scale_a is not None:
            merged_scales[name + "_scale"] = scale_a
        elif scale_b is not None:
            merged_scales[name + "_scale"] = scale_b

        # Estadísticas de la fusión
        abs_diff = float(np.mean(np.abs(a - b)))
        merge_details.append({
            "name": name,
            "shape": list(a.shape),
            "strategy": strategy,
            "detail": detail,
            "abs_diff_mean": round(abs_diff, 8),
            "params": int(a.size),
        })

    # Guardar modelo fusionado
    output_weights = output_dir / "model.safetensors"

    # Combinar tensores y escalas para guardar
    all_tensors_to_save = {}
    all_tensors_to_save.update(merged_tensors)
    all_tensors_to_save.update(merged_scales)

    save_file(all_tensors_to_save, str(output_weights))

    # Copiar configuración
    config_a = model_a_dir / "config.json"
    config_b = model_b_dir / "config.json"
    output_config = output_dir / "config.json"

    if config_a.exists():
        import json as _json
        with open(config_a, "r") as f:
            config = _json.load(f)
        config["merge_strategy"] = strategy
        config["weight_a"] = weight_a
        config["weight_b"] = weight_b
        with open(output_config, "w") as f:
            _json.dump(config, f, indent=2)

    # Estadísticas globales
    total_params = sum(d["params"] for d in merge_details)
    mean_diff = np.mean([d["abs_diff_mean"] for d in merge_details])

    result = {
        "strategy": strategy,
        "weight_a": weight_a,
        "weight_b": weight_b,
        "tensor_count": len(merged_tensors),
        "total_params": total_params,
        "mean_abs_diff": round(float(mean_diff), 8),
        "output_weights": str(output_weights),
        "output_config": str(output_config),
        "merge_details": merge_details,
    }

    return result


def generate_merge_report(
    merge_result: dict,
    output_path: str | Path,
) -> str:
    """
    Genera un reporte de fusión en Markdown.

    Returns
    -------
    str con el reporte en Markdown.
    """
    lines: List[str] = []
    lines.append("# Reporte de Fusión de Modelos")
    lines.append("")
    lines.append(f"Estrategia: `{merge_result['strategy']}`")
    lines.append(f"Peso A: {merge_result['weight_a']}, Peso B: {merge_result['weight_b']}")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Tensores fusionados | {merge_result['tensor_count']:,} |")
    lines.append(f"| Parámetros totales | {merge_result['total_params']:,} |")
    lines.append(f"| Diferencia absoluta media | {merge_result['mean_abs_diff']:.8f} |")
    lines.append(f"| Pesos de salida | `{merge_result['output_weights']}` |")
    lines.append("")

    lines.append("## Detalle por Tensor (Top 20 por diferencia)")
    lines.append("")
    lines.append("| Tensor | Forma | Estrategia | Diff Abs Media | Parámetros |")
    lines.append("|--------|-------|------------|----------------|------------|")

    details = sorted(
        merge_result["merge_details"],
        key=lambda x: x["abs_diff_mean"],
        reverse=True,
    )[:20]

    for d in details:
        shape_str = " × ".join(map(str, d["shape"]))
        lines.append(
            f"| {d['name']} | {shape_str} | {d['strategy']} | "
            f"{d['abs_diff_mean']:.8f} | {d['params']:,} |"
        )

    lines.append("")
    return "\n".join(lines)


__all__ = [
    "merge_models",
    "generate_merge_report",
]
