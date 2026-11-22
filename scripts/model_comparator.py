#!/usr/bin/env python3
"""
model_comparator.py — Comparador de modelos 800M.

Permite comparar dos versiones del modelo (o dos archivos .safetensors)
y generar reportes detallados de las diferencias:

  - Diferencia absoluta media por tensor.
  - Similitud de coseno entre vectores de pesos.
  - Tensores que más cambiaron.
  - Diferencias en configuración (config.json).
  - Reporte en JSON y Markdown.
  - Visualización textual de cambios (heatmap simplificado).

Diseñado para comparar el modelo antes y después de run_stage5.py,
o entre diferentes etapas de síntesis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


# Usamos dict en lugar de dataclass para evitar import de dataclasses
# (se puede cambiar, pero funcionan igual)

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


def _load_config(
    model_dir: str | Path,
) -> dict:
    """Carga config.json de un directorio de modelo."""
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def compare_models(
    model_a_dir: str | Path,
    model_b_dir: str | Path,
    weights_a: str = "model.safetensors",
    weights_b: str = "model.safetensors",
) -> dict:
    """
    Compara dos modelos directorio a directorio.

    Returns
    -------
    dict con resultados de la comparación.
    """
    model_a_dir = Path(model_a_dir)
    model_b_dir = Path(model_b_dir)

    path_a = model_a_dir / weights_a
    path_b = model_b_dir / weights_b

    tensors_a = _load_tensors(path_a)
    tensors_b = _load_tensors(path_b)

    config_a = _load_config(model_a_dir)
    config_b = _load_config(model_b_dir)

    # Comparar configuración
    config_diff = _compare_configs(config_a, config_b)

    # Encontrar tensores comunes y únicos
    keys_a = set(tensors_a.keys())
    keys_b = set(tensors_b.keys())
    common_keys = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    # Comparar tensores comunes
    tensor_diffs: List[dict] = []
    total_abs_diff = 0.0
    total_cosine_sim = 0.0
    n_compared = 0

    for name in sorted(common_keys):
        a = tensors_a[name].astype(np.float64).flatten()
        b = tensors_b[name].astype(np.float64).flatten()

        if a.shape != b.shape:
            tensor_diffs.append({
                "name": name,
                "shape_a": list(tensors_a[name].shape),
                "shape_b": list(tensors_b[name].shape),
                "error": "formas diferentes",
            })
            continue

        abs_diff = np.abs(a - b)
        mean_abs_diff = float(np.mean(abs_diff))
        max_abs_diff = float(np.max(abs_diff))

        # Similitud de coseno
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a > 0 and norm_b > 0:
            cosine_sim = dot / (norm_a * norm_b)
        else:
            cosine_sim = 1.0 if np.array_equal(a, b) else 0.0

        # Diferencia relativa
        mean_abs_a = float(np.mean(np.abs(a)))
        rel_diff = mean_abs_diff / mean_abs_a if mean_abs_a > 0 else 0.0

        total_abs_diff += mean_abs_diff
        total_cosine_sim += cosine_sim
        n_compared += 1

        tensor_diffs.append({
            "name": name,
            "shape": list(tensors_a[name].shape),
            "mean_abs_diff": round(mean_abs_diff, 8),
            "max_abs_diff": round(max_abs_diff, 8),
            "cosine_similarity": round(cosine_sim, 6),
            "relative_diff": round(rel_diff, 6),
            "params": int(a.size),
        })

    # Ordenar por mayor diferencia absoluta
    tensor_diffs.sort(key=lambda x: x.get("mean_abs_diff", 0), reverse=True)

    # Estadísticas globales
    mean_abs_diff_all = total_abs_diff / n_compared if n_compared > 0 else 0.0
    mean_cosine_all = total_cosine_sim / n_compared if n_compared > 0 else 0.0

    # Tensores que más cambiaron (top 10)
    top_changed = tensor_diffs[:10]

    # Tensores más similares (bottom 10, excluyendo errores)
    valid_diffs = [d for d in tensor_diffs if "error" not in d]
    most_similar = sorted(valid_diffs, key=lambda x: x["mean_abs_diff"])[:10]

    result = {
        "model_a": str(model_a_dir),
        "model_b": str(model_b_dir),
        "weights_a": weights_a,
        "weights_b": weights_b,
        "tensor_count_a": len(tensors_a),
        "tensor_count_b": len(tensors_b),
        "common_tensors": len(common_keys),
        "tensors_only_a": len(only_a),
        "tensors_only_b": len(only_b),
        "tensors_only_a_names": sorted(only_a),
        "tensors_only_b_names": sorted(only_b),
        "config_diff": config_diff,
        "mean_abs_diff_all": round(mean_abs_diff_all, 8),
        "mean_cosine_similarity_all": round(mean_cosine_all, 6),
        "n_tensors_compared": n_compared,
        "tensor_diffs": tensor_diffs,
        "top_changed": top_changed,
        "most_similar": most_similar,
    }

    return result


def _compare_configs(
    config_a: dict,
    config_b: dict,
) -> dict:
    """Compara dos configuraciones de modelo."""
    all_keys = set(config_a.keys()) | set(config_b.keys())
    diffs: List[dict] = []
    same = 0
    changed = 0
    added = 0
    removed = 0

    for key in sorted(all_keys):
        in_a = key in config_a
        in_b = key in config_b

        if in_a and in_b:
            if config_a[key] == config_b[key]:
                same += 1
            else:
                changed += 1
                diffs.append({
                    "key": key,
                    "value_a": config_a[key],
                    "value_b": config_b[key],
                    "change": "modified",
                })
        elif in_a:
            removed += 1
            diffs.append({
                "key": key,
                "value_a": config_a[key],
                "value_b": None,
                "change": "removed",
            })
        else:
            added += 1
            diffs.append({
                "key": key,
                "value_a": None,
                "value_b": config_b[key],
                "change": "added",
            })

    return {
        "same": same,
        "changed": changed,
        "added": added,
        "removed": removed,
        "diffs": diffs,
    }


def generate_markdown_report(
    comparison: dict,
) -> str:
    """
    Genera un reporte en Markdown a partir de los resultados de comparación.

    Returns
    -------
    str con el reporte en Markdown.
    """
    lines: List[str] = []
    lines.append("# Comparación de Modelos")
    lines.append("")
    lines.append(f"- Modelo A: `{comparison['model_a']}`")
    lines.append(f"- Modelo B: `{comparison['model_b']}`")
    lines.append(f"- Pesos A: `{comparison['weights_a']}`")
    lines.append(f"- Pesos B: `{comparison['weights_b']}`")
    lines.append("")

    lines.append("## Resumen")
    lines.append("")
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Tensores en A | {comparison['tensor_count_a']:,} |")
    lines.append(f"| Tensores en B | {comparison['tensor_count_b']:,} |")
    lines.append(f"| Tensores comunes | {comparison['common_tensors']:,} |")
    lines.append(f"| Solo en A | {comparison['tensors_only_a']} |")
    lines.append(f"| Solo en B | {comparison['tensors_only_b']} |")
    lines.append(f"| Tensores comparados | {comparison['n_tensors_compared']:,} |")
    lines.append(f"| Diferencia absoluta media | {comparison['mean_abs_diff_all']:.8f} |")
    lines.append(f"| Similitud de coseno media | {comparison['mean_cosine_similarity_all']:.6f} |")
    lines.append("")

    # Configuración
    config_diff = comparison["config_diff"]
    lines.append("## Diferencias de Configuración")
    lines.append("")
    lines.append(f"| Iguales | Cambiadas | Añadidas | Eliminadas |")
    lines.append(f"|---------|-----------|----------|------------|")
    lines.append(
        f"| {config_diff['same']} | {config_diff['changed']} | "
        f"{config_diff['added']} | {config_diff['removed']} |"
    )
    lines.append("")

    if config_diff["diffs"]:
        lines.append("### Detalle de cambios")
        lines.append("")
        lines.append("| Clave | Valor A | Valor B | Cambio |")
        lines.append("|-------|---------|---------|--------|")
        for d in config_diff["diffs"]:
            va = d["value_a"]
            vb = d["value_b"]
            lines.append(f"| {d['key']} | {va} | {vb} | {d['change']} |")
        lines.append("")

    # Top cambiados
    lines.append("## Tensores que Más Cambiaron (Top 10)")
    lines.append("")
    lines.append("| Tensor | Forma | Diff Abs Media | Diff Abs Máx | Similitud Coseno | Diff Relativa | Parámetros |")
    lines.append("|--------|-------|----------------|--------------|------------------|---------------|------------|")

    for d in comparison["top_changed"]:
        if "error" in d:
            lines.append(
                f"| {d['name']} | {d['shape_a']} vs {d['shape_b']} | "
                f"Error: {d['error']} | | | | |"
            )
        else:
            lines.append(
                f"| {d['name']} | {' × '.join(map(str, d['shape']))} | "
                f"{d['mean_abs_diff']:.8f} | {d['max_abs_diff']:.8f} | "
                f"{d['cosine_similarity']:.6f} | {d['relative_diff']:.6f} | "
                f"{d['params']:,} |"
            )

    lines.append("")

    # Más similares
    lines.append("## Tensores Más Similares (Bottom 10)")
    lines.append("")
    lines.append("| Tensor | Diff Abs Media | Similitud Coseno |")
    lines.append("|--------|----------------|------------------|")

    for d in comparison["most_similar"]:
        lines.append(
            f"| {d['name']} | {d['mean_abs_diff']:.8f} | {d['cosine_similarity']:.6f} |"
        )

    lines.append("")

    # Tensores únicos
    if comparison["tensors_only_a_names"]:
        lines.append("## Tensores Solo en A")
        lines.append("")
        for name in comparison["tensors_only_a_names"][:20]:
            lines.append(f"- {name}")
        if len(comparison["tensors_only_a_names"]) > 20:
            lines.append(f"- ... y {len(comparison['tensors_only_a_names']) - 20} más")
        lines.append("")

    if comparison["tensors_only_b_names"]:
        lines.append("## Tensores Solo en B")
        lines.append("")
        for name in comparison["tensors_only_b_names"][:20]:
            lines.append(f"- {name}")
        if len(comparison["tensors_only_b_names"]) > 20:
            lines.append(f"- ... y {len(comparison['tensors_only_b_names']) - 20} más")
        lines.append("")

    return "\n".join(lines)


def save_comparison(
    comparison: dict,
    output_dir: str | Path,
    base_name: str = "model_comparison",
) -> dict:
    """
    Guarda los resultados de comparación en JSON y Markdown.

    Returns
    -------
    dict con las rutas de los archivos guardados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{base_name}.json"
    md_path = output_dir / f"{base_name}.md"

    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)

    md_report = generate_markdown_report(comparison)
    with open(md_path, "w") as f:
        f.write(md_report)

    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }


def quick_compare(
    model_a_dir: str | Path,
    model_b_dir: str | Path,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """
    Función de conveniencia: compara dos modelos y opcionalmente guarda
    los resultados.

    Returns
    -------
    dict con el resultado de la comparación (y rutas de archivos si se guardó).
    """
    comparison = compare_models(model_a_dir, model_b_dir)

    if output_dir:
        paths = save_comparison(comparison, output_dir)
        comparison["saved_to"] = paths

    return comparison


__all__ = [
    "compare_models",
    "generate_markdown_report",
    "save_comparison",
    "quick_compare",
]
