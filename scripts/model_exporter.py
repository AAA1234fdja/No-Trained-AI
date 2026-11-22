#!/usr/bin/env python3
"""
model_exporter.py — Exportador del modelo 800M a otros formatos.

Proporciona funciones para exportar el modelo desde .safetensors a:
  - NumPy NPZ (para uso directo en NumPy).
  - JSON con pesos desquantizados (para inspección, archivos grandes).
  - Formato de pesos crudos (raw binary) para carga en C/C++.
  - Configuración del modelo en múltiples formatos (JSON, YAML, INI).
  - Reporte de arquitectura en Markdown.
  - Script de carga en PyTorch (plantilla, sin ejecutar PyTorch).
  - Script de carga en TensorFlow (plantilla, sin ejecutar TF).

Diseñado para que el modelo pueda ser usado en otros entornos sin
depender del formato .safetensors original.
"""

from __future__ import annotations

import configparser
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


# ---------------------------------------------------------------------------
# Exportación a NumPy NPZ
# ---------------------------------------------------------------------------

def export_to_npz(
    safetensors_path: str | Path,
    output_path: str | Path,
    dequantize: bool = True,
) -> Dict[str, object]:
    """
    Exporta un archivo .safetensors a formato NumPy NPZ.

    Parameters
    ----------
    safetensors_path : str | Path
        Ruta al archivo .safetensors de entrada.
    output_path : str | Path
        Ruta al archivo NPZ de salida.
    dequantize : bool
        Si True, desquantiza los pesos INT8 a float32 antes de guardar.

    Returns
    -------
    dict con información de la exportación.
    """
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    safetensors_path = Path(safetensors_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    export_info: Dict[str, object] = {
        "source": str(safetensors_path),
        "output": str(output_path),
        "dequantize": dequantize,
        "tensor_count": 0,
        "total_params": 0,
        "tensor_details": [],
    }

    with safe_open(str(safetensors_path), framework="numpy") as f:
        for name in f.keys():
            t = f.get_tensor(name)

            if dequantize and t.dtype == np.int8:
                scale_name = name + "_scale"
                if scale_name in f.keys():
                    scale = f.get_tensor(scale_name).astype(np.float32)
                    arr = t.astype(np.float32) * scale.T
                    arrays[name] = arr
                    # No guardar la escala por separado si ya está desquantizada
                    export_info["tensor_details"].append({
                        "name": name,
                        "original_dtype": "int8",
                        "exported_dtype": "float32",
                        "shape": list(arr.shape),
                        "params": int(arr.size),
                    })
                    export_info["total_params"] += int(arr.size)
                else:
                    # Sin escala, guardar tal cual
                    arrays[name] = t
                    export_info["tensor_details"].append({
                        "name": name,
                        "original_dtype": str(t.dtype),
                        "exported_dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "params": int(t.size),
                    })
                    export_info["total_params"] += int(t.size)
            else:
                arrays[name] = t
                export_info["tensor_details"].append({
                    "name": name,
                    "original_dtype": str(t.dtype),
                    "exported_dtype": str(t.dtype),
                    "shape": list(t.shape),
                    "params": int(t.size),
                })
                export_info["total_params"] += int(t.size)

            export_info["tensor_count"] += 1

    np.savez(str(output_path), **arrays)
    export_info["output_size_bytes"] = output_path.stat().st_size
    export_info["output_size_mb"] = round(
        output_path.stat().st_size / (1024 * 1024), 2
    )

    return export_info


# ---------------------------------------------------------------------------
# Exportación a JSON (pesos desquantizados)
# ---------------------------------------------------------------------------

def export_to_json(
    safetensors_path: str | Path,
    output_path: str | Path,
    max_tensors: Optional[int] = None,
    only_metadata: bool = False,
) -> Dict[str, object]:
    """
    Exporta pesos a archivo JSON.

    ADVERTENCIA: Los archivos JSON con pesos completos son muy grandes
    (el modelo 800M ocuparía varios GB). Usar solo_metadata=True para
    exportar solo metadados y estadísticas.

    Parameters
    ----------
    safetensors_path : str | Path
    output_path : str | Path
    max_tensors : int opcional
        Limitar el número de tensores a exportar.
    only_metadata : bool
        Si True, exportar solo estadísticas, no los pesos.
    """
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    safetensors_path = Path(safetensors_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_info: Dict[str, object] = {
        "source": str(safetensors_path),
        "output": str(output_path),
        "only_metadata": only_metadata,
        "tensor_count": 0,
        "total_params": 0,
        "tensors": [],
    }

    if only_metadata:
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for i, name in enumerate(f.keys()):
                if max_tensors is not None and i >= max_tensors:
                    break
                t = f.get_tensor(name)
                info = {
                    "name": name,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "params": int(t.size),
                    "mean": float(np.mean(t.astype(np.float64))),
                    "std": float(np.std(t.astype(np.float64))),
                    "min": float(np.min(t.astype(np.float64))),
                    "max": float(np.max(t.astype(np.float64))),
                }
                export_info["tensors"].append(info)
                export_info["total_params"] += int(t.size)
                export_info["tensor_count"] += 1

        with open(output_path, "w") as f:
            json.dump(export_info, f, indent=2)

        return export_info

    # Exportación completa (pesos incluidos) - solo para modelos pequeños
    arrays: Dict[str, list] = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        for i, name in enumerate(f.keys()):
            if max_tensors is not None and i >= max_tensors:
                break
            t = f.get_tensor(name)
            if t.dtype == np.int8:
                scale_name = name + "_scale"
                if scale_name in f.keys():
                    scale = f.get_tensor(scale_name).astype(np.float32)
                    arr = t.astype(np.float32) * scale.T
                else:
                    arr = t.astype(np.float32)
            else:
                arr = t.astype(np.float32)
            arrays[name] = arr.tolist()

    export_info["arrays"] = arrays
    export_info["total_params"] = sum(len(v) for v in arrays.values() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list))

    with open(output_path, "w") as f:
        json.dump(export_info, f)

    export_info["output_size_bytes"] = output_path.stat().st_size
    export_info["output_size_mb"] = round(
        output_path.stat().st_size / (1024 * 1024), 2
    )

    return export_info


# ---------------------------------------------------------------------------
# Exportación a binario crudo (raw) para C/C++
# ---------------------------------------------------------------------------

def export_to_raw_binary(
    safetensors_path: str | Path,
    output_dir: str | Path,
    dequantize: bool = True,
) -> Dict[str, object]:
    """
    Exporta cada tensor como un archivo binario crudo separado.

    Cada archivo contiene:
    - 4 bytes: número de dimensiones (uint32)
    - N * 4 bytes: dimensiones (uint32 cada una)
    - datos en float32 (little-endian)

    También se genera un archivo index.txt con los nombres y formas.
    """
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    safetensors_path = Path(safetensors_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines: List[str] = []
    total_params = 0
    tensor_count = 0

    with safe_open(str(safetensors_path), framework="numpy") as f:
        for name in f.keys():
            t = f.get_tensor(name)

            if dequantize and t.dtype == np.int8:
                scale_name = name + "_scale"
                if scale_name in f.keys():
                    scale = f.get_tensor(scale_name).astype(np.float32)
                    arr = (t.astype(np.float32) * scale.T).astype(np.float32)
                else:
                    arr = t.astype(np.float32)
            else:
                arr = t.astype(np.float32)

            # Nombre del archivo de salida (reemplazar puntos por guiones bajos)
            safe_name = name.replace(".", "_").replace("/", "_")
            bin_path = output_dir / f"{safe_name}.bin"

            with open(bin_path, "wb") as binf:
                # Escribir número de dimensiones
                ndim = len(arr.shape)
                binf.write(struct.pack("<I", ndim))
                # Escribir dimensiones
                for dim in arr.shape:
                    binf.write(struct.pack("<I", dim))
                # Escribir datos float32
                binf.write(arr.tobytes())

            total_params += int(arr.size)
            tensor_count += 1
            index_lines.append(f"{name}\t{safe_name}.bin\t{','.join(map(str, arr.shape))}\tfloat32")

    # Escribir índice
    index_path = output_dir / "index.txt"
    with open(index_path, "w") as f:
        f.write("nombre_tensor\tarchivo_binario\tforma\tdtype\n")
        for line in index_lines:
            f.write(line + "\n")

    return {
        "source": str(safetensors_path),
        "output_dir": str(output_dir),
        "tensor_count": tensor_count,
        "total_params": total_params,
        "dequantize": dequantize,
    }


# ---------------------------------------------------------------------------
# Exportación de configuración
# ---------------------------------------------------------------------------

def export_config_to_yaml(
    config_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Exporta config.json a YAML (si PyYAML está disponible).
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ImportError("PyYAML no está instalado. Ejecutar: pip install pyyaml")

    config_path = Path(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def export_config_to_ini(
    config_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Exporta config.json a formato INI (secciones planas).
    """
    config_path = Path(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    config_parser = configparser.ConfigParser()

    def _flatten(d: Dict[str, object], prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                # Usar la primera parte del key como sección
                parts = key.split(".")
                section = parts[0] if len(parts) > 1 else "DEFAULT"
                option = ".".join(parts[1:]) if len(parts) > 1 else key
                if section not in config_parser:
                    config_parser[section] = {}
                config_parser[section][option] = str(v)

    _flatten(config)

    with open(output_path, "w") as f:
        config_parser.write(f)


# ---------------------------------------------------------------------------
# Reporte de arquitectura en Markdown
# ---------------------------------------------------------------------------

def export_architecture_markdown(
    config_path: str | Path,
    safetensors_path: Optional[str | Path] = None,
    output_path: str | Path = "architecture.md",
) -> None:
    """
    Genera un reporte de arquitectura del modelo en formato Markdown.

    Incluye:
        - Hiperparámetros del modelo.
        - Conteo de parámetros por capa.
        - Especificación de tensores en .safetensors.
        - Estadísticas de cuantización.
    """
    config_path = Path(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    lines: List[str] = []
    lines.append("# Arquitectura del Modelo AI 800M")
    lines.append("")
    lines.append("## Hiperparámetros")
    lines.append("")
    lines.append("| Parámetro | Valor |")
    lines.append("|-----------|-------|")
    for key, value in config.items():
        if key in ("quantization_config", "experiment_metadata"):
            continue
        lines.append(f"| {key} | {value} |")
    lines.append("")

    if "quantization_config" in config:
        lines.append("## Cuantización")
        lines.append("")
        for key, value in config["quantization_config"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    lines.append("## Conteo de Parámetros")
    lines.append("")
    lines.append("| Componente | Parámetros |")
    lines.append("|------------|------------|")

    # Calcular desglose usando model_builder
    from scripts.model_builder import (
        count_embedding_params, count_layer_params,
        count_final_norm_params, count_total_params, L, D, DFF,
    )

    lines.append(f"| Embeddings de entrada | {count_embedding_params()//2:,} |")
    lines.append(f"| Cabeza de salida (lm_head) | {count_embedding_params()//2:,} |")
    lines.append(f"| Por capa Transformer | {count_layer_params():,} |")
    lines.append(f"| Número de capas | {L()} |")
    lines.append(f"| Todas las capas | {count_layer_params()*L():,} |")
    lines.append(f"| Norma final | {count_final_norm_params():,} |")
    lines.append(f"| **Total** | **{count_total_params():,}** |")
    lines.append("")

    lines.append("## Estructura de Tensores (.safetensors)")
    lines.append("")
    lines.append("| Tensor | Forma | Tipo | Parámetros |")
    lines.append("|--------|-------|------|------------|")

    if safetensors_path is not None and safe_open is not None:
        from scripts.model_builder import build_tensor_spec
        spec = build_tensor_spec()
        spec_dict = {name: (shape, dtype) for name, shape, dtype in spec}

        with safe_open(str(safetensors_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                shape_str = " × ".join(str(s) for s in t.shape)
                n_params = int(t.size)
                dtype_str = str(t.dtype)
                lines.append(f"| {name} | {shape_str} | {dtype_str} | {n_params:,} |")
    else:
        lines.append("| (requiere archivo .safetensors para detalles completos) | | | | |")

    lines.append("")
    lines.append("## Notas del Experimento")
    lines.append("")
    lines.append("- Creado sin entrenamiento (0 pasos de gradiente).")
    lines.append("- Pesos sintetizados y editados analíticamente.")
    lines.append("- Formato: INT8 simétrico per-canal + escalas FP16.")
    lines.append("- Arquitectura: Llama/GQA/RoPE/SwiGLU/RMSNorm.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def export_pytorch_loader_template(
    output_path: str | Path,
) -> None:
    """
    Genera una plantilla de script Python para cargar el modelo en PyTorch.

    Nota: Esto es solo una plantilla; requiere PyTorch instalado para ejecutar.
    """
    template = '''#!/usr/bin/env python3
"""
Plantilla para cargar el modelo 800M en PyTorch.
Requiere: pip install torch safetensors
"""
import json
import torch
from safetensors import safe_open


def load_model_800m(model_dir: str, device: str = "cpu"):
    model_dir = Path(model_dir)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # Crear modelo vacío (aquí iría la definición de tu clase Llama)
    # model = LlamaForCausalLM(config)
    # model.to(device)

    # Cargar pesos
    tensors = {}
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    # Asignar pesos al modelo
    # model.load_state_dict(tensors, strict=False)

    return tensors, config


if __name__ == "__main__":
    tensors, config = load_model_800m("model")
    print(f"Cargados {len(tensors)} tensores")
'''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(template)


def export_tensorflow_loader_template(
    output_path: str | Path,
) -> None:
    """
    Genera una plantilla de script Python para cargar el modelo en TensorFlow.

    Nota: Esto es solo una plantilla; requiere TensorFlow instalado.
    """
    template = '''#!/usr/bin/env python3
"""
Plantilla para cargar el modelo 800M en TensorFlow.
Requiere: pip install tensorflow safetensors
"""
import json
import numpy as np
import tensorflow as tf
from safetensors import safe_open


def load_model_800m_tf(model_dir: str):
    model_dir = Path(model_dir)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    tensors = {}
    with safe_open(str(model_dir / "model.safetensors"), framework="numpy") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name).numpy()

    # Aquí iría la asignación a capas de Keras/TensorFlow
    # Por ejemplo: model.get_layer(name).set_weight(tensors[name])

    return tensors, config


if __name__ == "__main__":
    tensors, config = load_model_800m_tf("model")
    print(f"Cargados {len(tensors)} tensores")
'''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(template)


# ---------------------------------------------------------------------------
# Función de conveniencia
# ---------------------------------------------------------------------------

def export_all_formats(
    model_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, object]:
    """
    Exporta el modelo a todos los formatos disponibles.

    Returns
    -------
    dict con resultados de cada exportación.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    results: Dict[str, object] = {}

    # NPZ
    try:
        npz_result = export_to_npz(
            safetensors_path,
            output_dir / "model.npz",
            dequantize=True,
        )
        results["npz"] = {"status": "success", "details": npz_result}
    except Exception as e:
        results["npz"] = {"status": "error", "message": str(e)}

    # JSON solo metadatos
    try:
        json_result = export_to_json(
            safetensors_path,
            output_dir / "model_metadata.json",
            only_metadata=True,
        )
        results["json_metadata"] = {"status": "success", "details": json_result}
    except Exception as e:
        results["json_metadata"] = {"status": "error", "message": str(e)}

    # Binario crudo
    try:
        raw_result = export_to_raw_binary(
            safetensors_path,
            output_dir / "raw_tensors",
            dequantize=True,
        )
        results["raw_binary"] = {"status": "success", "details": raw_result}
    except Exception as e:
        results["raw_binary"] = {"status": "error", "message": str(e)}

    # Config YAML
    try:
        export_config_to_yaml(config_path, output_dir / "config.yaml")
        results["config_yaml"] = {"status": "success"}
    except Exception as e:
        results["config_yaml"] = {"status": "error", "message": str(e)}

    # Config INI
    try:
        export_config_to_ini(config_path, output_dir / "config.ini")
        results["config_ini"] = {"status": "success"}
    except Exception as e:
        results["config_ini"] = {"status": "error", "message": str(e)}

    # Arquitectura Markdown
    try:
        export_architecture_markdown(
            config_path,
            safetensors_path,
            output_dir / "architecture.md",
        )
        results["architecture_md"] = {"status": "success"}
    except Exception as e:
        results["architecture_md"] = {"status": "error", "message": str(e)}

    # Plantillas de carga
    try:
        export_pytorch_loader_template(output_dir / "load_pytorch.py")
        results["pytorch_loader"] = {"status": "success"}
    except Exception as e:
        results["pytorch_loader"] = {"status": "error", "message": str(e)}

    try:
        export_tensorflow_loader_template(output_dir / "load_tensorflow.py")
        results["tensorflow_loader"] = {"status": "success"}
    except Exception as e:
        results["tensorflow_loader"] = {"status": "error", "message": str(e)}

    return results


__all__ = [
    "export_to_npz",
    "export_to_json",
    "export_to_raw_binary",
    "export_config_to_yaml",
    "export_config_to_ini",
    "export_architecture_markdown",
    "export_pytorch_loader_template",
    "export_tensorflow_loader_template",
    "export_all_formats",
]
