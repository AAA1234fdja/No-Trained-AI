#!/usr/bin/env python3
"""
config_manager.py — Gestor de configuración para el experimento AI 800M.

Proporciona:
  - Carga, guardado y validación de config.json.
  - Gestión de hiperparámetros por etapa de síntesis.
  - Sobrescritura de configuración vía argumentos de línea de comandos.
  - Exportación de configuración a YAML (opcional).
  - Versionado de configuraciones experimentales.
  - Plantillas de configuración para diferentes tamaños de modelo.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


class ExperimentConfig:
    """
    Gestor de configuración para el experimento.

    Almacena hiperparámetros del modelo, rutas de archivos, y
    configuración de etapas de síntesis.
    """

    DEFAULT_CONFIG: Dict[str, object] = {
        "model": {
            "vocab_size": 32000,
            "hidden_size": 1536,
            "intermediate_size": 5888,
            "num_hidden_layers": 21,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "quantization": {
            "method": "symmetric_per_channel_int8",
            "bits": 8,
            "scale_suffix": ".weight_scale",
            "unquantized_1d_dtype": "float16",
        },
        "experiment": {
            "name": "AI-Experiment-800M-Analytical",
            "total_weight_parameters": 800260608,
            "training_steps": 0,
            "creation_method": "pure_analytical_weight_synthesis_and_editing",
        },
        "paths": {
            "weights_file": "model.safetensors",
            "config_file": "config.json",
            "tokenizer_file": "tokenizer.json",
            "results_file": "experiment_results.json",
        },
        "inference": {
            "max_new_tokens": 20,
            "temperature": 0.0,
            "top_k": 40,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "stages": {
            "n_stages": 6,
            "stage_0": "base_orthogonal",
            "stage_1": "analytical_embeddings",
            "stage_2": "induction_head",
            "stage_3": "swiglu_calibration",
            "stage_4": "final_calibrated",
            "stage_5": "antipodal_fix",
        },
    }

    def __init__(
        self,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        if config is None:
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        else:
            self.config = copy.deepcopy(config)

    def get(
        self,
        key_path: str,
        default: object = None,
    ) -> object:
        """
        Obtiene un valor de configuración usando notación de puntos.

        Ejemplo: config.get("model.hidden_size") -> 1536
        """
        keys = key_path.split(".")
        val: object = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(
        self,
        key_path: str,
        value: object,
    ) -> None:
        """Establece un valor de configuración usando notación de puntos."""
        keys = key_path.split(".")
        d = self.config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def to_dict(self) -> Dict[str, object]:
        """Devuelve una copia profunda del diccionario de configuración."""
        return copy.deepcopy(self.config)

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        """Guarda la configuración en archivo JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)
            f.write("\n")

    def save_yaml(
        self,
        path: str | Path,
    ) -> None:
        """Guarda la configuración en archivo YAML (si PyYAML está instalado)."""
        if yaml is None:
            raise ImportError("PyYAML no está instalado. Ejecutar: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "ExperimentConfig":
        """Carga una configuración desde archivo JSON."""
        path = Path(path)
        with open(path, "r") as f:
            config = json.load(f)
        return cls(config)

    def merge(
        self,
        other: Dict[str, object],
    ) -> "ExperimentConfig":
        """
        Fusiona otro diccionario de configuración sobre el actual.
        Los valores de `other` sobrescriben los existentes.
        """
        self._deep_merge(self.config, other)
        return self

    @staticmethod
    def _deep_merge(
        base: Dict[str, object],
        override: Dict[str, object],
    ) -> None:
        """Fusión profunda recursiva."""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ExperimentConfig._deep_merge(base[k], v)
            else:
                base[k] = v


def model_size_config(
    size_name: str,
) -> ExperimentConfig:
    """
    Devuelve una configuración predefinida para diferentes tamaños de modelo.

    Soportados: "800M", "100M", "1B".
    """
    configs: Dict[str, Dict[str, object]] = {
        "800M": {
            "vocab_size": 32000,
            "hidden_size": 1536,
            "intermediate_size": 5888,
            "num_hidden_layers": 21,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "head_dim": 128,
        },
        "100M": {
            "vocab_size": 16000,
            "hidden_size": 768,
            "intermediate_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "head_dim": 64,
        },
        "1B": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
        },
    }

    if size_name not in configs:
        raise ValueError(
            f"Tamaño '{size_name}' no soportado. Opciones: {list(configs.keys())}"
        )

    cfg = ExperimentConfig()
    cfg.config["model"].update(configs[size_name])
    cfg.config["experiment"]["name"] = f"AI-Experiment-{size_name}-Analytical"
    return cfg


def validate_config_keys(
    config: Dict[str, object],
    required_keys: List[str],
) -> List[str]:
    """
    Valida que un diccionario de configuración contenga todas las claves
    requeridas (usando notación de puntos).

    Returns
    -------
    lista de claves faltantes.
    """
    missing: List[str] = []
    for key in required_keys:
        keys = key.split(".")
        val: object = config
        found = True
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                found = False
                break
        if not found:
            missing.append(key)
    return missing


__all__ = [
    "ExperimentConfig",
    "model_size_config",
    "validate_config_keys",
]
