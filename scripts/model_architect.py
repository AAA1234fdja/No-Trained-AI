#!/usr/bin/env python3
"""
model_architect.py — Diseñador de arquitecturas de modelo.

Proporciona herramientas para definir, validar y comparar arquitecturas
de modelos de lenguaje (transformer, decoder-only, encoder-decoder):

  - Especificación de arquitectura (capas, cabezas, dimensiones).
  - Conteo automático de parámetros.
  - Estimación de tamaño en disco y memoria.
  - Validación de consistencia (dimensiones compatibles).
  - Generación de config.json para el modelo.
  - Comparación entre arquitecturas.
  - Plantillas de arquitecturas comunes (800M, 1B, 3B, 7B).
  - Cálculo de FLOPs por forward pass.
  - Reporte de arquitectura en Markdown/JSON.
  - Generación de código de inicialización de pesos.

Todas las operaciones son analíticas y no requieren entrenamiento.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuración de arquitectura de un modelo de lenguaje."""
    vocab_size: int = 32000
    hidden_size: int = 1536
    num_hidden_layers: int = 21
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None
    intermediate_size: int = 5888
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_base: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    activation_function: str = "swiglu"
    norm_type: str = "rmsnorm"
    tie_word_embeddings: bool = False
    quantization: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_base": self.rope_base,
            "rope_scaling": self.rope_scaling,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "activation_function": self.activation_function,
            "norm_type": self.norm_type,
            "tie_word_embeddings": self.tie_word_embeddings,
            "quantization": self.quantization,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ArchitectureStats:
    """Estadísticas de una arquitectura de modelo."""
    total_params: int
    embedding_params: int
    lm_head_params: int
    per_layer_params: int
    all_layers_params: int
    final_norm_params: int
    size_fp32_mb: float
    size_fp16_mb: float
    size_int8_mb: float
    flops_per_token: int
    memory_per_seq_mb: float  # memoria por token de secuencia

    def to_dict(self) -> dict:
        return {
            "total_params": self.total_params,
            "embedding_params": self.embedding_params,
            "lm_head_params": self.lm_head_params,
            "per_layer_params": self.per_layer_params,
            "all_layers_params": self.all_layers_params,
            "final_norm_params": self.final_norm_params,
            "size_fp32_mb": round(self.size_fp32_mb, 2),
            "size_fp16_mb": round(self.size_fp16_mb, 2),
            "size_int8_mb": round(self.size_int8_mb, 2),
            "flops_per_token": self.flops_per_token,
            "memory_per_seq_mb": round(self.memory_per_seq_mb, 2),
        }


class ModelArchitect:
    """
    Diseñador de arquitecturas de modelos de lenguaje.

    Permite definir arquitecturas personalizadas, calcular estadísticas,
    validar consistencia y generar configuraciones para modelos.
    """

    # Plantillas de arquitecturas predefinidas
    PREDEFINED_ARCHITECTURES: Dict[str, ModelConfig] = {}

    def __init__(self) -> None:
        self._register_predefined()

    def _register_predefined(self) -> None:
        """Registra arquitecturas predefinidas."""
        self.PREDEFINED_ARCHITECTURES["800M"] = ModelConfig(
            vocab_size=32000,
            hidden_size=1536,
            num_hidden_layers=21,
            num_attention_heads=12,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=5888,
            max_position_embeddings=2048,
        )
        self.PREDEFINED_ARCHITECTURES["100M"] = ModelConfig(
            vocab_size=16000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            head_dim=64,
            intermediate_size=2048,
            max_position_embeddings=1024,
        )
        self.PREDEFINED_ARCHITECTURES["1B"] = ModelConfig(
            vocab_size=32000,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=8192,
            max_position_embeddings=2048,
        )
        self.PREDEFINED_ARCHITECTURES["3B"] = ModelConfig(
            vocab_size=32000,
            hidden_size=2560,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=80,
            intermediate_size=6912,
            max_position_embeddings=2048,
        )
        self.PREDEFINED_ARCHITECTURES["7B"] = ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=11008,
            max_position_embeddings=2048,
        )

    def create_config(
        self,
        name: str = "custom",
        **kwargs,
    ) -> ModelConfig:
        """
        Crea una configuración de modelo personalizada.

        Parameters
        ----------
        name : str
            Nombre de la configuración.
        **kwargs : dict
            Parámetros de arquitectura (vocab_size, hidden_size, etc.).

        Returns
        -------
        ModelConfig con los parámetros especificados.
        """
        # Comenzar desde una plantilla predefinida si se especifica
        template_name = kwargs.pop("_template", None)
        if template_name and template_name in self.PREDEFINED_ARCHITECTURES:
            base = self.PREDEFINED_ARCHITECTURES[template_name]
            config = ModelConfig(**base.to_dict())
        else:
            config = ModelConfig()

        # Aplicar parámetros personalizados
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Recalcular head_dim si es necesario
        if "hidden_size" in kwargs or "num_attention_heads" in kwargs:
            config.head_dim = config.hidden_size // config.num_attention_heads

        return config

    def get_predefined(
        self,
        name: str,
    ) -> Optional[ModelConfig]:
        """
        Obtiene una arquitectura predefinida por nombre.

        Returns
        -------
        ModelConfig o None si no existe.
        """
        return self.PREDEFINED_ARCHITECTURES.get(name)

    def list_predefined(self) -> List[str]:
        """Devuelve la lista de arquitecturas predefinidas."""
        return sorted(self.PREDEFINED_ARCHITECTURES.keys())

    def validate_config(
        self,
        config: ModelConfig,
    ) -> dict:
        """
        Valida la consistencia de una configuración de modelo.

        Checks:
        - hidden_size divisible por num_attention_heads.
        - num_key_value_heads divides num_attention_heads (para GQA).
        - intermediate_size es múltiplo razonable de hidden_size.
        - head_dim es potencia de 2 (recomendado para RoPE).
        - max_position_embeddings es razonable.

        Returns
        -------
        dict con resultados de validación.
        """
        issues: List[str] = []
        warnings: List[str] = []

        # 1. hidden_size divisible por num_attention_heads
        if config.hidden_size % config.num_attention_heads != 0:
            issues.append(
                f"hidden_size ({config.hidden_size}) no es divisible por "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        # 2. head_dim calculado
        expected_head_dim = config.hidden_size // config.num_attention_heads
        if config.head_dim != expected_head_dim:
            warnings.append(
                f"head_dim ({config.head_dim}) no coincide con "
                f"hidden_size/num_attention_heads ({expected_head_dim})"
            )

        # 3. GQA: num_key_value_heads debe dividir num_attention_heads
        if config.num_attention_heads % config.num_key_value_heads != 0:
            issues.append(
                f"num_attention_heads ({config.num_attention_heads}) no es divisible por "
                f"num_key_value_heads ({config.num_key_value_heads})"
            )

        # 4. intermediate_size razonable
        ratio = config.intermediate_size / config.hidden_size
        if ratio < 1.5 or ratio > 5.0:
            warnings.append(
                f"intermediate_size/hidden_size ratio ({ratio:.2f}) está fuera del "
                f"rango típico (1.5-5.0)"
            )

        # 5. head_dim potencia de 2
        if config.head_dim & (config.head_dim - 1) != 0:
            warnings.append(
                f"head_dim ({config.head_dim}) no es potencia de 2, "
                f"lo cual puede afectar el rendimiento de RoPE"
            )

        # 6. max_position_embeddings razonable
        if config.max_position_embeddings < 128:
            warnings.append(
                f"max_position_embeddings ({config.max_position_embeddings}) es muy bajo"
            )

        # 7. vocab_size razonable
        if config.vocab_size < 1000:
            warnings.append(
                f"vocab_size ({config.vocab_size}) es muy bajo para un modelo de lenguaje"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    def calculate_stats(
        self,
        config: ModelConfig,
        tie_embeddings: bool = False,
    ) -> ArchitectureStats:
        """
        Calcula estadísticas de parámetros y tamaño para una arquitectura.

        Parameters
        ----------
        config : ModelConfig
            Configuración del modelo.
        tie_embeddings : bool
            Si True, los embeddings de entrada y salida comparten pesos.

        Returns
        -------
        ArchitectureStats con conteo de parámetros y tamaños.
        """
        # Embeddings de entrada
        embedding_params = config.vocab_size * config.hidden_size

        # lm_head (puede compartir con embeddings si tie_embeddings)
        if tie_embeddings:
            lm_head_params = 0
        else:
            lm_head_params = config.vocab_size * config.hidden_size

        # Norma final
        final_norm_params = config.hidden_size

        # Por capa Transformer
        # Attention: Q, K, V, O projections
        attn_params = (
            4 * config.hidden_size * config.hidden_size  # Q, K, V, O matrices
        )

        # MLP: gate, up, down projections (SwiGLU)
        mlp_params = (
            config.hidden_size * config.intermediate_size * 2  # gate + up
            + config.intermediate_size * config.hidden_size  # down
        )

        # Layer norms (2 por capa)
        layernorm_params = 2 * config.hidden_size

        per_layer_params = attn_params + mlp_params + layernorm_params
        all_layers_params = per_layer_params * config.num_hidden_layers

        total_params = (
            embedding_params + lm_head_params + final_norm_params + all_layers_params
        )

        # Tamaños en diferentes precisiones
        size_fp32_mb = total_params * 4 / (1024 * 1024)
        size_fp16_mb = total_params * 2 / (1024 * 1024)
        size_int8_mb = total_params * 1 / (1024 * 1024)  # INT8 aproximado

        # FLOPs por token (forward pass aproximado)
        # Attention: 4 * batch * seq_len * hidden^2 * layers
        # MLP: 8 * batch * seq_len * hidden * intermediate * layers (SwiGLU)
        # Por token (seq_len=1, batch=1):
        flops_per_token = (
            4 * config.hidden_size ** 2 * config.num_hidden_layers +  # atención
            8 * config.hidden_size * config.intermediate_size * config.num_hidden_layers  # MLP
        )

        # Memoria por secuencia (activaciones aproximadas)
        memory_per_seq_mb = (
            config.num_hidden_layers * config.hidden_size * 4  # activaciones por capa
        ) / (1024 * 1024)

        return ArchitectureStats(
            total_params=total_params,
            embedding_params=embedding_params,
            lm_head_params=lm_head_params,
            per_layer_params=per_layer_params,
            all_layers_params=all_layers_params,
            final_norm_params=final_norm_params,
            size_fp32_mb=size_fp32_mb,
            size_fp16_mb=size_fp16_mb,
            size_int8_mb=size_int8_mb,
            flops_per_token=flops_per_token,
            memory_per_seq_mb=memory_per_seq_mb,
        )

    def compare_architectures(
        self,
        config_a: ModelConfig,
        config_b: ModelConfig,
    ) -> dict:
        """
        Compara dos arquitecturas de modelo.

        Returns
        -------
        dict con diferencias en parámetros, tamaño, FLOPs, etc.
        """
        stats_a = self.calculate_stats(config_a)
        stats_b = self.calculate_stats(config_b)

        return {
            "config_a": config_a.to_dict(),
            "config_b": config_b.to_dict(),
            "stats_a": stats_a.to_dict(),
            "stats_b": stats_b.to_dict(),
            "params_diff": stats_b.total_params - stats_a.total_params,
            "params_diff_percentage": (
                (stats_b.total_params - stats_a.total_params) / stats_a.total_params * 100
                if stats_a.total_params > 0 else 0
            ),
            "size_diff_mb": stats_b.size_int8_mb - stats_a.size_int8_mb,
            "flops_diff": stats_b.flops_per_token - stats_a.flops_per_token,
            "memory_diff_mb": stats_b.memory_per_seq_mb - stats_a.memory_per_seq_mb,
        }

    def generate_config_json(
        self,
        config: ModelConfig,
        output_path: str | Path,
        include_quantization: bool = True,
    ) -> str:
        """
        Genera un archivo config.json para el modelo.

        Parameters
        ----------
        config : ModelConfig
            Configuración del modelo.
        output_path : str | Path
            Ruta para guardar el archivo JSON.
        include_quantization : bool
            Si True, incluye configuración de cuantización.

        Returns
        -------
        str con la ruta del archivo guardado.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.to_dict()

        if include_quantization:
            config_dict["quantization_config"] = {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_enable_fp32_cpu_offload": False,
            }

        config_dict["model_type"] = "llama"
        config_dict["architectures"] = ["LlamaForCausalLM"]
        config_dict["bos_token_id"] = 1
        config_dict["eos_token_id"] = 2
        config_dict["pad_token_id"] = 0
        config_dict["pretraining_tp"] = 1

        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return str(output_path)

    def generate_weight_initialization_code(
        self,
        config: ModelConfig,
        seed: int = 42,
    ) -> str:
        """
        Genera código Python para inicializar pesos aleatoriamente.

        Returns
        -------
        str con el código de inicialización.
        """
        lines: List[str] = []
        lines.append("#!/usr/bin/env python3")
        lines.append('"""')
        lines.append(f"Código de inicialización de pesos para {config.__class__.__name__}")
        lines.append("Generado automáticamente por model_architect.py")
        lines.append('"""')
        lines.append("")
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"rng = np.random.RandomState({seed})")
        lines.append("")
        lines.append("def initialize_weights():")
        lines.append("    weights = {}")
        lines.append("")
        lines.append(f"    # Embeddings")
        lines.append(f"    weights['embed_tokens.weight'] = rng.randn({config.vocab_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append("")
        lines.append(f"    # lm_head")
        lines.append(f"    weights['lm_head.weight'] = rng.randn({config.vocab_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append("")
        lines.append(f"    # Final norm")
        lines.append(f"    weights['final_norm.weight'] = np.ones({config.hidden_size}, dtype=np.float32)")
        lines.append("")
        lines.append("    # Por capa")
        lines.append(f"    for layer_idx in range({config.num_hidden_layers}):")
        lines.append(f"        prefix = f'layers.{{layer_idx}}.'")
        lines.append("")
        lines.append(f"        # Input layernorm")
        lines.append(f"        weights[prefix + 'input_layernorm.weight'] = np.ones({config.hidden_size}, dtype=np.float32)")
        lines.append("")
        lines.append(f"        # Attention projections")
        lines.append(f"        weights[prefix + 'self_attn.q_proj.weight'] = rng.randn({config.hidden_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append(f"        weights[prefix + 'self_attn.k_proj.weight'] = rng.randn({config.hidden_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append(f"        weights[prefix + 'self_attn.v_proj.weight'] = rng.randn({config.hidden_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append(f"        weights[prefix + 'self_attn.o_proj.weight'] = rng.randn({config.hidden_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append("")
        lines.append(f"        # Post-attention layernorm")
        lines.append(f"        weights[prefix + 'post_attention_layernorm.weight'] = np.ones({config.hidden_size}, dtype=np.float32)")
        lines.append("")
        lines.append(f"        # MLP projections")
        lines.append(f"        weights[prefix + 'mlp.gate_proj.weight'] = rng.randn({config.intermediate_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append(f"        weights[prefix + 'mlp.up_proj.weight'] = rng.randn({config.intermediate_size}, {config.hidden_size}).astype(np.float32) * 0.02")
        lines.append(f"        weights[prefix + 'mlp.down_proj.weight'] = rng.randn({config.hidden_size}, {config.intermediate_size}).astype(np.float32) * 0.02")
        lines.append("")
        lines.append("    return weights")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    weights = initialize_weights()")
        lines.append("    print(f'Inicializados {len(weights)} tensores de pesos')")

        return "\n".join(lines)

    def generate_architecture_report(
        self,
        config: ModelConfig,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de arquitectura en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        stats = self.calculate_stats(config)
        validation = self.validate_config(config)

        lines: List[str] = []
        lines.append("# Reporte de Arquitectura de Modelo")
        lines.append("")
        lines.append("## Configuración")
        lines.append("")
        lines.append("| Parámetro | Valor |")
        lines.append("|-----------|-------|")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

        lines.append("## Estadísticas de Parámetros")
        lines.append("")
        lines.append("| Componente | Parámetros | Porcentaje |")
        lines.append("|------------|------------|------------|")

        total = stats.total_params
        components = [
            ("Embeddings de entrada", stats.embedding_params),
            ("Cabeza de salida (lm_head)", stats.lm_head_params),
            ("Todas las capas Transformer", stats.all_layers_params),
            ("Norma final", stats.final_norm_params),
        ]

        for name, count in components:
            pct = count / total * 100.0 if total > 0 else 0.0
            lines.append(f"| {name} | {count:,} | {pct:.1f}% |")

        lines.append(f"| **Total** | **{total:,}** | **100.0%** |")
        lines.append("")

        lines.append("## Tamaño Estimado")
        lines.append("")
        lines.append("| Precisión | Tamaño |")
        lines.append("|-----------|--------|")
        lines.append(f"| FP32 | {stats.size_fp32_mb:.2f} MB |")
        lines.append(f"| FP16 | {stats.size_fp16_mb:.2f} MB |")
        lines.append(f"| INT8 | {stats.size_int8_mb:.2f} MB |")
        lines.append("")

        lines.append("## Rendimiento Estimado")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| FLOPs por token | {stats.flops_per_token:,} |")
        lines.append(f"| Parámetros por capa | {stats.per_layer_params:,} |")
        lines.append(f"| Memoria por secuencia | {stats.memory_per_seq_mb:.2f} MB |")
        lines.append("")

        lines.append("## Validación de Consistencia")
        lines.append("")
        lines.append(f"- **Válido:** {'Sí' if validation['valid'] else 'No'}")
        if validation["issues"]:
            lines.append("- **Problemas:**")
            for issue in validation["issues"]:
                lines.append(f"  - {issue}")
        if validation["warnings"]:
            lines.append("- **Advertencias:**")
            for warning in validation["warnings"]:
                lines.append(f"  - {warning}")
        lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text


def quick_architecture_stats(
    config_dict: Optional[dict] = None,
) -> ArchitectureStats:
    """
    Función de conveniencia: calcula estadísticas de arquitectura.

    Parameters
    ----------
    config_dict : dict opcional
        Diccionario con parámetros de configuración.

    Returns
    -------
    ArchitectureStats con las estadísticas.
    """
    architect = ModelArchitect()
    if config_dict:
        config = ModelConfig(**config_dict)
    else:
        config = architect.get_predefined("800M") or ModelConfig()
    return architect.calculate_stats(config)


__all__ = [
    "ModelConfig",
    "ArchitectureStats",
    "ModelArchitect",
    "quick_architecture_stats",
]
