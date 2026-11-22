#!/usr/bin/env python3
"""
model_builder.py — Constructor y validador de arquitectura del modelo 800M.

Define la especificación completa del modelo Llama/GQA (800,260,608 parámetros)
y proporciona funciones para:

  - Generar el tensor_spec (lista de nombre, forma, dtype) para .safetensors.
  - Validar que un config.json sea consistente con la arquitectura esperada.
  - Calcular el conteo exacto de parámetros por capa y en total.
  - Generar un config.json desde cero con los hiperparámetros del experimento.
  - Verificar que un archivo .safetensors cargado coincida con el tensor_spec.
  - Estimar el tamaño del archivo en disco.

La arquitectura es:
  - LlamaForCausalLM, 21 capas, hidden=1536, 12 cabezas de query, 4 KV (GQA 3:1),
    intermedio=5888, vocab=32000, RoPE theta=10000, RMSNorm eps=1e-6,
    SwiGLU, tie_word_embeddings=false.
  - Cuantización: INT8 simétrico per-canal para pesos 2D, FP16 para escalas
    y parámetros 1D (normas).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Hiperparámetros del modelo 800M
# ---------------------------------------------------------------------------

MODEL_CONFIG: Dict[str, object] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 32000,
    "hidden_size": 1536,
    "intermediate_size": 5888,
    "num_hidden_layers": 21,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "torch_dtype": "int8",
    "quantization_config": {
        "quant_method": "symmetric_per_channel_int8",
        "bits": 8,
        "scale_suffix": ".weight_scale",
        "unquantized_1d_dtype": "float16",
    },
    "experiment_metadata": {
        "name": "AI-Experiment-800M-Analytical",
        "total_weight_parameters": 800260608,
        "training_steps": 0,
        "creation_method": "pure_analytical_weight_synthesis_and_editing",
    },
}


def get_config() -> Dict[str, object]:
    """Devuelve una copia del diccionario de configuración del modelo."""
    return json.loads(json.dumps(MODEL_CONFIG))


# ---------------------------------------------------------------------------
# Cálculo de parámetros
# ---------------------------------------------------------------------------

def V() -> int:
    return int(MODEL_CONFIG["vocab_size"])


def D() -> int:
    return int(MODEL_CONFIG["hidden_size"])


def DFF() -> int:
    return int(MODEL_CONFIG["intermediate_size"])


def L() -> int:
    return int(MODEL_CONFIG["num_hidden_layers"])


def H() -> int:
    return int(MODEL_CONFIG["num_attention_heads"])


def HKV() -> int:
    return int(MODEL_CONFIG["num_key_value_heads"])


def HD() -> int:
    return int(MODEL_CONFIG["head_dim"])


def KV_D() -> int:
    return HKV() * HD()


def count_embedding_params() -> int:
    """Parámetros de embeddings de entrada y salida (sin atar)."""
    return V() * D() * 2  # embed_tokens + lm_head


def count_layer_params() -> int:
    """Parámetros por capa Transformer."""
    # input_layernorm + post_attention_layernorm: 2 * D
    norms = 2 * D()

    # q_proj: D x D
    q = D() * D()
    # k_proj: KV_D x D
    k = KV_D() * D()
    # v_proj: KV_D x D
    v = KV_D() * D()
    # o_proj: D x D
    o = D() * D()
    attn_total = q + k + v + o

    # gate_proj: DFF x D, up_proj: DFF x D, down_proj: D x DFF
    mlp = DFF() * D() * 2 + D() * DFF()

    return norms + attn_total + mlp


def count_final_norm_params() -> int:
    """Parámetros de la norma final (model.norm.weight)."""
    return D()


def count_total_params() -> int:
    """Conteo total de parámetros del modelo."""
    return (
        count_embedding_params()
        + L() * count_layer_params()
        + count_final_norm_params()
    )


def param_breakdown() -> Dict[str, int]:
    """Desglose detallado de parámetros por componente."""
    return {
        "embed_tokens": V() * D(),
        "lm_head": V() * D(),
        "per_layer": count_layer_params(),
        "num_layers": L(),
        "all_layers": L() * count_layer_params(),
        "final_norm": count_final_norm_params(),
        "total": count_total_params(),
    }


# ---------------------------------------------------------------------------
# Tensor specification para .safetensors
# ---------------------------------------------------------------------------

def build_tensor_spec() -> List[Tuple[str, Tuple[int, ...], str]]:
    """
    Construye la especificación completa de tensores para el archivo
    .safetensors, en el orden exacto en que deben escribirse.

    Cada entrada es (nombre, forma, dtype) donde dtype es "I8" (INT8 per-row)
    o "F16" (float16 para escalas y parámetros 1D).

    Returns
    -------
    lista de (nombre, forma, dtype).
    """
    spec: List[Tuple[str, Tuple[int, ...], str]] = []

    # Embeddings de entrada y salida (INT8 per-row + escala FP16)
    spec.append(("model.embed_tokens.weight", (V(), D()), "I8"))
    spec.append(("model.embed_tokens.weight_scale", (V(), 1), "F16"))

    for li in range(L()):
        lp = f"model.layers.{li}"

        # Normas de entrada y post-attention (FP16, 1D)
        spec.append((f"{lp}.input_layernorm.weight", (D(),), "F16"))

        # Atención: q, k, v, o (INT8 per-row + escala FP16)
        for proj_name, shape in [
            ("self_attn.q_proj", (D(), D())),
            ("self_attn.k_proj", (KV_D(), D())),
            ("self_attn.v_proj", (KV_D(), D())),
            ("self_attn.o_proj", (D(), D())),
        ]:
            spec.append((f"{lp}.{proj_name}.weight", shape, "I8"))
            spec.append((f"{lp}.{proj_name}.weight_scale", (shape[0], 1), "F16"))

        # Norma post-attention (FP16, 1D)
        spec.append((f"{lp}.post_attention_layernorm.weight", (D(),), "F16"))

        # MLP: gate, up, down (INT8 per-row + escala FP16)
        for proj_name, shape in [
            ("mlp.gate_proj", (DFF(), D())),
            ("mlp.up_proj", (DFF(), D())),
            ("mlp.down_proj", (D(), DFF())),
        ]:
            spec.append((f"{lp}.{proj_name}.weight", shape, "I8"))
            spec.append((f"{lp}.{proj_name}.weight_scale", (shape[0], 1), "F16"))

    # Norma final (FP16, 1D)
    spec.append(("model.norm.weight", (D(),), "F16"))

    # Cabeza de salida (INT8 per-row + escala FP16)
    spec.append(("lm_head.weight", (V(), D()), "I8"))
    spec.append(("lm_head.weight_scale", (V(), 1), "F16"))

    return spec


def verify_tensor_spec_consistency(
    spec: List[Tuple[str, Tuple[int, ...], str]],
) -> Dict[str, object]:
    """
    Verifica que el tensor_spec sea consistente con los hiperparámetros del
    modelo y que el conteo de parámetros coincida.

    Returns
    -------
    dict con resultado de verificación.
    """
    total_core = 0
    total_scale = 0
    tensor_count = len(spec)
    issues: List[str] = []

    expected_total = count_total_params()
    expected_tensors = 2  # embed + lm_head
    expected_tensors += L() * (
        2  # input_layernorm + post_attention_layernorm
        + 4 * 2  # q, k, v, o (weight + scale)
        + 1  # post_attention_layernorm
        + 3 * 2  # gate, up, down (weight + scale)
    )
    expected_tensors += 1  # final norm
    expected_tensors += 2  # lm_head + lm_head scale

    for name, shape, dtype in spec:
        n_elems = 1
        for s in shape:
            n_elems *= s

        if dtype == "I8":
            total_core += n_elems
        elif dtype == "F16":
            if name.endswith("_scale"):
                total_scale += n_elems
            else:
                total_core += n_elems
        else:
            issues.append(f"dtype desconocido en {name}: {dtype}")

    if total_core != expected_total:
        issues.append(
            f"Parámetros core ({total_core}) != esperados ({expected_total})"
        )

    if tensor_count != expected_tensors:
        issues.append(
            f"Conteo de tensores ({tensor_count}) != esperado ({expected_tensors})"
        )

    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "tensor_count": tensor_count,
        "expected_tensors": expected_tensors,
        "total_core_params": total_core,
        "expected_core_params": expected_total,
        "total_scale_params": total_scale,
    }


# ---------------------------------------------------------------------------
# Generación de config.json
# ---------------------------------------------------------------------------

def write_config(
    path: str | Path,
    config: Optional[Dict[str, object]] = None,
) -> None:
    """
    Escribe un archivo config.json para el modelo.
    """
    if config is None:
        config = get_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def validate_config(
    config: Dict[str, object],
) -> Dict[str, object]:
    """
    Valida que un diccionario de configuración tenga todas las claves
    requeridas y valores consistentes.

    Returns
    -------
    dict con claves: valid (bool), issues (list).
    """
    required_keys = [
        "architectures", "model_type", "vocab_size", "hidden_size",
        "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings",
        "rms_norm_eps", "rope_theta", "tie_word_embeddings",
    ]

    issues: List[str] = []

    for key in required_keys:
        if key not in config:
            issues.append(f"Clave faltante: {key}")

    if "vocab_size" in config and config["vocab_size"] != 32000:
        issues.append(f"vocab_size = {config['vocab_size']}, se esperaba 32000")
    if "hidden_size" in config and config["hidden_size"] != 1536:
        issues.append(f"hidden_size = {config['hidden_size']}, se esperaba 1536")
    if "num_hidden_layers" in config and config["num_hidden_layers"] != 21:
        issues.append(f"num_hidden_layers = {config['num_hidden_layers']}, se esperaba 21")
    if "num_attention_heads" in config and config["num_attention_heads"] != 12:
        issues.append(f"num_attention_heads = {config['num_attention_heads']}, se esperaba 12")
    if "num_key_value_heads" in config and config["num_key_value_heads"] != 4:
        issues.append(f"num_key_value_heads = {config['num_key_value_heads']}, se esperaba 4")
    if "head_dim" in config and config["head_dim"] != 128:
        issues.append(f"head_dim = {config['head_dim']}, se esperaba 128")
    if "intermediate_size" in config and config["intermediate_size"] != 5888:
        issues.append(f"intermediate_size = {config['intermediate_size']}, se esperaba 5888")

    # Consistencia: num_heads debe ser múltiplo de num_kv_heads
    if "num_attention_heads" in config and "num_key_value_heads" in config:
        if config["num_attention_heads"] % config["num_key_value_heads"] != 0:
            issues.append(
                "num_attention_heads no es múltiplo de num_key_value_heads"
            )

    # Consistencia: head_dim * num_kv_heads debe ser coherente con hidden_size / num_heads
    if "head_dim" in config and "num_attention_heads" in config:
        if config["head_dim"] * config["num_attention_heads"] != config.get("hidden_size", 0):
            issues.append(
                "head_dim * num_attention_heads != hidden_size"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Verificación de archivo .safetensors contra spec
# ---------------------------------------------------------------------------

def verify_safetensors_matches_spec(
    path: str | Path,
    spec: Optional[List[Tuple[str, Tuple[int, ...], str]]] = None,
) -> Dict[str, object]:
    """
    Verifica que un archivo .safetensors coincida con el tensor_spec esperado.

    Checks:
        - Mismos nombres de tensor.
        - Mismas formas.
        - Mismos dtypes (I8 -> int8, F16 -> float16).
        - Conteo de parámetros coincide.

    Returns
    -------
    dict con resultado de verificación.
    """
    if safe_open is None:
        raise ImportError("safetensors no está instalado")

    if spec is None:
        spec = build_tensor_spec()

    spec_dict = {name: (shape, dtype) for name, shape, dtype in spec}
    issues: List[str] = []

    total_core = 0
    total_scale = 0
    tensors_found = 0

    with safe_open(str(path), framework="numpy") as f:
        file_keys = set(f.keys())
        spec_keys = set(spec_dict.keys())

        missing = spec_keys - file_keys
        extra = file_keys - spec_keys

        if missing:
            issues.append(f"Tensores faltantes: {sorted(missing)}")
        if extra:
            issues.append(f"Tensores extra: {sorted(extra)}")

        for name in sorted(spec_keys & file_keys):
            expected_shape, expected_dtype = spec_dict[name]
            t = f.get_tensor(name)
            tensors_found += 1

            if tuple(t.shape) != tuple(expected_shape):
                issues.append(
                    f"{name}: forma {tuple(t.shape)} != esperada {tuple(expected_shape)}"
                )

            expected_np_dtype = (
                np.int8 if expected_dtype == "I8" else np.float16
            )
            if t.dtype != expected_np_dtype:
                issues.append(
                    f"{name}: dtype {t.dtype} != esperado {expected_np_dtype}"
                )

            n_elems = 1
            for s in t.shape:
                n_elems *= s
            if expected_dtype == "I8":
                total_core += n_elems
            elif name.endswith("_scale"):
                total_scale += n_elems
            else:
                total_core += n_elems

    expected_total = count_total_params()
    if total_core != expected_total:
        issues.append(
            f"Parámetros core ({total_core}) != esperados ({expected_total})"
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "tensors_found": tensors_found,
        "tensors_expected": len(spec),
        "missing": sorted(spec_keys - file_keys) if 'spec_keys' in dir() else [],
        "extra": sorted(extra) if 'extra' in dir() else [],
        "total_core_params": total_core,
        "expected_core_params": expected_total,
        "total_scale_params": total_scale,
    }


# ---------------------------------------------------------------------------
# Estimación de tamaño
# ---------------------------------------------------------------------------

def estimate_disk_size_mb() -> float:
    """
    Estima el tamaño en MB del archivo .safetensors completo.
    """
    from scripts.quantization_tools import estimate_safetensors_size
    spec = build_tensor_spec()
    bytes_est = estimate_safetensors_size(spec)
    return bytes_est / (1024 * 1024)


__all__ = [
    "get_config",
    "V", "D", "DFF", "L", "H", "HKV", "HD", "KV_D",
    "count_embedding_params", "count_layer_params",
    "count_final_norm_params", "count_total_params",
    "param_breakdown",
    "build_tensor_spec",
    "verify_tensor_spec_consistency",
    "write_config",
    "validate_config",
    "verify_safetensors_matches_spec",
    "estimate_disk_size_mb",
    "MODEL_CONFIG",
]
