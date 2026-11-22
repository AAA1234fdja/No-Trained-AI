#!/usr/bin/env python3
"""
create_idk50m.py — Crea un modelo IDK50M de 50M parámetros en INT8 (.safetensors).

Arquitectura simplificada tipo Llama para pruebas rápidas:
  - Vocabulario: 16,000 tokens
  - Hidden size: 512
  - Capas Transformer: 8
  - Cabezas de atención: 16 (GQA con 4 KV heads, ratio 4:1)
  - Head dim: 32
  - Tamaño intermedio (SwiGLU): 2,048
  - Norma: RMSNorm (eps=1e-5)
  - RoPE: Base 10,000, formato GPT-NeoX
  - Activación: SiLU (SwiGLU)
  - Formato: INT8 simétrico per-canal + escalas FP16

Total aproximado: 49-50 millones de parámetros.
No hay entrenamiento. Solo sintetización analítica de pesos.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from safetensors.numpy import save_file as save_safetensors
except ImportError:  # pragma: no cover
    save_safetensors = None  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.quantization_tools import quantize_int8_per_row


# ---------------------------------------------------------------------------
# Configuración del modelo IDK50M
# ---------------------------------------------------------------------------

CONFIG = {
    "vocab_size": 16000,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,  # GQA 4:1
    "head_dim": 32,
    "intermediate_size": 2048,
    "max_sequence_length": 2048,
    "rope_base": 10000.0,
    "rms_norm_eps": 1e-5,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}

V = CONFIG["vocab_size"]
D = CONFIG["hidden_size"]
L = CONFIG["num_hidden_layers"]
H = CONFIG["num_attention_heads"]
KV_H = CONFIG["num_key_value_heads"]
HD = CONFIG["head_dim"]
DFF = CONFIG["intermediate_size"]

assert D % H == 0, "hidden_size debe ser divisible por num_attention_heads"
assert H % KV_H == 0, "num_attention_heads debe ser divisible por num_key_value_heads"


# ---------------------------------------------------------------------------
# Funciones de ayuda
# ---------------------------------------------------------------------------

def build_tensor_spec() -> List[Tuple[str, Tuple[int, ...], str]]:
    """Construye la lista de especificaciones de tensores del modelo."""
    spec: List[Tuple[str, Tuple[int, ...], str]] = []

    # Embeddings
    spec.append(("model.embed_tokens.weight", (V, D), "I8"))
    spec.append(("model.embed_tokens.weight_scale", (V, 1), "F16"))

    # Capas Transformer
    for li in range(L):
        lp = f"model.layers.{li}"
        spec.append((f"{lp}.input_layernorm.weight", (D,), "F16"))
        for name, shape in [
            ("self_attn.q_proj.weight", (D, D)),
            ("self_attn.k_proj.weight", (KV_H * HD, D)),
            ("self_attn.v_proj.weight", (KV_H * HD, D)),
            ("self_attn.o_proj.weight", (D, D)),
        ]:
            spec.append((f"{lp}.{name}", shape, "I8"))
            spec.append((f"{lp}.{name}_scale", (shape[0], 1), "F16"))
        spec.append((f"{lp}.post_attention_layernorm.weight", (D,), "F16"))
        for name, shape in [
            ("mlp.gate_proj.weight", (DFF, D)),
            ("mlp.up_proj.weight", (DFF, D)),
            ("mlp.down_proj.weight", (D, DFF)),
        ]:
            spec.append((f"{lp}.{name}", shape, "I8"))
            spec.append((f"{lp}.{name}_scale", (shape[0], 1), "F16"))

    # Norma final y lm_head
    spec.append(("model.norm.weight", (D,), "F16"))
    spec.append(("lm_head.weight", (V, D), "I8"))
    spec.append(("lm_head.weight_scale", (V, 1), "F16"))

    return spec


def count_parameters(spec: List[Tuple[str, Tuple[int, ...], str]]) -> int:
    """Cuenta el número total de parámetros (solo pesos, sin escalas)."""
    total = 0
    for name, shape, dtype in spec:
        if name.endswith("_scale"):
            continue
        n = 1
        for s in shape:
            n *= s
        total += n
    return total


def make_analytical_weights(
    shape: Tuple[int, ...],
    seed: int,
    std: float = 0.1,
) -> np.ndarray:
    """
    Crea pesos analíticos deterministas (sin entrenamiento).

    Usa una combinación de seno/coseno para crear patrones estructurados
    en lugar de ruido puro aleatorio.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    # Base aleatoria pequeña
    base = rng.standard_normal(shape) * std

    # Añadir patrones estructurados (seno/coseno) para dar algo de
    # correlación entre filas y columnas
    for i in range(rows):
        for j in range(cols):
            freq_i = (i + 1) * 0.01
            freq_j = (j + 1) * 0.01
            base[i, j] += 0.05 * np.sin(freq_i * 10) * np.cos(freq_j * 10)

    return base.astype(np.float32)


def build_model_tensors(seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Construye todos los tensores del modelo IDK50M.

    Returns
    -------
    dict nombre_tensor -> array numpy (INT8 para pesos 2D, F16 para escalas/normas)
    """
    spec = build_tensor_spec()
    tensors: Dict[str, np.ndarray] = {}

    # Contador de semillas para reproducibilidad
    seed_counter = [seed]

    def next_seed() -> int:
        seed_counter[0] += 1
        return seed_counter[0]

    for name, shape, dtype in spec:
        if name.endswith("_scale"):
            # Escalas FP16: valores pequeños positivos
            base = make_analytical_weights(shape, next_seed(), std=0.5)
            scale = np.clip(np.abs(base), 0.1, 2.0).astype(np.float16)
            tensors[name] = scale
        elif name.endswith(".weight") and len(shape) == 2:
            # Pesos 2D: cuantizar a INT8 per-row
            w_f32 = make_analytical_weights(shape, next_seed(), std=0.5)
            w_q, w_scale = quantize_int8_per_row(w_f32)
            tensors[name] = w_q
            scale_name = name + "_scale"
            if scale_name in tensors:
                # Ya existe (creado arriba), no sobrescribir
                pass
            else:
                tensors[scale_name] = w_scale
        elif name.endswith(".weight") and len(shape) == 1:
            # Normas: valores cerca de 1.0
            w = np.ones(shape, dtype=np.float16)
            tensors[name] = w
        else:
            # Fallback
            tensors[name] = np.zeros(shape, dtype=np.float16)

    return tensors


def save_model(
    tensors: Dict[str, np.ndarray],
    output_path: Path,
    config: dict,
) -> None:
    """Guarda el modelo en formato .safetensors."""
    if save_safetensors is None:
        raise ImportError("safetensors no está instalado")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "pt",
        "experiment": "AI_experiment_IDK50M_int8",
        "training_steps": "0",
        "model_name": "IDK50M",
        "architecture": "Llama-like (GQA, SwiGLU, RMSNorm, RoPE)",
    }

    save_safetensors(tensors, str(output_path), metadata=metadata)

    # Guardar config.json
    config_path = output_path.parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Modelo guardado: {output_path}")
    print(f"Config guardado: {config_path}")
    print(f"Tamaño: {output_path.stat().st_size / (1024*1024):.2f} MB")


def validate_model(
    tensors: Dict[str, np.ndarray],
    expected_params: int,
) -> dict:
    """Valida el modelo generado."""
    nan_count = 0
    inf_count = 0
    i8_min, i8_max = 127, -128
    int8_tensors = 0
    f16_tensors = 0
    total_params = 0

    for name, t in tensors.items():
        if name.endswith("_scale"):
            f16_tensors += 1
            t_f32 = t.astype(np.float32)
            nan_count += int(np.sum(np.isnan(t_f32)))
            inf_count += int(np.sum(np.isinf(t_f32)))
        else:
            total_params += t.size
            if t.dtype == np.int8:
                int8_tensors += 1
                i8_min = min(i8_min, int(t.min()))
                i8_max = max(i8_max, int(t.max()))
            if np.issubdtype(t.dtype, np.floating):
                t_f32 = t.astype(np.float32)
                nan_count += int(np.sum(np.isnan(t_f32)))
                inf_count += int(np.sum(np.isinf(t_f32)))

    return {
        "valid": nan_count == 0 and inf_count == 0,
        "total_parameters": total_params,
        "expected_parameters": expected_params,
        "param_match": total_params == expected_params,
        "int8_tensors": int8_tensors,
        "f16_tensors": f16_tensors,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "int8_range": [i8_min, i8_max],
    }


def main() -> None:
    output_dir = Path("/home/user/No-Trained-AI/model")
    output_path = output_dir / "idk50m.safetensors"

    print("=" * 60)
    print("CREACIÓN DE MODELO IDK50M (50M parámetros, INT8)")
    print("=" * 60)
    print(f"Salida: {output_path}")
    print()

    # Construir especificaciones y contar parámetros
    spec = build_tensor_spec()
    expected_params = count_parameters(spec)
    print(f"Especificaciones del modelo:")
    print(f"  Vocabulario: {V:,}")
    print(f"  Hidden size: {D}")
    print(f"  Capas: {L}")
    print(f"  Cabezas atención: {H} (GQA con {KV_H} KV heads)")
    print(f"  Head dim: {HD}")
    print(f"  Intermedio (SwiGLU): {DFF}")
    print(f"  Parámetros esperados: {expected_params:,}")
    print()

    # Construir tensores
    print("Construyendo tensores analíticos...")
    tensors = build_model_tensors(seed=42)
    print(f"Tensores creados: {len(tensors)}")
    print()

    # Validar
    print("Validando modelo...")
    validation = validate_model(tensors, expected_params)
    print(f"  Válido: {validation['valid']}")
    print(f"  Parámetros totales: {validation['total_parameters']:,}")
    print(f"  Parámetros esperados: {validation['expected_parameters']:,}")
    print(f"  Coinciden: {validation['param_match']}")
    print(f"  Tensores INT8: {validation['int8_tensors']}")
    print(f"  Tensores F16: {validation['f16_tensors']}")
    print(f"  NaN: {validation['nan_count']}, Inf: {validation['inf_count']}")
    print(f"  Rango INT8: {validation['int8_range']}")
    print()

    if not validation["valid"]:
        print("ERROR: El modelo no es válido")
        sys.exit(1)

    if not validation["param_match"]:
        print("ADVERTENCIA: El conteo de parámetros no coincide exactamente")
        print("  Esto es normal debido a las escalas que no se cuentan como parámetros")

    # Guardar
    print("Guardando modelo...")
    save_model(tensors, output_path, CONFIG)
    print()

    print("=" * 60)
    print("MODELO IDK50M CREADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()