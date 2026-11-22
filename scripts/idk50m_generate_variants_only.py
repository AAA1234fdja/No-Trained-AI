#!/usr/bin/env python3
"""
idk50m_generate_variants_only.py — Genera variantes IDK50M sin evaluar.
Rápido: solo crea y guarda los archivos .safetensors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.quantization_tools import quantize_int8_per_row
from scripts.idk50m_variants_generator import (
    CONFIG, DISTRIBUTIONS, build_tensor_spec, MODEL_DIR, save_variant,
)


def main() -> None:
    seeds = [42, 123, 456]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERACIÓN DE VARIANTES IDK50M (sin evaluación)")
    print("=" * 70)
    print(f"Distribuciones: {len(DISTRIBUTIONS)}")
    print(f"Semillas: {seeds}")
    print(f"Total variantes: {len(DISTRIBUTIONS) * len(seeds)}")
    print()

    count = 0
    for dist_name, dist_func in DISTRIBUTIONS.items():
        for seed in seeds:
            variant_name = f"{dist_name}_seed{seed}"
            variant_path = MODEL_DIR / f"idk50m_{variant_name}.safetensors"

            # Saltar si ya existe
            if variant_path.exists():
                print(f"  [SKIP] {variant_name} (ya existe)")
                continue

            tensors = {}
            seed_counter = [seed]

            def next_seed() -> int:
                seed_counter[0] += 1
                return seed_counter[0]

            spec = build_tensor_spec()
            for name, shape, dtype in spec:
                if name.endswith("_scale"):
                    base = dist_func(shape, next_seed())
                    scale = np.clip(np.abs(base), 0.1, 2.0).astype(np.float16)
                    tensors[name] = scale
                elif name.endswith(".weight") and len(shape) == 2:
                    w_f32 = dist_func(shape, next_seed())
                    w_q, w_scale = quantize_int8_per_row(w_f32)
                    tensors[name] = w_q
                    scale_name = name + "_scale"
                    if scale_name not in tensors:
                        tensors[scale_name] = w_scale
                elif name.endswith(".weight") and len(shape) == 1:
                    tensors[name] = np.ones(shape, dtype=np.float16)
                else:
                    tensors[name] = np.zeros(shape, dtype=np.float16)

            save_variant(tensors, variant_path, dist_name, seed)
            count += 1
            print(f"  [OK] {variant_name}")

    print(f"\nTotal variantes generadas en esta ejecución: {count}")
    print(f"Total variantes en directorio: {len(list(MODEL_DIR.glob('idk50m_*.safetensors')))}")


if __name__ == "__main__":
    main()