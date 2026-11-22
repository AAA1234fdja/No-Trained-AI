#!/usr/bin/env python3
"""
test_idk50m.py — Prueba rápida de inferencia en el modelo IDK50M.

Prueba el modelo IDK50M (50M parámetros, INT8) con prompts clave
para verificar que la generación funciona sin loops excesivos.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference_engine import MmapInt8LlamaModel, stable_softmax


def detect_token_loop(text: str, min_repeats: int = 3) -> bool:
    """Detecta repetición de token completo."""
    pattern = r"(.{2,30}?)\1{" + str(min_repeats - 1) + r",}"
    return bool(re.search(pattern, text))


def loop_severity(text: str) -> float:
    """Puntuación de severidad de loop (0-1)."""
    pattern = r"(.{2,30}?)\1{2,}"
    match = re.search(pattern, text)
    if not match:
        return 0.0
    token = match.group(1)
    repeats = len(match.group(0)) // len(token)
    return min(1.0, (repeats - 2) * 0.3 + (len(token.strip()) / 50) * 0.2)


def test_idk50m(
    model_dir: Path,
    prompts: List[str],
    max_new_tokens: int = 10,
    temperature: float = 0.3,
    top_k: int = 6,
    repetition_penalty: float = 1.5,
) -> dict:
    """
    Ejecuta pruebas de inferencia en IDK50M.
    """
    weights_path = model_dir / "idk50m.safetensors"
    tokenizer_path = model_dir / "idk50m_tokenizer.json"
    config_path = model_dir / "config.json"

    if not weights_path.exists():
        print(f"ERROR: No se encontró {weights_path}")
        sys.exit(1)

    print("Cargando modelo IDK50M...")
    model = MmapInt8LlamaModel(
        model_dir,
        weights_filename="idk50m.safetensors",
        config_filename="idk50m_config.json",
        tokenizer_filename="idk50m_tokenizer.json",
    )
    print("Modelo cargado.")
    print()

    results = []

    for prompt in prompts:
        t0 = time.time()
        try:
            res = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            elapsed = time.time() - t0
            text = res["generated_text"]
            has_loop = detect_token_loop(text)
            severity = loop_severity(text)
            entropy = res["first_token_entropy_nats"]

            loop_icon = "🔴" if has_loop else "🟢"
            print(
                f"  {loop_icon} [{prompt[:25]:<25}] -> "
                f"{repr(text[:50])} "
                f"(loop={has_loop}, sev={severity:.2f}, H={entropy:.2f}, "
                f"{elapsed:.1f}s)"
            )

            results.append({
                "prompt": prompt,
                "generated_text": text,
                "has_loop": has_loop,
                "severity": severity,
                "entropy": entropy,
                "elapsed_sec": elapsed,
            })
        except Exception as e:
            print(f"  ❌ [{prompt[:25]:<25}] ERROR: {e}")
            results.append({
                "prompt": prompt,
                "error": str(e),
            })

    return results


def main() -> None:
    model_dir = Path("/home/user/No-Trained-AI/model")

    print("=" * 60)
    print("PRUEBA RÁPIDA DE MODELO IDK50M (50M parámetros, INT8)")
    print("=" * 60)
    print()

    prompts = [
        "def ",
        "for i in range(",
        "return ",
        "import ",
        "Complexity:",
        "Analyze data:",
        "Step 1:",
        "Hello",
        "alpha = 42 ; beta = 99 ; alpha =",
    ]

    print("Configuración: temp=0.3, top_k=6, rep_pen=1.5")
    print()

    results = test_idk50m(model_dir, prompts)

    # Resumen
    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        total = len(valid_results)
        loop_count = sum(1 for r in valid_results if r["has_loop"])
        avg_severity = float(np.mean([r["severity"] for r in valid_results]))
        avg_entropy = float(np.mean([r["entropy"] for r in valid_results]))
        avg_elapsed = float(np.mean([r["elapsed_sec"] for r in valid_results]))

        print(f"  Pruebas totales: {total}")
        print(f"  Con loops: {loop_count} ({loop_count/total*100:.1f}%)")
        print(f"  Severidad media: {avg_severity:.3f}")
        print(f"  Entropía media: {avg_entropy:.2f}")
        print(f"  Tiempo medio: {avg_elapsed:.1f}s")

    # Guardar resultados
    out_path = Path("/home/user/ai_experiment/idk50m_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()