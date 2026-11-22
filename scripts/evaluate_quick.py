#!/usr/bin/env python3
"""
evaluate_quick.py — Evaluación rápida del modelo 800M editado.
Prueba limitada para verificar si las ediciones de pesos surtieron efecto.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference_engine import MmapInt8LlamaModel


def quick_test(model: MmapInt8LlamaModel, name: str, prompt: str,
               max_new: int = 8, temp: float = 0.3, top_k: int = 8,
               rep_pen: float = 1.5) -> str:
    """Ejecuta una prueba rápida y devuelve el texto generado."""
    t0 = time.time()
    res = model.generate(prompt, max_new_tokens=max_new, temperature=temp,
                         top_k=top_k, repetition_penalty=rep_pen)
    elapsed = time.time() - t0
    text = res["generated_text"]
    entropy = res["first_token_entropy_nats"]
    print(f"  [{name}] {repr(prompt)} -> {repr(text)} "
          f"(H={entropy:.2f}, {elapsed:.1f}s)")
    return text


def main() -> None:
    model_dir = Path("/home/user/No-Trained-AI/model")
    print("Cargando modelo...")
    model = MmapInt8LlamaModel(model_dir)
    print("Modelo cargado.")
    print()

    results = {}

    print("=" * 50)
    print("PRUEBA 1: Python")
    print("=" * 50)
    results["def_func"] = quick_test(model, "def", "def ", max_new=10, temp=0.2, top_k=5, rep_pen=1.6)
    results["for_loop"] = quick_test(model, "for", "for i in range(", max_new=10, temp=0.2, top_k=5, rep_pen=1.6)
    results["return"] = quick_test(model, "return", "return ", max_new=10, temp=0.2, top_k=5, rep_pen=1.6)
    results["import"] = quick_test(model, "import", "import ", max_new=10, temp=0.2, top_k=5, rep_pen=1.6)

    print()
    print("=" * 50)
    print("PRUEBA 2: Análisis")
    print("=" * 50)
    results["complexity"] = quick_test(model, "complexity", "Complexity:", max_new=10, temp=0.3, top_k=8, rep_pen=1.5)
    results["analyze"] = quick_test(model, "analyze", "Analyze data:", max_new=10, temp=0.3, top_k=8, rep_pen=1.5)
    results["step"] = quick_test(model, "step", "Step 1:", max_new=10, temp=0.3, top_k=8, rep_pen=1.5)

    print()
    print("=" * 50)
    print("PRUEBA 3: Habla mínima (debería generar poco)")
    print("=" * 50)
    results["hello"] = quick_test(model, "hello", "Hello", max_new=8, temp=0.5, top_k=10, rep_pen=1.8)
    results["thanks"] = quick_test(model, "thanks", "Thank you", max_new=8, temp=0.5, top_k=10, rep_pen=1.8)

    print()
    print("=" * 50)
    print("PRUEBA 4: Inducción [A][B]...[A]->[B]")
    print("=" * 50)
    results["induction"] = quick_test(model, "induction", "alpha = 42 ; beta = 99 ; alpha =", max_new=8, temp=0.1, top_k=3, rep_pen=2.0)

    # Guardar resultados
    out_path = Path("/home/user/ai_experiment/evaluation_quick.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()