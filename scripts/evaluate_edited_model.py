#!/usr/bin/env python3
"""
evaluate_edited_model.py — Evaluación del modelo 800M editado.

Prueba el modelo en tres dimensiones:
  1. Bueno en Python (def, return, range, for, if, etc.)
  2. Habla mínima (menos palabras de relleno conversacional)
  3. Bueno en análisis (step, complexity, variance, compare, etc.)

Compara con resultados previos si existen experiment_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference_engine import MmapInt8LlamaModel


def evaluate_python_capability(model: MmapInt8LlamaModel) -> dict:
    """Evalúa capacidad de generación de código Python."""
    print("=" * 60)
    print("EVALUACIÓN: Capacidad en Python")
    print("=" * 60)

    prompts = [
        ("def_func", "def "),
        ("for_loop", "for i in range("),
        ("if_statement", "if "),
        ("return_stmt", "return "),
        ("import_stmt", "import "),
        ("class_def", "class "),
        ("list_comp", "[x for x in "),
        ("try_except", "try:"),
    ]

    results = {}
    for name, prompt in prompts:
        res = model.generate(prompt, max_new_tokens=15, temperature=0.3,
                             top_k=10, repetition_penalty=1.4)
        results[name] = res
        print(f"  [{name}] prompt={repr(prompt)} -> {repr(res['generated_text'])} "
              f"(H={res['first_token_entropy_nats']:.2f})")

    return results


def evaluate_analysis_capability(model: MmapInt8LlamaModel) -> dict:
    """Evalúa capacidad de análisis de datos."""
    print()
    print("=" * 60)
    print("EVALUACIÓN: Capacidad en Análisis")
    print("=" * 60)

    prompts = [
        ("complexity", "Analyze the time complexity of binary search:"),
        ("variance", "Compare variance and outliers in numerical data:"),
        ("step_by_step", "Step 1: Step 2: Step 3:"),
        ("statistics", "Calculate mean, median, and std deviation:"),
        ("correlation", "Find the correlation between variables:"),
    ]

    results = {}
    for name, prompt in prompts:
        res = model.generate(prompt, max_new_tokens=20, temperature=0.4,
                             top_k=12, repetition_penalty=1.4)
        results[name] = res
        print(f"  [{name}] -> {repr(res['generated_text'])} "
              f"(H={res['first_token_entropy_nats']:.2f})")

    return results


def evaluate_minimal_speech(model: MmapInt8LlamaModel) -> dict:
    """Evalúa que el modelo tenga habla mínima (menos conversación innecesaria)."""
    print()
    print("=" * 60)
    print("EVALUACIÓN: Habla Mínima")
    print("=" * 60)

    prompts = [
        ("greeting", "Hello! How can you help me?"),
        ("thanks", "Thank you for your assistance."),
        ("how_are_you", "How are you today?"),
        ("please_help", "Please help me with this problem."),
    ]

    results = {}
    for name, prompt in prompts:
        res = model.generate(prompt, max_new_tokens=15, temperature=0.5,
                             top_k=15, repetition_penalty=1.5)
        results[name] = res
        print(f"  [{name}] prompt={repr(prompt)} -> {repr(res['generated_text'])} "
              f"(H={res['first_token_entropy_nats']:.2f})")

    return results


def evaluate_induction(model: MmapInt8LlamaModel) -> dict:
    """Evalúa la cabeza de inducción (patrón [A][B]...[A]->[B])."""
    print()
    print("=" * 60)
    print("EVALUACIÓN: Cabeza de Inducción")
    print("=" * 60)

    prompts = [
        ("copy_alpha", "alpha = 42 ; beta = 99 ; alpha ="),
        ("copy_xyz", "x = 10 ; y = 20 ; x ="),
        ("pattern_ab", "[A][B] ... [A] ->"),
    ]

    results = {}
    for name, prompt in prompts:
        res = model.generate(prompt, max_new_tokens=10, temperature=0.2,
                             top_k=8, repetition_penalty=1.5)
        results[name] = res
        print(f"  [{name}] -> {repr(res['generated_text'])} "
              f"(H={res['first_token_entropy_nats']:.2f})")

    return results


def main() -> None:
    model_dir = Path("/home/user/No-Trained-AI/model")
    weights_path = model_dir / "model.safetensors"

    if not weights_path.exists():
        print(f"ERROR: No se encontró {weights_path}")
        sys.exit(1)

    print("Cargando modelo...")
    model = MmapInt8LlamaModel(model_dir)
    print(f"Modelo cargado: {weights_path}")
    print()

    # Ejecutar todas las evaluaciones
    py_results = evaluate_python_capability(model)
    analysis_results = evaluate_analysis_capability(model)
    speech_results = evaluate_minimal_speech(model)
    induction_results = evaluate_induction(model)

    # Resumen
    print()
    print("=" * 60)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 60)

    # Contar tokens de Python generados
    py_keywords = {"def", "return", "range", "for", "if", "import", "class",
                    "with", "try", "except", "lambda", "yield", "print", "len"}
    py_count = 0
    for name, res in py_results.items():
        text = res["generated_text"].lower()
        for kw in py_keywords:
            if kw in text:
                py_count += 1

    # Contar tokens de análisis generados
    analysis_keywords = {"step", "complexity", "variance", "compare", "analyze",
                          "data", "mean", "median", "std", "correlation",
                          "distribution", "statistic"}
    analysis_count = 0
    for name, res in analysis_results.items():
        text = res["generated_text"].lower()
        for kw in analysis_keywords:
            if kw in text:
                analysis_count += 1

    # Contar palabras de conversación (deberían ser mínimas)
    conv_keywords = {"hello", "hi", "hey", "thank", "please", "sorry",
                      "welcome", "awesome", "great", "wonderful", "amazing"}
    conv_count = 0
    for name, res in speech_results.items():
        text = res["generated_text"].lower()
        for kw in conv_keywords:
            if kw in text:
                conv_count += 1

    print(f"Tokens de Python detectados: {py_count}")
    print(f"Tokens de análisis detectados: {analysis_count}")
    print(f"Palabras de conversación detectadas: {conv_count} (menor = mejor)")
    print()

    # Evaluación subjetiva
    print(">>> EVALUACIÓN SUBJETIVA:")
    if py_count >= 3:
        print("  [PYTHON] ✓ Bueno - genera palabras clave de Python")
    else:
        print("  [PYTHON] ✗ Necesita mejora - pocas palabras clave de Python")

    if analysis_count >= 2:
        print("  [ANÁLISIS] ✓ Bueno - genera términos de análisis")
    else:
        print("  [ANÁLISIS] ✗ Necesita mejora - pocos términos de análisis")

    if conv_count <= 1:
        print("  [HABLA MÍNIMA] ✓ Bueno - poca conversación innecesaria")
    else:
        print("  [HABLA MÍNIMA] ✗ Necesita mejora - todavía genera conversación")

    # Guardar resultados
    results = {
        "python": {k: v["generated_text"] for k, v in py_results.items()},
        "analysis": {k: v["generated_text"] for k, v in analysis_results.items()},
        "minimal_speech": {k: v["generated_text"] for k, v in speech_results.items()},
        "induction": {k: v["generated_text"] for k, v in induction_results.items()},
        "metrics": {
            "python_keywords_count": py_count,
            "analysis_keywords_count": analysis_count,
            "conversation_keywords_count": conv_count,
        },
    }

    out_path = Path("/home/user/ai_experiment/evaluation_edited.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()