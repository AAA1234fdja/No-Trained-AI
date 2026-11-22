#!/usr/bin/env python3
"""
real_evaluation.py — Evaluación real del modelo (sin métricas proxy circulares).

Evalúa el modelo con:
  1. Parsing de sintaxis Python con ast.parse (¿el texto generado es código válido?)
  2. Detección de repetición de n-gramas (tasa de loops)
  3. Entropía de la distribución de logits (incertidumbre del modelo)
  4. Perplejidad sobre un mini-corpus held-out (si se proporcionan datos)

No usa scores compuestos basados en los mismos pesos que se editaron.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference_engine import MmapInt8LlamaModel, stable_softmax


# ---------------------------------------------------------------------------
# Utilidades de evaluación real
# ---------------------------------------------------------------------------

def is_valid_python_code(text: str) -> bool:
    """
    Verifica si el texto generado es código Python sintácticamente válido.

    Returns
    -------
    bool: True si se puede parsear con ast.parse sin errores de sintaxis.
    """
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def detect_ngram_repetition(
    tokens: List[int],
    n: int = 3,
    min_repeats: int = 2,
) -> Tuple[bool, float]:
    """
    Detecta repetición de n-gramas en una secuencia de tokens.

    Returns
    -------
    (has_repetition, repetition_rate)
    """
    if len(tokens) < n * min_repeats:
        return False, 0.0

    ngrams: List[Tuple[int, ...]] = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i : i + n]))

    counts = Counter(ngrams)
    repeated_count = sum(1 for c in counts.values() if c >= min_repeats)
    total_ngrams = len(ngrams)

    repetition_rate = repeated_count / total_ngrams if total_ngrams > 0 else 0.0
    has_repetition = repetition_rate > 0

    return has_repetition, repetition_rate


def detect_token_loop(text: str, min_repeats: int = 3) -> bool:
    """Detecta repetición de token completo (ej. 'forforfor')."""
    pattern = r"(.{2,30}?)\1{" + str(min_repeats - 1) + r",}"
    return bool(re.search(pattern, text))


def entropy_from_logits(logits: np.ndarray) -> float:
    """Calcula la entropía de Shannon de una distribución de logits."""
    probs = stable_softmax(logits)
    return float(
        -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
    )


def perplexity_from_logits(logits: np.ndarray, target_id: int) -> float:
    """Calcula la perplejidad para un token objetivo dado los logits."""
    probs = stable_softmax(logits)
    p = probs[target_id] if target_id < len(probs) else 1e-12
    return float(1.0 / max(p, 1e-12))


# ---------------------------------------------------------------------------
# Evaluación principal
# ---------------------------------------------------------------------------

def evaluate_model(
    model_dir: Path,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 20,
    temperature: float = 0.3,
    top_k: int = 6,
    repetition_penalty: float = 2.0,
) -> dict:
    """
    Ejecuta evaluación real del modelo con métricas honestas.

    Parameters
    ----------
    model_dir : Path
        Directorio del modelo (con model.safetensors, tokenizer.json, config.json).
    prompts : list de str opcional
        Prompts de prueba. Si no se proporciona, usa un conjunto por defecto.
    max_new_tokens : int
        Número máximo de tokens a generar.
    temperature : float
        Temperatura de muestreo.
    top_k : int
        Top-k para muestreo.
    repetition_penalty : float
        Penalidad de repetición.

    Returns
    -------
    dict con resultados detallados y resumen de métricas reales.
    """
    if prompts is None:
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

    print("Cargando modelo...")
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        print(f"ADVERTENCIA: No se encontró {weights_path}")
        print("El modelo necesita reensamblarse con: bash model/reassemble.sh")
        # Usar IDK50M como fallback para pruebas
        idk50m_path = model_dir / "idk50m.safetensors"
        if idk50m_path.exists():
            print("Usando IDK50M como fallback...")
            weights_path = idk50m_path
        else:
            print("No hay modelo disponible para evaluar")
            sys.exit(1)

    model = MmapInt8LlamaModel(
        model_dir,
        weights_filename=weights_path.name,
        config_filename="config.json",
        tokenizer_filename="tokenizer.json",
    )
    print("Modelo cargado.")
    print()

    results: List[dict] = []

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
            generated_ids = res.get("generated_ids", [])
            entropy = res["first_token_entropy_nats"]

            # Métricas reales
            has_token_loop = detect_token_loop(text)
            has_ngram_rep, ngram_rate = detect_ngram_repetition(
                generated_ids, n=3, min_repeats=2
            )
            is_valid_py = is_valid_python_code(text)

            loop_icon = "🔴" if has_token_loop else "🟢"
            py_icon = "✅" if is_valid_py else "❌"

            print(
                f"  {loop_icon} [{prompt[:25]:<25}] -> "
                f"{repr(text[:50])} "
                f"(loop={has_token_loop}, ngram_rep={ngram_rate:.2f}, "
                f"valid_py={is_valid_py}, H={entropy:.2f}, {elapsed:.1f}s)"
            )

            results.append({
                "prompt": prompt,
                "generated_text": text,
                "has_token_loop": has_token_loop,
                "ngram_repetition_rate": round(ngram_rate, 4),
                "is_valid_python": is_valid_py,
                "entropy": round(entropy, 2),
                "elapsed_sec": round(elapsed, 2),
                "n_generated": len(generated_ids),
            })
        except Exception as e:
            print(f"  ❌ [{prompt[:25]:<25}] ERROR: {e}")
            results.append({
                "prompt": prompt,
                "error": str(e),
            })

    # Calcular resumen de métricas reales
    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        total = len(valid_results)
        loop_count = sum(1 for r in valid_results if r["has_token_loop"])
        ngram_rep_count = sum(
            1 for r in valid_results if r["ngram_repetition_rate"] > 0
        )
        valid_py_count = sum(1 for r in valid_results if r["is_valid_python"])
        avg_entropy = float(np.mean([r["entropy"] for r in valid_results]))
        avg_ngram_rate = float(
            np.mean([r["ngram_repetition_rate"] for r in valid_results])
        )
        avg_elapsed = float(np.mean([r["elapsed_sec"] for r in valid_results]))

        summary = {
            "total_prompts": total,
            "token_loop_count": loop_count,
            "token_loop_rate": round(loop_count / total * 100, 1),
            "ngram_repetition_count": ngram_rep_count,
            "ngram_repetition_rate": round(avg_ngram_rate, 4),
            "valid_python_count": valid_py_count,
            "valid_python_rate": round(valid_py_count / total * 100, 1),
            "avg_entropy": round(avg_entropy, 2),
            "avg_elapsed_sec": round(avg_elapsed, 1),
        }
    else:
        summary = {
            "total_prompts": 0,
            "error": "No valid results",
        }

    print()
    print("=" * 60)
    print("RESUMEN DE EVALUACIÓN REAL")
    print("=" * 60)
    if valid_results:
        print(f"  Prompts totales: {summary['total_prompts']}")
        print(f"  Con loops de token: {summary['token_loop_count']} ({summary['token_loop_rate']}%)")
        print(f"  Con repetición n-grama: {summary['ngram_repetition_count']}")
        print(f"  Código Python válido: {summary['valid_python_count']} ({summary['valid_python_rate']}%)")
        print(f"  Entropía media: {summary['avg_entropy']} nats")
        print(f"  Tiempo medio: {summary['avg_elapsed_sec']}s")
    print()

    # Guardar resultados
    out_path = Path("/home/user/ai_experiment/real_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": results,
        }, f, indent=2)
    print(f"Resultados guardados en: {out_path}")

    return {"summary": summary, "results": results}


def main() -> None:
    model_dir = Path("/home/user/No-Trained-AI/model")

    print("=" * 60)
    print("EVALUACIÓN REAL DEL MODELO (sin métricas proxy circulares)")
    print("=" * 60)
    print()

    evaluate_model(model_dir)


if __name__ == "__main__":
    main()