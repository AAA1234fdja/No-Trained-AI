#!/usr/bin/env python3
"""
idk50m_evaluate_variants.py — Evalúa variantes IDK50M existentes de forma incremental.

Guarda resultados después de cada variante para poder reanudar si se interrumpe.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.inference_engine import MmapInt8LlamaModel, stable_softmax


MODEL_DIR = Path("/home/user/No-Trained-AI/model")
RESULTS_DIR = Path("/home/user/ai_experiment")
RESULTS_PATH = RESULTS_DIR / "idk50m_variants_evaluation.json"

PROMPTS = [
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


def detect_token_loop(text: str, min_repeats: int = 3) -> bool:
    pattern = r"(.{2,30}?)\1{" + str(min_repeats - 1) + r",}"
    return bool(re.search(pattern, text))


def detect_ngram_repetition(tokens: List[int], n: int = 3, min_repeats: int = 2) -> float:
    if len(tokens) < n * min_repeats:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c >= min_repeats)
    return repeated / len(ngrams) if ngrams else 0.0


def is_valid_python_code(text: str) -> bool:
    import ast
    try:
        ast.parse(text)
        return True
    except (SyntaxError, Exception):
        return False


def evaluate_single_variant(
    model: MmapInt8LlamaModel,
    variant_name: str,
    weights_filename: str,
    max_new_tokens: int = 10,
    temperature: float = 0.3,
    top_k: int = 6,
    repetition_penalty: float = 2.0,
) -> dict:
    """Evalúa una sola variante y devuelve sus resultados."""
    result = {
        "variant": variant_name,
        "weights": weights_filename,
        "results": [],
        "summary": {},
    }

    for prompt in PROMPTS:
        try:
            res = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            text = res["generated_text"]
            generated_ids = res.get("generated_ids", [])
            entropy = res["first_token_entropy_nats"]

            has_loop = detect_token_loop(text)
            ngram_rate = detect_ngram_repetition(generated_ids, n=3, min_repeats=2)
            valid_py = is_valid_python_code(text)

            result["results"].append({
                "prompt": prompt,
                "generated_text": text,
                "has_loop": has_loop,
                "ngram_repetition_rate": round(ngram_rate, 4),
                "is_valid_python": valid_py,
                "entropy": round(entropy, 2),
            })
        except Exception as e:
            result["results"].append({
                "prompt": prompt,
                "error": str(e),
            })

    # Resumen
    valid = [r for r in result["results"] if "error" not in r]
    if valid:
        total = len(valid)
        loop_count = sum(1 for r in valid if r["has_loop"])
        valid_py_count = sum(1 for r in valid if r["is_valid_python"])
        avg_ngram = float(np.mean([r["ngram_repetition_rate"] for r in valid]))
        avg_entropy = float(np.mean([r["entropy"] for r in valid]))

        result["summary"] = {
            "total": total,
            "loop_count": loop_count,
            "loop_rate": round(loop_count / total * 100, 1),
            "valid_python_count": valid_py_count,
            "valid_python_rate": round(valid_py_count / total * 100, 1),
            "avg_ngram_repetition": round(avg_ngram, 4),
            "avg_entropy": round(avg_entropy, 2),
        }
    else:
        result["summary"] = {"total": 0, "error": "No valid results"}

    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar resultados existentes si hay
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r") as f:
            all_results = json.load(f)
        print(f"Cargados {len(all_results)} resultados previos")
    else:
        all_results = {}

    # Encontrar variantes no evaluadas
    variant_files = sorted(MODEL_DIR.glob("idk50m_*.safetensors"))
    print(f"Encontradas {len(variant_files)} variantes de modelo")

    already_done = set(all_results.keys())
    to_evaluate = [f for f in variant_files if f.name not in already_done]

    print(f"Ya evaluadas: {len(already_done)}")
    print(f"Pendientes: {len(to_evaluate)}")

    if not to_evaluate:
        print("Todas las variantes ya están evaluadas.")
        # Mostrar ranking
        show_ranking(all_results)
        return

    # Cargar modelo una vez (se reusa para todas las variantes)
    # Nota: MmapInt8LlamaModel carga un modelo específico, así que necesitamos
    # recrearlo para cada variante. Para ahorrar tiempo, cargamos cada variante
    # individualmente.

    for i, variant_path in enumerate(to_evaluate):
        variant_name = variant_path.name
        print(f"\n[{i + 1}/{len(to_evaluate)}] Evaluando {variant_name}...")

        try:
            model = MmapInt8LlamaModel(
                MODEL_DIR,
                weights_filename=variant_name,
                config_filename="idk50m_config.json",
                tokenizer_filename="idk50m_tokenizer.json",
            )

            result = evaluate_single_variant(model, variant_name, variant_name)
            all_results[variant_name] = result

            # Guardar después de cada variante
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

            summary = result["summary"]
            if "error" not in summary:
                print(f"  Loop rate: {summary['loop_rate']}% | "
                      f"Valid Python: {summary['valid_python_rate']}% | "
                      f"Avg n-gram: {summary['avg_ngram_repetition']} | "
                      f"Avg entropy: {summary['avg_entropy']} nats")
            else:
                print(f"  ERROR: {summary.get('error', 'unknown')}")

        except Exception as e:
            print(f"  ERROR cargando/evaluando: {e}")
            all_results[variant_name] = {"error": str(e)}
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("RANKING FINAL")
    print("=" * 70)
    show_ranking(all_results)


def show_ranking(all_results: dict) -> None:
    """Muestra el ranking de variantes por score compuesto."""
    scored: List[dict] = []

    for name, result in all_results.items():
        if "error" in result or "summary" not in result:
            continue
        summary = result["summary"]
        if "error" in summary:
            continue

        score = (
            (100 - summary["loop_rate"]) * 0.4 +
            summary["valid_python_rate"] * 0.4 +
            (10 - min(summary["avg_entropy"], 10)) * 0.2
        )

        scored.append({
            "variant": name,
            "score": round(score, 2),
            "loop_rate": summary["loop_rate"],
            "valid_python_rate": summary["valid_python_rate"],
            "avg_ngram_repetition": summary["avg_ngram_repetition"],
            "avg_entropy": summary["avg_entropy"],
            "distribution": name.rsplit("_seed", 1)[0] if "_seed" in name else name,
            "seed": name.rsplit("_seed", 1)[-1] if "_seed" in name else "unknown",
        })

    scored.sort(key=lambda x: -x["score"])

    print(f"{'Rango':<6} {'Variante':<40} {'Score':<8} {'Loop%':<7} {'Py%':<6} {'N-gram':<8} {'Entropía':<8} {'Dist':<20}")
    print("-" * 110)
    for i, v in enumerate(scored[:20], 1):
        print(
            f"{i:<6} {v['variant']:<40} {v['score']:<8} "
            f"{v['loop_rate']:<7} {v['valid_python_rate']:<6} "
            f"{v['avg_ngram_repetition']:<8} {v['avg_entropy']:<8} "
            f"{v['distribution']:<20}"
        )

    if scored:
        best = scored[0]
        print(f"\n🏆 MEJOR VARIANTE: {best['variant']}")
        print(f"   Distribución: {best['distribution']}")
        print(f"   Seed: {best['seed']}")
        print(f"   Score: {best['score']}")
        print(f"   Loop rate: {best['loop_rate']}%")
        print(f"   Valid Python: {best['valid_python_rate']}%")


if __name__ == "__main__":
    main()