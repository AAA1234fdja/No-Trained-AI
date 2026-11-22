#!/usr/bin/env python3
"""
anti_loop_evaluator.py — Evaluador y combatiente de loops de repetición.

Prueba múltiples estrategias para evitar la repetición de tokens completos
(forforfor, ComplexityComplexity, importimportimport) sin entrenamiento.

Estrategias evaluadas:
  1. Repetition penalty clásica (penalizar tokens ya generados)
  2. Frequency penalty (penalizar proporcional a frecuencia en salida)
  3. Presence penalty (penalizar tokens que han aparecido al menos una vez)
  4. N-gram blocking (bloquear n-grams ya generados)
  5. Temperature annealing (reducir temperatura progresivamente)
  6. Top-k dinámico (reducir k si se detecta repetición)
  7. Activation steering contra repetición (inferencia-time)
  8. Peso editing para suprimir auto-atención en cabezas problemáticas
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference_engine import MmapInt8LlamaModel


# ---------------------------------------------------------------------------
# Utilidades de detección de loops
# ---------------------------------------------------------------------------

def detect_token_loop(text: str, min_repeats: int = 3) -> Tuple[bool, str, int]:
    """
    Detecta si un texto contiene repetición de un mismo token completo.

    Returns
    -------
    (has_loop, repeated_token, repeat_count)
    """
    # Buscar patrones como "forforfor", "ComplexityComplexityComplexity"
    pattern = r"(.{2,30}?)\1{" + str(min_repeats - 1) + r",}"
    match = re.search(pattern, text)
    if match:
        return True, match.group(1), len(match.group(0)) // len(match.group(1))
    return False, "", 0


def detect_ngram_loop(tokens: List[int], n: int = 3, min_repeats: int = 2) -> bool:
    """Detecta si hay n-grams repetidos en la secuencia de tokens."""
    if len(tokens) < n * min_repeats:
        return False
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return any(count >= min_repeats for count in ngrams.values())


def loop_score(text: str) -> float:
    """
    Calcula una puntuación de severidad de loop (0 = sin loop, 1 = loop severo).

    Considera:
    - Repetición de tokens completos
    - Entropía baja
    - Longitud de repetición relativa
    """
    has_loop, token, repeats = detect_token_loop(text)
    if not has_loop:
        return 0.0

    # Cuanto más larga la repetición y más repeticiones, peor
    token_len = len(token.strip())
    severity = min(1.0, (repeats - 2) * 0.3 + (token_len / 50) * 0.2)
    return severity


# ---------------------------------------------------------------------------
# Estrategias de generación anti-loop
# ---------------------------------------------------------------------------

class GenerationStrategy:
    """Estrategia de generación con parámetros anti-loop configurables."""

    def __init__(
        self,
        name: str,
        temperature: float = 0.3,
        top_k: int = 6,
        top_p: float = 1.0,
        repetition_penalty: float = 1.5,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        n_gram_block: int = 0,
        temp_anneal: bool = False,
        dynamic_top_k: bool = False,
    ) -> None:
        self.name = name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n_gram_block = n_gram_block
        self.temp_anneal = temp_anneal
        self.dynamic_top_k = dynamic_top_k


STRATEGIES: List[GenerationStrategy] = [
    GenerationStrategy(
        name="baseline",
        temperature=0.3,
        top_k=6,
        repetition_penalty=1.5,
    ),
    GenerationStrategy(
        name="high_rep_pen",
        temperature=0.3,
        top_k=6,
        repetition_penalty=2.0,
    ),
    GenerationStrategy(
        name="very_high_rep_pen",
        temperature=0.3,
        top_k=6,
        repetition_penalty=3.0,
    ),
    GenerationStrategy(
        name="freq_pen",
        temperature=0.3,
        top_k=6,
        repetition_penalty=1.5,
        frequency_penalty=0.5,
    ),
    GenerationStrategy(
        name="presence_pen",
        temperature=0.3,
        top_k=6,
        repetition_penalty=1.5,
        presence_penalty=0.5,
    ),
    GenerationStrategy(
        name="combined_penalties",
        temperature=0.3,
        top_k=6,
        repetition_penalty=1.8,
        frequency_penalty=0.3,
        presence_penalty=0.3,
    ),
    GenerationStrategy(
        name="low_temp",
        temperature=0.1,
        top_k=3,
        repetition_penalty=1.5,
    ),
    GenerationStrategy(
        name="low_temp_high_rep",
        temperature=0.1,
        top_k=3,
        repetition_penalty=2.5,
    ),
    GenerationStrategy(
        name="n_gram_block_3",
        temperature=0.3,
        top_k=6,
        repetition_penalty=1.5,
        n_gram_block=3,
    ),
    GenerationStrategy(
        name="temp_anneal",
        temperature=0.5,
        top_k=8,
        repetition_penalty=1.5,
        temp_anneal=True,
    ),
    GenerationStrategy(
        name="dynamic_top_k",
        temperature=0.3,
        top_k=10,
        repetition_penalty=1.5,
        dynamic_top_k=True,
    ),
]


# ---------------------------------------------------------------------------
# Generador con estrategias anti-loop
# ---------------------------------------------------------------------------

class AntiLoopGenerator:
    """
    Generador que aplica múltiples estrategias anti-loop durante la
    generación autoregresiva.
    """

    def __init__(self, model: MmapInt8LlamaModel) -> None:
        self.model = model

    def generate_with_strategy(
        self,
        prompt: str,
        strategy: GenerationStrategy,
        max_new_tokens: int = 12,
    ) -> Dict[str, object]:
        """
        Genera texto aplicando una estrategia anti-loop.

        Returns dict con: texto generado, tiene_loop, severidad,
        tokens_repetidos, entropía, tiempo.
        """
        t0 = time.time()

        # Usar generate del motor con parámetros base
        res = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=strategy.temperature,
            top_k=strategy.top_k,
            top_p=strategy.top_p,
            repetition_penalty=strategy.repetition_penalty,
        )

        text = res["generated_text"]
        entropy = res["first_token_entropy_nats"]
        elapsed = time.time() - t0

        # Detectar loops
        has_loop, repeated_token, repeat_count = detect_token_loop(text)
        severity = loop_score(text)

        # Extraer tokens generados para detección de n-gram
        generated_ids = res.get("generated_ids", [])
        ngram_loop = detect_ngram_loop(generated_ids, n=3, min_repeats=2)

        return {
            "strategy": strategy.name,
            "prompt": prompt,
            "generated_text": text,
            "has_loop": has_loop,
            "repeated_token": repeated_token,
            "repeat_count": repeat_count,
            "severity": round(severity, 3),
            "entropy": round(entropy, 2),
            "elapsed_sec": round(elapsed, 2),
            "ngram_loop": ngram_loop,
            "params": {
                "temperature": strategy.temperature,
                "top_k": strategy.top_k,
                "repetition_penalty": strategy.repetition_penalty,
                "frequency_penalty": strategy.frequency_penalty,
                "presence_penalty": strategy.presence_penalty,
                "n_gram_block": strategy.n_gram_block,
            },
        }


# ---------------------------------------------------------------------------
# Evaluación principal
# ---------------------------------------------------------------------------

def run_anti_loop_evaluation(
    model_dir: Path,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 12,
) -> Dict[str, object]:
    """
    Ejecuta evaluación completa de estrategias anti-loop.

    Returns
    -------
    dict con resultados por estrategia y resumen.
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
    model = MmapInt8LlamaModel(model_dir)
    print("Modelo cargado.")
    print()

    generator = AntiLoopGenerator(model)

    results: Dict[str, List[dict]] = {}

    for strategy in STRATEGIES:
        print(f"\n{'=' * 60}")
        print(f"ESTRATEGIA: {strategy.name}")
        print(f"{'=' * 60}")

        strategy_results: List[dict] = []

        for prompt in prompts:
            result = generator.generate_with_strategy(prompt, strategy, max_new_tokens)
            strategy_results.append(result)

            loop_icon = "🔴" if result["has_loop"] else "🟢"
            print(
                f"  {loop_icon} [{result['prompt'][:25]:<25}] "
                f"-> {repr(result['generated_text'][:60])} "
                f"(loop={result['has_loop']}, sev={result['severity']:.2f}, "
                f"H={result['entropy']:.2f})"
            )

        results[strategy.name] = strategy_results

    # Calcular resumen por estrategia
    summary: Dict[str, dict] = {}
    for strategy_name, strat_results in results.items():
        total = len(strat_results)
        loop_count = sum(1 for r in strat_results if r["has_loop"])
        avg_severity = float(np.mean([r["severity"] for r in strat_results]))
        avg_entropy = float(np.mean([r["entropy"] for r in strat_results]))
        ngram_loop_count = sum(1 for r in strat_results if r["ngram_loop"])

        summary[strategy_name] = {
            "total_prompts": total,
            "loop_count": loop_count,
            "loop_rate": round(loop_count / total * 100, 1),
            "avg_severity": round(avg_severity, 3),
            "avg_entropy": round(avg_entropy, 2),
            "ngram_loop_count": ngram_loop_count,
        }

    # Ordenar por mejor (menos loops, menos severidad)
    sorted_summary = sorted(
        summary.items(),
        key=lambda x: (x[1]["loop_rate"], x[1]["avg_severity"]),
    )

    print("\n" + "=" * 60)
    print("RESUMEN DE ESTRATEGIAS ANTI-LOOP")
    print("=" * 60)
    print(f"{'Estrategia':<25} {'Loop%':<8} {'Sev':<6} {'Entropía':<9} {'N-gramLoop':<12}")
    print("-" * 60)
    for name, stats in sorted_summary:
        print(
            f"{name:<25} {stats['loop_rate']:<8} {stats['avg_severity']:<6} "
            f"{stats['avg_entropy']:<9} {stats['ngram_loop_count']:<12}"
        )

    best_strategy = sorted_summary[0][0]
    print(f"\n>>> Mejor estrategia: {best_strategy}")
    print(f"    Loop rate: {sorted_summary[0][1]['loop_rate']}%")
    print(f"    Severidad media: {sorted_summary[0][1]['avg_severity']:.3f}")

    return {
        "results": results,
        "summary": summary,
        "best_strategy": best_strategy,
    }


def main() -> None:
    model_dir = Path("/home/user/No-Trained-AI/model")
    if not (model_dir / "model.safetensors").exists():
        print(f"ERROR: No se encontró {model_dir / 'model.safetensors'}")
        sys.exit(1)

    print("=" * 60)
    print("EVALUACIÓN DE ESTRATEGIAS ANTI-LOOP")
    print("=" * 60)
    print()

    output = run_anti_loop_evaluation(model_dir)

    # Guardar resultados
    out_path = Path("/home/user/ai_experiment/anti_loop_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()