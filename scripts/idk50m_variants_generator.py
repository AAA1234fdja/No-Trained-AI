#!/usr/bin/env python3
"""
idk50m_variants_generator.py — Genera múltiples variantes del modelo IDK50M
con diferentes distribuciones estadísticas de pesos para encontrar el
mejor caso que salga del loop de repetición.

Estrategia:
  1. Generar N variantes del modelo IDK50M con diferentes fórmulas de
     distribución de pesos (normal, uniforme, mezclada, sparse, etc.)
  2. Evaluar cada variante con inferencia real en prompts clave
  3. Identificar cuál variante produce menos loops y más código válido
  4. Analizar qué factor de la distribución hace la diferencia
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors.numpy import save_file as save_safetensors
except ImportError:  # pragma: no cover
    save_safetensors = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.inference_engine import MmapInt8LlamaModel, stable_softmax


# ---------------------------------------------------------------------------
# Configuración IDK50M
# ---------------------------------------------------------------------------

CONFIG = {
    "vocab_size": 16000,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
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

MODEL_DIR = Path("/home/user/No-Trained-AI/model")
VARIANTS_DIR = MODEL_DIR
RESULTS_DIR = Path("/home/user/ai_experiment")


# ---------------------------------------------------------------------------
# Funciones de distribución de pesos
# ---------------------------------------------------------------------------

def dist_normal(shape: Tuple[int, ...], seed: int, std: float = 0.5) -> np.ndarray:
    """Distribución normal centrada en 0."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) * std


def dist_uniform(shape: Tuple[int, ...], seed: int, scale: float = 0.5) -> np.ndarray:
    """Distribución uniforme en [-scale, scale]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, shape)


def dist_mixed_sinusoidal(
    shape: Tuple[int, ...],
    seed: int,
    std: float = 0.5,
    freq_factor: float = 10.0,
) -> np.ndarray:
    """
    Distribución mixta: ruido gaussiano + patrón senoidal estructurado.

    Esto crea correlaciones entre filas y columnas que podrían
    ayudar al modelo a aprender patrones de repetición de forma más
    estructurada.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    base = rng.standard_normal(shape) * std

    # Patrón estructurado: seno/coseno con frecuencias diferentes
    for i in range(rows):
        for j in range(cols):
            freq_i = (i + 1) * freq_factor / max(rows, 1)
            freq_j = (j + 1) * freq_factor / max(cols, 1)
            base[i, j] += 0.1 * np.sin(freq_i) * np.cos(freq_j)

    return base.astype(np.float32)


def dist_sparse(
    shape: Tuple[int, ...],
    seed: int,
    std: float = 0.5,
    sparsity: float = 0.1,
) -> np.ndarray:
    """
    Distribución con sparsity: un porcentaje de pesos se establece a cero.

    Esto puede ayudar a reducir la repetición al hacer que el modelo
    no tenga caminos "demasiado fáciles" para repetir el mismo token.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape) * std

    # Aplicar máscara de sparsity
    mask = rng.random(shape) > sparsity
    w = w * mask

    return w.astype(np.float32)


def dist_heavy_tail(
    shape: Tuple[int, ...],
    seed: int,
    std: float = 0.5,
    df: float = 3.0,
) -> np.ndarray:
    """
    Distribución t de Student (colas pesadas).

    Las colas pesadas crean algunos pesos con valores extremos que
    podrían actuar como "anclas" para romper loops de repetición.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_t(df, size=shape) * std
    return w.astype(np.float32)


def dist_antilog(
    shape: Tuple[int, ...],
    seed: int,
    std: float = 0.5,
) -> np.ndarray:
    """
    Distribución anti-log: pesos concentrados cerca de 0 con algunos
    valores extremos. Útil para crear dinámicas de atención asimétricas.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    # Base gaussiana pequeña
    base = rng.standard_normal(shape) * std * 0.3

    # Añadir algunos picos extremos
    n_spikes = max(1, int(rows * cols * 0.001))
    spike_rows = rng.integers(0, rows, n_spikes)
    spike_cols = rng.integers(0, cols, n_spikes)
    spike_vals = rng.standard_normal(n_spikes) * std * 5.0

    for r, c, v in zip(spike_rows, spike_cols, spike_vals):
        base[r, c] += v

    return base.astype(np.float32)


DISTRIBUTIONS: Dict[str, Callable] = {
    "normal_std0.5": lambda s, sd: dist_normal(s, sd, std=0.5),
    "normal_std1.0": lambda s, sd: dist_normal(s, sd, std=1.0),
    "normal_std2.0": lambda s, sd: dist_normal(s, sd, std=2.0),
    "uniform_scale0.5": lambda s, sd: dist_uniform(s, sd, scale=0.5),
    "uniform_scale1.0": lambda s, sd: dist_uniform(s, sd, scale=1.0),
    "mixed_sinusoidal": lambda s, sd: dist_mixed_sinusoidal(s, sd, std=0.5, freq_factor=10.0),
    "mixed_sinusoidal_strong": lambda s, sd: dist_mixed_sinusoidal(s, sd, std=0.5, freq_factor=50.0),
    "sparse_0.1": lambda s, sd: dist_sparse(s, sd, std=0.5, sparsity=0.1),
    "sparse_0.3": lambda s, sd: dist_sparse(s, sd, std=0.5, sparsity=0.3),
    "sparse_0.5": lambda s, sd: dist_sparse(s, sd, std=0.5, sparsity=0.5),
    "heavy_tail_df3": lambda s, sd: dist_heavy_tail(s, sd, std=0.5, df=3.0),
    "heavy_tail_df5": lambda s, sd: dist_heavy_tail(s, sd, std=0.5, df=5.0),
    "antilog": lambda s, sd: dist_antilog(s, sd, std=0.5),
}


# ---------------------------------------------------------------------------
# Construcción de variantes del modelo
# ---------------------------------------------------------------------------

def build_tensor_spec() -> List[Tuple[str, Tuple[int, ...], str]]:
    """Construye especificaciones de tensores del modelo IDK50M."""
    spec: List[Tuple[str, Tuple[int, ...], str]] = []
    spec.append(("model.embed_tokens.weight", (V, D), "I8"))
    spec.append(("model.embed_tokens.weight_scale", (V, 1), "F16"))

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

    spec.append(("model.norm.weight", (D,), "F16"))
    spec.append(("lm_head.weight", (V, D), "I8"))
    spec.append(("lm_head.weight_scale", (V, 1), "F16"))

    return spec


def build_variant(
    dist_name: str,
    dist_func: Callable,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Construye una variante del modelo IDK50M con una distribución dada.

    Parameters
    ----------
    dist_name : str
        Nombre de la distribución (para el log).
    dist_func : callable
        Función que genera pesos (shape, seed) -> np.ndarray float32.
    seed : int
        Semilla base para reproducibilidad.

    Returns
    -------
    dict nombre_tensor -> array numpy.
    """
    spec = build_tensor_spec()
    tensors: Dict[str, np.ndarray] = {}
    seed_counter = [seed]

    def next_seed() -> int:
        seed_counter[0] += 1
        return seed_counter[0]

    for name, shape, dtype in spec:
        if name.endswith("_scale"):
            base = dist_func(shape, next_seed())
            scale = np.clip(np.abs(base), 0.1, 2.0).astype(np.float16)
            tensors[name] = scale
        elif name.endswith(".weight") and len(shape) == 2:
            w_f32 = dist_func(shape, next_seed())
            from scripts.quantization_tools import quantize_int8_per_row
            w_q, w_scale = quantize_int8_per_row(w_f32)
            tensors[name] = w_q
            scale_name = name + "_scale"
            if scale_name not in tensors:
                tensors[scale_name] = w_scale
        elif name.endswith(".weight") and len(shape) == 1:
            tensors[name] = np.ones(shape, dtype=np.float16)
        else:
            tensors[name] = np.zeros(shape, dtype=np.float16)

    return tensors


def save_variant(
    tensors: Dict[str, np.ndarray],
    output_path: Path,
    dist_name: str,
    seed: int,
) -> None:
    """Guarda una variante en formato .safetensors."""
    if save_safetensors is None:
        raise ImportError("safetensors no está instalado")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "pt",
        "experiment": "AI_experiment_IDK50M_variant",
        "training_steps": "0",
        "distribution": dist_name,
        "seed": str(seed),
        "model_name": "IDK50M_variant",
    }

    save_safetensors(tensors, str(output_path), metadata=metadata)


# ---------------------------------------------------------------------------
# Evaluación de variantes
# ---------------------------------------------------------------------------

def detect_token_loop(text: str, min_repeats: int = 3) -> bool:
    """Detecta repetición de token completo."""
    import re
    pattern = r"(.{2,30}?)\1{" + str(min_repeats - 1) + r",}"
    return bool(re.search(pattern, text))


def detect_ngram_repetition(tokens: List[int], n: int = 3, min_repeats: int = 2) -> float:
    """Calcula la tasa de repetición de n-gramas."""
    from collections import Counter
    if len(tokens) < n * min_repeats:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c >= min_repeats)
    return repeated / len(ngrams) if ngrams else 0.0


def is_valid_python_code(text: str) -> bool:
    """Verifica si el texto es Python sintácticamente válido."""
    import ast
    try:
        ast.parse(text)
        return True
    except (SyntaxError, Exception):
        return False


def evaluate_variant(
    model_dir: Path,
    weights_filename: str,
    prompts: List[str],
    max_new_tokens: int = 10,
    temperature: float = 0.3,
    top_k: int = 6,
    repetition_penalty: float = 2.0,
) -> dict:
    """
    Evalúa una variante del modelo con prompts clave.

    Returns
    -------
    dict con métricas de evaluación.
    """
    model = MmapInt8LlamaModel(
        model_dir,
        weights_filename=weights_filename,
        config_filename="idk50m_config.json",
        tokenizer_filename="idk50m_tokenizer.json",
    )

    results: List[dict] = []

    for prompt in prompts:
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

            results.append({
                "prompt": prompt,
                "generated_text": text,
                "has_loop": has_loop,
                "ngram_repetition_rate": round(ngram_rate, 4),
                "is_valid_python": valid_py,
                "entropy": round(entropy, 2),
            })
        except Exception as e:
            results.append({
                "prompt": prompt,
                "error": str(e),
            })

    # Resumen
    valid = [r for r in results if "error" not in r]
    if valid:
        total = len(valid)
        loop_count = sum(1 for r in valid if r["has_loop"])
        valid_py_count = sum(1 for r in valid if r["is_valid_python"])
        avg_ngram = float(np.mean([r["ngram_repetition_rate"] for r in valid]))
        avg_entropy = float(np.mean([r["entropy"] for r in valid]))

        summary = {
            "total": total,
            "loop_count": loop_count,
            "loop_rate": round(loop_count / total * 100, 1),
            "valid_python_count": valid_py_count,
            "valid_python_rate": round(valid_py_count / total * 100, 1),
            "avg_ngram_repetition": round(avg_ngram, 4),
            "avg_entropy": round(avg_entropy, 2),
        }
    else:
        summary = {"total": 0, "error": "No valid results"}

    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# Bucle principal de generación y evaluación de variantes
# ---------------------------------------------------------------------------

def run_variant_experiment(
    n_variants_per_dist: int = 3,
    seeds: Optional[List[int]] = None,
) -> dict:
    """
    Genera y evalúa múltiples variantes del modelo IDK50M.

    Parameters
    ----------
    n_variants_per_dist : int
        Número de variantes por distribución (diferentes semillas).
    seeds : list de int opcional
        Semillas a usar. Si None, usa [42, 123, 456].

    Returns
    -------
    dict con todos los resultados.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    # VARIANTS_DIR is MODEL_DIR, no separate mkdir needed
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    all_results: Dict[str, dict] = {}
    variant_scores: List[dict] = []

    print("=" * 70)
    print("GENERACIÓN Y EVALUACIÓN DE VARIANTES IDK50M")
    print("=" * 70)
    print(f"Distribuciones: {len(DISTRIBUTIONS)}")
    print(f"Variantes por distribución: {n_variants_per_dist}")
    print(f"Semillas: {seeds}")
    print(f"Prompts de prueba: {len(PROMPTS)}")
    print()

    for dist_name, dist_func in DISTRIBUTIONS.items():
        print(f"\n{'=' * 70}")
        print(f"DISTRIBUCIÓN: {dist_name}")
        print(f"{'=' * 70}")

        dist_results: List[dict] = []

        for seed_idx, seed in enumerate(seeds[:n_variants_per_dist]):
            variant_name = f"{dist_name}_seed{seed}"
            print(f"\n  Variante {seed_idx + 1}/{n_variants_per_dist}: {variant_name}")

            # Construir variante
            tensors = build_variant(dist_name, dist_func, seed=seed)

            # Guardar variante
            variant_path = VARIANTS_DIR / f"idk50m_{variant_name}.safetensors"
            save_variant(tensors, variant_path, dist_name, seed)

            # Evaluar
            eval_result = evaluate_variant(
                MODEL_DIR,
                variant_path.name,
                PROMPTS,
            )

            summary = eval_result["summary"]
            print(f"    Loop rate: {summary.get('loop_rate', 'N/A')}%")
            print(f"    Valid Python: {summary.get('valid_python_rate', 'N/A')}%")
            print(f"    Avg n-gram rep: {summary.get('avg_ngram_repetition', 'N/A')}")
            print(f"    Avg entropy: {summary.get('avg_entropy', 'N/A')} nats")

            dist_results.append({
                "variant": variant_name,
                "distribution": dist_name,
                "seed": seed,
                "path": str(variant_path),
                "evaluation": eval_result,
            })

            # Score compuesto para ranking
            if "error" not in summary:
                score = (
                    (100 - summary["loop_rate"]) * 0.4 +
                    summary["valid_python_rate"] * 0.4 +
                    (10 - min(summary["avg_entropy"], 10)) * 0.2
                )
                variant_scores.append({
                    "variant": variant_name,
                    "distribution": dist_name,
                    "seed": seed,
                    "score": round(score, 2),
                    "loop_rate": summary["loop_rate"],
                    "valid_python_rate": summary["valid_python_rate"],
                    "avg_ngram_repetition": summary["avg_ngram_repetition"],
                    "avg_entropy": summary["avg_entropy"],
                })

        all_results[dist_name] = dist_results

    # Ranking de variantes
    variant_scores.sort(key=lambda x: -x["score"])

    print("\n" + "=" * 70)
    print("RANKING DE VARIANTES (mejor a peor)")
    print("=" * 70)
    print(f"{'Rango':<6} {'Variante':<35} {'Score':<8} {'Loop%':<7} {'Py%':<6} {'N-gram':<8} {'Entropía':<8}")
    print("-" * 70)
    for i, v in enumerate(variant_scores[:15], 1):
        print(
            f"{i:<6} {v['variant']:<35} {v['score']:<8} "
            f"{v['loop_rate']:<7} {v['valid_python_rate']:<6} "
            f"{v['avg_ngram_repetition']:<8} {v['avg_entropy']:<8}"
        )

    # Análisis por distribución
    print("\n" + "=" * 70)
    print("ANÁLISIS POR DISTRIBUCIÓN")
    print("=" * 70)

    dist_stats: Dict[str, dict] = {}
    for v in variant_scores:
        dist = v["distribution"]
        if dist not in dist_stats:
            dist_stats[dist] = {
                "variants": 0,
                "avg_score": 0.0,
                "avg_loop_rate": 0.0,
                "avg_valid_py": 0.0,
                "best_score": -1.0,
                "best_variant": "",
            }
        dist_stats[dist]["variants"] += 1
        dist_stats[dist]["avg_score"] += v["score"]
        dist_stats[dist]["avg_loop_rate"] += v["loop_rate"]
        dist_stats[dist]["avg_valid_py"] += v["valid_python_rate"]
        if v["score"] > dist_stats[dist]["best_score"]:
            dist_stats[dist]["best_score"] = v["score"]
            dist_stats[dist]["best_variant"] = v["variant"]

    for dist, stats in sorted(
        dist_stats.items(),
        key=lambda x: -x[1]["avg_score"] / max(x[1]["variants"], 1),
    ):
        n = stats["variants"]
        print(
            f"  {dist:<25} variants={n} "
            f"avg_score={stats['avg_score'] / n:.2f} "
            f"avg_loop={stats['avg_loop_rate'] / n:.1f}% "
            f"avg_py={stats['avg_valid_py'] / n:.1f}% "
            f"best={stats['best_variant']} ({stats['best_score']:.2f})"
        )

    # Guardar todos los resultados
    output = {
        "config": CONFIG,
        "distributions_tested": list(DISTRIBUTIONS.keys()),
        "seeds": seeds,
        "n_variants_per_dist": n_variants_per_dist,
        "all_results": all_results,
        "variant_scores": variant_scores,
        "distribution_stats": dist_stats,
        "best_variant": variant_scores[0] if variant_scores else None,
    }

    output_path = RESULTS_DIR / "idk50m_variants_experiment.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {output_path}")

    return output


def main() -> None:
    print("=" * 70)
    print("EXPERIMENTO DE VARIANTES IDK50M")
    print("=" * 70)
    print()

    output = run_variant_experiment(
        n_variants_per_dist=3,
        seeds=[42, 123, 456],
    )

    if output["best_variant"]:
        print("\n" + "=" * 70)
        print("MEJOR VARIANTE")
        print("=" * 70)
        best = output["best_variant"]
        print(f"  Nombre: {best['variant']}")
        print(f"  Distribución: {best['distribution']}")
        print(f"  Score: {best['score']}")
        print(f"  Loop rate: {best['loop_rate']}%")
        print(f"  Valid Python: {best['valid_python_rate']}%")
        print(f"  Ruta: {best.get('path', 'N/A')}")


if __name__ == "__main__":
    main()