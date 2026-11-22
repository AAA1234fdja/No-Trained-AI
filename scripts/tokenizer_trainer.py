#!/usr/bin/env python3
"""
tokenizer_trainer.py — Entrenador simulado de tokenizador (sin gradientes).

Este módulo proporciona herramientas para "entrenar" o ajustar un tokenizador
sin usar gradientes, basado en análisis estadístico de un corpus de texto:

  - Construcción de vocabulario a partir de un corpus.
  - Análisis de frecuencia de tokens (Zipf, ley de potencias).
  - Generación de reglas de tokenización basadas en frecuencias.
  - Ajuste de tokens especiales (BOS, EOS, PAD, UNK).
  - Entrenamiento simulado de BPE (Byte Pair Encoding) sin gradientes.
  - Evaluación de cobertura del vocabulario sobre un corpus.
  - Reporte de estadísticas del tokenizador en JSON/Markdown.

Nota: Esto es un simulador, no un entrenamiento real de tokenizador.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TokenStats:
    """Estadísticas de un token individual."""
    token: str
    frequency: int
    rank: int
    probability: float
    cumulative_probability: float


@dataclass
class TokenizerTrainingReport:
    """Reporte del entrenamiento simulado del tokenizador."""
    corpus_size_chars: int
    corpus_size_tokens: int
    vocab_size: int
    special_tokens: Dict[str, str]
    top_tokens: List[TokenStats] = field(default_factory=list)
    zipf_exponent: float = 0.0
    coverage_percentage: float = 0.0
    oov_rate: float = 0.0  # out-of-vocabulary rate
    bpe_merges: List[Tuple[str, str, int]] = field(default_factory=list)
    training_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "corpus_size_chars": self.corpus_size_chars,
            "corpus_size_tokens": self.corpus_size_tokens,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "top_tokens": [
                {
                    "token": t.token,
                    "frequency": t.frequency,
                    "rank": t.rank,
                    "probability": round(t.probability, 6),
                    "cumulative_probability": round(t.cumulative_probability, 6),
                }
                for t in self.top_tokens
            ],
            "zipf_exponent": round(self.zipf_exponent, 4),
            "coverage_percentage": round(self.coverage_percentage, 2),
            "oov_rate": round(self.oov_rate, 4),
            "n_bpe_merges": len(self.bpe_merges),
            "training_time_sec": round(self.training_time_sec, 4),
        }


class TokenizerTrainer:
    """
    Entrenador simulado de tokenizador.

    Analiza un corpus de texto y genera estadísticas y reglas para
    un tokenizador, sin usar gradientes ni backpropagation.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        }

    def train_from_corpus(
        self,
        corpus_path: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> TokenizerTrainingReport:
        """
        Entrena un tokenizador simulado a partir de un corpus de texto.

        Parameters
        ----------
        corpus_path : str | Path
            Ruta al archivo de texto del corpus.
        output_dir : str | Path opcional
            Directorio para guardar el vocabulario generado.

        Returns
        -------
        TokenizerTrainingReport con estadísticas del entrenamiento.
        """
        import time
        start = time.time()

        corpus_path = Path(corpus_path)
        text = corpus_path.read_text(encoding="utf-8")

        # Tokenización simple por palabras y puntuación
        tokens = self._tokenize(text)

        # Contar frecuencias
        freq_counter = Counter(tokens)
        total_tokens = len(tokens)

        # Ordenar por frecuencia descendente
        sorted_items = freq_counter.most_common()

        # Asignar ranks y probabilidades
        token_stats: List[TokenStats] = []
        cumulative = 0.0
        for rank, (token, freq) in enumerate(sorted_items, start=1):
            prob = freq / total_tokens if total_tokens > 0 else 0.0
            cumulative += prob
            token_stats.append(TokenStats(
                token=token,
                frequency=freq,
                rank=rank,
                probability=prob,
                cumulative_probability=cumulative,
            ))

        # Calcular exponente de Zipf (regresión log-log de los top tokens)
        zipf_exp = self._calculate_zipf_exponent(token_stats[:1000])

        # Calcular cobertura del vocabulario
        top_vocab = set(t.token for t in token_stats[:self.vocab_size])
        oov_count = sum(1 for t in tokens if t not in top_vocab)
        coverage = (1.0 - oov_count / total_tokens) * 100.0 if total_tokens > 0 else 0.0
        oov_rate = oov_count / total_tokens if total_tokens > 0 else 0.0

        # Entrenamiento simulado de BPE (top merges)
        bpe_merges = self._simulate_bpe_merges(tokens, n_merges=100)

        elapsed = time.time() - start

        report = TokenizerTrainingReport(
            corpus_size_chars=len(text),
            corpus_size_tokens=total_tokens,
            vocab_size=min(self.vocab_size, len(token_stats)),
            special_tokens=self.special_tokens,
            top_tokens=token_stats[:50],
            zipf_exponent=zipf_exp,
            coverage_percentage=coverage,
            oov_rate=oov_rate,
            bpe_merges=bpe_merges,
            training_time_sec=elapsed,
        )

        # Guardar vocabulario si se solicita
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            vocab = {t.token: t.rank for t in token_stats[:self.vocab_size]}
            for i, (key, val) in enumerate(self.special_tokens.items()):
                vocab[val] = self.vocab_size + i

            with open(output_dir / "vocab.json", "w") as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)

            with open(output_dir / "training_report.json", "w") as f:
                json.dump(report.to_dict(), f, indent=2)

        return report

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenización simple: separa por espacios y puntuación.
        """
        # Dividir por espacios y mantener puntuación como tokens separados
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def _calculate_zipf_exponent(
        self,
        top_stats: List[TokenStats],
    ) -> float:
        """
        Calcula el exponente de Zipf mediante regresión log-log.

        Zipf: freq(rank) ~ C / rank^s
        log(freq) = log(C) - s * log(rank)
        """
        import math

        if len(top_stats) < 2:
            return 0.0

        # Filtrar tokens con frecuencia > 0
        points = [
            (math.log(t.rank), math.log(t.frequency))
            for t in top_stats
            if t.frequency > 0 and t.rank > 0
        ]

        if len(points) < 2:
            return 0.0

        # Regresión lineal simple: y = a + b*x, b = -s
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_x2 = sum(p[0] ** 2 for p in points)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        b = (n * sum_xy - sum_x * sum_y) / denominator
        return -b  # exponente de Zipf

    def _simulate_bpe_merges(
        self,
        tokens: List[str],
        n_merges: int = 100,
    ) -> List[Tuple[str, str, int]]:
        """
        Simula el entrenamiento de BPE (Byte Pair Encoding).

        Encuentra los pares de tokens adjacentes más frecuentes.

        Returns
        -------
        Lista de (token1, token2, frecuencia) ordenada por frecuencia.
        """
        pair_counter: Counter = Counter()

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counter[pair] += 1

        top_pairs = pair_counter.most_common(n_merges)
        return [(p[0], p[1], freq) for p, freq in top_pairs]

    def evaluate_coverage(
        self,
        vocab: Dict[str, int],
        test_text: str,
    ) -> dict:
        """
        Evalúa la cobertura de un vocabulario sobre un texto de prueba.

        Returns
        -------
        dict con tasa de cobertura, tokens OOV, etc.
        """
        tokens = self._tokenize(test_text)
        total = len(tokens)
        oov = sum(1 for t in tokens if t not in vocab)

        return {
            "total_tokens": total,
            "oov_tokens": oov,
            "coverage_percentage": (1.0 - oov / total) * 100.0 if total > 0 else 0.0,
            "oov_rate": oov / total if total > 0 else 0.0,
        }

    def generate_report(
        self,
        report: TokenizerTrainingReport,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de entrenamiento en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Entrenamiento de Tokenizador")
        lines.append("")
        lines.append("## Estadísticas del Corpus")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Tamaño del corpus (caracteres) | {report.corpus_size_chars:,} |")
        lines.append(f"| Tamaño del corpus (tokens) | {report.corpus_size_tokens:,} |")
        lines.append(f"| Tamaño del vocabulario | {report.vocab_size:,} |")
        lines.append(f"| Cobertura del vocabulario | {report.coverage_percentage:.2f}% |")
        lines.append(f"| Tasa OOV (out-of-vocabulary) | {report.oov_rate:.4f} |")
        lines.append(f"| Exponente de Zipf | {report.zipf_exponent:.4f} |")
        lines.append(f"| Tiempo de entrenamiento | {report.training_time_sec:.4f}s |")
        lines.append("")

        lines.append("## Tokens Especiales")
        lines.append("")
        for key, value in report.special_tokens.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

        lines.append("## Top 20 Tokens por Frecuencia")
        lines.append("")
        lines.append("| Rang | Token | Frecuencia | Probabilidad | Prob. Acumulada |")
        lines.append("|------|-------|------------|--------------|-----------------|")
        for t in report.top_tokens[:20]:
            lines.append(
                f"| {t.rank} | `{t.token}` | {t.frequency:,} | "
                f"{t.probability:.6f} | {t.cumulative_probability:.6f} |"
            )
        lines.append("")

        lines.append("## Top 10 Merges BPE Simulados")
        lines.append("")
        lines.append("| Token 1 | Token 2 | Frecuencia |")
        lines.append("|---------|---------|------------|")
        for t1, t2, freq in report.bpe_merges[:10]:
            lines.append(f"| `{t1}` | `{t2}` | {freq:,} |")
        lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text


def quick_train_tokenizer(
    corpus_path: str | Path,
    output_dir: Optional[str | Path] = None,
    vocab_size: int = 32000,
) -> TokenizerTrainingReport:
    """
    Función de conveniencia: entrena un tokenizador simulado.

    Returns
    -------
    TokenizerTrainingReport con el resultado.
    """
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    return trainer.train_from_corpus(corpus_path, output_dir)


__all__ = [
    "TokenStats",
    "TokenizerTrainingReport",
    "TokenizerTrainer",
    "quick_train_tokenizer",
]
