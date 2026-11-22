#!/usr/bin/env python3
"""
create_idk50m_tokenizer.py — Crea un tokenizer simple para el modelo IDK50M.

Crea un tokenizer JSON básico con vocabulario de 16,000 tokens
incluyendo tokens especiales y tokens comunes de Python/análisis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VOCAB_SIZE = 16000

# Tokens especiales
SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "<|unk|>": 3,
}

# Tokens de Python comunes (para pruebas)
PYTHON_TOKENS = [
    "def", "return", "range", "for", "if", "import", "class", "with",
    "try", "except", "lambda", "yield", "async", "await", "print", "len",
    "list", "dict", "set", "enumerate", "zip", "map", "filter", "open",
    "read", "write", "True", "False", "None", "and", "or", "not", "in",
    "is", "as", "from", "del", "global", "nonlocal", "assert", "break",
    "continue", "pass", "raise", "while",
]

# Tokens de análisis
ANALYSIS_TOKENS = [
    "step", "complexity", "variance", "compare", "analyze", "data",
    "outlier", "mean", "median", "std", "correlation", "regression",
    "distribution", "statistic", "hypothesis", "significance", "confidence",
    "interval", "sample", "population", "bias", "error", "metric", "kpi",
    "insight", "pattern", "trend", "anomaly", "causation",
]

# Tokens de conversación (para suprimir)
CONVERSATION_TOKENS = [
    "hello", "hi", "hey", "howdy", "greetings", "welcome", "please",
    "thank", "thanks", "sorry", "excuse", "can you", "could you",
    "would you", "help me", "what is", "how do", "i think", "i believe",
    "maybe", "perhaps", "probably", "definitely", "absolutely", "awesome",
    "great", "wonderful", "amazing", "fantastic", "superb", "excellent",
]

# Palabras comunes
COMMON_WORDS = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those", "i",
    "you", "he", "she", "it", "we", "they", "what", "who", "when", "where",
    "why", "how", "all", "each", "every", "some", "any", "no", "yes", "not",
    "just", "only", "also", "very", "really", "quite", "so", "then", "now",
    "today", "tomorrow", "yesterday", "here", "there", "where", "when",
    "time", "day", "week", "month", "year", "number", "word", "letter",
    "character", "string", "text", "file", "code", "program", "function",
    "variable", "string", "integer", "float", "boolean", "list", "array",
    "dictionary", "object", "class", "method", "property", "attribute",
    "module", "package", "library", "framework", "api", "database", "sql",
    "query", "table", "row", "column", "index", "key", "value", "item",
    "element", "index", "position", "first", "last", "next", "previous",
    "current", "new", "old", "big", "small", "large", "huge", "tiny",
    "good", "bad", "better", "worse", "best", "worst", "easy", "hard",
    "simple", "complex", "difficult", "fast", "slow", "quick", "rapid",
    "high", "low", "top", "bottom", "left", "right", "up", "down",
    "start", "end", "begin", "finish", "stop", "go", "run", "walk",
    "work", "play", "sleep", "eat", "drink", "read", "write", "speak",
    "listen", "see", "hear", "feel", "think", "know", "learn", "teach",
    "study", "practice", "test", "check", "verify", "confirm", "deny",
    "accept", "reject", "agree", "disagree", "like", "love", "hate",
    "prefer", "choose", "select", "pick", "find", "search", "seek",
    "discover", "create", "make", "build", "destroy", "break", "fix",
    "solve", "problem", "solution", "answer", "question", "ask", "tell",
    "say", "tell", "show", "show", "give", "take", "get", "set", "put",
    "place", "move", "turn", "keep", "hold", "let", "make", "do", "be",
    "become", "seem", "appear", "look", "sound", "feel", "taste", "smell",
]


def build_vocab() -> dict:
    """Construye el vocabulario del tokenizer."""
    vocab: dict = {}

    # Tokens especiales primero
    for token, tid in SPECIAL_TOKENS.items():
        vocab[token] = tid

    # Tokens de Python
    tid = len(SPECIAL_TOKENS)
    for token in PYTHON_TOKENS:
        if tid < VOCAB_SIZE:
            vocab[token] = tid
            tid += 1

    # Tokens de análisis
    for token in ANALYSIS_TOKENS:
        if tid < VOCAB_SIZE:
            vocab[token] = tid
            tid += 1

    # Tokens de conversación
    for token in CONVERSATION_TOKENS:
        if tid < VOCAB_SIZE:
            vocab[token] = tid
            tid += 1

    # Palabras comunes
    for token in COMMON_WORDS:
        if tid < VOCAB_SIZE:
            vocab[token] = tid
            tid += 1

    # Rellenar con tokens genéricos hasta completar el vocabulario
    while tid < VOCAB_SIZE:
        vocab[f"<token_{tid}>"] = tid
        tid += 1

    return vocab


def main() -> None:
    output_dir = Path("/home/user/No-Trained-AI/model")
    output_path = output_dir / "idk50m_tokenizer.json"
    config_path = output_dir / "idk50m_tokenizer_config.json"

    print("=" * 60)
    print("CREACIÓN DE TOKENIZER IDK50M")
    print("=" * 60)

    vocab = build_vocab()
    print(f"Vocabulario construido: {len(vocab)} tokens")

    # Crear tokenizer.json en formato compatible con `tokenizers` library
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<|unk|>",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tokenizer_json, f, indent=2)
    print(f"Tokenizer guardado: {output_path}")

    # Config del tokenizer
    tokenizer_config = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "model_max_length": 2048,
    }
    with open(config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Config guardado: {config_path}")
    print()

    print("=" * 60)
    print("TOKENIZER IDK50M CREADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()