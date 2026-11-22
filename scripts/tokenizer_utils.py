#!/usr/bin/env python3
"""
tokenizer_utils.py — Utilidades para el tokenizador BPE de 32,000 tokens.

Proporciona:
  - Carga y guardado de tokenizer.json (formato HuggingFace tokenizers).
  - Codificación / decodificación de texto.
  - Análisis del vocabulario: clasificación por categorías, longitud de tokens.
  - Construcción de tablas de características analíticas (usado por synthesize_weights.py).
  - Detección de tokens especiales (BOS, EOS, PAD, UNK, etc.).
  - Round-trip check: encode(decode(x)) == x.
  - Estadísticas de cobertura del vocabulario (cuántos bytes/caracteres cubiertos).
  - Generación de reports del vocabulario para depuración.

Diseñado para trabajar con el tokenizador programático construido por
build_tokenizer.py (ByteLevel BPE sin entrenamiento sobre corpus externo).
"""

from __future__ import annotations

import json
import keyword
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.processors import ByteLevel as ByteLevelProcessor
except ImportError:  # pragma: no cover - permite importar sin tokenizers instalado
    Tokenizer = None  # type: ignore


# ---------------------------------------------------------------------------
# Carga / guardado
# ---------------------------------------------------------------------------

def load_tokenizer(
    path: str | Path,
) -> "Tokenizer":
    """
    Carga un tokenizer desde archivo JSON (formato HuggingFace tokenizers).

    Raises FileNotFoundError si el archivo no existe.
    """
    if Tokenizer is None:
        raise ImportError(
            "La librería 'tokenizers' no está instalada. "
            "Ejecuta: pip install tokenizers"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el tokenizer: {path}")
    return Tokenizer.from_file(str(path))


def save_tokenizer(
    tok: "Tokenizer",
    path: str | Path,
) -> None:
    """Guarda un tokenizer en archivo JSON."""
    tok.save(str(path))


def get_vocab(
    tok: "Tokenizer",
) -> Dict[str, int]:
    """Devuelve el diccionario token -> id."""
    return tok.get_vocab()


def get_id_to_token(
    tok: "Tokenizer",
) -> Dict[int, str]:
    """Devuelve el diccionario id -> token."""
    vocab = tok.get_vocab()
    return {v: k for k, v in vocab.items()}


# ---------------------------------------------------------------------------
# Codificación / decodificación
# ---------------------------------------------------------------------------

def encode(
    tok: "Tokenizer",
    text: str,
    add_special_tokens: bool = True,
) -> List[int]:
    """Codifica texto a lista de IDs de token."""
    return tok.encode(text, add_special_tokens=add_special_tokens).ids


def decode(
    tok: "Tokenizer",
    ids: List[int],
    skip_special_tokens: bool = False,
) -> str:
    """Decodifica lista de IDs a texto."""
    return tok.decode(ids, skip_special_tokens=skip_special_tokens)


def roundtrip_check(
    tok: "Tokenizer",
    text: str,
) -> bool:
    """
    Verifica que encode(decode(encode(text))) == encode(text).

    Útil para asegurar que el tokenizador no pierde información.
    """
    ids = encode(tok, text)
    decoded = decode(tok, ids)
    re_ids = encode(tok, decoded)
    return ids == re_ids


# ---------------------------------------------------------------------------
# Análisis del vocabulario
# ---------------------------------------------------------------------------

def categorize_token(
    token: str,
) -> str:
    """
    Clasifica un token en una categoría según su contenido.

    Categorías:
        - special: tokens especiales <|...|>
        - byte: tokens de un solo byte (ByteLevel base)
        - python_kw: palabra clave de Python
        - python_builtin: builtin de Python
        - python_syntax: paréntesis, dos puntos, operadores, etc.
        - python_indent: espacios / indentación / nueva línea
        - python_number: número puro
        - python_ident: identificador (variable, función)
        - analysis: palabra de análisis / razonamiento
        - conversation: palabra conversacional
        - punct: puntuación
        - subword: resto (bigramas, sufijos, prefijos)
    """
    if token.startswith("<|") and token.endswith("|>"):
        return "special"

    clean = token.strip().replace(" ", "").replace("\n", "")

    # Tokens de un solo carácter de puntuación
    if token in {"(", ")", "[", "]", "{", "}", ":", ",", ".", ";", "!"}:
        return "punct"

    # Operadores Python
    if token in {"+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<",
                 "<=", ">=", "->", ":", "+=", "-=", "*=", "/=", "//=", "%=",
                 "**=", ":="}:
        return "python_syntax"

    # Palabras clave de Python
    py_kws = {
        "def", "class", "return", "if", "elif", "else", "for", "while",
        "try", "except", "finally", "with", "yield", "lambda", "async",
        "await", "pass", "break", "continue", "raise", "assert", "import",
        "from", "in", "is", "not", "and", "or", "True", "False", "None",
    }
    if clean in py_kws:
        return "python_kw"

    # Builtins comunes
    py_builtins = {
        "print", "len", "sum", "range", "sorted", "enumerate", "zip",
        "min", "max", "abs", "int", "float", "str", "list", "dict",
        "set", "tuple", "open", "type", "isinstance", "hasattr",
    }
    if clean in py_builtins:
        return "python_builtin"

    # Números
    if clean.isdigit():
        return "python_number"

    # Indentación / espacios / nueva línea
    if "\n" in token or token in {" ", "    ", "        ", "\t"}:
        return "python_indent"

    # Identificadores (alfabeto + guión bajo, empieza por letra o _)
    if clean and (clean[0].isalpha() or clean[0] == "_") and clean.replace("_", "").isalnum():
        return "python_ident"

    return "subword"


def vocab_statistics(
    tok: "Tokenizer",
) -> Dict[str, object]:
    """
    Calcula estadísticas completas del vocabulario.

    Returns
    -------
    dict con:
        - vocab_size: número total de tokens
        - category_counts: conteo por categoría
        - avg_token_length: longitud promedio de tokens (en caracteres)
        - special_tokens: lista de tokens especiales
        - byte_coverage: cuántos de los 256 bytes están representados
    """
    vocab = get_vocab(tok)
    id_to_tok = get_id_to_token(tok)

    category_counts: Dict[str, int] = defaultdict(int)
    special_tokens: List[str] = []
    byte_chars: Set[str] = set()
    total_len = 0

    for token, tid in vocab.items():
        cat = categorize_token(token)
        category_counts[cat] += 1
        total_len += len(token)

        if cat == "special":
            special_tokens.append(token)

        # Detectar tokens de un solo byte (ByteLevel)
        if len(token) == 1 and ord(token) < 256:
            byte_chars.add(token)

    return {
        "vocab_size": len(vocab),
        "category_counts": dict(category_counts),
        "avg_token_length": total_len / len(vocab) if vocab else 0.0,
        "special_tokens": special_tokens,
        "byte_coverage": len(byte_chars),
        "byte_coverage_pct": len(byte_chars) / 256.0 * 100.0,
    }


def print_vocab_report(
    tok: "Tokenizer",
) -> None:
    """Imprime un reporte legible del vocabulario."""
    stats = vocab_statistics(tok)
    print("=" * 60)
    print("REPORTE DEL TOKENIZADOR")
    print("=" * 60)
    print(f"Tamaño del vocabulario: {stats['vocab_size']:,}")
    print(f"Longitud promedio de token: {stats['avg_token_length']:.2f} caracteres")
    print(f"Cobertura de bytes: {stats['byte_coverage']}/256 ({stats['byte_coverage_pct']:.1f}%)")
    print()
    print("Conteo por categoría:")
    for cat, count in sorted(stats["category_counts"].items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {count:,}")
    print()
    print("Tokens especiales:")
    for t in stats["special_tokens"]:
        print(f"  {t}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Tabla de características analíticas
# ---------------------------------------------------------------------------

def build_feature_table(
    tok: "Tokenizer",
    hidden_dim: int = 1536,
    seed: int = 20261122,
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """
    Construye una tabla de características analíticas para los tokens del
    vocabulario, usada por synthesize_weights.py para generar embeddings
    sin entrenamiento.

    Cada fila del embedding contiene:
        - [0:512]   firma léxica ortogonalizada
        - [512:640] flags de categoría (uno-hot suave)
        - [640:896] estado de sintaxis Python
        - [896:1152] estado analítico / razonamiento
        - [1152:1280] intención conversacional
        - [1280:1408] buffer previo-token
        - [1408:1536] buffer de inducción

    Returns
    -------
    E : np.ndarray de forma (vocab_size, hidden_dim)
    groups : dict nombre_grupo -> lista de IDs de token
    """
    vocab = get_vocab(tok)
    id_to_tok = get_id_to_token(tok)
    V = len(vocab)
    D = hidden_dim

    rng = np.random.default_rng(seed)
    E = np.zeros((V, D), dtype=np.float32)

    # Subespacio léxico [0:512]: base ortogonal aleatoria normalizada
    lex_primary = rng.standard_normal((V, 128)).astype(np.float32)
    lex_primary /= (np.linalg.norm(lex_primary, axis=1, keepdims=True) + 1e-8)
    lex_sec = rng.standard_normal((V, 256)).astype(np.float32)
    lex_sec /= (np.linalg.norm(lex_sec, axis=1, keepdims=True) + 1e-8)

    # Proyección de caracteres en [128:256]
    char_proj = np.zeros((V, 128), dtype=np.float32)

    groups: Dict[str, List[int]] = {
        "def_kw": [], "return_kw": [], "for_kw": [], "if_kw": [], "in_kw": [],
        "py_ident_good": [], "py_param": [], "open_paren": [], "close_colon": [],
        "newline_indent": [], "py_builtins": [], "py_ops": [],
        "analysis_step1": [], "analysis_step2": [], "analysis_step3": [],
        "complexity_terms": [], "comparison_terms": [], "stats_terms": [],
        "conv_greet_reply": [], "conv_help_words": [], "punct_end": [],
    }

    # Conjuntos de palabras para clasificación
    py_kws = {
        "def", "class", "return", "if", "elif", "else", "for", "while",
        "try", "except", "finally", "with", "yield", "lambda", "async",
        "await", "pass", "break", "continue", "raise", "assert", "import",
        "from", "in", "is", "not", "and", "or", "True", "False", "None",
    }
    def_kws = {"def", "class", "async def"}
    ret_kws = {"return", "yield"}
    ctrl_kws = {"if", "elif", "else", "for", "while", "try", "except",
                "finally", "with"}
    analysis_set = {
        "analysis", "analyze", "analytical", "evaluate", "compare",
        "comparison", "complexity", "algorithm", "hypothesis", "evidence",
        "conclusion", "reason", "reasoning", "logic", "logical",
        "variance", "mean", "median", "outlier", "step", "steps",
        "because", "therefore", "thus", "hence", "however", "first",
        "second", "third", "finally", "explain", "problem", "solution",
        "time", "space", "memory", "linear", "logarithmic", "quadratic",
        "binary", "search", "sort", "data", "pattern", "trend",
        "correlation", "factor",
    }
    logic_conn_set = {
        "first", "second", "third", "finally", "because", "therefore",
        "thus", "hence", "however", "moreover", "furthermore",
        "consequently", "specifically", "if", "then", "implies", "and",
        "or", "but",
    }
    complexity_set = {
        "complexity", "time", "space", "memory", "linear", "logarithmic",
        "quadratic", "constant", "asymptotic", "algorithm", "binary",
        "search", "sort", "runtime", "latency", "fast", "slow", "optimal",
    }
    stats_set = {
        "mean", "median", "variance", "std", "average", "sum", "count",
        "ratio", "rate", "distribution", "outlier", "outliers",
        "correlation", "probability", "data",
    }
    greet_set = {
        "hello", "hi", "hey", "greetings", "welcome", "thanks", "thank",
        "please", "good", "morning", "fine", "great", "ready", "help",
        "assistant",
    }
    wh_set = {
        "what", "why", "how", "which", "who", "where", "when", "can",
        "could", "would", "explain", "describe", "analyze", "compare",
    }

    for tid_str, token in vocab.items():
        tid = int(tid_str)
        clean = token.strip().lower()
        cat = categorize_token(token)

        # Proyección de n-gramas de caracteres
        for i in range(len(token)):
            h = (ord(token[i]) * 31 + i * 7) % 128
            char_proj[tid, h] += 1.0
            if i + 1 < len(token):
                h2 = (ord(token[i]) * 131 + ord(token[i + 1]) * 17) % 128
                char_proj[tid, h2] += 0.7

        # Normalizar proyección de caracteres
        nrm = np.linalg.norm(char_proj[tid])
        if nrm > 0:
            char_proj[tid] /= nrm

        # Señal constante de presencia del token
        E[tid, 512] = 2.0  # F_BIAS

        # Clasificación por categoría
        if cat == "python_kw" or clean in py_kws:
            E[tid, 513] = 3.0  # F_PY_KW

        if clean in def_kws or token in ("def ", "class ", "async def "):
            E[tid, 514] = 4.0  # F_DEF_CLASS
            groups["def_kw"].append(tid)

        if clean in ret_kws or token in ("return ", "yield "):
            E[tid, 515] = 4.0  # F_RETURN
            groups["return_kw"].append(tid)

        if clean in ctrl_kws:
            E[tid, 516] = 3.5  # F_CTRL_FLOW
            if clean == "for":
                groups["for_kw"].append(tid)
            if clean == "if":
                groups["if_kw"].append(tid)

        if clean in ("in",):
            groups["in_kw"].append(tid)

        if cat == "python_builtin":
            E[tid, 517] = 3.0  # F_PY_BUILTIN

        if cat == "python_syntax":
            E[tid, 518] = 3.0  # F_PY_OP
            if token in ("(", " (", "()", "):"):
                E[tid, 521] = 4.0  # F_OPEN_PAREN
                groups["open_paren"].append(tid)
            if token in (")", "):", "()"):
                E[tid, 522] = 4.0  # F_CLOSE_PAREN
            if ":" in token:
                E[tid, 519] = 4.0  # F_COLON
                if token in (")", "):", "()", ":\n"):
                    groups["close_colon"].append(tid)

        if "\n" in token or token in {" ", "    ", "\n    ", "\n        "}:
            E[tid, 520] = 4.0  # F_INDENT
            if token in {"\n    ", "    "}:
                groups["newline_indent"].append(tid)

        if clean in analysis_set:
            E[tid, 523] = 3.5  # F_ANALYSIS

        if clean in logic_conn_set:
            E[tid, 524] = 3.5  # F_LOGIC_CONN

        if clean in complexity_set:
            E[tid, 525] = 3.5  # F_COMPLEXITY
            if token.startswith(" ") and len(clean) >= 4:
                groups["complexity_terms"].append(tid)

        if clean in stats_set:
            if token.startswith(" ") and len(clean) >= 4:
                groups["stats_terms"].append(tid)

        if clean in {"compare", "comparison", "versus", "however", "whereas",
                     "better", "faster", "tradeoff"}:
            if token.startswith(" "):
                groups["comparison_terms"].append(tid)

        if clean in greet_set:
            E[tid, 526] = 3.5  # F_GREETING

        if "?" in token or clean in wh_set:
            E[tid, 527] = 3.5  # F_QUESTION

        if cat == "python_ident" or (token.strip().isidentifier()
                                      and clean not in py_kws):
            E[tid, 529] = 2.0  # F_IDENT
            if token.strip() in {"n", "x", "arr", "data", "self"}:
                groups["py_param"].append(tid)
            if token.strip() in {"compute", "calculate", "solve", "process",
                                  "analyze"}:
                groups["py_ident_good"].append(tid)

        if clean.isdigit():
            E[tid, 530] = 3.0  # F_NUMBER

        if token in {".", ". ", ".\n", "!", "! ", "?", "? "}:
            E[tid, 531] = 3.0  # F_PUNCT_END
            groups["punct_end"].append(tid)

        # Palabras específicas para pasos analíticos
        if token.strip() in {"First", " Step", "Analysis", "Step"}:
            groups["analysis_step1"].append(tid)
        if token.strip() in {" because", " therefore", " thus", " since",
                              " implies"}:
            groups["analysis_step2"].append(tid)
        if token.strip() in {" conclusion", " Therefore", " summary",
                              " optimal", " efficient"}:
            groups["analysis_step3"].append(tid)

        # Palabras de conversación
        if token.strip() in {"Hello", " hello", "ready", "I", "can", "help",
                              "analyze", "Python"}:
            groups["conv_greet_reply"].append(tid)
        if token.strip() in {"I", "can", "help", "explain", "analyze", "write",
                              "Python", "code", "data"}:
            groups["conv_help_words"].append(tid)

    # Ensamblar subespacio léxico [0:512]
    E[:, 0:128] = lex_primary * 18.0
    E[:, 128:256] = char_proj * 12.0
    E[:, 256:512] = lex_sec * 16.0

    # Subespacios de estado están en ceros por defecto (se escriben por las capas)

    return E, groups


def save_feature_table(
    E: np.ndarray,
    groups: Dict[str, List[int]],
    path: str | Path,
) -> None:
    """Guarda la tabla de características en NPZ."""
    np.savez(str(path), E=E, groups=json.dumps(groups))


def load_feature_table(
    path: str | Path,
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """Carga la tabla de características desde NPZ."""
    data = np.load(str(path), allow_pickle=True)
    E = data["E"]
    groups = json.loads(data["groups"])
    return E, groups


__all__ = [
    "load_tokenizer",
    "save_tokenizer",
    "get_vocab",
    "get_id_to_token",
    "encode",
    "decode",
    "roundtrip_check",
    "categorize_token",
    "vocab_statistics",
    "print_vocab_report",
    "build_feature_table",
    "save_feature_table",
    "load_feature_table",
]
