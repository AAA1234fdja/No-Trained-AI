#!/usr/bin/env python3
"""
prompt_processor.py — Procesamiento de prompts para el modelo 800M.

Proporciona utilidades para:
  - Construir prompts con plantillas (sistema, usuario, asistente).
  - Formatear prompts de Python (inyección de código, autocompletado).
  - Formatear prompts de análisis (pregunta + contexto de datos).
  - Formatear prompts de conversación (historial de turnos).
  - Sanitización de prompts (eliminar caracteres de control, truncar).
  - Extracción de bloques de código de la generación.
  - Conteo de tokens y estimación de longitud.
  - Búsqueda de prompts similares (por solapamiento de tokens).

Todo está diseñado para usarse con el tokenizador del modelo y el motor
de inferencia.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Plantillas de prompt
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATES: Dict[str, str] = {
    "python_coder": (
        "You are a Python programming assistant. "
        "Write clean, correct, and efficient Python code. "
        "Include comments explaining complex logic. "
        "Do not generate incomplete code."
    ),
    "data_analyst": (
        "You are a data analysis assistant. "
        "Help users understand datasets, compute statistics, "
        "and interpret results. Use clear, structured explanations."
    ),
    "conversational": (
        "You are a helpful, polite, and concise conversational assistant. "
        "Answer questions directly and ask clarifying questions when needed."
    ),
    "general": (
        "You are a helpful AI assistant. "
        "Answer questions accurately and concisely."
    ),
}

USER_TEMPLATES: Dict[str, str] = {
    "python_complete": (
        "Complete the following Python code:\n\n{code}\n"
    ),
    "python_explain": (
        "Explain what the following Python code does:\n\n{code}\n"
    ),
    "python_debug": (
        "The following Python code has an error. Identify and fix it:\n\n{code}\n"
    ),
    "analysis_question": (
        "Analyze the following question and provide a structured answer:\n\n{question}\n"
    ),
    "data_summary": (
        "Summarize the following dataset description:\n\n{data_desc}\n"
    ),
    "chat": (
        "{user_message}\n"
    ),
}


@dataclass
class Prompt:
    """
    Representa un prompt procesado listo para ser enviado al modelo.

    Attributes
    ----------
    system : str
        Instrucción del sistema (opcional).
    user : str
        Mensaje del usuario.
    assistant_prefix : str
        Prefijo de respuesta del asistente (opcional, para few-shot).
    full_text : str
        Texto completo concatenado listo para tokenizar.
    metadata : dict
        Metadatos adicionales (tipo de prompt, longitud, etc.).
    """
    system: str = ""
    user: str = ""
    assistant_prefix: str = ""
    full_text: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_string(self) -> str:
        """Devuelve el texto completo del prompt."""
        return self.full_text

    def token_count_estimate(self) -> int:
        """
        Estima el número de tokens sin tokenizador real:
        aproximadamente 1 token por 4 caracteres (inglés).
        """
        return max(1, len(self.full_text) // 4)


def build_python_completion_prompt(
    code_snippet: str,
    system: Optional[str] = None,
) -> Prompt:
    """
    Construye un prompt de autocompletado de código Python.

    Parameters
    ----------
    code_snippet : str
        Código parcial a completar.
    system : str opcional
        Instrucción del sistema (usa python_coder por defecto).

    Returns
    -------
    Prompt listo para inferencia.
    """
    if system is None:
        system = SYSTEM_TEMPLATES["python_coder"]

    user = USER_TEMPLATES["python_complete"].format(code=code_snippet)
    full_text = f"{system}\n\n{user}\nAssistant:"

    return Prompt(
        system=system,
        user=user,
        assistant_prefix="",
        full_text=full_text,
        metadata={
            "type": "python_completion",
            "code_length": len(code_snippet),
        },
    )


def build_python_explain_prompt(
    code_snippet: str,
    system: Optional[str] = None,
) -> Prompt:
    """Construye un prompt de explicación de código Python."""
    if system is None:
        system = SYSTEM_TEMPLATES["python_coder"]

    user = USER_TEMPLATES["python_explain"].format(code=code_snippet)
    full_text = f"{system}\n\n{user}\nAssistant:"

    return Prompt(
        system=system,
        user=user,
        full_text=full_text,
        metadata={"type": "python_explain", "code_length": len(code_snippet)},
    )


def build_analysis_prompt(
    question: str,
    context: Optional[str] = None,
    system: Optional[str] = None,
) -> Prompt:
    """
    Construye un prompt de análisis de datos o razonamiento.

    Parameters
    ----------
    question : str
        Pregunta de análisis.
    context : str opcional
        Contexto adicional (descripción de datos, etc.).
    system : str opcional
    """
    if system is None:
        system = SYSTEM_TEMPLATES["data_analyst"]

    if context:
        user = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n"
        )
    else:
        user = USER_TEMPLATES["analysis_question"].format(question=question)

    full_text = f"{system}\n\n{user}\nAssistant:"

    return Prompt(
        system=system,
        user=user,
        full_text=full_text,
        metadata={
            "type": "analysis",
            "question_length": len(question),
            "has_context": context is not None,
        },
    )


def build_conversation_prompt(
    user_message: str,
    history: Optional[List[Tuple[str, str]]] = None,
    system: Optional[str] = None,
) -> Prompt:
    """
    Construye un prompt conversacional con historial opcional.

    history: lista de tuplas (rol, mensaje) donde rol es "user" o "assistant".
    """
    if system is None:
        system = SYSTEM_TEMPLATES["conversational"]

    parts: List[str] = [system]

    if history:
        for role, message in history:
            parts.append(f"{role.capitalize()}: {message}")

    parts.append(f"User: {user_message}")
    parts.append("Assistant:")

    full_text = "\n".join(parts)

    return Prompt(
        system=system,
        user=user_message,
        full_text=full_text,
        metadata={
            "type": "conversation",
            "history_turns": len(history) if history else 0,
        },
    )


def build_general_prompt(
    query: str,
    system: Optional[str] = None,
) -> Prompt:
    """Construye un prompt de propósito general."""
    if system is None:
        system = SYSTEM_TEMPLATES["general"]
    full_text = f"{system}\n\n{query}\nAssistant:"
    return Prompt(
        system=system,
        user=query,
        full_text=full_text,
        metadata={"type": "general"},
    )


# ---------------------------------------------------------------------------
# Sanitización y utilidades
# ---------------------------------------------------------------------------

def sanitize_prompt(
    text: str,
    max_length: int = 4000,
) -> str:
    """
    Limpia un texto de entrada para usarlo como prompt:
    - Elimina caracteres de control (excepto tab, nueva línea).
    - Colapsa múltiples espacios en blanco.
    - Trunca a max_length caracteres si excede el límite.
    """
    # Eliminar caracteres de control excepto \t \n \r
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Colapsar espacios múltiples (pero preservar nuevas líneas individuales)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit("\n", 1)[0] + "\n..."

    return cleaned


def extract_code_blocks(
    text: str,
) -> List[str]:
    """
    Extrae bloques de código Python de un texto generado.

    Busca bloques delimitados por ```python ... ``` o ``` ... ```.
    Si no hay bloques delimitados, devuelve una lista vacía.
    """
    pattern = r"```(?:\w*\n)?(.*?)```"
    blocks = re.findall(pattern, text, flags=re.DOTALL)
    return [b.strip() for b in blocks]


def extract_inline_code(
    text: str,
) -> List[str]:
    """
    Extrae código en línea (entre backticks simples).
    """
    pattern = r"`([^`]+)`"
    return re.findall(pattern, text)


def count_tokens_approx(
    text: str,
    chars_per_token: float = 4.0,
) -> int:
    """
    Estimación aproximada de tokens basada en caracteres.

    Nota: esto es una aproximación. Para conteo exacto usar el tokenizador
    real del modelo.
    """
    return max(1, int(len(text) / chars_per_token))


def find_similar_prompts(
    query: str,
    candidates: List[str],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Encuentra prompts similares al query basándose en solapamiento de
    palabras (similitud de Jaccard simplificada).

    Returns lista de (prompt, puntuación) ordenada por puntuación descendente.
    """
    query_words = set(query.lower().split())
    if not query_words:
        return []

    scored: List[Tuple[str, float]] = []
    for cand in candidates:
        cand_words = set(cand.lower().split())
        if not cand_words:
            continue
        intersection = len(query_words & cand_words)
        union = len(query_words | cand_words)
        jaccard = intersection / union if union > 0 else 0.0
        scored.append((cand, jaccard))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


def split_into_turns(
    text: str,
) -> List[Tuple[str, str]]:
    """
    Divide un texto de conversación en turnos (rol, mensaje).

    Busca patrones como "User: ..." y "Assistant: ...".
    """
    turns: List[Tuple[str, str]] = []
    # Dividir por marcadores de rol
    parts = re.split(r"(?:^|\n)(User|Assistant):\s*", text, flags=re.MULTILINE)

    current_role: Optional[str] = None
    current_msg: List[str] = []

    for part in parts:
        part = part.strip()
        if part in ("User", "Assistant"):
            if current_role is not None and current_msg:
                turns.append((current_role, "\n".join(current_msg).strip()))
            current_role = part.lower()
            current_msg = []
        elif part:
            current_msg.append(part)

    if current_role is not None and current_msg:
        turns.append((current_role, "\n".join(current_msg).strip()))

    return turns


# ---------------------------------------------------------------------------
# Plantillas de few-shot para Python
# ---------------------------------------------------------------------------

PYTHON_FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = [
    (
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\n# Example usage:\nprint(factorial(5))  # 120",
    ),
    (
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(is_prime(17))  # True",
    ),
]


def build_few_shot_python_prompt(
    code_snippet: str,
    examples: Optional[List[Tuple[str, str]]] = None,
) -> Prompt:
    """
    Construye un prompt de Python con ejemplos few-shot para mejorar
    la calidad de la completación.
    """
    if examples is None:
        examples = PYTHON_FEW_SHOT_EXAMPLES

    parts: List[str] = [SYSTEM_TEMPLATES["python_coder"]]

    for incomplete, complete in examples:
        parts.append(f"Input:\n{incomplete}")
        parts.append(f"Output:\n{complete}")

    parts.append(f"Input:\n{code_snippet}")
    parts.append("Output:")

    full_text = "\n\n".join(parts)

    return Prompt(
        system=SYSTEM_TEMPLATES["python_coder"],
        user=code_snippet,
        full_text=full_text,
        metadata={
            "type": "python_few_shot",
            "num_examples": len(examples),
        },
    )


# ---------------------------------------------------------------------------
# Estadísticas de prompts
# ---------------------------------------------------------------------------

def prompt_statistics(
    prompts: List[Prompt],
) -> Dict[str, object]:
    """
    Calcula estadísticas de una lista de prompts.

    Returns
    -------
    dict con conteo total, longitud promedio, tipos de prompt, etc.
    """
    if not prompts:
        return {"total": 0}

    lengths = [len(p.full_text) for p in prompts]
    token_estimates = [p.token_count_estimate() for p in prompts]
    types = [p.metadata.get("type", "unknown") for p in prompts]
    type_counts: Dict[str, int] = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total": len(prompts),
        "avg_chars": statistics_mean(lengths),
        "max_chars": max(lengths),
        "min_chars": min(lengths),
        "avg_tokens_est": statistics_mean(token_estimates),
        "type_distribution": type_counts,
    }


def statistics_mean(values: List[float]) -> float:
    """Media simple sin importar statistics (evita dependencia)."""
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "Prompt",
    "SYSTEM_TEMPLATES",
    "USER_TEMPLATES",
    "build_python_completion_prompt",
    "build_python_explain_prompt",
    "build_analysis_prompt",
    "build_conversation_prompt",
    "build_general_prompt",
    "build_few_shot_python_prompt",
    "sanitize_prompt",
    "extract_code_blocks",
    "extract_inline_code",
    "count_tokens_approx",
    "find_similar_prompts",
    "split_into_turns",
    "prompt_statistics",
]
