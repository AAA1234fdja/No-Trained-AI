#!/usr/bin/env python3
"""
prompt_optimizer.py — Optimizador de prompts para el modelo 800M.

Proporciona herramientas para optimizar prompts de entrada:

  - Plantillas de prompt por tarea (Python, análisis, conversación).
  - Inyección de few-shot examples automática.
  - Optimización de longitud de prompt (truncamiento inteligente).
  - Deteción de intención del usuario para seleccionar plantilla.
  - Generación de variaciones de prompt para testing A/B.
  - Puntuación de calidad de prompts (claridad, especificidad, contexto).
  - Cache de prompts frecuentes.
  - Reporte de optimización en Markdown/JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PromptTemplate:
    """Plantilla de prompt para una tarea específica."""
    name: str
    task_type: str  # "python", "analysis", "conversation", "general"
    system_prompt: str
    user_template: str
    fewshot_examples: List[dict] = field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.0

    def format(
        self,
        user_input: str,
        fewshot: Optional[List[dict]] = None,
    ) -> str:
        """
        Formatea el prompt con la entrada del usuario y opcionalmente
        ejemplos few-shot.

        Returns
        -------
        str con el prompt completo listo para el modelo.
        """
        parts: List[str] = []

        # Sistema
        parts.append(f"System: {self.system_prompt}")

        # Few-shot examples
        examples = fewshot if fewshot is not None else self.fewshot_examples
        for ex in examples:
            parts.append(f"User: {ex.get('user', '')}")
            parts.append(f"Assistant: {ex.get('assistant', '')}")

        # Usuario actual
        parts.append(f"User: {user_input}")
        parts.append("Assistant:")

        return "\n".join(parts)


@dataclass
class PromptQualityScore:
    """Puntuación de calidad de un prompt."""
    clarity: float  # 0-1
    specificity: float  # 0-1
    context: float  # 0-1
    length_score: float  # 0-1
    overall: float  # 0-1

    def to_dict(self) -> dict:
        return {
            "clarity": round(self.clarity, 3),
            "specificity": round(self.specificity, 3),
            "context": round(self.context, 3),
            "length_score": round(self.length_score, 3),
            "overall": round(self.overall, 3),
        }


class PromptOptimizer:
    """
    Optimizador de prompts para el modelo AI 800M.

    Selecciona plantillas, inyecta few-shot examples, puntúa la calidad
    de los prompts y genera variaciones para pruebas A/B.
    """

    def __init__(self) -> None:
        self.templates: Dict[str, PromptTemplate] = {}
        self._register_default_templates()

        self.cache: Dict[str, str] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _register_default_templates(self) -> None:
        """Registra plantillas por defecto para cada tipo de tarea."""

        # Python
        self.templates["python"] = PromptTemplate(
            name="python_code",
            task_type="python",
            system_prompt=(
                "Eres un asistente experto en programación Python. "
                "Responde solo con código Python válido, sin explicaciones "
                "adicionales a menos que se te pida. Usa nombres de variables "
                "descriptivos, incluye docstrings y maneja errores adecuadamente."
            ),
            user_template="Escribe código Python para: {user_input}",
            fewshot_examples=[
                {
                    "user": "Función para calcular factorial",
                    "assistant": "def factorial(n):\n    if n < 0:\n        raise ValueError('n debe ser no negativo')\n    resultado = 1\n    for i in range(2, n + 1):\n        resultado *= i\n    return resultado",
                },
                {
                    "user": "Función para verificar si un número es primo",
                    "assistant": "def es_primo(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                },
        ],
        max_tokens=256,
        temperature=0.1,
    )

    # Análisis
    self.templates["analysis"] = PromptTemplate(
        name="data_analysis",
        task_type="analysis",
        system_prompt=(
            "Eres un analista de datos experto. Proporciona análisis "
            "claro, conciso y basado en evidencia. Usa listas con viñetas "
            "para hallazgos principales y evita repeticiones innecesarias."
        ),
        user_template="Analiza lo siguiente: {user_input}",
        fewshot_examples=[
            {
                "user": "Datos de ventas: [100, 150, 200, 180, 220]",
                "assistant": "Hallazgos:\n- Crecimiento general del 120%\n- Pico en el tercer mes\n- Leve caída en el cuarto mes\n- Recuperación en el quinto mes",
            },
        ],
        max_tokens=256,
        temperature=0.2,
    )

    # Conversación
    self.templates["conversation"] = PromptTemplate(
        name="conversation",
        task_type="conversation",
        system_prompt=(
            "Eres un asistente conversacional amable y conciso. "
            "Responde de forma breve y directa. Evita repeticiones. "
            "Si no sabes algo, admítelo honestamente."
        ),
        user_template="{user_input}",
        fewshot_examples=[
            {
                "user": "Hola, ¿cómo estás?",
                "assistant": "Hola. Estoy bien, gracias. ¿En qué puedo ayudarte?",
            },
            {
                "user": "¿Cuál es tu nombre?",
                "assistant": "Soy un asistente de IA basado en el modelo 800M.",
            },
        ],
        max_tokens=128,
        temperature=0.5,
    )

    # General
    self.templates["general"] = PromptTemplate(
        name="general_purpose",
        task_type="general",
        system_prompt=(
            "Eres un asistente de IA útil. Responde de forma clara, "
            "precisa y concisa a las preguntas del usuario."
        ),
        user_template="{user_input}",
        fewshot_examples=[],
        max_tokens=256,
        temperature=0.3,
    )

    def detect_task_type(self, user_input: str) -> str:
        """
        Detecta el tipo de tarea basado en el contenido del prompt.

        Returns
        -------
        str: "python", "analysis", "conversation", o "general".
        """
        text = user_input.lower()

        # Palabras clave de Python
        python_keywords = [
            "def ", "import ", "class ", "función", "código python",
            "programa", "script", "algoritmo", "bucle", "condicional",
            "lista", "diccionario", "tupla", "decorador",
        ]
        if any(kw in text for kw in python_keywords):
            return "python"

        # Palabras clave de análisis
        analysis_keywords = [
            "analiza", "análisis", "datos", "estadística", "promedio",
            "mediana", "varianza", "correlación", "regresión", "tendencia",
            "informe", "reporte", "métrica",
        ]
        if any(kw in text for kw in analysis_keywords):
            return "analysis"

        # Palabras clave de conversación
        conv_keywords = [
            "hola", "adiós", "gracias", "nombre", "cómo estás",
            "quién eres", "qué puedes hacer", "ayuda",
        ]
        if any(kw in text for kw in conv_keywords) and len(user_input.split()) <= 10:
            return "conversation"

        return "general"

    def get_template(
        self,
        task_type: Optional[str] = None,
        user_input: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Obtiene la plantilla adecuada para una tarea.

        Parameters
        ----------
        task_type : str opcional
            Tipo de tarea conocido.
        user_input : str opcional
            Entrada del usuario para detectar automáticamente el tipo.

        Returns
        -------
        PromptTemplate adecuada.
        """
        if task_type is None and user_input is not None:
            task_type = self.detect_task_type(user_input)

        if task_type is None:
            task_type = "general"

        return self.templates.get(task_type, self.templates["general"])

    def format_prompt(
        self,
        user_input: str,
        task_type: Optional[str] = None,
        fewshot: Optional[List[dict]] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Formatea un prompt completo usando la plantilla adecuada.

        Parameters
        ----------
        user_input : str
            Entrada del usuario.
        task_type : str opcional
            Tipo de tarea (se detecta automáticamente si no se especifica).
        fewshot : list de dict opcional
            Ejemplos few-shot personalizados.
        use_cache : bool
            Si True, usa caché para prompts repetidos.

        Returns
        -------
        str con el prompt formateado.
        """
        cache_key = f"{task_type}:{user_input}:{json.dumps(fewshot, sort_keys=True)}"

        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1

        template = self.get_template(task_type, user_input)
        formatted = template.format(user_input, fewshot)

        if use_cache:
            self.cache[cache_key] = formatted

        return formatted

    def score_prompt_quality(self, prompt: str) -> PromptQualityScore:
        """
        Evalúa la calidad de un prompt en varias dimensiones.

        Returns
        -------
        PromptQualityScore con puntuaciones de 0 a 1.
        """
        words = prompt.split()
        n_words = len(words)

        # Claridad: presencia de verbos de acción y sustantivos concretos
        action_verbs = [
            "escribe", "analiza", "calcula", "explica", "describe",
            "genera", "crea", "implementa", "resume", "traduce",
        ]
        clarity_words = sum(1 for w in words if w.lower() in action_verbs)
        clarity = min(clarity_words / 3.0, 1.0)

        # Especificidad: longitud del prompt (más detalle = más específico)
        specificity = min(n_words / 50.0, 1.0)

        # Contexto: presencia de palabras de contexto
        context_words = [
            "contexto", "datos", "ejemplo", "caso", "escenario",
            "situación", "proyecto", "requisito",
        ]
        context_count = sum(1 for w in words if w.lower() in context_words)
        context = min(context_count / 3.0, 1.0)

        # Longitud: prompts muy cortos o muy largos puntuación menor
        if 10 <= n_words <= 100:
            length_score = 1.0
        elif n_words < 10:
            length_score = n_words / 10.0
        else:
            length_score = max(0.5, 100.0 / n_words)

        # Puntuación general
        overall = (clarity + specificity + context + length_score) / 4.0

        return PromptQualityScore(
            clarity=clarity,
            specificity=specificity,
            context=context,
            length_score=length_score,
            overall=overall,
        )

    def generate_variations(
        self,
        user_input: str,
        n_variations: int = 3,
    ) -> List[dict]:
        """
        Genera variaciones de un prompt para pruebas A/B.

        Returns
        -------
        lista de dicts con "name", "prompt", "task_type", "score".
        """
        variations: List[dict] = []

        # Variación 1: directa
        prompt_direct = self.format_prompt(user_input, task_type="general", fewshot=None)
        score_direct = self.score_prompt_quality(prompt_direct)
        variations.append({
            "name": "directa",
            "prompt": prompt_direct,
            "task_type": "general",
            "score": score_direct.to_dict(),
        })

        # Variación 2: con contexto
        prompt_context = self.format_prompt(
            f"Proporciona contexto detallado antes de responder: {user_input}",
            task_type="general",
            fewshot=None,
        )
        score_context = self.score_prompt_quality(prompt_context)
        variations.append({
            "name": "con_contexto",
            "prompt": prompt_context,
            "task_type": "general",
            "score": score_context.to_dict(),
        })

        # Variación 3: paso a paso
        prompt_step = self.format_prompt(
            f"Responde paso a paso: {user_input}",
            task_type="general",
            fewshot=None,
        )
        score_step = self.score_prompt_quality(prompt_step)
        variations.append({
            "name": "paso_a_paso",
            "prompt": prompt_step,
            "task_type": "general",
            "score": score_step.to_dict(),
        })

        return variations[:n_variations]

    def get_cache_stats(self) -> dict:
        """Devuelve estadísticas del caché de prompts."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": round(hit_rate, 4),
        }


def quick_optimize_prompt(
    user_input: str,
    task_type: Optional[str] = None,
) -> dict:
    """
    Función de conveniencia: optimiza un prompt para el modelo.

    Returns
    -------
    dict con el prompt optimizado, puntuación y variaciones.
    """
    optimizer = PromptOptimizer()
    prompt = optimizer.format_prompt(user_input, task_type)
    score = optimizer.score_prompt_quality(prompt)
    variations = optimizer.generate_variations(user_input)
    cache_stats = optimizer.get_cache_stats()

    return {
        "optimized_prompt": prompt,
        "task_type": task_type or optimizer.detect_task_type(user_input),
        "quality_score": score.to_dict(),
        "variations": variations,
        "cache_stats": cache_stats,
    }


__all__ = [
    "PromptTemplate",
    "PromptQualityScore",
    "PromptOptimizer",
    "quick_optimize_prompt",
]
