#!/usr/bin/env python3
"""
conversation_engine.py — Motor de conversación basado en reglas para el modelo 800M.

Implementa un sistema de conversación mínima basado en reglas y patrones
que puede usarse como capa de post-procesamiento sobre la generación
del modelo, o como sistema de conversación independiente cuando el modelo
no produce respuestas coherentes.

Incluye:
  - Deteción de intención (saludo, pregunta, ayuda, despedida, etc.).
  - Plantillas de respuesta por intención.
  - Manejo de historial de conversación.
  - Extracción de entidades simples (nombres, números, temas).
  - Generación de respuestas estructuradas.
  - Integración con el motor de inferencia (fallback cuando el modelo
    colapsa o genera texto incoherente).

Todas las respuestas son determinísticas y no requieren entrenamiento.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Intenciones y patrones
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """Representa una intención detectada en un mensaje del usuario."""
    name: str
    confidence: float
    entities: Dict[str, List[str]] = field(default_factory=dict)
    keywords_matched: List[str] = field(default_factory=list)


INTENT_PATTERNS: Dict[str, List[str]] = {
    "greeting": [
        r"\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b",
        r"\b(howdy|yo|sup)\b",
    ],
    "goodbye": [
        r"\b(goodbye|bye|see you|farewell|take care|good night)\b",
    ],
    "help": [
        r"\b(help|assist|support|aid)\b",
        r"\b(can you help|could you help|i need help)\b",
    ],
    "question": [
        r"\b(what|why|how|which|who|where|when|can you|could you|would you)\b",
        r"\?\s*$",
    ],
    "analysis": [
        r"\b(analyze|analysis|evaluate|assess|compare|contrast)\b",
        r"\b(complexity|time complexity|space complexity|algorithm)\b",
        r"\b(variance|mean|median|std|deviation|correlation)\b",
    ],
    "python": [
        r"\b(python|code|program|script|function|def |class )\b",
        r"\b(import |from |print\(|len\(|range\(|\.py\b)\b",
        r"\b(debug|refactor|optimize|review)\b",
    ],
    "data": [
        r"\b(data|dataset|statistics|stats|numbers|values|column|row)\b",
        r"\b(csv|excel|spreadsheet|table|df|dataframe)\b",
    ],
    "thanks": [
        r"\b(thank|thanks|thank you|appreciate)\b",
    ],
    "confirmation": [
        r"\b(yes|yeah|yep|correct|right|exactly)\b",
    ],
    "denial": [
        r"\b(no|nope|not really|incorrect|wrong)\b",
    ],
}


RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Greetings! I'm here to help with Python, analysis, or conversation.",
        "Hello! Feel free to ask me about code, data analysis, or anything else.",
    ],
    "goodbye": [
        "Goodbye! Have a great day.",
        "Bye! Feel free to come back if you have more questions.",
        "Take care! Happy coding!",
    ],
    "help": [
        "I can help you with Python programming, data analysis, and general questions. What do you need?",
        "Sure! I can assist with code completion, code explanation, data analysis, or just chat. What would you like to work on?",
        "I'm here to help. Try asking me to analyze something, write Python code, or explain a concept.",
    ],
    "question": [
        "That's an interesting question. Could you provide more context so I can give a better answer?",
        "I'd be happy to help with that. Could you clarify what specifically you'd like to know?",
        "That's a good question. Let me think about the best way to answer it.",
    ],
    "analysis": [
        "I can help you analyze data, compute statistics, or evaluate algorithmic complexity. Please share the data or question you'd like to analyze.",
        "For analysis, I can compute mean, median, variance, correlation, and more. What data do you have?",
        "I can perform statistical analysis, detect outliers, and help interpret results. Share your dataset or analysis question.",
    ],
    "python": [
        "I can help you write, debug, or explain Python code. Share the code snippet you're working on.",
        "Python is my strength! I can help with functions, classes, algorithms, data structures, and more. What do you need?",
        "Sure, I can assist with Python code. Paste your code and tell me what you'd like to do with it.",
    ],
    "data": [
        "I can help you analyze datasets, compute statistics, and detect patterns. Please share your data or describe it.",
        "For data analysis, I can compute descriptive statistics, correlations, and visualizations (described in text). What's your dataset?",
    ],
    "thanks": [
        "You're welcome!",
        "Glad I could help!",
        "No problem! Let me know if you need anything else.",
    ],
    "confirmation": [
        "Great! Let's proceed with that then.",
        "Understood. What would you like to do next?",
    ],
    "denial": [
        "I see. Is there something else you'd like to try instead?",
        "No problem. Let me know how I can help differently.",
    ],
    "fallback": [
        "I'm not sure I understand. Could you rephrase that?",
        "Could you provide more details about what you need?",
        "I can help with Python, data analysis, or general questions. What would you like to talk about?",
    ],
}

ENTITY_PATTERNS: Dict[str, str] = {
    "person_name": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b",
    "number": r"\b\d+(?:\.\d+)?\b",
    "python_topic": r"\b(pandas|numpy|matplotlib|scikit|scikit-learn|tensorflow|torch|django|flask|fastapi)\b",
    "file_extension": r"\b\w+\.(py|csv|json|txt|md)\b",
}


# ---------------------------------------------------------------------------
# Motor de conversación
# ---------------------------------------------------------------------------

class ConversationEngine:
    """
    Motor de conversación basado en reglas y patrones.

    Detecta la intención del usuario, extrae entidades, y genera una
    respuesta determinística. Puede usarse de forma independiente o como
    fallback del modelo de IA cuando su salida es incoherente.
    """

    def __init__(
        self,
        name: str = "AI Assistant",
        language: str = "en",
        random_seed: int = 42,
    ) -> None:
        self.name = name
        self.language = language
        self.rng = random.Random(random_seed)
        self.history: List[Dict[str, str]] = []
        self.session_start = datetime.now().isoformat()
        self.message_count = 0
        self.intent_counts: Dict[str, int] = {}

    def detect_intent(
        self,
        text: str,
    ) -> Intent:
        """
        Detecta la intención principal en el texto del usuario.

        Returns
        -------
        Intent con nombre, confianza, entidades y palabras clave coincidentes.
        """
        text_lower = text.lower()
        best_intent = "fallback"
        best_confidence = 0.0
        best_keywords: List[str] = []

        for intent_name, patterns in INTENT_PATTERNS.items():
            matched_keywords: List[str] = []
            match_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
                if matches:
                    match_count += len(matches)
                    # Extraer palabras clave del patrón (simplificado)
                    for m in matches:
                        if isinstance(m, str):
                            matched_keywords.append(m)

            if match_count > 0:
                # Confianza basada en número de coincidencias y longitud del texto
                confidence = min(0.95, 0.4 + 0.1 * match_count)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent_name
                    best_keywords = matched_keywords

        # Extraer entidades
        entities = self._extract_entities(text)

        return Intent(
            name=best_intent,
            confidence=best_confidence,
            entities=entities,
            keywords_matched=best_keywords,
        )

    def _extract_entities(
        self,
        text: str,
    ) -> Dict[str, List[str]]:
        """Extrae entidades nombradas del texto."""
        entities: Dict[str, List[str]] = {}
        for entity_name, pattern in ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_name] = matches
        return entities

    def generate_response(
        self,
        user_message: str,
        use_model_fallback: bool = True,
        model_generation: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Genera una respuesta a un mensaje del usuario.

        Parameters
        ----------
        user_message : str
            Mensaje del usuario.
        use_model_fallback : bool
            Si usar la generación del modelo cuando el motor de reglas
            detecte que la respuesta del modelo es mejor (no implementado
            aquí; se deja como gancho).
        model_generation : str opcional
            Texto generado por el modelo de IA (para decidir si usar fallback).

        Returns
        -------
        dict con: respuesta, intención, confianza, entidades, usar_modelo.
        """
        self.message_count += 1
        cleaned = user_message.strip()

        intent = self.detect_intent(cleaned)
        self.intent_counts[intent.name] = self.intent_counts.get(intent.name, 0) + 1

        # Registrar en historial
        self.history.append({
            "role": "user",
            "message": user_message,
            "intent": intent.name,
            "timestamp": datetime.now().isoformat(),
        })

        # Decidir si usar respuesta basada en reglas o del modelo
        use_rules = True
        if model_generation is not None:
            # Si el modelo generó algo razonable (no colapso), usarlo
            if self._is_model_response_good(model_generation, intent):
                use_rules = False

        if use_rules:
            response = self._rule_based_response(intent, cleaned)
            source = "rules"
        else:
            response = model_generation
            source = "model"

        # Registrar respuesta en historial
        self.history.append({
            "role": "assistant",
            "message": response,
            "intent": intent.name,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })

        return {
            "response": response,
            "intent": intent.name,
            "confidence": intent.confidence,
            "entities": intent.entities,
            "source": source,
            "message_id": self.message_count,
        }

    def _is_model_response_good(
        self,
        generation: str,
        intent: Intent,
    ) -> bool:
        """
        Determina si la generación del modelo es lo suficientemente buena
        como para usarse directamente en lugar de la respuesta basada en reglas.
        """
        if not generation or generation.isspace():
            return False

        stripped = generation.strip()

        # Detectar colapso a un solo carácter
        if len(stripped) > 1 and len(set(stripped)) == 1:
            return False

        # Detectar repetición de palabra única
        tokens = stripped.split()
        if len(tokens) > 3 and len(set(tokens)) == 1:
            return False

        # Detectar repetición excesiva de la misma palabra (más del 70%)
        if len(tokens) > 5:
            from collections import Counter
            counts = Counter(tokens)
            most_common_count = counts.most_common(1)[0][1]
            if most_common_count / len(tokens) > 0.7:
                return False

        # Para preguntas, el modelo debe generar algo sustancial
        if intent.name == "question" and len(tokens) < 2:
            return False

        # Para saludos, debe haber al menos una palabra reconocible
        if intent.name == "greeting":
            greeting_words = {"hello", "hi", "hey", "greetings", "welcome",
                              "good", "help", "can", "you"}
            if not any(w in stripped.lower() for w in greeting_words):
                return False

        return True

    def _rule_based_response(
        self,
        intent: Intent,
        user_message: str,
    ) -> str:
        """
        Genera una respuesta basada en reglas para una intención dada.
        """
        templates = RESPONSE_TEMPLATES.get(intent.name, RESPONSE_TEMPLATES["fallback"])
        base_response = self.rng.choice(templates)

        # Personalizar según entidades detectadas
        entities = intent.entities

        if intent.name == "greeting" and "person_name" in entities:
            name = entities["person_name"][0]
            base_response = f"Hello {name}! How can I help you today?"

        if intent.name == "analysis" and "python_topic" in entities:
            topic = entities["python_topic"][0]
            base_response += f" I see you mentioned {topic} — I can help with that library specifically."

        if intent.name == "python" and "file_extension" in entities:
            ext = entities["file_extension"][0]
            base_response += f" I see you're working with a {ext} file."

        if intent.name == "data" and "number" in entities:
            nums = entities["number"][:3]
            base_response += f" I notice some numbers in your message: {', '.join(nums)}."

        return base_response

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Devuelve el historial completo de la conversación."""
        return self.history.copy()

    def clear_history(self) -> None:
        """Limpia el historial de la conversación."""
        self.history = []
        self.message_count = 0

    def get_statistics(self) -> Dict[str, object]:
        """Devuelve estadísticas de la conversación."""
        return {
            "message_count": self.message_count,
            "turns": len(self.history) // 2,
            "intent_distribution": dict(self.intent_counts),
            "session_start": self.session_start,
            "language": self.language,
            "name": self.name,
        }

    def save_history(
        self,
        path: str | Path,
    ) -> None:
        """Guarda el historial en archivo JSON."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_start": self.session_start,
            "name": self.name,
            "language": self.language,
            "message_count": self.message_count,
            "intent_distribution": self.intent_counts,
            "history": self.history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Integración con motor de inferencia
# ---------------------------------------------------------------------------

def conversation_with_model_fallback(
    user_message: str,
    model_generation: str,
    engine: Optional[ConversationEngine] = None,
) -> Dict[str, object]:
    """
    Función de conveniencia que combina el motor de reglas con una
    generación del modelo, usando reglas cuando el modelo falla.

    Returns
    -------
    dict con la respuesta final y metadatos.
    """
    if engine is None:
        engine = ConversationEngine()
    return engine.generate_response(
        user_message,
        use_model_fallback=True,
        model_generation=model_generation,
    )


__all__ = [
    "Intent",
    "ConversationEngine",
    "INTENT_PATTERNS",
    "RESPONSE_TEMPLATES",
    "ENTITY_PATTERNS",
    "conversation_with_model_fallback",
]
