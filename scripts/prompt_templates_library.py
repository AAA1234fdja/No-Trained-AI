#!/usr/bin/env python3
"""
prompt_templates_library.py — Librería extensa de plantillas de prompt.

Proporciona una colección organizada de plantillas de prompt para diferentes
tipos de tareas y dominios. Cada plantilla incluye:

  - System prompt (instrucciones para el modelo).
  - User template (plantilla con marcadores {variable}).
  - Ejemplos few-shot (opcional).
  - Parámetros recomendados (temperatura, top-k, max_tokens).
  - Metadata (dominio, dificultad, idioma).

Dominios cubiertos:
  - Python: programación, depuración, refactorización, algoritmos.
  - Análisis de datos: estadística, visualización, inferencia.
  - Conversación: chat general, asistencia, preguntas frecuentes.
  - Código general: SQL, Bash, JavaScript, HTML/CSS.
  - Documentación: README, comentarios, documentación de API.
  - Pruebas: unit tests, integración, casos de prueba.
  - Explicación: conceptos técnicos, tutoriales, guías.
  - Traducción: inglés-español, español-inglés, programación.
  - Resumen: resumen de texto, artículos, documentación.
  - Brainstorming: generación de ideas, planificación.

Todas las plantillas son cadenas de texto listas para usar con el
modelo de inferencia o con el PromptOptimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptTemplate:
    """Una plantilla de prompt individual."""
    name: str
    domain: str  # python, analysis, conversation, code, docs, tests, etc.
    system_prompt: str
    user_template: str  # con marcadores {variable}
    fewshot_examples: List[dict] = field(default_factory=list)
    recommended_temperature: float = 0.0
    recommended_top_k: int = 0
    recommended_max_tokens: int = 256
    language: str = "es"
    difficulty: str = "intermedio"  # basico, intermedio, avanzado
    tags: List[str] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        """
        Formatea la plantilla con los argumentos proporcionados.

        Returns
        -------
        str con el prompt de usuario formateado.
        """
        return self.user_template.format(**kwargs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "fewshot_examples": self.fewshot_examples,
            "recommended_temperature": self.recommended_temperature,
            "recommended_top_k": self.recommended_top_k,
            "recommended_max_tokens": self.recommended_max_tokens,
            "language": self.language,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }


class PromptTemplatesLibrary:
    """
    Librería de plantillas de prompt.

    Proporciona acceso a plantillas organizadas por dominio y permite
    buscar, filtrar y formatear plantillas fácilmente.
    """

    def __init__(self) -> None:
        self.templates: Dict[str, PromptTemplate] = {}
        self._register_all_templates()

    # ------------------------------------------------------------------
    # Registro de plantillas
    # ------------------------------------------------------------------

    def _register_all_templates(self) -> None:
        """Registra todas las plantillas predefinidas."""

        # ============ PYTHON ============

        self.register(PromptTemplate(
            name="python_func_from_description",
            domain="python",
            system_prompt=(
                "Eres un experto programador Python. Escribe código Python "
                "limpio, eficiente y bien documentado. Incluye docstrings, "
                "type hints y manejo de errores. No incluyas explicaciones "
                "además del código a menos que se te pida."
            ),
            user_template=(
                "Escribe una función Python que {description}.\n"
                "Requisitos:\n{requirements}\n"
                "Entrada de ejemplo: {example_input}\n"
                "Salida esperada: {example_output}"
            ),
            fewshot_examples=[
                {
                    "user": "Escribe una función Python que calcule el factorial de un número.\nRequisitos: Debe manejar números negativos con ValueError.\nEntrada de ejemplo: 5\nSalida esperada: 120",
                    "assistant": "def factorial(n: int) -> int:\n    '''Calcula el factorial de un número entero no negativo.'''\n    if n < 0:\n        raise ValueError('n debe ser un número no negativo')\n    if n == 0 or n == 1:\n        return 1\n    resultado = 1\n    for i in range(2, n + 1):\n        resultado *= i\n    return resultado",
                },
            ],
            recommended_temperature=0.1,
            recommended_max_tokens=256,
            tags=["función", "algoritmo", "matemáticas"],
        ))

        self.register(PromptTemplate(
            name="python_class_design",
            domain="python",
            system_prompt=(
                "Diseña clases Python siguiendo principios POO (herencia, "
                "encapsulamiento, polimorfismo). Incluye __init__, métodos, "
                "propiedades y docstrings completos."
            ),
            user_template=(
                "Diseña una clase Python llamada {class_name} que represente {description}.\n"
                "Atributos: {attributes}\n"
                "Métodos requeridos: {methods}\n"
                "Casos de uso: {use_cases}"
            ),
            recommended_temperature=0.2,
            recommended_max_tokens=300,
            tags=["clase", "POO", "diseño"],
        ))

        self.register(PromptTemplate(
            name="python_debug",
            domain="python",
            system_prompt=(
                "Analiza el código Python proporcionado y encuentra errores "
                "sintácticos, lógicos y de tiempo de ejecución. Proporciona "
                "la versión corregida con explicación de los cambios."
            ),
            user_template=(
                "El siguiente código Python tiene un error:\n\n```python\n{code}\n```\n\n"
                "Error: {error_message}\n"
                "Por favor, corrige el código y explica qué estaba mal."
            ),
            recommended_temperature=0.1,
            recommended_max_tokens=400,
            tags=["debug", "errores", "corrección"],
        ))

        self.register(PromptTemplate(
            name="python_optimize",
            domain="python",
            system_prompt=(
                "Optimiza el código Python proporcionado para mejor "
                "rendimiento (tiempo de ejecución y memoria). Explica los "
                "cambios realizados y por qué mejoran el rendimiento."
            ),
            user_template=(
                "Optimiza el siguiente código Python:\n\n```python\n{code}\n```\n\n"
                "Objetivo de optimización: {optimization_goal}\n"
                "Restricciones: {constraints}"
            ),
            recommended_temperature=0.2,
            recommended_max_tokens=400,
            tags=["optimización", "rendimiento"],
        ))

        self.register(PromptTemplate(
            name="python_algorithm",
            domain="python",
            system_prompt=(
                "Implementa algoritmos clásicos en Python con complejidad "
                "temporal y espacial documentada. Incluye casos de prueba "
                "básicos y explicación del enfoque."
            ),
            user_template=(
                "Implementa el algoritmo {algorithm_name} en Python.\n"
                "Descripción: {description}\n"
                "Complejidad esperada: {expected_complexity}\n"
                "Casos de prueba: {test_cases}"
            ),
            recommended_temperature=0.1,
            recommended_max_tokens=400,
            tags=["algoritmo", "complejidad"],
        ))

        # ============ ANÁLISIS DE DATOS ============

        self.register(PromptTemplate(
            name="data_statistical_analysis",
            domain="analysis",
            system_prompt=(
                "Realiza análisis estadístico de datos proporcionados. "
                "Calcula media, mediana, moda, varianza, desviación estándar, "
                "y detecta outliers. Presenta hallazgos en listas con viñetas."
            ),
            user_template=(
                "Analiza estadísticamente los siguientes datos:\n{data}\n\n"
                "Preguntas específicas: {questions}\n"
                "Formato de salida: {output_format}"
            ),
            recommended_temperature=0.2,
            recommended_max_tokens=300,
            tags=["estadística", "datos", "análisis"],
        ))

        self.register(PromptTemplate(
            name="data_correlation",
            domain="analysis",
            system_prompt=(
                "Analiza correlaciones entre variables en un dataset. "
                "Identifica correlaciones fuertes, débiles y nulas. "
                "Explica posibles relaciones causales vs. correlacionales."
            ),
            user_template=(
                "Analiza las correlaciones en estos datos:\n{data}\n\n"
                "Variables a considerar: {variables}\n"
                "Método de correlación: {method}"
            ),
            recommended_temperature=0.2,
            recommended_max_tokens=300,
            tags=["correlación", "variables"],
        ))

        self.register(PromptTemplate(
            name="data_visualization_recommendation",
            domain="analysis",
            system_prompt=(
                "Recomienda el tipo de gráfico más apropiado para visualizar "
                "datos dados. Explica por qué cada tipo es adecuado y qué "
                "aspectos de los datos resalta."
            ),
            user_template=(
                "Tengo los siguientes datos:\n{data_description}\n\n"
                "Objetivo de visualización: {goal}\n"
                "Audience: {audience}\n"
                "Recomienda los mejores tipos de gráficos y por qué."
            ),
            recommended_temperature=0.3,
            recommended_max_tokens=256,
            tags=["visualización", "gráficos"],
        ))

        # ============ CONVERSACIÓN ============

        self.register(PromptTemplate(
            name="conversation_greeting",
            domain="conversation",
            system_prompt=(
                "Eres un asistente amable y conversacional. Responde de forma "
                "breve, directa y útil. Evita repeticiones innecesarias."
            ),
            user_template="{user_message}",
            fewshot_examples=[
                {
                    "user": "Hola, ¿cómo estás?",
                    "assistant": "Hola. Estoy bien, gracias. ¿En qué puedo ayudarte hoy?",
                },
                {
                    "user": "¿Cuál es tu nombre?",
                    "assistant": "Soy un asistente de IA basado en un modelo de 800M parámetros.",
                },
            ],
            recommended_temperature=0.5,
            recommended_max_tokens=128,
            tags=["saludo", "conversación"],
        ))

        self.register(PromptTemplate(
            name="conversation_explain_concept",
            domain="conversation",
            system_prompt=(
                "Explica conceptos técnicos de forma simple y clara. "
                "Usa analogías cuando sea útil. Evita jerga innecesaria."
            ),
            user_template=(
                "Explica el concepto de {concept} de forma que un principiante pueda entenderlo.\n"
                "Incluye un ejemplo práctico.\n"
                "Nivel: {level}"
            ),
            recommended_temperature=0.4,
            recommended_max_tokens=256,
            tags=["explicación", "concepto"],
        ))

        self.register(PromptTemplate(
            name="conversation_advice",
            domain="conversation",
            system_prompt=(
                "Proporciona consejos prácticos y útiles. Sé honesto, "
                "directo y evita promesas exageradas."
            ),
            user_template=(
                "Necesito consejo sobre: {topic}\n"
                "Mi situación: {situation}\n"
                "¿Qué me recomiendas hacer?"
            ),
            recommended_temperature=0.5,
            recommended_max_tokens=200,
            tags=["consejo", "ayuda"],
        ))

        # ============ CÓDIGO GENERAL ============

        self.register(PromptTemplate(
            name="sql_query",
            domain="code",
            language="sql",
            system_prompt=(
                "Escribe consultas SQL eficientes y correctas. "
                "Incluye comentarios explicando las joins, filtros y agregaciones."
            ),
            user_template=(
                "Escribe una consulta SQL para: {query_description}\n"
                "Tablas disponibles: {tables}\n"
                "Columnas relevantes: {columns}"
            ),
            recommended_temperature=0.1,
            recommended_max_tokens=200,
            tags=["sql", "base de datos", "consulta"],
        ))

        self.register(PromptTemplate(
            name="bash_script",
            domain="code",
            language="bash",
            system_prompt=(
                "Escribe scripts de Bash seguros y eficientes. "
                "Incluye manejo de errores, comentarios y uso de variables."
            ),
            user_template=(
                "Escribe un script de Bash para: {script_description}\n"
                "Requisitos: {requirements}\n"
                "Entorno: {environment}"
            ),
            recommended_temperature=0.1,
            recommended_max_tokens=256,
            tags=["bash", "script", "automatización"],
        ))

        self.register(PromptTemplate(
            name="javascript_function",
            domain="code",
            language="javascript",
            system_prompt=(
                "Escribe funciones JavaScript modernas (ES6+). "
                "Usa const/let, arrow functions, template literals y async/await."
            ),
            user_template=(
                "Escribe una función JavaScript que {description}.\n"
                "Parámetros: {parameters}\n"
                "Retorno: {return_value}"
            ),
            recommended_temperature=0.1,
            recommended_max_tokens=200,
            tags=["javascript", "web", "función"],
        ))

        # ============ DOCUMENTACIÓN ============

        self.register(PromptTemplate(
            name="docs_readme",
            domain="docs",
            system_prompt=(
                "Escribe un archivo README.md completo y bien estructurado "
                "para un proyecto de software. Incluye instalación, uso, "
                "ejemplos y contribución."
            ),
            user_template=(
                "Escribe un README para el proyecto: {project_name}\n"
                "Descripción: {description}\n"
                "Características: {features}\n"
                "Instalación: {installation}\n"
                "Uso: {usage}"
            ),
            recommended_temperature=0.3,
            recommended_max_tokens=400,
            tags=["readme", "documentación", "proyecto"],
        ))

        self.register(PromptTemplate(
            name="docs_api",
            domain="docs",
            system_prompt=(
                "Escribe documentación de API clara y completa. "
                "Incluye endpoints, parámetros, respuestas, códigos de error "
                "y ejemplos de uso."
            ),
            user_template=(
                "Documenta la siguiente API:\n{api_description}\n\n"
                "Endpoints: {endpoints}\n"
                "Autenticación: {auth}\n"
                "Formato de respuesta: {response_format}"
            ),
            recommended_temperature=0.2,
            recommended_max_tokens=400,
            tags=["api", "documentación", "endpoints"],
        ))

        self.register(PromptTemplate(
            name="docs_code_comments",
            domain="docs",
            system_prompt=(
                "Añade comentarios claros y útiles al código fuente. "
                "Explica el 'porqué' no el 'qué'. Mantén comentarios concisos."
            ),
            user_template=(
                "Añade comentarios al siguiente código:\n\n```{language}\n{code}\n```\n\n"
                "Estilo de comentarios: {comment_style}"
            ),
            recommended_temperature=0.2,
            recommended_max_tokens=300,
            tags=["comentarios", "código", "documentación"],
        ))

        # ============ PRUEBAS ============

        self.register(PromptTemplate(
            name="tests_unit",
            domain="tests",
            system_prompt=(
                "Escribe pruebas unitarias en Python usando pytest o unittest. "
                "Incluye casos positivos, negativos y edge cases. "
                "Usa fixtures y parametrización cuando sea apropiado."
            ),
            user_template=(
                "Escribe pruebas unitarias para el siguiente código:\n\n```python\n{code}\n```\n\n"
                "Framework: {framework}\n"
                "Casos de prueba requeridos: {test_cases}"
            ),
            recommended_temperature=0.1,
            recommended_max_tokens=300,
            tags=["pruebas", "pytest", "unit"],
        ))

        self.register(PromptTemplate(
            name="tests_edge_cases",
            domain="tests",
            system_prompt=(
                "Identifica y escribe casos de prueba para edge cases "
                "y situaciones límite en el código proporcionado."
            ),
            user_template=(
                "Identifica edge cases para esta función:\n\n```python\n{code}\n```\n\n"
                "Función: {function_name}\n"
                "Entradas esperadas: {expected_inputs}"
            ),
            recommended_temperature=0.2,
            recommended_max_tokens=256,
            tags=["edge cases", "pruebas", "límite"],
        ))

        # ============ EXPLICACIÓN ============

        self.register(PromptTemplate(
            name="explain_code",
            domain="explanation",
            system_prompt=(
                "Explica qué hace un fragmento de código línea por línea. "
                "Usa lenguaje claro y evita jerga innecesaria."
            ),
            user_template=(
                "Explica el siguiente código:\n\n```{language}\n{code}\n```\n\n"
                "Nivel de detalle: {detail_level}\n"
                "Público objetivo: {audience}"
            ),
            recommended_temperature=0.3,
            recommended_max_tokens=300,
            tags=["explicación", "código"],
        ))

        self.register(PromptTemplate(
            name="explain_concept",
            domain="explanation",
            system_prompt=(
                "Explica un concepto técnico con analogías, ejemplos y "
                "diagramas de texto cuando sea útil. Estructura con secciones claras."
            ),
            user_template=(
                "Explica el concepto de {concept}.\n"
                "Incluye: definición, analogía, ejemplo práctico, y por qué es importante.\n"
                "Nivel: {level}"
            ),
            recommended_temperature=0.4,
            recommended_max_tokens=300,
            tags=["concepto", "educación"],
        ))

        # ============ TRADUCCIÓN ============

        self.register(PromptTemplate(
            name="translate_es_en",
            domain="translation",
            language="es",
            system_prompt=(
                "Traduce texto de español a inglés manteniendo el significado, "
                "tono y estilo original. Para código, traduce comentarios y "
                "nombres de variables cuando sea apropiado."
            ),
            user_template="Traduce al inglés: {text}",
            recommended_temperature=0.1,
            recommended_max_tokens=256,
            tags=["traducción", "es-en"],
        ))

        self.register(PromptTemplate(
            name="translate_en_es",
            domain="translation",
            language="en",
            system_prompt=(
                "Translate text from English to Spanish maintaining meaning, "
                "tone, and style. For code, translate comments and variable "
                "names when appropriate."
            ),
            user_template="Translate to Spanish: {text}",
            recommended_temperature=0.1,
            recommended_max_tokens=256,
            tags=["translation", "en-es"],
        ))

        # ============ RESUMEN ============

        self.register(PromptTemplate(
            name="summarize_text",
            domain="summary",
            system_prompt=(
                "Resume el texto proporcionado extrayendo los puntos clave. "
                "Mantén la información importante y elimina redundancias."
            ),
            user_template=(
                "Resume el siguiente texto en {max_bullets} puntos clave:\n\n{text}\n\n"
                "Longitud máxima: {max_length} palabras"
            ),
            recommended_temperature=0.3,
            recommended_max_tokens=200,
            tags=["resumen", "texto"],
        ))

        self.register(PromptTemplate(
            name="summarize_article",
            domain="summary",
            system_prompt=(
                "Resume un artículo técnico extrayendo: problema, solución, "
                "metodología, resultados y conclusiones. Formato estructurado."
            ),
            user_template=(
                "Resume este artículo técnico:\n\n{article}\n\n"
                "Incluye: problema, solución, metodología, resultados, conclusiones."
            ),
            recommended_temperature=0.3,
            recommended_max_tokens=300,
            tags=["resumen", "artículo", "técnico"],
        ))

        # ============ BRAINSTORMING ============

        self.register(PromptTemplate(
            name="brainstorm_ideas",
            domain="brainstorming",
            system_prompt=(
                "Genera ideas creativas y prácticas para un proyecto o problema. "
                "Proporciona al menos 5 ideas con pros y contras de cada una."
            ),
            user_template=(
                "Genera ideas para: {problem_or_project}\n"
                "Restricciones: {constraints}\n"
                "Objetivo: {goal}\n"
                "Número de ideas: {n_ideas}"
            ),
            recommended_temperature=0.7,
            recommended_max_tokens=300,
            tags=["brainstorming", "ideas", "creatividad"],
        ))

        self.register(PromptTemplate(
            name="plan_project",
            domain="brainstorming",
            system_prompt=(
                "Crea un plan de proyecto estructurado con fases, hitos, "
                "tareas y entregables. Incluye estimaciones de tiempo."
            ),
            user_template=(
                "Planifica el proyecto: {project_name}\n"
                "Objetivo: {objective}\n"
                "Duración estimada: {duration}\n"
                "Recursos disponibles: {resources}"
            ),
            recommended_temperature=0.4,
            recommended_max_tokens=350,
            tags=["planificación", "proyecto"],
        ))

    # ------------------------------------------------------------------
    # Operaciones de la librería
    # ------------------------------------------------------------------

    def register(self, template: PromptTemplate) -> None:
        """Registra una plantilla en la librería."""
        self.templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Obtiene una plantilla por nombre."""
        return self.templates.get(name)

    def list_by_domain(
        self,
        domain: str,
    ) -> List[PromptTemplate]:
        """Lista todas las plantillas de un dominio."""
        return [t for t in self.templates.values() if t.domain == domain]

    def list_by_tag(
        self,
        tag: str,
    ) -> List[PromptTemplate]:
        """Lista todas las plantillas que tienen una etiqueta específica."""
        return [t for t in self.templates.values() if tag in t.tags]

    def list_by_language(
        self,
        language: str,
    ) -> List[PromptTemplate]:
        """Lista plantillas por idioma."""
        return [t for t in self.templates.values() if t.language == language]

    def list_by_difficulty(
        self,
        difficulty: str,
    ) -> List[PromptTemplate]:
        """Lista plantillas por nivel de dificultad."""
        return [t for t in self.templates.values() if t.difficulty == difficulty]

    def search(
        self,
        query: str,
    ) -> List[PromptTemplate]:
        """
        Busca plantillas por nombre, dominio o etiquetas.

        Returns
        -------
        Lista de plantillas que coinciden con la consulta.
        """
        query_lower = query.lower()
        results: List[PromptTemplate] = []
        for t in self.templates.values():
            if (
                query_lower in t.name.lower()
                or query_lower in t.domain.lower()
                or any(query_lower in tag.lower() for tag in t.tags)
                or query_lower in t.system_prompt.lower()
            ):
                results.append(t)
        return results

    def format_prompt(
        self,
        template_name: str,
        **kwargs,
    ) -> Optional[str]:
        """
        Obtiene una plantilla y la formatea con los argumentos dados.

        Returns
        -------
        str con el prompt formateado o None si la plantilla no existe.
        """
        template = self.get(template_name)
        if template is None:
            return None
        return template.format(**kwargs)

    def get_all_domains(self) -> List[str]:
        """Devuelve la lista de dominios únicos."""
        return sorted(set(t.domain for t in self.templates.values()))

    def get_all_tags(self) -> List[str]:
        """Devuelve la lista de etiquetas únicas."""
        all_tags: set = set()
        for t in self.templates.values():
            all_tags.update(t.tags)
        return sorted(all_tags)

    def get_stats(self) -> dict:
        """Devuelve estadísticas de la librería."""
        by_domain: Dict[str, int] = {}
        by_language: Dict[str, int] = {}
        by_difficulty: Dict[str, int] = {}

        for t in self.templates.values():
            by_domain[t.domain] = by_domain.get(t.domain, 0) + 1
            by_language[t.language] = by_language.get(t.language, 0) + 1
            by_difficulty[t.difficulty] = by_difficulty.get(t.difficulty, 0) + 1

        return {
            "total_templates": len(self.templates),
            "by_domain": by_domain,
            "by_language": by_language,
            "by_difficulty": by_difficulty,
        }


def quick_library_stats() -> dict:
    """
    Función de conveniencia: devuelve estadísticas de la librería por defecto.

    Returns
    -------
    dict con estadísticas de la librería.
    """
    library = PromptTemplatesLibrary()
    return library.get_stats()


__all__ = [
    "PromptTemplate",
    "PromptTemplatesLibrary",
    "quick_library_stats",
]
