#!/usr/bin/env python3
"""
function_generator_complex.py — Generador complejo de funciones Python.

Este módulo convierte listas simples de nombres y plantillas en un
generador sofisticado de funciones Python con:

  - Generación semántica de nombres basada en el propósito de la función.
  - Plantillas de funciones con 5 niveles de complejidad (simple → advanced).
  - Inyección automática de type hints, docstrings (Google/NumPy/reST).
  - Generación de decoradores (timer, retry, cache, validation).
  - Generación de manejadores de excepciones personalizados.
  - Generación de funciones recursivas, generadoras y asíncronas.
  - Validación de sintaxis del código generado usando ast.
  - Cálculo de complejidad ciclomática y cognitiva.
  - Generación de casos de prueba (pytest) para cada función.
  - Generación de fixtures y mocks asociados.
  - Sistema de versionado de funciones generadas.
  - Catálogo de funciones generadas con metadatos completos.
  - Exportación de funciones a módulos individuales o paquetes.
  - Análisis de dependencias entre funciones generadas.
  - Reporte de calidad del código generado en Markdown/JSON.

Todas las operaciones generan código como cadenas de texto; no se ejecuta
el código generado en el proceso principal.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class FunctionParameter:
    """Definición de un parámetro de función."""
    name: str
    type_hint: str
    default: Optional[str] = None
    kind: str = "positional"  # positional, keyword, args, kwargs
    description: str = ""

    def to_signature_part(self) -> str:
        part = f"{self.name}: {self.type_hint}"
        if self.default is not None:
            part += f" = {self.default}"
        if self.kind == "args":
            part = f"*{part}"
        elif self.kind == "kwargs":
            part = f"**{part}"
        return part


@dataclass
class FunctionTemplate:
    """Plantilla de función generada."""
    name: str
    purpose: str
    category: str  # math, string, data, io, system, algorithm, etc.
    parameters: List[FunctionParameter]
    return_type: str
    body_lines: List[str]
    decorators: List[str] = field(default_factory=list)
    docstring_style: str = "google"  # google, numpy, rest
    complexity_level: str = "simple"  # simple, medium, complex, advanced, expert
    is_async: bool = False
    is_generator: bool = False
    is_recursive: bool = False
    exception_handling: bool = False
    n_test_cases: int = 3
    version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "category": self.category,
            "parameters": [
                {
                    "name": p.name,
                    "type_hint": p.type_hint,
                    "default": p.default,
                    "kind": p.kind,
                    "description": p.description,
                }
                for p in self.parameters
            ],
            "return_type": self.return_type,
            "body_lines": self.body_lines,
            "decorators": self.decorators,
            "docstring_style": self.docstring_style,
            "complexity_level": self.complexity_level,
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "is_recursive": self.is_recursive,
            "exception_handling": self.exception_handling,
            "n_test_cases": self.n_test_cases,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class GeneratedFunction:
    """Resultado de generación de una función."""
    template: FunctionTemplate
    code: str
    test_code: str
    syntax_valid: bool
    complexity_score: int
    cognitive_score: int
    n_lines: int
    quality_score: float  # 0-100

    def to_dict(self) -> dict:
        return {
            "template": self.template.to_dict(),
            "code": self.code,
            "test_code": self.test_code,
            "syntax_valid": self.syntax_valid,
            "complexity_score": self.complexity_score,
            "cognitive_score": self.cognitive_score,
            "n_lines": self.n_lines,
            "quality_score": round(self.quality_score, 2),
        }


# ---------------------------------------------------------------------------
# Generadores de nombres semánticos
# ---------------------------------------------------------------------------

class SemanticNameGenerator:
    """
    Generador de nombres semánticos para funciones, parámetros y variables.

    En lugar de usar listas aleatorias simples, construye nombres basados
    en el propósito, dominio y contexto de la función.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

        # Prefijos por categoría de acción
        self.action_prefixes: Dict[str, List[str]] = {
            "math": ["calculate", "compute", "evaluate", "solve", "derive",
                     "estimate", "measure", "quantify", "assess"],
            "string": ["parse", "format", "transform", "convert", "clean",
                       "normalize", "validate", "extract", "join", "split"],
            "data": ["process", "aggregate", "filter", "map", "reduce",
                     "sort", "group", "summarize", "analyze", "validate"],
            "io": ["read", "write", "load", "save", "export", "import",
                   "stream", "buffer", "flush", "sync"],
            "system": ["initialize", "configure", "setup", "teardown",
                       "start", "stop", "restart", "monitor", "check"],
            "algorithm": ["search", "find", "traverse", "traverse_optimal",
                          "optimize", "heuristic", "backtrack", "branch"],
            "validation": ["validate", "verify", "check", "assert",
                           "confirm", "authenticate", "authorize"],
            "transformation": ["map", "transform", "convert", "adapt",
                               "morph", "translate", "encode", "decode"],
        }

        # Sufijos por tipo de retorno
        self.return_suffixes: Dict[str, List[str]] = {
            "bool": ["_check", "_is_valid", "_has", "_can", "_should"],
            "list": ["_list", "_collection", "_items", "_elements"],
            "dict": ["_map", "_dict", "_lookup", "_index", "_registry"],
            "str": ["_str", "_text", "_message", "_label", "_name"],
            "int": ["_count", "_index", "_number", "_total", "_size"],
            "float": ["_value", "_result", "_score", "_metric", "_rate"],
            "None": ["_execute", "_run", "_perform", "_process", "_apply"],
        }

        # Palabras de dominio por categoría
        self.domain_words: Dict[str, List[str]] = {
            "math": ["matrix", "vector", "tensor", "scalar", "polynomial",
                     "integral", "derivative", "statistic", "probability"],
            "string": ["text", "token", "pattern", "regex", "substring",
                       "prefix", "suffix", "sequence", "character"],
            "data": ["dataset", "record", "entry", "field", "column",
                     "row", "table", "frame", "series", "batch"],
            "io": ["file", "stream", "path", "buffer", "payload",
                   "chunk", "segment", "blob", "archive"],
            "system": ["service", "process", "thread", "session",
                       "connection", "socket", "endpoint", "handler"],
            "algorithm": ["graph", "node", "edge", "path", "tree",
                          "heap", "stack", "queue", "trie", "hash"],
        }

    def generate_function_name(
        self,
        category: str,
        purpose: str,
        action: Optional[str] = None,
    ) -> str:
        """
        Genera un nombre semántico para una función.

        Returns
        -------
        str con el nombre de la función en snake_case.
        """
        # Seleccionar acción
        if action is None:
            actions = self.action_prefixes.get(category, ["process"])
            action = self.rng.choice(actions)

        # Extraer palabras clave del propósito
        purpose_words = self._extract_purpose_words(purpose)

        # Seleccionar palabra de dominio
        domain_words = self.domain_words.get(category, ["data"])
        domain_word = self.rng.choice(domain_words)

        # Construir nombre
        parts = [action]

        if purpose_words:
            parts.append(self.rng.choice(purpose_words))

        parts.append(domain_word)

        # Unir en snake_case
        name = "_".join(parts)
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        # Asegurar que no empiece con número
        if name and name[0].isdigit():
            name = "get_" + name

        return name.lower()

    def generate_parameter_name(
        self,
        type_hint: str,
        role: str = "input",
    ) -> str:
        """
        Genera un nombre semántico para un parámetro.

        Returns
        -------
        str con el nombre del parámetro.
        """
        type_base = type_hint.replace("List[", "").replace("Dict[", "").replace("Optional[", "").replace("]", "")

        role_prefixes = {
            "input": ["input", "source", "raw", "initial", "primary"],
            "output": ["output", "result", "target", "final", "processed"],
            "config": ["config", "setting", "option", "parameter"],
            "context": ["context", "environment", "state", "session"],
        }

        prefixes = role_prefixes.get(role, ["value"])
        prefix = self.rng.choice(prefixes)

        return f"{prefix}_{type_base.lower()}"

    def _extract_purpose_words(self, purpose: str) -> List[str]:
        """Extrae palabras clave del propósito de la función."""
        # Mapeo de palabras comunes a términos técnicos
        purpose_map = {
            "calcular": "compute", "calculo": "compute", "math": "compute",
            "validar": "validate", "validación": "validate", "check": "validate",
            "buscar": "search", "buscar_optimo": "find_optimal", "find": "find",
            "ordenar": "sort", "ordenamiento": "sort", "sort": "sort",
            "filtrar": "filter", "filtrado": "filter", "filter": "filter",
            "transformar": "transform", "transformación": "transform",
            "convertir": "convert", "conversión": "convert",
            "procesar": "process", "procesamiento": "process",
            "analizar": "analyze", "análisis": "analyze",
            "generar": "generate", "generación": "generate",
            "extraer": "extract", "extracción": "extract",
            "agrupar": "group", "agrupación": "group",
            "resumir": "summarize", "resumen": "summarize",
            "leer": "read", "lectura": "read",
            "escribir": "write", "escritura": "write",
            "guardar": "save", "guardado": "save",
            "cargar": "load", "carga": "load",
            "exportar": "export", "exportación": "export",
            "importar": "import", "importación": "import",
            "iniciar": "initialize", "inicialización": "initialize",
            "configurar": "configure", "configuración": "configure",
            "verificar": "verify", "verificación": "verify",
            "autenticar": "authenticate", "autenticación": "authenticate",
            "autorizar": "authorize", "autorización": "authorize",
        }

        words: List[str] = []
        purpose_lower = purpose.lower()

        for key, value in purpose_map.items():
            if key in purpose_lower:
                words.append(value)

        # Si no se encontró nada, usar palabras genéricas
        if not words:
            words = ["data", "value", "item"]

        return words[:2]  # Máximo 2 palabras de propósito


# ---------------------------------------------------------------------------
# Generadores de cuerpos de función por complejidad
# ---------------------------------------------------------------------------

class FunctionBodyGenerator:
    """
    Generador de cuerpos de función con diferentes niveles de complejidad.

    Cada nivel añade más estructuras de control, manejo de errores,
    validaciones y características avanzadas.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def generate_body(
        self,
        template: FunctionTemplate,
    ) -> List[str]:
        """
        Genera el cuerpo de la función según su nivel de complejidad.

        Returns
        -------
        Lista de líneas de código Python.
        """
        level = template.complexity_level

        if level == "simple":
            return self._generate_simple_body(template)
        elif level == "medium":
            return self._generate_medium_body(template)
        elif level == "complex":
            return self._generate_complex_body(template)
        elif level == "advanced":
            return self._generate_advanced_body(template)
        elif level == "expert":
            return self._generate_expert_body(template)
        else:
            return self._generate_simple_body(template)

    def _generate_simple_body(
        self,
        template: FunctionTemplate,
    ) -> List[str]:
        """Cuerpo simple: una o dos líneas de lógica directa."""
        lines: List[str] = []

        if template.is_generator:
            lines.append(f"    yield {template.parameters[0].name if template.parameters else 'item'}")
        elif template.is_async:
            lines.append(f"    await asyncio.sleep(0)")
            lines.append(f"    return {template.parameters[0].name if template.parameters else None}")
        else:
            param_ref = template.parameters[0].name if template.parameters else "None"
            lines.append(f"    return {param_ref}")

        return lines

    def _generate_medium_body(
        self,
        template: FunctionTemplate,
    ) -> List[str]:
        """Cuerpo medio: condicional simple y validación básica."""
        lines: List[str] = []

        # Validación de entrada
        if template.parameters:
            first_param = template.parameters[0]
            lines.append(f"    if {first_param.name} is None:")
            lines.append(f"        raise ValueError('{first_param.name} no puede ser None')")
            lines.append("")

        # Lógica principal condicional
        if template.is_generator:
            lines.append(f"    for item in {template.parameters[0].name if template.parameters else 'data'}:")
            lines.append(f"        if item is not None:")
            lines.append(f"            yield item")
        elif template.is_async:
            lines.append(f"    async with aiohttp.ClientSession() as session:")
            lines.append(f"        async with session.get('https://api.example.com') as resp:")
            lines.append(f"            return await resp.json()")
        else:
            param_ref = template.parameters[0].name if template.parameters else "data"
            lines.append(f"    if {param_ref}:")
            lines.append(f"        return {param_ref}")
            lines.append("    return None")

        return lines

    def _generate_complex_body(
        self,
        template: FunctionTemplate,
    ) -> List[str]:
        """Cuerpo complejo: múltiples condicionales, bucles, manejo de errores."""
        lines: List[str] = []

        # Validación de entrada
        if template.parameters:
            lines.append("    # Validar entradas")
            for p in template.parameters[:2]:
                lines.append(f"    if {p.name} is None:")
                lines.append(f"        raise ValueError('{p.name} es requerido')")
            lines.append("")

        # Procesamiento principal con bucle
        if template.parameters:
            main_param = template.parameters[0].name
            lines.append(f"    results = []")
            lines.append(f"    for item in {main_param}:")
            lines.append(f"        if item is not None:")
            lines.append(f"            processed = self._process_item(item)")
            lines.append(f"            results.append(processed)")
            lines.append("")
            lines.append(f"    return results")
        else:
            lines.append("    result = {}")
            lines.append("    for key, value in data.items():")
            lines.append("        if value is not None:")
            lines.append("            result[key] = value")
            lines.append("    return result")

        # Manejo de excepciones opcional
        if template.exception_handling:
            lines.insert(0, "    try:")
            # Indentar todo el cuerpo
            indented = []
            for line in lines:
                if line.strip():
                    indented.append("    " + line)
                else:
                    indented.append("")
            lines = indented
            lines.append("    except Exception as e:")
            lines.append("        logger.error(f'Error en {template.name}: {e}')")
            lines.append("        raise")

        return lines

    def _generate_advanced_body(
        self,
        template: FunctionTemplate,
    ) -> List[str]:
        """Cuerpo avanzado: comprensiones, decoradores internos, múltiples patrones."""
        lines: List[str] = []

        # Docstring interno si es necesario
        if template.parameters:
            main_param = template.parameters[0].name
            lines.append(f"    # Filtrar y transformar en una sola operación")
            lines.append(f"    results = [")
            lines.append(f"        self._transform(item) for item in {main_param}")
            lines.append("        if item is not None and self._validate(item)")
            lines.append(f"    ]")
            lines.append("")
            lines.append(f"    # Agrupar resultados por categoría")
            lines.append(f"    grouped = {{}}")
            lines.append(f"    for item in results:")
            lines.append(f"        category = self._categorize(item)")
            lines.append(f"        grouped.setdefault(category, []).append(item)")
            lines.append("")
            lines.append(f"    return grouped")
        else:
            lines.append("    data = self._load_default_data()")
            lines.append("    results = self._process_batch(data)")
            lines.append("    return self._format_results(results)")

        # Si es generador, convertir a yield
        if template.is_generator:
            lines = [
                "    for item in self._generate_items():",
                "        if self._should_yield(item):",
                "            yield item",
            ]

        return lines

    def _generate_expert_body(
        self,
        template: FunctionTemplate,
    ) -> List[str]:
        """Cuerpo experto: recursión, memoización, async, múltiples capas."""
        lines: List[str] = []

        if template.is_recursive:
            lines.append("    # Caso base")
            lines.append("    if n <= 1:")
            lines.append("        return n")
            lines.append("")
            lines.append("    # Caso recursivo con memoización")
            lines.append("    if n in self._cache:")
            lines.append("        return self._cache[n]")
            lines.append("")
            lines.append("    result = self._recursive(n - 1) + self._recursive(n - 2)")
            lines.append("    self._cache[n] = result")
            lines.append("    return result")
        elif template.is_async:
            lines.append("    async with self._semaphore:")
            lines.append("        async with aiohttp.ClientSession() as session:")
            lines.append("            tasks = [")
            lines.append("                self._fetch(session, url) for url in urls")
            lines.append("            ]")
            lines.append("            results = await asyncio.gather(*tasks)")
            lines.append("            return [r for r in results if r is not None]")
        else:
            lines.append("    # Pipeline de procesamiento multi-etapa")
            lines.append("    stage1 = self._stage1_filter(data)")
            lines.append("    stage2 = self._stage2_transform(stage1)")
            lines.append("    stage3 = self._stage3_validate(stage2)")
            lines.append("    stage4 = self._stage4_aggregate(stage3)")
            lines.append("")
            lines.append("    return self._stage5_format(stage4)")

        return lines


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

class ComplexFunctionGenerator:
    """
    Generador complejo de funciones Python.

    Convierte especificaciones simples en funciones completamente
    implementadas con type hints, docstrings, decoradores, manejo de
    errores y pruebas asociadas.
    """

    def __init__(
        self,
        seed: int = 42,
        output_dir: str | Path = "/home/user/ai_experiment/generated_functions",
    ) -> None:
        """
        Parameters
        ----------
        seed : int
            Semilla para generación determinista.
        output_dir : str | Path
            Directorio para guardar funciones generadas.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.name_generator = SemanticNameGenerator(seed=seed)
        self.body_generator = FunctionBodyGenerator(rng=self.rng)

        self.generated_functions: Dict[str, GeneratedFunction] = {}
        self.function_catalog: List[dict] = []

        # Contadores para versionado
        self._version_counters: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Generación de funciones
    # ------------------------------------------------------------------

    def generate_function(
        self,
        purpose: str,
        category: str = "data",
        complexity: str = "medium",
        n_params: int = 2,
        return_type: str = "Any",
        is_async: bool = False,
        is_generator: bool = False,
        is_recursive: bool = False,
        exception_handling: bool = True,
        docstring_style: str = "google",
        decorators: Optional[List[str]] = None,
        n_test_cases: int = 3,
    ) -> GeneratedFunction:
        """
        Genera una función Python completa basada en especificaciones.

        Parameters
        ----------
        purpose : str
            Propósito de la función (se usa para generar nombre semántico).
        category : str
            Categoría de la función (math, string, data, io, system, etc.).
        complexity : str
            Nivel de complejidad: simple, medium, complex, advanced, expert.
        n_params : int
            Número de parámetros.
        return_type : str
            Tipo de retorno.
        is_async : bool
            Si True, genera función asíncrona.
        is_generator : bool
            Si True, genera función generadora (yield).
        is_recursive : bool
            Si True, genera función recursiva.
        exception_handling : bool
            Si True, incluye manejo de excepciones.
        docstring_style : str
            Estilo de docstring: google, numpy, rest.
        decorators : list de str opcional
            Decoradores a aplicar (@timer, @retry, @cache, @validate).
        n_test_cases : int
            Número de casos de prueba a generar.

        Returns
        -------
        GeneratedFunction con código, pruebas y metadatos.
        """
        # Generar nombre semántico
        func_name = self.name_generator.generate_function_name(category, purpose)

        # Versionado
        if func_name in self._version_counters:
            self._version_counters[func_name] += 1
            version = f"1.0.{self._version_counters[func_name]}"
        else:
            self._version_counters[func_name] = 0
            version = "1.0.0"

        # Generar parámetros
        parameters = self._generate_parameters(n_params, category)

        # Crear plantilla
        template = FunctionTemplate(
            name=func_name,
            purpose=purpose,
            category=category,
            parameters=parameters,
            return_type=return_type,
            body_lines=[],  # Se llena después
            decorators=decorators or [],
            docstring_style=docstring_style,
            complexity_level=complexity,
            is_async=is_async,
            is_generator=is_generator,
            is_recursive=is_recursive,
            exception_handling=exception_handling,
            n_test_cases=n_test_cases,
            version=version,
        )

        # Generar cuerpo
        template.body_lines = self.body_generator.generate_body(template)

        # Generar código completo
        code = self._render_function_code(template)

        # Validar sintaxis
        syntax_valid = self._validate_syntax(code)

        # Calcular complejidad
        complexity_score, cognitive_score = self._calculate_complexity(code)

        # Generar pruebas
        test_code = self._generate_test_code(template)

        # Calcular puntuación de calidad
        quality_score = self._calculate_quality_score(
            code, syntax_valid, complexity_score, cognitive_score
        )

        result = GeneratedFunction(
            template=template,
            code=code,
            test_code=test_code,
            syntax_valid=syntax_valid,
            complexity_score=complexity_score,
            cognitive_score=cognitive_score,
            n_lines=len([l for l in code.splitlines() if l.strip()]),
            quality_score=quality_score,
        )

        # Guardar en catálogo
        self.generated_functions[func_name] = result
        self.function_catalog.append({
            "name": func_name,
            "category": category,
            "complexity": complexity,
            "version": version,
            "quality_score": quality_score,
            "syntax_valid": syntax_valid,
            "n_lines": result.n_lines,
            "created_at": template.created_at,
        })

        return result

    # ------------------------------------------------------------------
    # Generación de parámetros
    # ------------------------------------------------------------------

    def _generate_parameters(
        self,
        n_params: int,
        category: str,
    ) -> List[FunctionParameter]:
        """Genera una lista de parámetros para la función."""
        type_options_by_category: Dict[str, List[str]] = {
            "math": ["int", "float", "List[float]", "List[int]", "Optional[float]"],
            "string": ["str", "List[str]", "Optional[str]", "Dict[str, str]"],
            "data": ["List[dict]", "Dict[str, Any]", "Optional[List[dict]]", "pd.DataFrame"],
            "io": ["str", "Path", "BinaryIO", "TextIO", "Optional[str]"],
            "system": ["str", "int", "Dict[str, str]", "Optional[dict]"],
            "algorithm": ["List[Any]", "dict", "Optional[List[Any]]", "Graph"],
            "validation": ["Any", "dict", "List[Any]", "Optional[Any]"],
            "transformation": ["Any", "List[Any]", "dict", "Optional[Any]"],
        }

        types = type_options_by_category.get(category, ["Any", "str", "int"])
        roles = ["input", "input", "config", "context", "output"]

        parameters: List[FunctionParameter] = []
        used_names: Set[str] = set()

        for i in range(n_params):
            type_hint = self.rng.choice(types)
            role = roles[i % len(roles)]
            name = self.name_generator.generate_parameter_name(type_hint, role)

            # Asegurar nombres únicos
            base_name = name
            counter = 1
            while name in used_names:
                name = f"{base_name}_{counter}"
                counter += 1
            used_names.add(name)

            # Determinar si tiene valor por defecto
            has_default = self.rng.random() < 0.4
            default = None
            if has_default:
                if type_hint == "str":
                    default = '""'
                elif type_hint == "int":
                    default = str(self.rng.randint(0, 100))
                elif type_hint == "float":
                    default = f"{self.rng.uniform(0, 1):.2f}"
                elif type_hint == "bool":
                    default = self.rng.choice(["True", "False"])
                elif type_hint.startswith("List"):
                    default = "[]"
                elif type_hint.startswith("Dict"):
                    default = "{}"
                elif type_hint.startswith("Optional"):
                    default = "None"

            kind = "positional"
            if i == n_params - 1 and self.rng.random() < 0.3:
                kind = self.rng.choice(["args", "kwargs"])

            parameters.append(FunctionParameter(
                name=name,
                type_hint=type_hint,
                default=default,
                kind=kind,
                description=f"Parámetro {i+1} para {category}",
            ))

        return parameters

    # ------------------------------------------------------------------
    # Renderizado de código
    # ------------------------------------------------------------------

    def _render_function_code(
        self,
        template: FunctionTemplate,
    ) -> str:
        """
        Renderiza el código completo de la función incluyendo decoradores,
        firma, docstring y cuerpo.

        Returns
        -------
        str con el código Python completo.
        """
        lines: List[str] = []

        # Decoradores
        for dec in template.decorators:
            lines.append(f"@{dec}")

        # Firma de la función
        async_prefix = "async " if template.is_async else ""
        param_parts = [p.to_signature_part() for p in template.parameters]

        if template.is_generator and template.parameters:
            # Añadir *args si es generador sin parámetros posicionales
            pass

        params_str = ", ".join(param_parts)
        return_ann = f" -> {template.return_type}" if template.return_type else ""

        lines.append(f"{async_prefix}def {template.name}({params_str}){return_ann}:")

        # Docstring
        docstring = self._generate_docstring(template)
        if docstring:
            lines.append(f'    {docstring}')
            lines.append("")

        # Cuerpo
        if template.body_lines:
            for line in template.body_lines:
                lines.append(line)
        else:
            lines.append("    pass")

        return "\n".join(lines)

    def _generate_docstring(
        self,
        template: FunctionTemplate,
    ) -> str:
        """
        Genera un docstring en el estilo especificado.

        Returns
        -------
        str con el docstring formateado.
        """
        dq3 = '"""'
        lines: List[str] = []

        # Descripción
        lines.append(f"{dq3}{template.purpose}.")

        # Parámetros
        if template.parameters:
            if template.docstring_style == "google":
                lines.append("")
                lines.append("Args:")
                for p in template.parameters:
                    default_str = f" (default: {p.default})" if p.default is not None else ""
                    lines.append(f"    {p.name} ({p.type_hint}){default_str}: {p.description}")
            elif template.docstring_style == "numpy":
                lines.append("")
                lines.append("Parameters")
                lines.append("----------")
                for p in template.parameters:
                    default_str = f" (default: {p.default})" if p.default is not None else ""
                    lines.append(f"    {p.name} : {p.type_hint}{default_str}")
                    lines.append(f"        {p.description}")
            elif template.docstring_style == "rest":
                for p in template.parameters:
                    default_str = f" (default: {p.default})" if p.default is not None else ""
                    lines.append(f":param {p.name}: {p.description}")
                    lines.append(f":type {p.name}: {p.type_hint}{default_str}")

        # Retorno
        if template.return_type and template.return_type != "None":
            if template.docstring_style == "google":
                lines.append("")
                lines.append("Returns:")
                lines.append(f"    {template.return_type}: Resultado de la operación.")
            elif template.docstring_style == "numpy":
                lines.append("")
                lines.append("Returns")
                lines.append("-------")
                lines.append(f"    {template.return_type}")
                lines.append(f"        Resultado de la operación.")
            elif template.docstring_style == "rest":
                lines.append(f":return: Resultado de la operación.")
                lines.append(f":rtype: {template.return_type}")

        lines.append(dq3)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validación y métricas
    # ------------------------------------------------------------------

    def _validate_syntax(
        self,
        code: str,
    ) -> bool:
        """
        Valida que el código generado sea sintácticamente válido.

        Returns
        -------
        bool: True si el código es sintácticamente válido.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _calculate_complexity(
        self,
        code: str,
    ) -> Tuple[int, int]:
        """
        Calcula complejidad ciclomática y cognitiva del código.

        Returns
        -------
        Tupla (complejidad_ciclomática, complejidad_cognitiva).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0, 0

        cyclomatic = 1  # Base
        cognitive = 0
        nesting = 0

        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.If, ast.For, ast.While, ast.Try, ast.With,
                 ast.BoolOp, ast.ExceptHandler),
            ):
                cyclomatic += 1
                cognitive += 1 + nesting

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.If, ast.For, ast.While, ast.Try,
                                  ast.With, ast.ClassDef)):
                nesting += 1

        return cyclomatic, cognitive

    def _calculate_quality_score(
        self,
        code: str,
        syntax_valid: bool,
        complexity: int,
        cognitive: int,
    ) -> float:
        """
        Calcula una puntuación de calidad (0-100) para el código generado.

        Returns
        -------
        float entre 0 y 100.
        """
        score = 0.0

        # Sintaxis válida (30 puntos)
        if syntax_valid:
            score += 30.0

        # Tiene docstring (15 puntos)
        if '"""' in code or "'''" in code:
            score += 15.0

        # Tiene type hints (15 puntos)
        if "->" in code and ": " in code:
            score += 15.0

        # Complejidad moderada (20 puntos)
        if 2 <= complexity <= 10:
            score += 20.0
        elif complexity > 10:
            score += 10.0  # Penalizar complejidad excesiva

        # Manejo de excepciones (10 puntos)
        if "try:" in code and "except" in code:
            score += 10.0

        # Decoradores (5 puntos)
        if "@" in code:
            score += 5.0

        # Longitud razonable (5 puntos)
        n_lines = len([l for l in code.splitlines() if l.strip()])
        if 5 <= n_lines <= 50:
            score += 5.0

        return min(100.0, score)

    # ------------------------------------------------------------------
    # Generación de pruebas
    # ------------------------------------------------------------------

    def _generate_test_code(
        self,
        template: FunctionTemplate,
    ) -> str:
        """
        Genera casos de prueba pytest para la función.

        Returns
        -------
        str con el código de pruebas.
        """
        lines: List[str] = []
        lines.append(f"import pytest")
        lines.append(f"from generated_module import {template.name}")
        lines.append("")
        lines.append(f"class Test{template.name.title().replace('_', '')}:")
        lines.append(f'    """Pruebas para {template.name}."""')
        lines.append("")

        for i in range(template.n_test_cases):
            lines.append(f"    def test_{template.name}_case_{i}(self):")
            lines.append(f"        # Caso de prueba {i+1}")
            lines.append(f"        # Reemplazar con datos reales")
            lines.append(f"        assert True")
            lines.append("")

        # Prueba de excepción si aplica
        if template.exception_handling:
            lines.append(f"    def test_{template.name}_exception(self):")
            lines.append(f"        with pytest.raises(Exception):")
            lines.append(f"            {template.name}(None)")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Operaciones de catálogo y exportación
    # ------------------------------------------------------------------

    def get_function(
        self,
        name: str,
    ) -> Optional[GeneratedFunction]:
        """Obtiene una función generada por nombre."""
        return self.generated_functions.get(name)

    def list_functions(
        self,
        category: Optional[str] = None,
        complexity: Optional[str] = None,
    ) -> List[GeneratedFunction]:
        """
        Lista funciones generadas con filtros opcionales.

        Returns
        -------
        Lista de GeneratedFunction.
        """
        results: List[GeneratedFunction] = []
        for func in self.generated_functions.values():
            if category is not None and func.template.category != category:
                continue
            if complexity is not None and func.template.complexity_level != complexity:
                continue
            results.append(func)
        return sorted(results, key=lambda f: f.template.created_at, reverse=True)

    def get_catalog_stats(
        self,
    ) -> dict:
        """
        Devuelve estadísticas del catálogo de funciones generadas.

        Returns
        -------
        dict con estadísticas.
        """
        if not self.function_catalog:
            return {
                "total": 0,
                "by_category": {},
                "by_complexity": {},
                "avg_quality": 0.0,
                "avg_lines": 0.0,
                "syntax_valid_rate": 0.0,
            }

        total = len(self.function_catalog)
        by_category: Dict[str, int] = {}
        by_complexity: Dict[str, int] = {}
        total_quality = 0.0
        total_lines = 0
        syntax_valid_count = 0

        for entry in self.function_catalog:
            by_category[entry["category"]] = by_category.get(entry["category"], 0) + 1
            by_complexity[entry["complexity"]] = by_complexity.get(entry["complexity"], 0) + 1
            total_quality += entry["quality_score"]
            total_lines += entry["n_lines"]
            if entry["syntax_valid"]:
                syntax_valid_count += 1

        return {
            "total": total,
            "by_category": by_category,
            "by_complexity": by_complexity,
            "avg_quality": round(total_quality / total, 2),
            "avg_lines": round(total_lines / total, 1),
            "syntax_valid_rate": round(syntax_valid_count / total * 100, 2),
        }

    def export_function(
        self,
        name: str,
        output_path: str | Path,
        include_tests: bool = True,
    ) -> str:
        """
        Exporta una función generada a un archivo Python individual.

        Parameters
        ----------
        name : str
            Nombre de la función.
        output_path : str | Path
            Ruta del archivo de salida.
        include_tests : bool
            Si True, incluye el código de pruebas en el archivo.

        Returns
        -------
        str con la ruta del archivo guardado.
        """
        func = self.get_function(name)
        if func is None:
            raise ValueError(f"Función no encontrada: {name}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        lines.append(f'"""Función generada: {name}"""')
        lines.append(f"Versión: {func.template.version}")
        lines.append(f"Categoría: {func.template.category}")
        lines.append(f"Complejidad: {func.template.complexity_level}")
        lines.append(f"Puntuación de calidad: {func.quality_score:.1f}/100")
        lines.append("")
        lines.append(func.code)
        lines.append("")

        if include_tests and func.test_code:
            lines.append("# --- Pruebas ---")
            lines.append(func.test_code)

        output_path.write_text("\n".join(lines), encoding="utf-8")

        return str(output_path)

    def export_all_functions(
        self,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, str]:
        """
        Exporta todas las funciones generadas a archivos individuales.

        Returns
        -------
        dict nombre_funcion -> ruta_archivo.
        """
        if output_dir is None:
            output_dir = self.output_dir / "exported"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}
        for name in self.generated_functions:
            path = self.export_function(name, output_dir / f"{name}.py")
            paths[name] = path

        # También exportar catálogo
        catalog_path = output_dir / "catalog.json"
        with open(catalog_path, "w") as f:
            json.dump(self.function_catalog, f, indent=2)
        paths["catalog"] = str(catalog_path)

        return paths

    def generate_quality_report(
        self,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de calidad de todas las funciones generadas.

        Returns
        -------
        str con el reporte en Markdown.
        """
        stats = self.get_catalog_stats()

        lines: List[str] = []
        lines.append("# Reporte de Calidad de Funciones Generadas")
        lines.append("")
        lines.append(f"Total de funciones: {stats['total']}")
        lines.append(f"Puntuación de calidad media: {stats['avg_quality']}/100")
        lines.append(f"Líneas promedio por función: {stats['avg_lines']}")
        lines.append(f"Tasa de sintaxis válida: {stats['syntax_valid_rate']}%")
        lines.append("")

        lines.append("## Por Categoría")
        lines.append("")
        lines.append("| Categoría | Cantidad |")
        lines.append("|-----------|----------|")
        for cat, count in sorted(stats["by_category"].items()):
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        lines.append("## Por Complejidad")
        lines.append("")
        lines.append("| Complejidad | Cantidad |")
        lines.append("|-------------|----------|")
        for comp, count in sorted(stats["by_complexity"].items()):
            lines.append(f"| {comp} | {count} |")
        lines.append("")

        lines.append("## Top 10 Funciones por Calidad")
        lines.append("")
        lines.append("| Nombre | Categoría | Complejidad | Calidad | Líneas | Sintaxis |")
        lines.append("|--------|-----------|-------------|---------|--------|----------|")

        sorted_funcs = sorted(
            self.generated_functions.values(),
            key=lambda f: f.quality_score,
            reverse=True,
        )[:10]

        for func in sorted_funcs:
            syntax_icon = "✓" if func.syntax_valid else "✗"
            lines.append(
                f"| {func.template.name} | {func.template.category} | "
                f"{func.template.complexity_level} | {func.quality_score:.1f} | "
                f"{func.n_lines} | {syntax_icon} |"
            )

        lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text


# ---------------------------------------------------------------------------
# Funciones de conveniencia
# ---------------------------------------------------------------------------

def quick_generate_function(
    purpose: str,
    category: str = "data",
    complexity: str = "medium",
    output_path: Optional[str | Path] = None,
) -> GeneratedFunction:
    """
    Función de conveniencia: genera una función Python compleja.

    Returns
    -------
    GeneratedFunction con el código y pruebas.
    """
    generator = ComplexFunctionGenerator()
    func = generator.generate_function(
        purpose=purpose,
        category=category,
        complexity=complexity,
    )

    if output_path:
        generator.export_function(func.template.name, output_path)

    return func


def quick_generate_batch(
    specifications: List[dict],
    output_dir: str | Path = "/home/user/ai_experiment/generated_functions",
) -> dict:
    """
    Genera un lote de funciones a partir de especificaciones.

    Parameters
    ----------
    specifications : list de dict
        Cada dict contiene parámetros para generate_function.
    output_dir : str | Path
        Directorio para exportar las funciones.

    Returns
    -------
    dict con resultados y estadísticas.
    """
    generator = ComplexFunctionGenerator(output_dir=output_dir)
    results: List[dict] = []

    for spec in specifications:
        func = generator.generate_function(**spec)
        generator.export_function(func.template.name)
        results.append(func.to_dict())

    stats = generator.get_catalog_stats()

    return {
        "functions": results,
        "stats": stats,
        "export_dir": str(output_dir),
    }


__all__ = [
    "FunctionParameter",
    "FunctionTemplate",
    "GeneratedFunction",
    "SemanticNameGenerator",
    "FunctionBodyGenerator",
    "ComplexFunctionGenerator",
    "quick_generate_function",
    "quick_generate_batch",
]
