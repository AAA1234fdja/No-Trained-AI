#!/usr/bin/env python3
"""
code_generator.py — Generador de código Python sintético para pruebas y entrenamiento del modelo 800M.

Este módulo genera fragmentos de código Python válidos que pueden usarse como:
  - Datos de prueba para evaluar la capacidad del modelo de completar código.
  - Ejemplos few-shot para mejorar la generación sin entrenamiento.
  - Casos de prueba para verificar sintaxis correcta (def, class, return, loops, etc.).
  - Prompts de few-shot en el motor de inferencia.

Todos los fragmentos generados son sintácticamente válidos y cubren patrones
comunes de Python: funciones, clases, bucles, condicionales, manejo de
excepciones, comprensiones de lista, decoradores, etc.

No requiere ejecución real de Python; solo genera texto de código.
"""

from __future__ import annotations

import random
import string
from typing import List, Optional

# ---------------------------------------------------------------------------
# Plantillas de código Python
# ---------------------------------------------------------------------------

FUNCTION_TEMPLATES: List[str] = [
    "def {name}({params}):\n    {body}\n    return {ret}\n",
    "def {name}({params}):\n    \"\"\"{doc}\"\"\"\n    {body}\n    return {ret}\n",
    "async def {name}({params}):\n    {body}\n    return {ret}\n",
    "def {name}({params}, *args, **kwargs):\n    {body}\n    return {ret}\n",
]

CLASS_TEMPLATES: List[str] = [
    "class {name}:\n    def __init__(self, {params}):\n        {body}\n",
    "class {name}({parent}):\n    def __init__(self, {params}):\n        {body}\n        super().__init__({super_params})\n",
    "class {name}:\n    {methods}\n",
]

LOOP_TEMPLATES: List[str] = [
    "for {var} in {iterable}:\n    {body}\n",
    "while {condition}:\n    {body}\n",
    "for {var} in range({start}, {end}):\n    {body}\n",
    "for {var} in enumerate({iterable}):\n    {body}\n",
]

CONDITIONAL_TEMPLATES: List[str] = [
    "if {condition}:\n    {body}\n",
    "if {condition}:\n    {body_if}\nelse:\n    {body_else}\n",
    "if {condition}:\n    {body_if}\nelif {condition2}:\n    {body_elif}\nelse:\n    {body_else}\n",
]

EXCEPTION_TEMPLATES: List[str] = [
    "try:\n    {body}\nexcept {exception}:\n    {handler}\n",
    "try:\n    {body}\nexcept {exception} as e:\n    {handler}\nfinally:\n    {final}\n",
    "try:\n    {body}\nfinally:\n    {final}\n",
]

LIST_COMP_TEMPLATES: List[str] = [
    "{var} = [{expr} for {item} in {iterable}]\n",
    "{var} = [{expr} for {item} in {iterable} if {condition}]\n",
    "{var} = {func}({expr} for {item} in {iterable})\n",
]

DECORATOR_TEMPLATES: List[str] = [
    "@{decorator}\ndef {name}({params}):\n    {body}\n    return {ret}\n",
    "@{decorator}\n@{decorator2}\ndef {name}({params}):\n    {body}\n    return {ret}\n",
]

IMPORT_TEMPLATES: List[str] = [
    "import {module}\n",
    "from {module} import {name}\n",
    "from {module} import {name} as {alias}\n",
    "import {module} as {alias}\n",
]

# ---------------------------------------------------------------------------
# Generadores de nombres e identificadores
# ---------------------------------------------------------------------------

def random_identifier(
    min_len: int = 3,
    max_len: int = 10,
    seed: Optional[int] = None,
) -> str:
    """Genera un identificador Python aleatorio válido."""
    rng = random.Random(seed)
    length = rng.randint(min_len, max_len)
    first = rng.choice(string.ascii_letters + "_")
    rest = "".join(rng.choice(string.ascii_letters + string.digits + "_")
                   for _ in range(length - 1))
    return first + rest


def random_variable_name(
    kind: str = "var",
    seed: Optional[int] = None,
) -> str:
    """Genera un nombre de variable semánticamente plausible."""
    rng = random.Random(seed)
    nouns = ["data", "result", "value", "item", "list", "dict", "set",
             "array", "matrix", "vector", "config", "params", "args",
             "kwargs", "response", "request", "response", "output",
             "input", "buffer", "cache", "index", "count", "total",
             "average", "sum", "max", "min", "count", "flag", "mode",
             "state", "context", "session", "user", "admin", "guest"]
    adjectives = ["final", "initial", "temp", "temp_var", "raw",
                  "clean", "processed", "sorted", "filtered", "unique",
                  "default", "custom", "global", "local", "static",
                  "dynamic", "active", "inactive", "pending", "completed"]
    kind_words = {
        "var": nouns,
        "func": ["process", "compute", "calculate", "analyze", "validate",
                 "transform", "filter", "map", "reduce", "sort", "search",
                 "load", "save", "read", "write", "create", "update",
                 "delete", "get", "set", "check", "test", "run"],
        "class": ["Data", "Result", "Processor", "Analyzer", "Manager",
                  "Handler", "Controller", "Service", "Client", "Server",
                  "Model", "View", "Builder", "Factory", "Generator",
                  "Validator", "Parser", "Renderer"],
        "module": ["utils", "helpers", "constants", "config", "settings",
                   "models", "views", "controllers", "services",
                   "database", "api", "client", "server"],
    }
    pool = kind_words.get(kind, nouns)
    adj = rng.choice(adjectives)
    noun = rng.choice(pool)
    return f"{adj}_{noun}"


# ---------------------------------------------------------------------------
# Generadores de fragmentos de código
# ---------------------------------------------------------------------------

def generate_function(
    name: Optional[str] = None,
    params: Optional[str] = None,
    body: Optional[str] = None,
    ret: str = "None",
    async_fn: bool = False,
    docstring: bool = False,
    seed: Optional[int] = None,
) -> str:
    """
    Genera una definición de función Python.

    Parameters
    ----------
    name : str opcional
        Nombre de la función. Se genera aleatoriamente si no se proporciona.
    params : str opcional
        Parámetros de la función. Default: "data, config".
    body : str opcional
        Cuerpo de la función. Default: cuerpo genérico.
    ret : str
        Expresión de retorno. Default: "None".
    async_fn : bool
        Si generar función async.
    docstring : bool
        Si incluir docstring.
    seed : int opcional
        Semilla para reproducibilidad.

    Returns
    -------
    str con el código de la función.
    """
    rng = random.Random(seed)
    if name is None:
        name = random_variable_name("func", seed=rng.randint(0, 10**6))
    if params is None:
        params = "data, config=None"
    if body is None:
        body_templates = [
            "result = process_data(data)",
            "result = transform(data, config)",
            "result = analyze(data)",
            "result = compute(data)",
            "result = validate(data)",
        ]
        body = rng.choice(body_templates)

    template = rng.choice(FUNCTION_TEMPLATES)
    if async_fn:
        template = "async def {name}({params}):\n    {body}\n    return {ret}\n"

    if docstring:
        doc = f"Process {name} with given parameters."
    else:
        doc = ""

    code = template.format(
        name=name,
        params=params,
        body=body,
        ret=ret,
        doc=doc,
    )
    return code


def generate_class(
    name: Optional[str] = None,
    parent: Optional[str] = None,
    methods: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Genera una definición de clase Python.

    Returns
    -------
    str con el código de la clase.
    """
    rng = random.Random(seed)
    if name is None:
        name = random_variable_name("class", seed=rng.randint(0, 10**6))

    if methods is None:
        methods = [
            "def process(self, data):\n        return self.transform(data)",
            "def validate(self, data):\n        return True",
            "def get_config(self):\n        return self.config",
        ]

    methods_code = "\n    ".join(methods)

    if parent:
        template = rng.choice([t for t in CLASS_TEMPLATES if "{parent}" in t])
        code = template.format(
            name=name,
            parent=parent,
            params="config=None",
            body="self.config = config",
            super_params="config",
            methods=methods_code,
        )
    else:
        template = "class {name}:\n    {methods}\n"
        code = template.format(name=name, methods=methods_code)

    return code


def generate_loop(
    loop_type: str = "for",
    seed: Optional[int] = None,
) -> str:
    """
    Genera un bucle Python (for o while).

    Returns
    -------
    str con el código del bucle.
    """
    rng = random.Random(seed)
    template = rng.choice(LOOP_TEMPLATES)

    if loop_type == "for":
        var = random_variable_name("var", seed=rng.randint(0, 10**6))
        iterable = rng.choice(["data", "items", "range(10)", "config.items()",
                               "enumerate(items)"])
        body = f"process({var})"
    else:
        var = "condition"
        iterable = ""
        body = "process()"

    code = template.format(
        var=var,
        iterable=iterable,
        body=body,
        condition="x > 0",
        start="0",
        end="10",
    )
    return code


def generate_conditional(
    seed: Optional[int] = None,
) -> str:
    """
    Genera una estructura condicional if/elif/else.

    Returns
    -------
    str con el código condicional.
    """
    rng = random.Random(seed)
    template = rng.choice(CONDITIONAL_TEMPLATES)
    code = template.format(
        condition="x > 0",
        condition2="x < 0",
        body_if="return 'positive'",
        body_else="return 'non-positive'",
        body_elif="return 'negative'",
    )
    return code


def generate_exception_handling(
    seed: Optional[int] = None,
) -> str:
    """
    Genera un bloque try/except/finally.

    Returns
    -------
    str con el código de manejo de excepciones.
    """
    rng = random.Random(seed)
    template = rng.choice(EXCEPTION_TEMPLATES)
    code = template.format(
        body="result = risky_operation()",
        exception="Exception",
        handler="log_error(e)",
        final="cleanup()",
    )
    return code


def generate_list_comprehension(
    seed: Optional[int] = None,
) -> str:
    """
    Genera una comprensión de lista.

    Returns
    -------
    str con el código de comprensión.
    """
    rng = random.Random(seed)
    template = rng.choice(LIST_COMP_TEMPLATES)
    code = template.format(
        var="results",
        expr="x * 2",
        item="x",
        iterable="data",
        condition="x > 0",
        func="sum",
    )
    return code


def generate_decorator(
    seed: Optional[int] = None,
) -> str:
    """
    Genera una función con decorador.

    Returns
    -------
    str con el código con decorador.
    """
    rng = random.Random(seed)
    template = rng.choice(DECORATOR_TEMPLATES)
    code = template.format(
        decorator="timing",
        decorator2="retry",
        name="process_data",
        params="data",
        body="return transform(data)",
        ret="result",
    )
    return code


def generate_import(
    seed: Optional[int] = None,
) -> str:
    """
    Genera una sentencia import.

    Returns
    -------
    str con el código de importación.
    """
    rng = random.Random(seed)
    template = rng.choice(IMPORT_TEMPLATES)
    module = rng.choice(["os", "sys", "json", "math", "collections",
                         "itertools", "functools", "typing", "ast",
                         "numpy", "pandas"])
    name = random_variable_name("var", seed=rng.randint(0, 10**6))
    alias = random_variable_name("var", seed=rng.randint(0, 10**6))
    code = template.format(module=module, name=name, alias=alias)
    return code


# ---------------------------------------------------------------------------
# Generador de programas completos
# ---------------------------------------------------------------------------

def generate_complete_program(
    n_functions: int = 3,
    n_classes: int = 1,
    include_imports: bool = True,
    include_main: bool = True,
    seed: Optional[int] = None,
) -> str:
    """
    Genera un programa Python completo con múltiples funciones, clases,
    importaciones y un bloque main.

    Returns
    -------
    str con el programa completo.
    """
    rng = random.Random(seed)
    parts: List[str] = []

    if include_imports:
        for _ in range(rng.randint(1, 3)):
            parts.append(generate_import(seed=rng.randint(0, 10**6)))
        parts.append("")

    for _ in range(n_functions):
        parts.append(generate_function(seed=rng.randint(0, 10**6)))
        parts.append("")

    for _ in range(n_classes):
        parts.append(generate_class(seed=rng.randint(0, 10**6)))
        parts.append("")

    if include_main:
        main_code = (
            "if __name__ == '__main__':\n"
            "    data = [1, 2, 3, 4, 5]\n"
            "    result = process_data(data)\n"
            "    print(result)\n"
        )
        parts.append(main_code)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generador de casos de prueba de sintaxis
# ---------------------------------------------------------------------------

def generate_syntax_test_cases(
    n_cases: int = 20,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Genera una lista de fragmentos de código Python para pruebas de sintaxis.

    Cada caso es un fragmento válido que cubre un patrón específico.
    """
    rng = random.Random(seed)
    cases: List[str] = []

    case_generators = [
        lambda: generate_function(seed=rng.randint(0, 10**6)),
        lambda: generate_class(seed=rng.randint(0, 10**6)),
        lambda: generate_loop(seed=rng.randint(0, 10**6)),
        lambda: generate_conditional(seed=rng.randint(0, 10**6)),
        lambda: generate_exception_handling(seed=rng.randint(0, 10**6)),
        lambda: generate_list_comprehension(seed=rng.randint(0, 10**6)),
        lambda: generate_decorator(seed=rng.randint(0, 10**6)),
        lambda: generate_import(seed=rng.randint(0, 10**6)),
    ]

    for _ in range(n_cases):
        gen = rng.choice(case_generators)
        cases.append(gen())

    return cases


# ---------------------------------------------------------------------------
# Generador de prompts de few-shot para código
# ---------------------------------------------------------------------------

def build_few_shot_code_prompt(
    incomplete_code: str,
    examples: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Construye un prompt de few-shot para completado de código.

    Parameters
    ----------
    incomplete_code : str
        Código incompleto que el modelo debe completar.
    examples : lista de (incompleto, completo) opcional.

    Returns
    -------
    str con el prompt formateado.
    """
    if examples is None:
        examples = [
            (
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return ",
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            ),
            (
                "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while ",
                "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            ),
            (
                "def analyze_data(data):\n    total = sum(data)\n    count = len(data)\n    return ",
                "def analyze_data(data):\n    total = sum(data)\n    count = len(data)\n    return total / count if count > 0 else 0",
            ),
        ]

    lines: List[str] = [
        "Complete the following Python code snippets.",
        "",
    ]

    for i, (inc, comp) in enumerate(examples):
        lines.append(f"Example {i + 1}:")
        lines.append(f"Input:\n{inc}")
        lines.append(f"Output:\n{comp}")
        lines.append("")

    lines.append(f"Example {len(examples) + 1}:")
    lines.append(f"Input:\n{incomplete_code}")
    lines.append("Output:")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validador sintáctico básico (sin ejecutar)
# ---------------------------------------------------------------------------

def basic_syntax_check(code: str) -> Tuple[bool, Optional[str]]:
    """
    Verificación sintáctica básica sin ejecutar el código.

    Checks:
        - Paréntesis, corchetes y llaves balanceados.
        - Comienzo de líneas con palabras clave válidas.
        - No hay caracteres de control extraños.

    Returns
    -------
    (bool, str o None): (True, None) si parece válido, (False, razón) si no.
    """
    # Verificar balance de delimitadores
    opens = "([{"
    closes = ")]}"
    match = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []

    for ch in code:
        if ch in opens:
            stack.append(ch)
        elif ch in closes:
            if not stack or stack[-1] != match[ch]:
                return False, f"Delimitador desbalanceado: {ch}"
            stack.pop()

    if stack:
        return False, f"Delimitadores sin cerrar: {stack}"

    # Verificar que no haya caracteres de control
    for ch in code:
        if ord(ch) < 32 and ch not in ("\n", "\t", "\r"):
            return False, f"Carácter de control encontrado: ord={ord(ch)}"

    return True, None


__all__ = [
    "random_identifier",
    "random_variable_name",
    "generate_function",
    "generate_class",
    "generate_loop",
    "generate_conditional",
    "generate_exception_handling",
    "generate_list_comprehension",
    "generate_decorator",
    "generate_import",
    "generate_complete_program",
    "generate_syntax_test_cases",
    "build_few_shot_code_prompt",
    "basic_syntax_check",
    "FUNCTION_TEMPLATES",
    "CLASS_TEMPLATES",
    "LOOP_TEMPLATES",
    "CONDITIONAL_TEMPLATES",
    "EXCEPTION_TEMPLATES",
    "LIST_COMP_TEMPLATES",
    "DECORATOR_TEMPLATES",
    "IMPORT_TEMPLATES",
]
