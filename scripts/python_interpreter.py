#!/usr/bin/env python3
"""
python_interpreter.py — Intérprete básico de Python para validar código generado.

Este módulo proporciona un analizador sintáctico y semántico ligero para
verificar que el código Python generado por el modelo es sintácticamente
válido y sigue patrones razonables, sin necesidad de ejecutar el código
(lo cual podría ser inseguro).

Incluye:
  - Análisis sintáctico usando el módulo ast de Python.
  - Detección de errores comunes (variables no definidas, imports faltantes).
  - Extracción de definiciones de funciones y clases.
  - Medición de complejidad ciclomática aproximada.
  - Puntuación de calidad del código generado.
  - Comparación entre código esperado y generado (similitud de AST).

Todas las operaciones son de solo lectura (no se ejecuta el código).
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SyntaxCheckResult:
    """Resultado de una verificación sintáctica."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ast_tree: Optional[ast.AST] = None

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


@dataclass
class CodeMetrics:
    """Métricas de un fragmento de código Python."""
    n_lines: int = 0
    n_functions: int = 0
    n_classes: int = 0
    n_imports: int = 0
    n_calls: int = 0
    complexity: int = 1  # complejidad ciclomática mínima
    has_main_block: bool = False
    has_docstrings: bool = False
    type_annotations: int = 0

    def to_dict(self) -> dict:
        return {
            "n_lines": self.n_lines,
            "n_functions": self.n_functions,
            "n_classes": self.n_classes,
            "n_imports": self.n_imports,
            "n_calls": self.n_calls,
            "complexity": self.complexity,
            "has_main_block": self.has_main_block,
            "has_docstrings": self.has_docstrings,
            "type_annotations": self.type_annotations,
        }


class PythonSyntaxChecker:
    """
    Verificador de sintaxis Python usando el módulo ast.

    No ejecuta el código, solo lo analiza estáticamente.
    """

    def __init__(self) -> None:
        self.builtins: Set[str] = set(dir(__builtins__))  # type: ignore
        self.common_modules: Set[str] = {
            "os", "sys", "json", "math", "collections", "itertools",
            "functools", "typing", "ast", "re", "datetime", "pathlib",
            "numpy", "pandas", "matplotlib", "scipy", "sklearn",
        }

    def check(
        self,
        code: str,
    ) -> SyntaxCheckResult:
        """
        Verifica la sintaxis de un fragmento de código Python.

        Returns
        -------
        SyntaxCheckResult con errores, advertencias y el árbol AST.
        """
        result = SyntaxCheckResult(is_valid=False)

        try:
            tree = ast.parse(code)
            result.ast_tree = tree
            result.is_valid = True
        except SyntaxError as e:
            result.errors.append(
                f"Error de sintaxis en línea {e.lineno}: {e.msg}"
            )
            return result
        except Exception as e:
            result.errors.append(f"Error al parsear: {e}")
            return result

        # Verificaciones semánticas básicas
        self._check_undefined_names(tree, result)
        self._check_unused_imports(tree, result)
        self._check_empty_blocks(tree, result)

        return result

    def _check_undefined_names(
        self,
        tree: ast.AST,
        result: SyntaxCheckResult,
    ) -> None:
        """Detecta nombres usados pero no definidos (aproximado)."""
        defined_names: Set[str] = set()
        used_names: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
                for arg in node.args.args:
                    defined_names.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    defined_names.add(node.module.split(".")[0])
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)

        # Filtrar builtins y módulos comunes
        used_names -= self.builtins
        used_names -= self.common_modules

        undefined = used_names - defined_names
        if undefined:
            result.warnings.append(
                f"Posibles nombres no definidos: {sorted(undefined)[:10]}"
            )

    def _check_unused_imports(
        self,
        tree: ast.AST,
        result: SyntaxCheckResult,
    ) -> None:
        """Detecta importaciones posiblemente no usadas."""
        imported: Set[str] = set()
        used: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
            elif isinstance(node, ast.Name):
                used.add(node.id)

        unused = imported - used
        if unused:
            result.warnings.append(
                f"Posibles imports no usados: {sorted(unused)[:10]}"
            )

    def _check_empty_blocks(
        self,
        tree: ast.AST,
        result: SyntaxCheckResult,
    ) -> None:
        """Detecta bloques vacíos que podrían indicar código incompleto."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If,
                                  ast.For, ast.While, ast.Try, ast.With)):
                if not node.body:
                    result.warnings.append(
                        f"Bloque vacío en {type(node).__name__} en línea {node.lineno}"
                    )


def calculate_code_metrics(
    code: str,
) -> CodeMetrics:
    """
    Calcula métricas de un fragmento de código Python.

    Returns
    -------
    CodeMetrics con conteo de líneas, funciones, clases, imports, etc.
    """
    metrics = CodeMetrics()
    lines = code.splitlines()
    metrics.n_lines = len([l for l in lines if l.strip()])

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return metrics

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            metrics.n_functions += 1
            metrics.complexity += 1  # cada función añade complejidad base
        elif isinstance(node, ast.ClassDef):
            metrics.n_classes += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            metrics.n_imports += 1
        elif isinstance(node, ast.Call):
            metrics.n_calls += 1
        elif isinstance(node, (ast.If, ast.For, ast.While,
                                ast.Try, ast.With, ast.BoolOp)):
            metrics.complexity += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                metrics.has_docstrings = True

    # Verificar bloque main
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare):
                if isinstance(test.left, ast.Name) and test.left.id == "__name__":
                    metrics.has_main_block = True

    return metrics


def score_code_quality(
    code: str,
) -> Dict[str, object]:
    """
    Calcula una puntuación de calidad (0-100) para código Python generado.

    Criterios:
        - Sintaxis válida (30 puntos)
        - Tiene funciones o clases (20 puntos)
        - Tiene imports apropiados (10 puntos)
        - No tiene bloques vacíos (15 puntos)
        - Complejidad moderada (15 puntos)
        - Tiene docstrings (10 puntos)
    """
    checker = PythonSyntaxChecker()
    check_result = checker.check(code)
    metrics = calculate_code_metrics(code)

    score = 0.0
    details: Dict[str, object] = {}

    # Sintaxis válida
    if check_result.is_valid:
        score += 30.0
        details["syntax_valid"] = True
    else:
        details["syntax_valid"] = False
        details["syntax_errors"] = check_result.errors

    # Tiene funciones o clases
    if metrics.n_functions > 0 or metrics.n_classes > 0:
        score += 20.0
    details["has_functions_or_classes"] = (
        metrics.n_functions > 0 or metrics.n_classes > 0
    )
    details["n_functions"] = metrics.n_functions
    details["n_classes"] = metrics.n_classes

    # Imports
    if metrics.n_imports > 0:
        score += 10.0
    details["n_imports"] = metrics.n_imports

    # Sin bloques vacíos
    if not any("bloque vacío" in w for w in check_result.warnings):
        score += 15.0
    details["no_empty_blocks"] = True

    # Complejidad moderada (entre 2 y 20)
    if 2 <= metrics.complexity <= 20:
        score += 15.0
    details["complexity"] = metrics.complexity
    details["complexity_score"] = 15.0 if 2 <= metrics.complexity <= 20 else 0.0

    # Docstrings
    if metrics.has_docstrings:
        score += 10.0
    details["has_docstrings"] = metrics.has_docstrings

    return {
        "score": round(score, 1),
        "max_score": 100.0,
        "percentage": round(score / 100.0 * 100, 1),
        "details": details,
        "metrics": metrics.to_dict(),
        "syntax": check_result.to_dict(),
    }


def compare_code_similarity(
    code_a: str,
    code_b: str,
) -> Dict[str, object]:
    """
    Compara dos fragmentos de código por similitud de estructura AST.

    Returns
    -------
    dict con puntuación de similitud (0-1) y diferencias detectadas.
    """
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
    except SyntaxError:
        return {
            "similarity": 0.0,
            "error": "Uno de los fragmentos tiene errores de sintaxis",
        }

    # Contar tipos de nodos en cada árbol
    def count_node_types(tree: ast.AST) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in ast.walk(tree):
            node_type = type(node).__name__
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    counts_a = count_node_types(tree_a)
    counts_b = count_node_types(tree_b)

    # Similitud de coseno entre vectores de tipos de nodos
    all_types = set(counts_a.keys()) | set(counts_b.keys())
    dot = sum(counts_a.get(t, 0) * counts_b.get(t, 0) for t in all_types)
    norm_a = math.sqrt(sum(v ** 2 for v in counts_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in counts_b.values()))

    if norm_a == 0 or norm_b == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = dot / (norm_a * norm_b)

    # Diferencias en conteos
    differences: Dict[str, dict] = {}
    for t in all_types:
        ca = counts_a.get(t, 0)
        cb = counts_b.get(t, 0)
        if ca != cb:
            differences[t] = {"code_a": ca, "code_b": cb, "diff": cb - ca}

    return {
        "similarity": round(cosine_sim, 4),
        "similarity_percentage": round(cosine_sim * 100, 2),
        "node_type_counts_a": counts_a,
        "node_type_counts_b": counts_b,
        "differences": differences,
        "n_differences": len(differences),
    }


def extract_function_definitions(
    code: str,
) -> List[Dict[str, object]]:
    """
    Extrae todas las definiciones de función de un fragmento de código.

    Returns
    -------
    lista de diccionarios con nombre, argumentos, línea y cuerpo (texto).
    """
    functions: List[Dict[str, object]] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return functions

    lines = code.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            decorators = [
                ast.unparse(d) if hasattr(ast, "unparse") else str(d)
                for d in node.decorator_list
            ]
            # Extraer el cuerpo como texto (aproximado por líneas)
            body_lines = []
            for i in range(node.lineno - 1, min(node.end_lineno or node.lineno, len(lines))):
                body_lines.append(lines[i])

            functions.append({
                "name": node.name,
                "args": args,
                "decorators": decorators,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "body_text": "\n".join(body_lines),
            })

    return functions


def extract_class_definitions(
    code: str,
) -> List[Dict[str, object]]:
    """
    Extrae todas las definiciones de clase de un fragmento de código.

    Returns
    -------
    lista de diccionarios con nombre, clases padre, métodos y línea.
    """
    classes: List[Dict[str, object]] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return classes

    lines = code.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [
                ast.unparse(b) if hasattr(ast, "unparse") else str(b)
                for b in node.bases
            ]
            methods = [
                n.name for n in node.body if isinstance(n, ast.FunctionDef)
            ]

            classes.append({
                "name": node.name,
                "bases": bases,
                "methods": methods,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "n_methods": len(methods),
            })

    return classes


__all__ = [
    "SyntaxCheckResult",
    "CodeMetrics",
    "PythonSyntaxChecker",
    "check",
    "calculate_code_metrics",
    "score_code_quality",
    "compare_code_similarity",
    "extract_function_definitions",
    "extract_class_definitions",
]

# Alias para mantener compatibilidad con imports existentes
check = PythonSyntaxChecker().check
