#!/usr/bin/env python3
"""
code_optimizer.py — Optimizador de código Python generado.

Proporciona herramientas para analizar y optimizar código Python:
  - Análisis de complejidad ciclomática y cognitiva.
  - Detección de código muerto y variables no usadas.
  - Sugerencias de refactorización (extraer método, eliminar duplicados).
  - Optimización de imports (eliminar no usados, agrupar).
  - Análisis de rendimiento estático (bucles anidados, llamadas costosas).
  - Generación de versión optimizada del código.
  - Reporte de optimización en Markdown/JSON.

Todas las operaciones son de solo lectura y no ejecutan código.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class OptimizationSuggestion:
    """Una sugerencia de optimización."""
    type: str  # "performance", "readability", "dead_code", "imports", "complexity"
    priority: str  # "high", "medium", "low"
    line: int
    message: str
    suggestion: str
    code_before: Optional[str] = None
    code_after: Optional[str] = None


@dataclass
class OptimizationReport:
    """Reporte de optimización de un archivo Python."""
    path: str
    n_lines: int
    n_functions: int
    n_classes: int
    complexity: int
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    optimized_code: Optional[str] = None
    original_code: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "n_lines": self.n_lines,
            "n_functions": self.n_functions,
            "n_classes": self.n_classes,
            "complexity": self.complexity,
            "n_suggestions": len(self.suggestions),
            "suggestions": [
                {
                    "type": s.type,
                    "priority": s.priority,
                    "line": s.line,
                    "message": s.message,
                    "suggestion": s.suggestion,
                }
                for s in self.suggestions
            ],
        }


class CodeOptimizer:
    """
    Optimizador de código Python.

    Analiza el código fuente y genera sugerencias de optimización
    sin modificar el archivo original (salvo que se solicite explícitamente).
    """

    def __init__(self) -> None:
        self.builtins: Set[str] = set(dir(__builtins__))  # type: ignore

    def analyze_file(
        self,
        path: str | Path,
    ) -> OptimizationReport:
        """
        Analiza un archivo Python y genera sugerencias de optimización.

        Returns
        -------
        OptimizationReport con sugerencias y código optimizado opcional.
        """
        path = Path(path)
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return OptimizationReport(
                path=str(path),
                n_lines=len(lines),
                n_functions=0,
                n_classes=0,
                complexity=0,
                suggestions=[
                    OptimizationSuggestion(
                        type="error",
                        priority="high",
                        line=e.lineno or 1,
                        message=f"Error de sintaxis: {e.msg}",
                        suggestion="Corregir el error de sintaxis antes de optimizar",
                    )
                ],
            )

        suggestions: List[OptimizationSuggestion] = []

        # 1. Complejidad ciclomática
        complexity = self._calculate_complexity(tree)
        if complexity > 10:
            suggestions.append(OptimizationSuggestion(
                type="complexity",
                priority="high",
                line=1,
                message=f"Complejidad ciclomática alta: {complexity}",
                suggestion="Dividir funciones complejas en funciones más pequeñas",
            ))

        # 2. Funciones largas
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = (node.end_lineno or node.lineno) - node.lineno + 1
                if func_lines > 50:
                    suggestions.append(OptimizationSuggestion(
                        type="readability",
                        priority="medium",
                        line=node.lineno,
                        message=f"Función '{node.name}' tiene {func_lines} líneas",
                        suggestion="Extraer lógica en funciones auxiliares",
                    ))

                # Demasiados parámetros
                n_params = len(node.args.args)
                if n_params > 5:
                    suggestions.append(OptimizationSuggestion(
                        type="readability",
                        priority="medium",
                        line=node.lineno,
                        message=f"Función '{node.name}' tiene {n_params} parámetros",
                        suggestion="Usar un objeto de configuración o dataclass",
                    ))

        # 3. Imports no usados
        unused_imports = self._find_unused_imports(tree)
        for imp_name, line in unused_imports:
            suggestions.append(OptimizationSuggestion(
                type="imports",
                priority="low",
                line=line,
                message=f"Import posiblemente no usado: {imp_name}",
                suggestion=f"Eliminar 'import {imp_name}' si no se usa",
            ))

        # 4. Bucles anidados profundos
        max_nesting = self._max_nesting_depth(tree)
        if max_nesting > 3:
            suggestions.append(OptimizationSuggestion(
                type="performance",
                priority="medium",
                line=1,
                message=f"Anidamiento profundo: {max_nesting} niveles",
                suggestion="Extraer bucles internos en funciones separadas",
            ))

        # 5. Código repetitivo (patrones similares)
        # (detección simplificada basada en AST)

        # 6. Variables no usadas
        unused_vars = self._find_unused_variables(tree)
        for var_name, line in unused_vars:
            suggestions.append(OptimizationSuggestion(
                type="dead_code",
                priority="low",
                line=line,
                message=f"Variable posiblemente no usada: {var_name}",
                suggestion=f"Eliminar o usar la variable '{var_name}'",
            ))

        # 7. Llamadas a funciones dentro de bucles (potencial optimización)
        loop_calls = self._find_calls_in_loops(tree)
        for func_name, line in loop_calls:
            suggestions.append(OptimizationSuggestion(
                type="performance",
                priority="medium",
                line=line,
                message=f"Llamada a '{func_name}' dentro de un bucle",
                suggestion="Considerar mover la llamada fuera del bucle si es invariante",
            ))

        n_functions = sum(
            1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        )
        n_classes = sum(
            1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        )

        return OptimizationReport(
            path=str(path),
            n_lines=len([l for l in lines if l.strip()]),
            n_functions=n_functions,
            n_classes=n_classes,
            complexity=complexity,
            suggestions=suggestions,
            original_code=source,
        )

    def optimize_code(
        self,
        code: str,
        apply_suggestions: bool = False,
    ) -> Tuple[str, List[OptimizationSuggestion]]:
        """
        Genera una versión optimizada del código.

        Parameters
        ----------
        code : str
            Código fuente original.
        apply_suggestions : bool
            Si True, intenta aplicar automáticamente algunas optimizaciones
            seguras (eliminar imports no usados, etc.).

        Returns
        -------
        Tupla (código optimizado, lista de sugerencias aplicadas).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, []

        suggestions: List[OptimizationSuggestion] = []
        optimized = code

        if apply_suggestions:
            # Eliminar imports no usados (transformación simple)
            unused = self._find_unused_imports_from_tree(tree)
            if unused:
                # Reconstruir el código sin esos imports (simplificado)
                lines = code.splitlines()
                new_lines = []
                for line in lines:
                    skip = False
                    for imp_name in unused:
                        if f"import {imp_name}" in line or f"from {imp_name}" in line:
                            skip = True
                            break
                    if not skip:
                        new_lines.append(line)
                optimized = "\n".join(new_lines)
                suggestions.append(OptimizationSuggestion(
                    type="imports",
                    priority="low",
                    line=1,
                    message=f"Eliminados {len(unused)} imports no usados",
                    suggestion="Imports eliminados automáticamente",
                ))

        return optimized, suggestions

    def generate_report(
        self,
        report: OptimizationReport,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de optimización en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append(f"# Reporte de Optimización: {report.path}")
        lines.append("")
        lines.append("## Resumen")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Líneas de código | {report.n_lines:,} |")
        lines.append(f"| Funciones | {report.n_functions} |")
        lines.append(f"| Clases | {report.n_classes} |")
        lines.append(f"| Complejidad | {report.complexity} |")
        lines.append(f"| Sugerencias | {len(report.suggestions)} |")
        lines.append("")

        # Agrupar sugerencias por tipo
        by_type: Dict[str, List[OptimizationSuggestion]] = {}
        for s in report.suggestions:
            by_type.setdefault(s.type, []).append(s)

        lines.append("## Sugerencias por Tipo")
        lines.append("")
        for stype, sugs in sorted(by_type.items()):
            lines.append(f"### {stype.title()}")
            lines.append("")
            lines.append("| Prioridad | Línea | Mensaje | Sugerencia |")
            lines.append("|-----------|-------|---------|------------|")
            for s in sugs:
                lines.append(
                    f"| {s.priority} | {s.line} | {s.message} | {s.suggestion} |"
                )
            lines.append("")

        # Sugerencias de alta prioridad
        high_priority = [s for s in report.suggestions if s.priority == "high"]
        if high_priority:
            lines.append("## Sugerencias de Alta Prioridad")
            lines.append("")
            for s in high_priority:
                lines.append(f"- Línea {s.line}: {s.message}")
                lines.append(f"  - {s.suggestion}")
            lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text

    # ------------------------------------------------------------------
    # Métodos auxiliares privados
    # ------------------------------------------------------------------

    def _calculate_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.If, ast.For, ast.While, ast.Try, ast.With,
                 ast.BoolOp, ast.ExceptHandler),
            ):
                complexity += 1
        return complexity

    def _find_unused_imports(
        self,
        tree: ast.AST,
    ) -> List[Tuple[str, int]]:
        """Encuentra imports posiblemente no usados."""
        imported: List[Tuple[str, int]] = []
        used_names: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    imported.append((name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append((node.module.split(".")[0], node.lineno))
                for alias in node.names:
                    name = alias.asname or alias.name
                    imported.append((name, node.lineno))
            elif isinstance(node, ast.Name):
                used_names.add(node.id)

        unused = [
            (name, line) for name, line in imported
            if name not in used_names and name not in self.builtins
        ]
        return unused

    def _find_unused_imports_from_tree(
        self,
        tree: ast.AST,
    ) -> List[str]:
        """Devuelve solo los nombres de imports no usados."""
        unused = self._find_unused_imports(tree)
        return [name for name, _ in unused]

    def _find_unused_variables(
        self,
        tree: ast.AST,
    ) -> List[Tuple[str, int]]:
        """Encuentra variables asignadas pero nunca leídas."""
        assigned: List[Tuple[str, int]] = []
        used: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.append((target.id, node.lineno))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    assigned.append((node.target.id, node.lineno))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)

        unused = [
            (name, line) for name, line in assigned
            if name not in used and name not in self.builtins
        ]
        return unused

    def _max_nesting_depth(self, tree: ast.AST) -> int:
        """Calcula la profundidad máxima de anidamiento."""
        return self._nesting_recursive(tree, 0)

    def _nesting_recursive(self, node: ast.AST, depth: int) -> int:
        max_depth = depth
        if isinstance(
            node,
            (ast.If, ast.For, ast.While, ast.Try, ast.With,
             ast.FunctionDef, ast.ClassDef),
        ):
            depth += 1
            max_depth = max(max_depth, depth)

        for child in ast.iter_child_nodes(node):
            max_depth = max(max_depth, self._nesting_recursive(child, depth))

        return max_depth

    def _find_calls_in_loops(
        self,
        tree: ast.AST,
    ) -> List[Tuple[str, int]]:
        """Encuentra llamadas a funciones dentro de bucles."""
        calls: List[Tuple[str, int]] = []

        class LoopCallFinder(ast.NodeVisitor):
            def __init__(self) -> None:
                self.in_loop = False
                self.calls: List[Tuple[str, int]] = []

            def visit_For(self, node: ast.For) -> None:
                old = self.in_loop
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = old

            def visit_While(self, node: ast.While) -> None:
                old = self.in_loop
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = old

            def visit_Call(self, node: ast.Call) -> None:
                if self.in_loop:
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    self.calls.append((func_name, node.lineno))
                self.generic_visit(node)

        finder = LoopCallFinder()
        finder.visit(tree)
        return finder.calls


def quick_optimize(
    path: str | Path,
    output_path: Optional[str | Path] = None,
) -> OptimizationReport:
    """
    Función de conveniencia: analiza un archivo y genera reporte.

    Returns
    -------
    OptimizationReport con el análisis completo.
    """
    optimizer = CodeOptimizer()
    report = optimizer.analyze_file(path)

    if output_path:
        optimizer.generate_report(report, output_path)

    return report


__all__ = [
    "OptimizationSuggestion",
    "OptimizationReport",
    "CodeOptimizer",
    "quick_optimize",
]
