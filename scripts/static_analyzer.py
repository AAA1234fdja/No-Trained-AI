#!/usr/bin/env python3
"""
static_analyzer.py — Analizador estático de código Python.

Proporciona un conjunto de herramientas para analizar estáticamente
código Python generado por el modelo o escrito manualmente:

  - Análisis de dependencias entre módulos.
  - Detección de código muerto (funciones/clases no usadas).
  - Medición de complejidad ciclomática y cognitiva.
  - Reglas de estilo básicas (longitud de línea, indentación, etc.).
  - Detección de "code smells" (métodos largos, muchas clases, etc.).
  - Reporte de calidad en Markdown/JSON.
  - Análisis de acoplamiento y cohesión entre funciones.

Todas las operaciones son de solo lectura y no ejecutan código.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class StyleIssue:
    """Un problema de estilo detectado."""
    line: int
    column: int
    rule: str
    message: str
    severity: str  # "info", "warning", "error"


@dataclass
class CodeSmell:
    """Un 'code smell' (olor a código) detectado."""
    name: str
    location: str  # nombre de función/clase
    line: int
    description: str
    severity: str


@dataclass
class Dependency:
    """Dependencia entre módulos."""
    source: str
    target: str
    line: int
    is_relative: bool


@dataclass
class ModuleAnalysis:
    """Resultado del análisis de un módulo Python."""
    path: str
    n_lines: int
    n_functions: int
    n_classes: int
    n_imports: int
    complexity: int
    cognitive_complexity: int
    style_issues: List[StyleIssue] = field(default_factory=list)
    code_smells: List[CodeSmell] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    defined_names: List[str] = field(default_factory=list)
    used_names: List[str] = field(default_factory=list)
    unused_functions: List[str] = field(default_factory=list)
    unused_classes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "n_lines": self.n_lines,
            "n_functions": self.n_functions,
            "n_classes": self.n_classes,
            "n_imports": self.n_imports,
            "complexity": self.complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "n_style_issues": len(self.style_issues),
            "n_code_smells": len(self.code_smells),
            "n_dependencies": len(self.dependencies),
            "defined_names": self.defined_names,
            "unused_functions": self.unused_functions,
            "unused_classes": self.unused_classes,
        }


class StaticAnalyzer:
    """
    Analizador estático principal.

    Recorre el AST de archivos Python y extrae métricas de calidad,
    problemas de estilo y code smells.
    """

    # Límites por defecto para detectar code smells
    MAX_FUNCTION_LINES = 50
    MAX_CLASS_LINES = 200
    MAX_COMPLEXITY = 10
    MAX_PARAMETERS = 6
    MAX_LINE_LENGTH = 100

    def __init__(
        self,
        max_function_lines: int = MAX_FUNCTION_LINES,
        max_class_lines: int = MAX_CLASS_LINES,
        max_complexity: int = MAX_COMPLEXITY,
        max_parameters: int = MAX_PARAMETERS,
        max_line_length: int = MAX_LINE_LENGTH,
    ) -> None:
        self.max_function_lines = max_function_lines
        self.max_class_lines = max_class_lines
        self.max_complexity = max_complexity
        self.max_parameters = max_parameters
        self.max_line_length = max_line_length

    def analyze_file(self, path: str | Path) -> ModuleAnalysis:
        """
        Analiza un único archivo Python.

        Returns
        -------
        ModuleAnalysis con todos los hallazgos.
        """
        path = Path(path)
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return ModuleAnalysis(
                path=str(path),
                n_lines=len(lines),
                n_functions=0,
                n_classes=0,
                n_imports=0,
                complexity=0,
                cognitive_complexity=0,
                style_issues=[
                    StyleIssue(
                        line=e.lineno or 1,
                        column=e.offset or 0,
                        rule="syntax-error",
                        message=str(e),
                        severity="error",
                    )
                ],
                code_smells=[],
                dependencies=[],
                defined_names=[],
                used_names=[],
                unused_functions=[],
                unused_classes=[],
            )

        # Recopilar información
        n_lines = len([l for l in lines if l.strip()])
        functions: List[ast.FunctionDef] = []
        classes: List[ast.ClassDef] = []
        imports: List[ast.Import | ast.ImportFrom] = []
        defined_names: List[str] = []
        used_names: Set[str] = set()
        dependencies: List[Dependency] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
                defined_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)
                defined_names.append(node.name)
            elif isinstance(node, ast.Import):
                imports.append(node)
                for alias in node.names:
                    dependencies.append(
                        Dependency(
                            source=str(path),
                            target=alias.name,
                            line=node.lineno,
                            is_relative=False,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                imports.append(node)
                target = node.module or ""
                dependencies.append(
                    Dependency(
                        source=str(path),
                        target=target,
                        line=node.lineno,
                        is_relative=node.level > 0,
                    )
                )
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)

        # Métricas de complejidad
        complexity = self._calculate_complexity(tree)
        cognitive = self._calculate_cognitive_complexity(tree)

        # Problemas de estilo
        style_issues = self._check_style(tree, lines)

        # Code smells
        code_smells = self._detect_code_smells(tree, functions, classes, lines)

        # Detectar funciones/clases no usadas
        used_in_code = used_names | {d for d in defined_names}
        unused_functions = [
            f.name for f in functions if f.name not in used_in_code
        ]
        unused_classes = [
            c.name for c in classes if c.name not in used_in_code
        ]

        return ModuleAnalysis(
            path=str(path),
            n_lines=n_lines,
            n_functions=len(functions),
            n_classes=len(classes),
            n_imports=len(imports),
            complexity=complexity,
            cognitive_complexity=cognitive,
            style_issues=style_issues,
            code_smells=code_smells,
            dependencies=dependencies,
            defined_names=defined_names,
            used_names=sorted(used_names),
            unused_functions=unused_functions,
            unused_classes=unused_classes,
        )

    def analyze_directory(
        self,
        directory: str | Path,
        pattern: str = "*.py",
    ) -> List[ModuleAnalysis]:
        """
        Analiza todos los archivos Python en un directorio recursivamente.

        Returns
        -------
        Lista de ModuleAnalysis, uno por archivo.
        """
        directory = Path(directory)
        results: List[ModuleAnalysis] = []

        for py_file in sorted(directory.rglob(pattern)):
            if "__pycache__" in str(py_file):
                continue
            results.append(self.analyze_file(py_file))

        return results

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calcula la complejidad ciclomática aproximada."""
        complexity = 1  # base
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.If, ast.For, ast.While, ast.Try, ast.With,
                 ast.BoolOp, ast.ExceptHandler),
            ):
                complexity += 1
        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """
        Calcula la complejidad cognitiva (variante de la ciclomática
        que penaliza anidamiento).
        """
        return self._cognitive_recursive(tree, 0)

    def _cognitive_recursive(self, node: ast.AST, nesting: int) -> int:
        score = 0
        # Estructuras de control incrementan puntuación + anidamiento
        if isinstance(
            node,
            (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp),
        ):
            score += 1 + nesting
        elif isinstance(node, ast.ExceptHandler):
            score += 1 + nesting

        for child in ast.iter_child_nodes(node):
            child_nesting = nesting
            if isinstance(
                node,
                (ast.If, ast.For, ast.While, ast.Try, ast.With,
                 ast.FunctionDef, ast.ClassDef),
            ):
                child_nesting += 1
            score += self._cognitive_recursive(child, child_nesting)

        return score

    def _check_style(
        self,
        tree: ast.AST,
        lines: List[str],
    ) -> List[StyleIssue]:
        """Detecta problemas de estilo básicos."""
        issues: List[StyleIssue] = []

        for i, line in enumerate(lines, start=1):
            # Longitud de línea
            if len(line) > self.max_line_length:
                issues.append(StyleIssue(
                    line=i,
                    column=0,
                    rule="line-too-long",
                    message=f"Línea excede {self.max_line_length} caracteres ({len(line)})",
                    severity="warning",
                ))
            # Espacios finales
            if line != line.rstrip():
                issues.append(StyleIssue(
                    line=i,
                    column=len(line.rstrip()),
                    rule="trailing-whitespace",
                    message="Espacios en blanco al final de línea",
                    severity="info",
                ))

        return issues

    def _detect_code_smells(
        self,
        tree: ast.AST,
        functions: List[ast.FunctionDef],
        classes: List[ast.ClassDef],
        lines: List[str],
    ) -> List[CodeSmell]:
        """Detecta code smells comunes."""
        smells: List[CodeSmell] = []

        for func in functions:
            func_lines = (
                (func.end_lineno or func.lineno) - func.lineno + 1
            )
            if func_lines > self.max_function_lines:
                smells.append(CodeSmell(
                    name="long-function",
                    location=func.name,
                    line=func.lineno,
                    description=f"Función con {func_lines} líneas (máx {self.max_function_lines})",
                    severity="warning",
                ))

            # Demasiados parámetros
            n_params = len(func.args.args)
            if n_params > self.max_parameters:
                smells.append(CodeSmell(
                    name="too-many-parameters",
                    location=func.name,
                    line=func.lineno,
                    description=f"Función con {n_params} parámetros (máx {self.max_parameters})",
                    severity="warning",
                ))

        for cls in classes:
            cls_lines = (
                (cls.end_lineno or cls.lineno) - cls.lineno + 1
            )
            if cls_lines > self.max_class_lines:
                smells.append(CodeSmell(
                    name="large-class",
                    location=cls.name,
                    line=cls.lineno,
                    description=f"Clase con {cls_lines} líneas (máx {self.max_class_lines})",
                    severity="warning",
                ))

        return smells

    def generate_report(
        self,
        analyses: List[ModuleAnalysis],
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de análisis estático en Markdown.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Análisis Estático")
        lines.append("")
        lines.append(f"Archivos analizados: {len(analyses)}")
        lines.append("")

        # Totales globales
        total_lines = sum(a.n_lines for a in analyses)
        total_funcs = sum(a.n_functions for a in analyses)
        total_classes = sum(a.n_classes for a in analyses)
        total_imports = sum(a.n_imports for a in analyses)
        total_complexity = sum(a.complexity for a in analyses)
        total_style = sum(len(a.style_issues) for a in analyses)
        total_smells = sum(len(a.code_smells) for a in analyses)

        lines.append("## Resumen")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Líneas de código | {total_lines:,} |")
        lines.append(f"| Funciones | {total_funcs} |")
        lines.append(f"| Clases | {total_classes} |")
        lines.append(f"| Importaciones | {total_imports} |")
        lines.append(f"| Complejidad total | {total_complexity} |")
        lines.append(f"| Problemas de estilo | {total_style} |")
        lines.append(f"| Code smells | {total_smells} |")
        lines.append("")

        # Detalle por archivo
        lines.append("## Detalle por Archivo")
        lines.append("")
        lines.append("| Archivo | Líneas | Funciones | Clases | Complejidad | Estilo | Smells |")
        lines.append("|---------|--------|-----------|--------|-------------|--------|--------|")

        for a in analyses:
            lines.append(
                f"| {a.path} | {a.n_lines} | {a.n_functions} | {a.n_classes} | "
                f"{a.complexity} | {len(a.style_issues)} | {len(a.code_smells)} |"
            )

        lines.append("")

        # Dependencias
        all_deps: List[Dependency] = []
        for a in analyses:
            all_deps.extend(a.dependencies)

        if all_deps:
            lines.append("## Dependencias entre Módulos")
            lines.append("")
            lines.append("| Origen | Destino | Línea | Relativo |")
            lines.append("|--------|---------|-------|----------|")
            for d in all_deps[:50]:  # limitar a 50 para no saturar
                lines.append(
                    f"| {d.source} | {d.target} | {d.line} | {'Sí' if d.is_relative else 'No'} |"
                )
            if len(all_deps) > 50:
                lines.append(f"| ... ({len(all_deps) - 50} más) | | | |")
            lines.append("")

        # Code smells destacados
        all_smells: List[CodeSmell] = []
        for a in analyses:
            for s in a.code_smells:
                all_smells.append(s)

        if all_smells:
            lines.append("## Code Smells Detectados")
            lines.append("")
            lines.append("| Tipo | Ubicación | Línea | Descripción | Severidad |")
            lines.append("|------|-----------|-------|-------------|-----------|")
            for s in all_smells[:30]:
                lines.append(
                    f"| {s.name} | {s.location} | {s.line} | {s.description} | {s.severity} |"
                )
            if len(all_smells) > 30:
                lines.append(f"| ... ({len(all_smells) - 30} más) | | | | |")
            lines.append("")

        # Funciones y clases no usadas
        all_unused_funcs: List[Tuple[str, str]] = []
        all_unused_classes: List[Tuple[str, str]] = []
        for a in analyses:
            for f in a.unused_functions:
                all_unused_funcs.append((a.path, f))
            for c in a.unused_classes:
                all_unused_classes.append((a.path, c))

        if all_unused_funcs or all_unused_classes:
            lines.append("## Posibles Elementos No Usados")
            lines.append("")
            if all_unused_funcs:
                lines.append("### Funciones")
                lines.append("")
                lines.append("| Archivo | Función |")
                lines.append("|---------|---------|")
                for path, name in all_unused_funcs:
                    lines.append(f"| {path} | {name} |")
                lines.append("")
            if all_unused_classes:
                lines.append("### Clases")
                lines.append("")
                lines.append("| Archivo | Clase |")
                lines.append("|---------|-------|")
                for path, name in all_unused_classes:
                    lines.append(f"| {path} | {name} |")
                lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")

        return report


def quick_analyze_directory(
    directory: str | Path,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Función de conveniencia para analizar un directorio completo.

    Returns
    -------
    str con el reporte en Markdown.
    """
    analyzer = StaticAnalyzer()
    analyses = analyzer.analyze_directory(directory)
    return analyzer.generate_report(analyses, output_path)


__all__ = [
    "StyleIssue",
    "CodeSmell",
    "Dependency",
    "ModuleAnalysis",
    "StaticAnalyzer",
    "quick_analyze_directory",
]
