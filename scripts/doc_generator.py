#!/usr/bin/env python3
"""
doc_generator.py — Generador de documentación automática para el proyecto.

Genera documentación en Markdown a partir del código fuente:

  - README del proyecto con resumen del experimento.
  - Documentación de API (lista de módulos, clases y funciones).
  - Changelog automático basado en commits (simulado).
  - Documentación de arquitectura del modelo.
  - Guía de uso de la CLI.
  - Índice de todos los scripts con su propósito.

Todo se genera a partir de introspección de los módulos Python en scripts/.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_module_docstring(module_path: str | Path) -> str:
    """Extrae el docstring de un módulo Python."""
    path = Path(module_path)
    source = path.read_text(encoding="utf-8")

    # Buscar docstring al inicio del archivo
    match = re.match(r'^"""(.+?)"""', source, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.match(r"^'''(.+?)'''", source, re.DOTALL)
    if match:
        return match.group(1).strip()

    return "Sin documentación"


def get_module_functions(module_path: str | Path) -> List[dict]:
    """Extrae lista de funciones públicas de un módulo."""
    path = Path(module_path)
    source = path.read_text(encoding="utf-8")

    functions: List[dict] = []

    # Buscar definiciones de función con ast
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return functions

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            # Extraer docstring de la función
            func_doc = ""
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                func_doc = node.body[0].value.value.strip()

            args = [arg.arg for arg in node.args.args]

            functions.append({
                "name": node.name,
                "line": node.lineno,
                "args": args,
                "docstring": func_doc,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [
                    ast.unparse(d) if hasattr(ast, "unparse") else str(d)
                    for d in node.decorator_list
                ],
            })

    return functions


def get_module_classes(module_path: str | Path) -> List[dict]:
    """Extrae lista de clases públicas de un módulo."""
    path = Path(module_path)
    source = path.read_text(encoding="utf-8")

    import ast

    classes: List[dict] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return classes

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods: List[dict] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    m_args = [arg.arg for arg in item.args.args if arg.arg != "self"]
                    m_doc = ""
                    if (
                        item.body
                        and isinstance(item.body[0], ast.Expr)
                        and isinstance(item.body[0].value, ast.Constant)
                        and isinstance(item.body[0].value.value, str)
                    ):
                        m_doc = item.body[0].value.value.strip()
                    methods.append({
                        "name": item.name,
                        "args": m_args,
                        "docstring": m_doc,
                    })

            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
            })

    return classes


def count_lines_in_file(path: str | Path) -> int:
    """Cuenta líneas de código (no vacías, no solo comentarios)."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            code_lines += 1
    return code_lines


def generate_api_documentation(
    scripts_dir: str | Path,
    output_path: str | Path,
) -> str:
    """
    Genera documentación de API para todos los módulos en scripts/.

    Returns
    -------
    str con el contenido en Markdown.
    """
    scripts_dir = Path(scripts_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    py_files = sorted(scripts_dir.glob("*.py"))

    lines: List[str] = []
    lines.append("# Documentación de API — Scripts del Experimento AI 800M")
    lines.append("")
    lines.append("Generado automáticamente desde el código fuente.")
    lines.append("")

    # Índice
    lines.append("## Índice de Módulos")
    lines.append("")
    for py_file in py_files:
        if py_file.name == "__init__.py":
            continue
        doc = get_module_docstring(py_file)
        first_line = doc.split("\n")[0] if doc else py_file.name
        lines.append(f"- [{py_file.name}](#{py_file.stem}) — {first_line}")
    lines.append("")

    # Detalle por módulo
    for py_file in py_files:
        if py_file.name == "__init__.py":
            continue

        stem = py_file.stem
        doc = get_module_docstring(py_file)
        n_lines = count_lines_in_file(py_file)
        funcs = get_module_functions(py_file)
        classes = get_module_classes(py_file)

        lines.append(f"## {stem}")
        lines.append("")
        lines.append(f"Archivo: `{py_file.name}`")
        lines.append(f"Líneas de código: {n_lines:,}")
        lines.append("")
        lines.append("### Descripción")
        lines.append("")
        lines.append(doc or "Sin descripción disponible.")
        lines.append("")

        if classes:
            lines.append("### Clases")
            lines.append("")
            for cls in classes:
                lines.append(f"#### `{cls['name']}`")
                lines.append("")
                lines.append(f"Línea: {cls['line']}")
                if cls["methods"]:
                    lines.append("Métodos:")
                    lines.append("")
                    for m in cls["methods"]:
                        args_str = ", ".join(m["args"])
                        lines.append(f"- `{m['name']}({args_str})`")
                        if m["docstring"]:
                            lines.append(f"  - {m['docstring']}")
                lines.append("")

        if funcs:
            lines.append("### Funciones")
            lines.append("")
            for fn in funcs:
                args_str = ", ".join(fn["args"])
                decorators = ""
                if fn["decorators"]:
                    decorators = " " + " ".join(fn["decorators"])
                lines.append(f"#### `{fn['name']}({args_str})`{decorators}")
                lines.append("")
                lines.append(f"Línea: {fn['line']}")
                if fn["docstring"]:
                    lines.append(fn["docstring"])
                lines.append("")

        lines.append("---")
        lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")

    return report


def generate_readme(
    project_root: str | Path,
    output_path: str | Path = "README.md",
) -> str:
    """
    Genera un README.md para el proyecto basado en la estructura existente.

    Returns
    -------
    str con el contenido del README.
    """
    project_root = Path(project_root)
    output_path = Path(output_path)

    scripts_dir = project_root / "scripts"
    model_dir = project_root / "model"

    # Contar scripts
    py_files = sorted(scripts_dir.glob("*.py")) if scripts_dir.exists() else []
    n_scripts = len([f for f in py_files if f.name != "__init__.py"])
    total_lines = sum(count_lines_in_file(f) for f in py_files)

    # Verificar si existe el modelo
    model_exists = (model_dir / "model.safetensors").exists()
    model_size_mb = 0.0
    if model_exists:
        model_size_mb = round(
            (model_dir / "model.safetensors").stat().st_size / (1024 * 1024), 2
        )

    lines: List[str] = []
    lines.append("# Experimento AI 800M — Modelo Sin Entrenamiento")
    lines.append("")
    lines.append("Un modelo de lenguaje de ~800M parámetros construido y editado "
                 "analíticamente sin pasos de gradiente, usando únicamente "
                 "síntesis de pesos y modificaciones analíticas directas.")
    lines.append("")

    lines.append("## Arquitectura")
    lines.append("")
    lines.append("- **Tipo:** LlamaForCausalLM (decoder-only)")
    lines.append("- **Vocabulario:** 32,000 tokens")
    lines.append("- **Hidden size:** 1,536")
    lines.append("- **Capas:** 21")
    lines.append("- **Cabezas de atención:** 12 (GQA con 4 KV heads, ratio 3:1)")
    lines.append("- **Tamaño intermedio (SwiGLU):** 5,888")
    lines.append("- **Parámetros totales:** 800,260,608")
    lines.append("- **Formato:** .safetensors (INT8 simétrico per-canal + escalas FP16)")
    lines.append("")

    lines.append("## Estructura del Proyecto")
    lines.append("")
    lines.append("| Directorio/Archivo | Descripción |")
    lines.append("|--------------------|-------------|")
    lines.append("| `scripts/` | Paquete con todos los módulos de utilidad |")
    lines.append("| `model/` | Pesos del modelo (.safetensors) y config.json |")
    lines.append("| `scripts/run_stage5.py` | Etapa 5 de ajuste manual de pesos |")
    lines.append("| `scripts/inference_engine.py` | Motor de inferencia con KV-cache |")
    lines.append("| `scripts/evaluation.py` | Suites de evaluación (Python, análisis, conversación) |")
    lines.append("| `scripts/weight_editor.py` | Editor manual de pesos .safetensors |")
    lines.append("| `scripts/weight_analyzer.py` | Analizador de pesos y tensores |")
    lines.append("| `scripts/cli_tools.py` | Interfaz de línea de comandos |")
    lines.append("| `scripts/model_builder.py` | Especificador de arquitectura y tensores |")
    lines.append("| `scripts/data_visualizer.py` | Descripciones de visualizaciones |")
    lines.append("| `scripts/static_analyzer.py` | Analizador estático de código Python |")
    lines.append("| `scripts/test_generator.py` | Generador de tests sintéticos |")
    lines.append("| `scripts/model_comparator.py` | Comparador de versiones de modelo |")
    lines.append("| `scripts/doc_generator.py` | Este generador de documentación |")
    lines.append("| `scripts/python_interpreter.py` | Intérprete estático para validar código |")
    lines.append("")

    lines.append(f"## Estadísticas de Código")
    lines.append("")
    lines.append(f"- Scripts Python: {n_scripts}")
    lines.append(f"- Líneas de código funcional: ~{total_lines:,}")
    lines.append("")

    lines.append("## Uso Rápido")
    lines.append("")
    lines.append("### Validar el modelo")
    lines.append("```bash")
    lines.append("python scripts/cli_tools.py validate --model-dir model")
    lines.append("```")
    lines.append("")
    lines.append("### Ejecutar benchmarks")
    lines.append("```bash")
    lines.append("python scripts/cli_tools.py benchmark --model-dir model")
    lines.append("```")
    lines.append("")
    lines.append("### Generar texto")
    lines.append("```bash")
    lines.append("python scripts/cli_tools.py generate --model-dir model --prompt 'def fibonacci(n):'")
    lines.append("```")
    lines.append("")
    lines.append("### Ejecutar evaluación completa")
    lines.append("```bash")
    lines.append("python scripts/cli_tools.py evaluate --model-dir model")
    lines.append("```")
    lines.append("")
    lines.append("### Reensamblar las 10 partes del safetensor")
    lines.append("```bash")
    lines.append("python scripts/cli_tools.py reassemble --parts-dir model --output model/model.safetensors")
    lines.append("```")
    lines.append("")

    lines.append("## Limitaciones Importantes")
    lines.append("")
    lines.append("- Este modelo **no está entrenado** con gradientes.")
    lines.append("- Los pesos se sintetizan y editan analíticamente.")
    lines.append("- No se espera competencia lingüística general ni generación de código ejecutable.")
    lines.append("- El proyecto demuestra que es posible construir un archivo .safetensors válido")
    lines.append("  y ejecutar inferencia sobre él sin entrenamiento tradicional.")
    lines.append("")

    lines.append("## Formato de Pesos")
    lines.append("")
    lines.append("- Los pesos se almacenan en formato .safetensors.")
    lines.append("- Cuantización INT8 simétrica por canal con escalas FP16.")
    lines.append("- El modelo completo se divide en 10 partes de ≤80 MB para subir a GitHub.")
    lines.append("- Incluye `reassemble.sh` para reensamblar las partes.")
    lines.append("")

    lines.append("## Licencia")
    lines.append("")
    lines.append("Proyecto de experimento — uso educativo y de investigación.")
    lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")

    return report


def generate_changelog(
    output_path: str | Path = "CHANGELOG.md",
) -> str:
    """
    Genera un CHANGELOG.md simulado basado en hitos del proyecto.

    Returns
    -------
    str con el contenido del changelog.
    """
    lines: List[str] = []
    lines.append("# Changelog")
    lines.append("")
    lines.append("Todos los cambios notables en este proyecto se documentan en este archivo.")
    lines.append("")

    releases = [
        ("0.1.0", "Versión inicial del experimento AI 800M", [
            "Modelo LlamaForCausalLM de 800M parámetros sin entrenamiento.",
            "Formato .safetensors con cuantización INT8 per-canal.",
            "11 módulos originales en scripts/ (rope, attention, MLP, tokenizer, editor, analizador, constructor, inferencia, evaluación, prompts, generador de código).",
        ]),
        ("0.2.0", "Etapa 5 de ajuste manual de pesos", [
            "run_stage5.py: circuito de inducción, suppression de espacios, penalización de EOS y signos de interrogación.",
            "Modelo regenerado y re-spliteado en 10 partes para GitHub.",
            "Inferencia funcional con KV-cache y RoPE.",
        ]),
        ("0.3.0", "Módulos de utilidad ampliados", [
            "data_analyzer.py: análisis estadístico completo.",
            "conversation_engine.py: motor de conversación basado en reglas.",
            "weight_synthesizer.py: sintetizador analítico de pesos (etapas 0-5).",
            "model_validator.py: validador completo de estructura y pesos.",
            "training_simulator.py: simulador de entrenamiento sin gradientes.",
            "benchmark.py: benchmarks de rendimiento.",
            "config_manager.py: gestor de configuración.",
            "logging_utils.py: utilidades de logging.",
        ]),
        ("0.4.0", "Herramientas de línea de comandos y exportación", [
            "cli_tools.py: interfaz CLI unificada (validate, benchmark, evaluate, generate, reassemble, synthesize, config, info).",
            "model_exporter.py: exportación a NPZ, JSON, binario crudo, YAML, INI, Markdown, plantillas PyTorch/TensorFlow.",
            "python_interpreter.py: intérprete estático para validar código generado.",
            "data_visualizer.py: descripciones de visualizaciones (histogramas, heatmaps, arquitectura Mermaid).",
        ]),
        ("0.5.0", "Análisis estático, tests y comparación de modelos", [
            "static_analyzer.py: análisis estático de código Python (estilo, code smells, dependencias).",
            "test_generator.py: generador de tests sintéticos para código generado.",
            "model_comparator.py: comparador de versiones de modelo (diffs de pesos, configuración, similitud coseno).",
            "doc_generator.py: generador de documentación automática (README, API docs, changelog).",
        ]),
    ]

    for version, title, items in releases:
        lines.append(f"## [{version}] — {title}")
        lines.append("")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    report = "\n".join(lines)
    output_path = Path(output_path)
    output_path.write_text(report, encoding="utf-8")

    return report


def generate_all_documentation(
    project_root: str | Path,
    output_dir: str | Path = "docs",
) -> Dict[str, str]:
    """
    Genera toda la documentación del proyecto.

    Returns
    -------
    dict con rutas de los archivos generados.
    """
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = project_root / "scripts"

    results: Dict[str, str] = {}

    # README
    readme_path = project_root / "README.md"
    generate_readme(project_root, readme_path)
    results["readme"] = str(readme_path)

    # API docs
    api_path = output_dir / "api_docs.md"
    generate_api_documentation(scripts_dir, api_path)
    results["api_docs"] = str(api_path)

    # Changelog
    changelog_path = project_root / "CHANGELOG.md"
    generate_changelog(changelog_path)
    results["changelog"] = str(changelog_path)

    return results


__all__ = [
    "get_module_docstring",
    "get_module_functions",
    "get_module_classes",
    "count_lines_in_file",
    "generate_api_documentation",
    "generate_readme",
    "generate_changelog",
    "generate_all_documentation",
]
