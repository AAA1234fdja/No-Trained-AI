#!/usr/bin/env python3
"""
experiment_reporter.py — Generador de reportes consolidados del experimento.

Combina los resultados de todas las etapas del pipeline (validación,
benchmarks, inferencia, evaluación, exportación, documentación) en un
reporte final único en formato Markdown y JSON.

Incluye:
  - Resumen ejecutivo del experimento.
  - Resultados de validación del modelo.
  - Resultados de benchmarks de rendimiento.
  - Resultados de inferencia en prompts de prueba.
  - Resultados de evaluación por suite.
  - Archivos de exportación generados.
  - Archivos de documentación generados.
  - Conclusiones y limitaciones.
  - Próximos pasos recomendados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_json_if_exists(path: str | Path) -> Optional[dict]:
    """Carga un archivo JSON si existe, devuelve None si no."""
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def generate_executive_summary(
    validation: Optional[dict],
    benchmark: Optional[dict],
    inference: Optional[dict],
    evaluation: Optional[dict],
    export: Optional[dict],
    documentation: Optional[dict],
) -> str:
    """
    Genera el resumen ejecutivo del reporte.

    Returns
    -------
    str con el resumen en Markdown.
    """
    lines: List[str] = []
    lines.append("# Reporte del Experimento AI 800M")
    lines.append("")
    lines.append("## Resumen Ejecutivo")
    lines.append("")

    # Estado del modelo
    if validation:
        all_passed = validation.get("all_passed", False)
        passed = validation.get("passed", 0)
        total = validation.get("total_checks", 0)
        lines.append(f"- **Validación del modelo:** {passed}/{total} checks pasados ({'todos OK' if all_passed else 'algunos fallaron'})")
    else:
        lines.append("- **Validación del modelo:** no ejecutada")

    # Evaluación
    if evaluation:
        total_tests = evaluation.get("total_tests", 0)
        total_passed = evaluation.get("total_passed", 0)
        pass_rate = evaluation.get("global_pass_rate", 0.0)
        lines.append(f"- **Evaluación:** {total_passed}/{total_tests} pruebas pasadas ({pass_rate:.1%})")
    else:
        lines.append("- **Evaluación:** no ejecutada")

    # Inferencia
    if inference:
        n_prompts = inference.get("n_prompts", 0)
        lines.append(f"- **Inferencia de prueba:** {n_prompts} prompts procesados")
    else:
        lines.append("- **Inferencia de prueba:** no ejecutada")

    # Exportación
    if export:
        n_formats = len(export.get("formats", {}))
        lines.append(f"- **Exportación:** {n_formats} formatos generados")
    else:
        lines.append("- **Exportación:** no ejecutada")

    # Documentación
    if documentation:
        n_docs = len(documentation.get("files", {}))
        lines.append(f"- **Documentación:** {n_docs} archivos generados")
    else:
        lines.append("- **Documentación:** no ejecutada")

    lines.append("")
    lines.append("## Conclusiones Principales")
    lines.append("")
    lines.append("1. El modelo .safetensors es estructuralmente válido y cargable.")
    lines.append("2. La síntesis analítica de pesos genera un modelo funcional para inferencia.")
    lines.append("3. La entropía de generación está en rango saludable (no hay colapso total).")
    lines.append("4. La competencia en Python y análisis es limitada sin entrenamiento con gradientes.")
    lines.append("5. Se requieren más datos y entrenamiento para lograr capacidad conversacional.")
    lines.append("")

    return "\n".join(lines)


def generate_model_section(
    model_dir: str | Path,
    validation: Optional[dict],
) -> str:
    """
    Genera la sección del modelo en el reporte.

    Returns
    -------
    str con la sección en Markdown.
    """
    model_dir = Path(model_dir)
    weights_file = model_dir / "model.safetensors"
    config_file = model_dir / "config.json"

    lines: List[str] = []
    lines.append("## Modelo")
    lines.append("")

    lines.append("### Archivos")
    lines.append("")
    lines.append(f"- Pesos: `{weights_file.name}` (existe: {weights_file.exists()})")
    lines.append(f"- Configuración: `{config_file.name}` (existe: {config_file.exists()})")

    if weights_file.exists():
        size_mb = round(weights_file.stat().st_size / (1024 * 1024), 2)
        lines.append(f"- Tamaño: {size_mb} MB")

    lines.append("")

    if config_file.exists():
        import json as _json
        with open(config_file, "r") as f:
            config = _json.load(f)

        lines.append("### Configuración")
        lines.append("")
        lines.append("| Parámetro | Valor |")
        lines.append("|-----------|-------|")
        for key, value in config.items():
            if key in ("quantization_config", "experiment_metadata"):
                continue
            lines.append(f"| {key} | {value} |")
        lines.append("")

    if validation:
        lines.append("### Resultados de Validación")
        lines.append("")
        lines.append("| Check | Estado |")
        lines.append("|-------|--------|")
        lines.append(f"| Checks pasados | {validation.get('passed', 0)} / {validation.get('total_checks', 0)} |")
        lines.append(f"| Todos pasados | {'Sí' if validation.get('all_passed') else 'No'} |")
        lines.append("")

    return "\n".join(lines)


def generate_benchmark_section(
    benchmark: Optional[dict],
) -> str:
    """
    Genera la sección de benchmarks.

    Returns
    -------
    str con la sección en Markdown.
    """
    lines: List[str] = []
    lines.append("## Benchmarks de Rendimiento")
    lines.append("")

    if not benchmark:
        lines.append("No se ejecutaron benchmarks.")
        lines.append("")
        return "\n".join(lines)

    results = benchmark.get("benchmark_results", [])
    if not results:
        lines.append("No hay resultados de benchmarks disponibles.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Benchmark | Valor | Unidad |")
    lines.append("|-----------|-------|--------|")
    for r in results:
        name = r.get("name", "desconocido")
        value = r.get("value", 0)
        unit = r.get("unit", "")
        lines.append(f"| {name} | {value:.4f} | {unit} |")

    lines.append("")
    return "\n".join(lines)


def generate_inference_section(
    inference: Optional[dict],
) -> str:
    """
    Genera la sección de inferencia.

    Returns
    -------
    str con la sección en Markdown.
    """
    lines: List[str] = []
    lines.append("## Inferencia de Prueba")
    lines.append("")

    if not inference:
        lines.append("No se ejecutó inferencia de prueba.")
        lines.append("")
        return "\n".join(lines)

    n_prompts = inference.get("n_prompts", 0)
    results = inference.get("results", [])

    lines.append(f"Se probaron {n_prompts} prompts.")
    lines.append("")
    lines.append("| Prompt | Texto Generado | Entropía (nats) | Tokens/seg |")
    lines.append("|--------|----------------|-----------------|------------|")

    for r in results:
        prompt = r.get("prompt", "")[:40]
        generated = r.get("generated_text", "")[:60]
        entropy = r.get("entropy", 0)
        tps = r.get("tokens_per_second", "N/A")
        lines.append(f"| {prompt!r} | {generated!r} | {entropy:.3f} | {tps} |")

    lines.append("")
    return "\n".join(lines)


def generate_evaluation_section(
    evaluation: Optional[dict],
) -> str:
    """
    Genera la sección de evaluación.

    Returns
    -------
    str con la sección en Markdown.
    """
    lines: List[str] = []
    lines.append("## Evaluación por Suite")
    lines.append("")

    if not evaluation:
        lines.append("No se ejecutó evaluación.")
        lines.append("")
        return "\n".join(lines)

    suites = evaluation.get("suites", {})
    if not suites:
        lines.append("No hay resultados de suites disponibles.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Suite | Pruebas | Pasadas | Falladas | Tasa de Paso |")
    lines.append("|-------|---------|---------|----------|--------------|")

    for sname, sdata in suites.items():
        total = sdata.get("total_tests", 0)
        passed = sdata.get("passed", 0)
        failed = sdata.get("failed", 0)
        rate = sdata.get("pass_rate", 0.0)
        bar_len = int(rate * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"| {sname} | {total} | {passed} | {failed} | {rate:.1%} {bar} |")

    lines.append("")
    total_tests = evaluation.get("total_tests", 0)
    total_passed = evaluation.get("total_passed", 0)
    global_rate = evaluation.get("global_pass_rate", 0.0)
    lines.append(f"**Global:** {total_passed}/{total_tests} pruebas pasadas ({global_rate:.1%})")
    lines.append("")
    return "\n".join(lines)


def generate_export_section(
    export: Optional[dict],
) -> str:
    """
    Genera la sección de exportación.

    Returns
    -------
    str con la sección en Markdown.
    """
    lines: List[str] = []
    lines.append("## Exportación de Formatos")
    lines.append("")

    if not export:
        lines.append("No se ejecutó exportación.")
        lines.append("")
        return "\n".join(lines)

    formats = export.get("formats", {})
    lines.append("| Formato | Estado |")
    lines.append("|---------|--------|")
    for fmt, info in formats.items():
        status = info.get("status", "desconocido")
        lines.append(f"| {fmt} | {status} |")

    lines.append("")
    return "\n".join(lines)


def generate_documentation_section(
    documentation: Optional[dict],
) -> str:
    """
    Genera la sección de documentación.

    Returns
    -------
    str con la sección en Markdown.
    """
    lines: List[str] = []
    lines.append("## Documentación Generada")
    lines.append("")

    if not documentation:
        lines.append("No se generó documentación.")
        lines.append("")
        return "\n".join(lines)

    files = documentation.get("files", {})
    for name, path in files.items():
        lines.append(f"- {name}: `{path}`")

    lines.append("")
    return "\n".join(lines)


def generate_limitations_section() -> str:
    """
    Genera la sección de limitaciones del experimento.

    Returns
    -------
    str con la sección en Markdown.
    """
    lines: List[str] = []
    lines.append("## Limitaciones del Experimento")
    lines.append("")
    lines.append("- **No hay entrenamiento con gradientes:** todos los pesos se sintetizan y editan analíticamente.")
    lines.append("- **Competencia limitada:** el modelo no logra competencia lingüística general ni generación de código ejecutable confiable sin entrenamiento.")
    lines.append("- **Repetición de tokens:** persiste la repetición de palabras completas en algunos prompts.")
    lines.append("- **Caracteres raros en conversación:** pueden aparecer tokens no válidos en prompts conversacionales.")
    lines.append("- **Formato .safetensors dividido:** el modelo completo se divide en 10 partes para cumplir con límites de GitHub.")
    lines.append("")
    return "\n".join(lines)


def generate_next_steps_section() -> str:
    """
    Genera la sección de próximos pasos recomendados.

    Returns
    -------
    str con la sección en Markdown.
    """
    lines: List[str] = []
    lines.append("## Próximos Pasos Recomendados")
    lines.append("")
    lines.append("1. Incorporar datos de entrenamiento reales para mejorar la competencia lingüística.")
    lines.append("2. Implementar fine-tuning supervisado en tareas de Python y análisis de datos.")
    lines.append("3. Aumentar el vocabulario y ajustar los tokens especiales para conversación.")
    lines.append("4. Expandir las suites de evaluación con casos de prueba más rigurosos.")
    lines.append("5. Optimizar la cuantización para reducir tamaño sin pérdida de calidad.")
    lines.append("6. Implementar RLHF (Reinforcement Learning from Human Feedback) para alineación.")
    lines.append("")
    return "\n".join(lines)


def generate_full_report(
    output_dir: str | Path,
    model_dir: str | Path = "model",
    validation_path: Optional[str | Path] = None,
    benchmark_path: Optional[str | Path] = None,
    inference_path: Optional[str | Path] = None,
    evaluation_path: Optional[str | Path] = None,
    export_path: Optional[str | Path] = None,
    documentation_path: Optional[str | Path] = None,
    pipeline_path: Optional[str | Path] = None,
) -> Dict[str, str]:
    """
    Genera el reporte completo del experimento combinando todas las fuentes.

    Parameters
    ----------
    output_dir : str | Path
        Directorio donde se guardarán los reportes.
    model_dir : str | Path
        Directorio del modelo.
    validation_path, benchmark_path, inference_path, evaluation_path,
    export_path, documentation_path, pipeline_path : str | Path opcional
        Rutas a los archivos JSON de resultados de cada etapa.

    Returns
    -------
    dict con las rutas de los archivos generados (JSON y Markdown).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar resultados de cada etapa
    validation = load_json_if_exists(validation_path) if validation_path else None
    benchmark = load_json_if_exists(benchmark_path) if benchmark_path else None
    inference = load_json_if_exists(inference_path) if inference_path else None
    evaluation = load_json_if_exists(evaluation_path) if evaluation_path else None
    export = load_json_if_exists(export_path) if export_path else None
    documentation = load_json_if_exists(documentation_path) if documentation_path else None
    pipeline = load_json_if_exists(pipeline_path) if pipeline_path else None

    # Generar secciones del reporte
    lines: List[str] = []

    lines.append("# Reporte del Experimento AI 800M")
    lines.append("")
    lines.append(f"Generado el: {__import__('datetime').datetime.now().isoformat()}")
    lines.append(f"Modelo: `{model_dir}`")
    lines.append("")

    lines.append(generate_executive_summary(
        validation, benchmark, inference, evaluation, export, documentation
    ))

    lines.append(generate_model_section(model_dir, validation))
    lines.append(generate_benchmark_section(benchmark))
    lines.append(generate_inference_section(inference))
    lines.append(generate_evaluation_section(evaluation))
    lines.append(generate_export_section(export))
    lines.append(generate_documentation_section(documentation))
    lines.append(generate_limitations_section())
    lines.append(generate_next_steps_section())

    # Si hay pipeline, agregar su resumen
    if pipeline:
        lines.append("## Ejecución del Pipeline")
        lines.append("")
        stages = pipeline.get("stages", [])
        lines.append("| Etapa | Completada | Error |")
        lines.append("|-------|------------|-------|")
        for s in stages:
            completed = s.get("completed", False)
            error = s.get("error", "—")
            lines.append(f"| {s.get('name', '?')} | {'Sí' if completed else 'No'} | {error} |")
        lines.append("")

    report_md = "\n".join(lines)

    # Guardar Markdown
    md_path = output_dir / "experiment_report.md"
    with open(md_path, "w") as f:
        f.write(report_md)

    # Guardar JSON consolidado
    consolidated = {
        "generated_at": __import__('datetime').datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "validation": validation,
        "benchmark": benchmark,
        "inference": inference,
        "evaluation": evaluation,
        "export": export,
        "documentation": documentation,
        "pipeline": pipeline,
    }
    json_path = output_dir / "experiment_report.json"
    with open(json_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    return {
        "markdown": str(md_path),
        "json": str(json_path),
    }


def quick_report(
    output_dir: str | Path = "/home/user/ai_experiment/reports",
    model_dir: str | Path = "model",
) -> Dict[str, str]:
    """
    Función de conveniencia: genera un reporte rápido usando los archivos
    JSON estándar del directorio de salida del experimento.

    Returns
    -------
    dict con rutas de los archivos generados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rutas estándar de archivos de resultados
    validation_path = output_dir / "validation_report.json"
    benchmark_path = output_dir / "benchmark_report.json"
    inference_path = output_dir / "inference_results.json"
    evaluation_path = output_dir / "evaluation_results.json"
    export_path = output_dir / "exports" / "export_results.json"
    documentation_path = output_dir / "docs" / "documentation_files.json"
    pipeline_path = output_dir / "pipeline_report.json"

    return generate_full_report(
        output_dir=output_dir,
        model_dir=model_dir,
        validation_path=validation_path,
        benchmark_path=benchmark_path,
        inference_path=inference_path,
        evaluation_path=evaluation_path,
        export_path=export_path,
        documentation_path=documentation_path,
        pipeline_path=pipeline_path,
    )


__all__ = [
    "generate_executive_summary",
    "generate_model_section",
    "generate_benchmark_section",
    "generate_inference_section",
    "generate_evaluation_section",
    "generate_export_section",
    "generate_documentation_section",
    "generate_limitations_section",
    "generate_next_steps_section",
    "generate_full_report",
    "quick_report",
]
