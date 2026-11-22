#!/usr/bin/env python3
"""
experiment_pipeline.py — Pipeline de experimento para el modelo AI 800M.

Orquesta la ejecución de todas las etapas del experimento en secuencia:

  1. Síntesis de pesos (etapas 0-5).
  2. Validación del modelo.
  3. Benchmarks de rendimiento.
  4. Ejecución de inferencia en prompts de prueba.
  5. Evaluación completa (todas las suites).
  6. Exportación a múltiples formatos.
  7. Generación de documentación.
  8. Guardado de reportes consolidados.

Cada etapa es opcional y puede ejecutarse de forma independiente.
El pipeline guarda un estado de ejecución para poder reanudarse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Importar módulos del proyecto
from scripts.logging_utils import setup_logging, timer, ProgressBar
from scripts.config_manager import ExperimentConfig
from scripts.model_validator import quick_validate
from scripts.benchmark import quick_benchmark
from scripts.evaluation import run_all_suites
from scripts.inference_engine import run_inference
from scripts.model_exporter import export_all_formats
from scripts.doc_generator import generate_all_documentation


logger = setup_logging(level=20)  # INFO


@dataclass
class PipelineStage:
    """Una etapa del pipeline."""
    name: str
    description: str
    enabled: bool = True
    completed: bool = False
    error: Optional[str] = None
    elapsed_sec: float = 0.0
    result: Optional[dict] = None


@dataclass
class PipelineRun:
    """Registro de una ejecución completa del pipeline."""
    start_time: str
    end_time: Optional[str] = None
    stages: List[PipelineStage] = field(default_factory=list)
    overall_success: bool = False
    output_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "overall_success": self.overall_success,
            "output_dir": self.output_dir,
            "stages": [
                {
                    "name": s.name,
                    "description": s.description,
                    "enabled": s.enabled,
                    "completed": s.completed,
                    "error": s.error,
                    "elapsed_sec": s.elapsed_sec,
                    "result": s.result,
                }
                for s in self.stages
            ],
        }


class ExperimentPipeline:
    """
    Pipeline principal del experimento AI 800M.

    Ejecuta todas las etapas en orden y guarda reportes consolidados.
    """

    def __init__(
        self,
        model_dir: str | Path = "model",
        output_dir: str | Path = "/home/user/ai_experiment",
        config: Optional[ExperimentConfig] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ExperimentConfig()

        self.stages: List[PipelineStage] = []
        self.run: Optional[PipelineRun] = None

    # ------------------------------------------------------------------
    # Etapas individuales
    # ------------------------------------------------------------------

    def _stage_synthesize(self) -> dict:
        """Etapa 1: Sintetizar pesos (ejecutar synthesize_weights.py + run_stage5.py)."""
        import subprocess
        import sys

        logger.info("Ejecutando síntesis de pesos...")

        # Ejecutar build_tokenizer.py
        subprocess.run(
            [sys.executable, "scripts/build_tokenizer.py"],
            check=True,
            cwd=str(Path.cwd()),
        )

        # Ejecutar synthesize_weights.py
        subprocess.run(
            [sys.executable, "scripts/synthesize_weights.py"],
            check=True,
            cwd=str(Path.cwd()),
        )

        # Ejecutar run_stage5.py
        subprocess.run(
            [sys.executable, "scripts/run_stage5.py"],
            check=True,
            cwd=str(Path.cwd()),
        )

        return {"status": "completed", "model_dir": str(self.model_dir)}

    def _stage_validate(self) -> dict:
        """Etapa 2: Validar el modelo."""
        logger.info("Validando modelo...")
        report = quick_validate(self.model_dir, include_dynamic=False)
        return {
            "passed": report["passed"],
            "total_checks": report["total_checks"],
            "all_passed": report["all_passed"],
        }

    def _stage_benchmark(self) -> dict:
        """Etapa 3: Benchmarks de rendimiento."""
        logger.info("Ejecutando benchmarks...")
        report = quick_benchmark(self.model_dir)
        return {
            "n_benchmarks": len(report["benchmark_results"]),
            "results": report["benchmark_results"],
        }

    def _stage_inference(self) -> dict:
        """Etapa 4: Ejecutar inferencia en prompts de prueba."""
        logger.info("Ejecutando inferencia de prueba...")

        test_prompts = [
            "def fibonacci(n):",
            "Hola, ¿cómo estás?",
            "Analiza la complejidad de este algoritmo:",
            "for i in range(10):",
            "Explica el concepto de recursividad:",
        ]

        results = run_inference(
            self.model_dir,
            test_prompts,
            weights_filename="model.safetensors",
            max_new_tokens=15,
            temperature=0.0,
        )

        return {
            "n_prompts": len(results),
            "results": [
                {
                    "prompt": r["prompt"],
                    "generated_text": r["generated_text"],
                    "entropy": r["first_token_entropy_nats"],
                    "tokens_per_second": r["tokens_per_second"],
                }
                for r in results
            ],
        }

    def _stage_evaluate(self) -> dict:
        """Etapa 5: Ejecutar evaluación completa."""
        logger.info("Ejecutando evaluación completa...")

        all_results = run_all_suites(
            str(self.model_dir),
            weights_file="model.safetensors",
        )

        # Calcular resumen global
        total_tests = 0
        total_passed = 0
        suite_summaries = {}

        for sname, suite_result in all_results.items():
            total_tests += suite_result["total_tests"]
            total_passed += suite_result["passed"]
            suite_summaries[sname] = {
                "total_tests": suite_result["total_tests"],
                "passed": suite_result["passed"],
                "failed": suite_result["failed"],
                "pass_rate": suite_result["pass_rate"],
            }

        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "global_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            "suites": suite_summaries,
        }

    def _stage_export(self) -> dict:
        """Etapa 6: Exportar a múltiples formatos."""
        logger.info("Exportando modelo a múltiples formatos...")

        export_dir = self.output_dir / "exports"
        results = export_all_formats(self.model_dir, export_dir)

        return {
            "export_dir": str(export_dir),
            "formats": {
                name: {"status": info["status"]}
                for name, info in results.items()
            },
        }

    def _stage_document(self) -> dict:
        """Etapa 7: Generar documentación."""
        logger.info("Generando documentación...")

        project_root = Path.cwd()
        docs_dir = self.output_dir / "docs"
        results = generate_all_documentation(project_root, docs_dir)

        return {
            "docs_dir": str(docs_dir),
            "files": results,
        }

    # ------------------------------------------------------------------
    # Ejecución del pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        stages_to_run: Optional[List[str]] = None,
        stop_on_error: bool = True,
    ) -> PipelineRun:
        """
        Ejecuta el pipeline completo o etapas específicas.

        Parameters
        ----------
        stages_to_run : list de str opcional
            Nombres de etapas a ejecutar. Si es None, se ejecutan todas.
        stop_on_error : bool
            Si True, se detiene en la primera etapa que falle.

        Returns
        -------
        PipelineRun con el registro completo de la ejecución.
        """
        start_time = datetime.now().isoformat()

        # Definir todas las etapas disponibles
        all_stages = [
            PipelineStage("synthesize", "Síntesis de pesos (etapas 0-5)", enabled=True),
            PipelineStage("validate", "Validación del modelo", enabled=True),
            PipelineStage("benchmark", "Benchmarks de rendimiento", enabled=True),
            PipelineStage("inference", "Inferencia de prueba", enabled=True),
            PipelineStage("evaluate", "Evaluación completa (todas las suites)", enabled=True),
            PipelineStage("export", "Exportación a múltiples formatos", enabled=True),
            PipelineStage("document", "Generación de documentación", enabled=True),
        ]

        # Filtrar etapas solicitadas
        if stages_to_run is not None:
            for s in all_stages:
                s.enabled = s.name in stages_to_run

        self.stages = all_stages
        self.run = PipelineRun(
            start_time=start_time,
            output_dir=str(self.output_dir),
        )

        stage_map = {
            "synthesize": self._stage_synthesize,
            "validate": self._stage_validate,
            "benchmark": self._stage_benchmark,
            "inference": self._stage_inference,
            "evaluate": self._stage_evaluate,
            "export": self._stage_export,
            "document": self._stage_document,
        }

        with ProgressBar(len(all_stages), prefix="Etapas del pipeline") as pb:
            for stage in all_stages:
                if not stage.enabled:
                    continue

                logger.info(f"Ejecutando etapa: {stage.name} — {stage.description}")

                with timer(f"etapa {stage.name}", logger):
                    try:
                        result = stage_map[stage.name]()
                        stage.completed = True
                        stage.result = result
                        logger.info(f"Etapa {stage.name} completada exitosamente")
                    except Exception as e:
                        stage.completed = False
                        stage.error = str(e)
                        logger.error(f"Etapa {stage.name} falló: {e}")
                        if stop_on_error:
                            break

                stage.elapsed_sec = 0.0  # el timer ya lo registró en logs
                pb.update(1)

        # Determinar éxito general
        completed_stages = [s for s in self.stages if s.enabled and s.completed]
        enabled_stages = [s for s in self.stages if s.enabled]
        self.run.overall_success = len(completed_stages) == len(enabled_stages)
        self.run.end_time = datetime.now().isoformat()
        self.run.stages = self.stages

        # Guardar reporte del pipeline
        pipeline_report_path = self.output_dir / "pipeline_report.json"
        with open(pipeline_report_path, "w") as f:
            json.dump(self.run.to_dict(), f, indent=2)

        logger.info(
            f"Pipeline finalizado: {len(completed_stages)}/{len(enabled_stages)} etapas completadas, "
            f"éxito global: {self.run.overall_success}"
        )
        logger.info(f"Reporte guardado en {pipeline_report_path}")

        return self.run

    def print_summary(self) -> None:
        """Imprime un resumen del pipeline en consola."""
        if self.run is None:
            print("El pipeline no se ha ejecutado todavía.")
            return

        print("\n" + "=" * 60)
        print("RESUMEN DEL PIPELINE DE EXPERIMENTO")
        print("=" * 60)
        print(f"Inicio: {self.run.start_time}")
        print(f"Fin: {self.run.end_time}")
        print(f"Éxito global: {'SÍ' if self.run.overall_success else 'NO'}")
        print(f"Directorio de salida: {self.run.output_dir}")
        print()

        print("| Etapa | Estado | Tiempo | Error |")
        print("|-------|--------|--------|-------|")
        for s in self.run.stages:
            estado = "✓" if s.completed else ("✗" if s.error else "—")
            print(f"| {s.name} | {estado} | {s.elapsed_sec:.2f}s | {s.error or '—'} |")

        print("=" * 60 + "\n")


def run_pipeline(
    model_dir: str | Path = "model",
    output_dir: str | Path = "/home/user/ai_experiment",
    stages: Optional[List[str]] = None,
) -> PipelineRun:
    """
    Función de conveniencia para ejecutar el pipeline completo.

    Returns
    -------
    PipelineRun con el registro de la ejecución.
    """
    pipeline = ExperimentPipeline(model_dir=model_dir, output_dir=output_dir)
    run = pipeline.run(stages_to_run=stages)
    pipeline.print_summary()
    return run


__all__ = [
    "PipelineStage",
    "PipelineRun",
    "ExperimentPipeline",
    "run_pipeline",
]
