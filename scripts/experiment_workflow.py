#!/usr/bin/env python3
"""
experiment_workflow.py — Flujo de trabajo de experimentos para el modelo 800M.

Proporciona herramientas para definir y ejecutar flujos de trabajo
compuestos de múltiples experimentos:

  - Definición de workflows con pasos secuenciales y paralelos.
  - Ejecución de experimentos en lotes (batch).
  - Comparación de resultados entre experimentos de un workflow.
  - Generación de reportes consolidados.
  - Guardado y carga de configuraciones de workflow.
  - Integración con experiment_tracker y model_pipeline_extended.
  - Análisis de tendencias entre experimentos relacionados.
  - Detección de mejoras y regresiones.
  - Exportación de resultados para análisis externo.
  - Plantillas de workflows comunes para el experimento AI 800M.

Todas las operaciones son analíticas y no requieren entrenamiento.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class WorkflowStep:
    """Un paso individual en un workflow de experimentos."""
    step_id: str
    name: str
    experiment_config: Dict[str, object]
    depends_on: List[str] = field(default_factory=list)
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "experiment_config": self.experiment_config,
            "depends_on": self.depends_on,
            "enabled": self.enabled,
            "description": self.description,
        }


@dataclass
class WorkflowExecution:
    """Registro de la ejecución de un workflow."""
    workflow_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed, partial
    step_results: Dict[str, dict] = field(default_factory=dict)
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "step_results": self.step_results,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
        }

    def summary(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status,
            "total_steps": self.total_steps,
            "completed": self.completed_steps,
            "failed": self.failed_steps,
            "success_rate": (
                self.completed_steps / self.total_steps
                if self.total_steps > 0 else 0.0
            ),
        }


@dataclass
class WorkflowComparison:
    """Comparación entre múltiples ejecuciones de workflow."""
    workflow_ids: List[str]
    metric_name: str
    best_workflow_id: str
    best_value: float
    comparison_table: List[dict]

    def to_dict(self) -> dict:
        return {
            "workflow_ids": self.workflow_ids,
            "metric_name": self.metric_name,
            "best_workflow_id": self.best_workflow_id,
            "best_value": round(self.best_value, 6),
            "comparison_table": self.comparison_table,
        }


class ExperimentWorkflow:
    """
    Gestor de flujos de trabajo de experimentos.

    Permite definir workflows con múltiples pasos de experimento,
    ejecutarlos en secuencia, y comparar los resultados.
    """

    def __init__(
        self,
        storage_dir: str | Path = "/home/user/ai_experiment/workflows",
    ) -> None:
        """
        Parameters
        ----------
        storage_dir : str | Path
            Directorio para guardar configuraciones y resultados de workflows.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.executions: Dict[str, WorkflowExecution] = {}

    def create_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str = "",
    ) -> None:
        """
        Crea un nuevo workflow vacío.

        Parameters
        ----------
        workflow_id : str
            Identificador único del workflow.
        name : str
            Nombre legible del workflow.
        description : str
            Descripción del workflow.
        """
        self.workflows[workflow_id] = []
        # Guardar metadata del workflow
        meta = {
            "workflow_id": workflow_id,
            "name": name,
            "description": description,
            "created_at": time.time(),
        }
        meta_path = self.storage_dir / f"{workflow_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def add_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
    ) -> bool:
        """
        Añade un paso a un workflow existente.

        Returns
        -------
        bool: True si se añadió correctamente.
        """
        if workflow_id not in self.workflows:
            return False

        self.workflows[workflow_id].append(step)
        return True

    def get_workflow(
        self,
        workflow_id: str,
    ) -> Optional[List[WorkflowStep]]:
        """Obtiene los pasos de un workflow."""
        return self.workflows.get(workflow_id)

    def list_workflows(
        self,
    ) -> List[dict]:
        """
        Lista todos los workflows definidos.

        Returns
        -------
        Lista de dicts con información de cada workflow.
        """
        result: List[dict] = []
        for wid, steps in self.workflows.items():
            meta_path = self.storage_dir / f"{wid}_meta.json"
            name = wid
            description = ""
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                name = meta.get("name", wid)
                description = meta.get("description", "")

            result.append({
                "workflow_id": wid,
                "name": name,
                "description": description,
                "n_steps": len(steps),
                "enabled_steps": sum(1 for s in steps if s.enabled),
            })
        return result

    def execute_workflow(
        self,
        workflow_id: str,
        execute_func: Callable[[Dict[str, object]], dict],
    ) -> WorkflowExecution:
        """
        Ejecuta un workflow completo paso a paso.

        Parameters
        ----------
        workflow_id : str
            ID del workflow a ejecutar.
        execute_func : Callable
            Función que recibe el config de un experimento y devuelve resultados.

        Returns
        -------
        WorkflowExecution con los resultados.
        """
        import uuid

        steps = self.workflows.get(workflow_id)
        if not steps:
            raise ValueError(f"Workflow no encontrado: {workflow_id}")

        enabled_steps = [s for s in steps if s.enabled]

        execution = WorkflowExecution(
            workflow_id=f"exec_{uuid.uuid4().hex[:8]}",
            name=workflow_id,
            start_time=time.time(),
            total_steps=len(enabled_steps),
        )

        completed_steps: set = set()

        for step in enabled_steps:
            # Verificar dependencias
            deps_met = all(
                dep in completed_steps for dep in step.depends_on
            )

            if not deps_met:
                execution.step_results[step.step_id] = {
                    "status": "skipped",
                    "reason": "dependencies_not_met",
                }
                execution.failed_steps += 1
                continue

            # Ejecutar paso
            try:
                result = execute_func(step.experiment_config)
                execution.step_results[step.step_id] = {
                    "status": "success",
                    "result": result,
                }
                execution.completed_steps += 1
                completed_steps.add(step.step_id)
            except Exception as e:
                execution.step_results[step.step_id] = {
                    "status": "failed",
                    "error": str(e),
                }
                execution.failed_steps += 1

        execution.end_time = time.time()

        if execution.failed_steps > 0:
            execution.status = (
                "partial" if execution.completed_steps > 0 else "failed"
            )
        else:
            execution.status = "completed"

        self.executions[execution.workflow_id] = execution
        self._save_execution(execution)

        return execution

    def _save_execution(
        self,
        execution: WorkflowExecution,
    ) -> None:
        """Guarda una ejecución en el almacenamiento."""
        exec_path = self.storage_dir / f"{execution.workflow_id}.json"
        with open(exec_path, "w") as f:
            json.dump(execution.to_dict(), f, indent=2)

    def get_execution(
        self,
        execution_id: str,
    ) -> Optional[WorkflowExecution]:
        """Obtiene una ejecución por ID."""
        return self.executions.get(execution_id)

    def list_executions(
        self,
        workflow_id: Optional[str] = None,
    ) -> List[WorkflowExecution]:
        """
        Lista todas las ejecuciones, opcionalmente filtradas por workflow.

        Returns
        -------
        Lista de WorkflowExecution.
        """
        result: List[WorkflowExecution] = []
        for exec in self.executions.values():
            if workflow_id is None or exec.name == workflow_id:
                result.append(exec)
        return sorted(result, key=lambda e: e.start_time, reverse=True)

    def compare_workflows(
        self,
        workflow_ids: List[str],
        metric_name: str,
    ) -> WorkflowComparison:
        """
        Compara múltiples workflows por una métrica específica.

        Parameters
        ----------
        workflow_ids : list de str
            IDs de los workflows a comparar.
        metric_name : str
            Nombre de la métrica para comparar.

        Returns
        -------
        WorkflowComparison con la tabla comparativa.
        """
        table: List[dict] = []
        best_id = ""
        best_value = float("-inf")

        for wid in workflow_ids:
            # Buscar la última ejecución de este workflow
            executions = [
                e for e in self.executions.values()
                if e.name == wid
            ]
            if not executions:
                continue

            latest = executions[0]  # Ya están ordenadas por tiempo descendente

            # Buscar la métrica en los resultados de pasos
            metric_value = 0.0
            for step_id, step_result in latest.step_results.items():
                if step_result.get("status") == "success":
                    result_data = step_result.get("result", {})
                    if metric_name in result_data:
                        metric_value = result_data[metric_name]
                        break

            table.append({
                "workflow_id": wid,
                "execution_id": latest.workflow_id,
                "metric_value": round(metric_value, 6),
                "status": latest.status,
            })

            if metric_value > best_value:
                best_value = metric_value
                best_id = wid

        return WorkflowComparison(
            workflow_ids=workflow_ids,
            metric_name=metric_name,
            best_workflow_id=best_id,
            best_value=best_value,
            comparison_table=table,
        )

    def generate_report(
        self,
        workflow_id: Optional[str] = None,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Genera un reporte de workflows en Markdown.

        Parameters
        ----------
        workflow_id : str opcional
            Si se especifica, solo genera reporte para ese workflow.
        output_path : str | Path opcional
            Ruta para guardar el reporte.

        Returns
        -------
        str con el reporte en Markdown.
        """
        lines: List[str] = []
        lines.append("# Reporte de Flujos de Trabajo de Experimentos")
        lines.append("")
        lines.append(f"Total de workflows: {len(self.workflows)}")
        lines.append(f"Total de ejecuciones: {len(self.executions)}")
        lines.append("")

        # Lista de workflows
        lines.append("## Workflows Definidos")
        lines.append("")
        lines.append("| ID | Nombre | Pasos | Habilitados |")
        lines.append("|----|--------|-------|-------------|")

        for wf in self.list_workflows():
            lines.append(
                f"| {wf['workflow_id']} | {wf['name']} | "
                f"{wf['n_steps']} | {wf['enabled_steps']} |"
            )

        lines.append("")

        # Ejecuciones recientes
        lines.append("## Ejecuciones Recientes")
        lines.append("")
        lines.append("| ID | Workflow | Estado | Pasos | Completadas | Falladas |")
        lines.append("|----|----------|--------|-------|-------------|----------|")

        recent_execs = self.list_executions(workflow_id)[:20]
        for exec in recent_execs:
            lines.append(
                f"| {exec.workflow_id} | {exec.name} | {exec.status} | "
                f"{exec.total_steps} | {exec.completed_steps} | {exec.failed_steps} |"
            )

        if len(self.executions) > 20:
            lines.append(f"| ... ({len(self.executions) - 20} más) | | | | | |")

        lines.append("")

        # Detalle de un workflow específico
        if workflow_id and workflow_id in self.workflows:
            lines.append(f"## Detalle: {workflow_id}")
            lines.append("")
            steps = self.workflows[workflow_id]
            lines.append("| Paso | Nombre | Depende de | Habilitado | Descripción |")
            lines.append("|------|--------|------------|------------|-------------|")

            for step in steps:
                deps = ", ".join(step.depends_on) if step.depends_on else "—"
                lines.append(
                    f"| {step.step_id} | {step.name} | {deps} | "
                    f"{'Sí' if step.enabled else 'No'} | {step.description} |"
                )

            lines.append("")

        # Ejecuciones del workflow específico
        if workflow_id:
            execs = [e for e in self.executions.values() if e.name == workflow_id]
            if execs:
                lines.append(f"## Ejecuciones de {workflow_id}")
                lines.append("")
                lines.append("| ID | Estado | Tasa de éxito | Inicio |")
                lines.append("|----|--------|---------------|--------|")

                for exec in execs[:10]:
                    summary = exec.summary()
                    lines.append(
                        f"| {exec.workflow_id} | {exec.status} | "
                        f"{summary['success_rate']:.1%} | "
                        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(exec.start_time))} |"
                    )

                lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")

        return report_text

    def export_executions(
        self,
        output_path: str | Path,
    ) -> str:
        """
        Exporta todas las ejecuciones a un archivo JSON.

        Returns
        -------
        str con la ruta del archivo exportado.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "workflows": {
                wid: [s.to_dict() for s in steps]
                for wid, steps in self.workflows.items()
            },
            "executions": {
                eid: e.to_dict()
                for eid, e in self.executions.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def import_executions(
        self,
        input_path: str | Path,
    ) -> None:
        """
        Importa ejecuciones desde un archivo JSON.

        Parameters
        ----------
        input_path : str | Path
            Ruta al archivo JSON a importar.
        """
        input_path = Path(input_path)
        with open(input_path, "r") as f:
            data = json.load(f)

        for wid, steps_data in data.get("workflows", {}).items():
            if wid not in self.workflows:
                self.workflows[wid] = []
            for s_data in steps_data:
                step = WorkflowStep(
                    step_id=s_data["step_id"],
                    name=s_data["name"],
                    experiment_config=s_data.get("experiment_config", {}),
                    depends_on=s_data.get("depends_on", []),
                    enabled=s_data.get("enabled", True),
                    description=s_data.get("description", ""),
                )
                self.workflows[wid].append(step)

        for eid, e_data in data.get("executions", {}).items():
            execution = WorkflowExecution(
                workflow_id=e_data["workflow_id"],
                name=e_data["name"],
                start_time=e_data["start_time"],
                end_time=e_data.get("end_time"),
                status=e_data.get("status", "running"),
                step_results=e_data.get("step_results", {}),
                total_steps=e_data.get("total_steps", 0),
                completed_steps=e_data.get("completed_steps", 0),
                failed_steps=e_data.get("failed_steps", 0),
            )
            self.executions[eid] = execution


def quick_workflow_demo() -> ExperimentWorkflow:
    """
    Función de demostración: crea un workflow con pasos de ejemplo.

    Returns
    -------
    ExperimentWorkflow con pasos configurados.
    """
    workflow = ExperimentWorkflow()
    workflow.create_workflow(
        workflow_id="demo_workflow",
        name="Workflow de Demostración",
        description="Workflow de ejemplo para probar el sistema",
    )

    def dummy_execute(config: dict) -> dict:
        return {
            "entropy": 5.0 + (hash(str(config)) % 100) / 100.0,
            "pass_rate": 0.3 + (hash(str(config)) % 50) / 100.0,
        }

    workflow.add_step(
        workflow_id="demo_workflow",
        step=WorkflowStep(
            step_id="step1",
            name="Síntesis base",
            experiment_config={"method": "base", "layers": 21},
            description="Síntesis de pesos base",
        ),
    )
    workflow.add_step(
        workflow_id="demo_workflow",
        step=WorkflowStep(
            step_id="step2",
            name="Etapa 5 ajustada",
            experiment_config={"method": "stage5", "induction": 0.06},
            depends_on=["step1"],
            description="Etapa 5 con circuito de inducción reducido",
        ),
    )
    workflow.add_step(
        workflow_id="demo_workflow",
        step=WorkflowStep(
            step_id="step3",
            name="Con suppression",
            experiment_config={"method": "stage5", "space_suppression": -6.0},
            depends_on=["step2"],
            description="Con suppression de espacios -6.0",
        ),
    )

    # Ejecutar workflow
    workflow.execute_workflow("demo_workflow", dummy_execute)

    return workflow


__all__ = [
    "WorkflowStep",
    "WorkflowExecution",
    "WorkflowComparison",
    "ExperimentWorkflow",
    "quick_workflow_demo",
]
