#!/usr/bin/env python3
"""
logging_utils.py — Utilidades de logging para el experimento AI 800M.

Proporciona:
  - Configuración de logging estándar (consola + archivo).
  - Clase ExperimentLogger para registrar eventos del experimento.
  - Métricas de tiempo (temporizadores con contexto).
  - Progreso de barras simples para operaciones largas.
  - Exportación de logs a JSON para análisis posterior.

Todo usando solo la librería estándar de Python.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class ExperimentLogger:
    """
    Logger especializado para el experimento de IA.

    Registra eventos con timestamps, niveles de severidad, y metadatos
    opcionales. Puede escribir a consola y/o a un archivo JSON.
    """

    def __init__(
        self,
        name: str = "ai_experiment",
        log_file: Optional[str | Path] = None,
        level: int = logging.INFO,
        console: bool = True,
    ) -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Limpiar handlers existentes
        self.logger.handlers.clear()

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setLevel(level)
            fformatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(fformatter)
            self.logger.addHandler(fh)

        self._events: List[Dict[str, Any]] = []

    def debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(message)
        self._record("DEBUG", message, kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message)
        self._record("INFO", message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message)
        self._record("WARNING", message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(message)
        self._record("ERROR", message, kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self.logger.critical(message)
        self._record("CRITICAL", message, kwargs)

    def _record(
        self,
        level: str,
        message: str,
        kwargs: Dict[str, Any],
    ) -> None:
        self._events.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "metadata": kwargs,
        })

    def log_stage_start(
        self,
        stage_name: str,
        stage_number: int,
    ) -> None:
        """Registra el inicio de una etapa de síntesis."""
        self.info(
            f"Iniciando etapa {stage_number}: {stage_name}",
            stage=stage_name,
            stage_number=stage_number,
            event="stage_start",
        )

    def log_stage_end(
        self,
        stage_name: str,
        stage_number: int,
        elapsed_sec: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra el final de una etapa de síntesis."""
        self.info(
            f"Etapa {stage_number} ({stage_name}) completada en {elapsed_sec:.2f}s",
            stage=stage_name,
            stage_number=stage_number,
            elapsed_sec=elapsed_sec,
            event="stage_end",
            metrics=metrics or {},
        )

    def log_generation(
        self,
        prompt: str,
        generated_text: str,
        entropy: float,
        elapsed_sec: float,
    ) -> None:
        """Registra una generación de texto."""
        self.info(
            f"Generación: '{prompt[:50]}...' -> '{generated_text[:50]}...' "
            f"(H={entropy:.3f}, {elapsed_sec:.2f}s)",
            prompt=prompt,
            generated_text=generated_text,
            entropy=entropy,
            elapsed_sec=elapsed_sec,
            event="generation",
        )

    def log_weight_edit(
        self,
        tensor_name: str,
        operation: str,
        num_edits: int,
    ) -> None:
        """Registra una edición de pesos."""
        self.info(
            f"Edición de pesos: {tensor_name} ({operation}, {num_edits} cambios)",
            tensor=tensor_name,
            operation=operation,
            num_edits=num_edits,
            event="weight_edit",
        )

    def save_events(
        self,
        path: str | Path,
    ) -> None:
        """Guarda todos los eventos registrados en archivo JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2, ensure_ascii=False)

    def get_event_count(self) -> int:
        """Devuelve el número de eventos registrados."""
        return len(self._events)


# ---------------------------------------------------------------------------
# Temporizador con contexto
# ---------------------------------------------------------------------------

@contextmanager
def timer(
    name: str,
    logger: Optional[ExperimentLogger] = None,
    level: int = logging.INFO,
) -> Any:
    """
    Context manager para medir el tiempo de un bloque de código.

    Ejemplo:
        with timer("carga de modelo", logger):
            model = MmapInt8LlamaModel("model")
    """
    start = time.perf_counter()
    if logger:
        logger.debug(f"Iniciando: {name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if logger:
            logger.info(f"Completado: {name} en {elapsed:.3f}s", event="timer", name=name, elapsed_sec=elapsed)
        else:
            print(f"[TIMER] {name}: {elapsed:.3f}s")


# ---------------------------------------------------------------------------
# Barra de progreso simple
# ---------------------------------------------------------------------------

class ProgressBar:
    """
    Barra de progreso simple para operaciones largas en consola.
    """

    def __init__(
        self,
        total: int,
        width: int = 40,
        prefix: str = "Progreso",
    ) -> None:
        self.total = max(1, total)
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()

    def update(
        self,
        current: int,
        message: Optional[str] = None,
    ) -> None:
        """Actualiza la barra de progreso."""
        self.current = min(current, self.total)
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.current) / rate if rate > 0 else 0.0

        msg = f"\r{self.prefix}: |{bar}| {percent:6.1%} ({self.current}/{self.total}) "
        msg += f"| {rate:.1f} it/s | ETA: {eta:.0f}s"
        if message:
            msg += f" | {message}"

        print(msg, end="", flush=True)

        if self.current >= self.total:
            print()  # nueva línea al finalizar

    def finish(
        self,
        message: Optional[str] = None,
    ) -> None:
        """Finaliza la barra de progreso."""
        self.update(self.total, message)


# ---------------------------------------------------------------------------
# Configuración de logging estándar
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> ExperimentLogger:
    """
    Configura y devuelve un ExperimentLogger listo para usar.
    """
    return ExperimentLogger(
        name="ai_experiment",
        log_file=log_file,
        level=level,
        console=True,
    )


__all__ = [
    "ExperimentLogger",
    "timer",
    "ProgressBar",
    "setup_logging",
]
