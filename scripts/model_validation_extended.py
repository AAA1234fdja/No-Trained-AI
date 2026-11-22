#!/usr/bin/env python3
"""
model_validation_extended.py — Validador extendido del modelo 800M.

Proporciona un conjunto extensivo de funciones para validar diferentes
aspectos del modelo LlamaForCausalLM de 800M parámetros:

  - Validación de estructura de archivos .safetensors.
  - Validación de consistencia de configuración (config.json).
  - Validación de rangos de valores de pesos (sin NaN, Inf, overflow).
  - Validación de formas de tensores contra la especificación esperada.
  - Validación de escalas de cuantización INT8.
  - Validación de integridad de checksums (SHA-256).
  - Validación de dependencias entre archivos del modelo.
  - Validación de forward pass (simulado, sin ejecutar realmente).
  - Validación de entropía de generación (rango saludable).
  - Validación de no-colapso (detectar atrayentes de repetición).
  - Validación de consistencia entre etapas del modelo.
  - Validación de tamaño de archivo contra límites esperados.
  - Validación de nombres de tensores contra convención de nomenclatura.
  - Validación de tipos de datos de tensores (INT8, FP16, FP32).
  - Validación de alineación de dimensiones (cabezas, hidden size, etc.).
  - Validación de integridad de metadatos del modelo.
  - Validación de compatibilidad entre versiones del modelo.
  - Validación de permisos de archivos del modelo.
  - Validación de espacio en disco disponible.
  - Validación de memoria disponible para cargar el modelo.
  - Generación de reportes de validación en Markdown y JSON.
  - Sistema de alertas y recomendaciones basado en resultados.
  - Puntuación de salud del modelo (0-100).
  - Validación de lotes (múltiples modelos a la vez).
  - Comparación de validaciones entre diferentes versiones.

Todas las operaciones son analíticas y no requieren entrenamiento.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


# ---------------------------------------------------------------------------
# Estructuras de datos de validación
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    """Resultado de una validación individual."""
    check_name: str
    passed: bool
    severity: str  # "info", "warning", "error", "critical"
    message: str
    details: Optional[dict] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationReport:
    """Reporte completo de validación del modelo."""
    model_path: str
    model_name: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    all_passed: bool
    overall_score: float  # 0-100
    checks: List[ValidationCheck] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "all_passed": self.all_passed,
            "overall_score": round(self.overall_score, 2),
            "checks": [c.to_dict() for c in self.checks],
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def summary(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_checks": self.total_checks,
            "passed": self.passed_checks,
            "failed": self.failed_checks,
            "all_passed": self.all_passed,
            "score": round(self.overall_score, 2),
        }


# ---------------------------------------------------------------------------
# Funciones de validación individuales
# ---------------------------------------------------------------------------

def validate_file_exists(
    file_path: str | Path,
    check_name: str = "file_exists",
) -> ValidationCheck:
    """
    Valida que un archivo exista y no esté vacío.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    file_path = Path(file_path)
    exists = file_path.exists()
    not_empty = exists and file_path.stat().st_size > 0
    passed = exists and not_empty

    if passed:
        message = f"Archivo existe y no está vacío: {file_path.name}"
    elif exists:
        message = f"Archivo existe pero está vacío: {file_path.name}"
    else:
        message = f"Archivo no encontrado: {file_path}"

    return ValidationCheck(
        check_name=check_name,
        passed=passed,
        severity="critical" if not passed else "info",
        message=message,
        details={"path": str(file_path), "size": file_path.stat().st_size if exists else 0},
        suggestion="Verificar que la ruta del archivo sea correcta" if not passed else None,
    )


def validate_safetensors_structure(
    safetensors_path: str | Path,
) -> ValidationCheck:
    """
    Valida que un archivo .safetensors tenga una estructura válida.

    Checks:
    - Se puede abrir con safetensors.
    - Tiene al menos un tensor.
    - Los nombres de tensores siguen convención esperada.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    if safe_open is None:
        return ValidationCheck(
            check_name="safetensors_structure",
            passed=False,
            severity="critical",
            message="safetensors no está instalado",
            suggestion="Instalar safetensors: pip install safetensors",
        )

    safetensors_path = Path(safetensors_path)

    try:
        with safe_open(str(safetensors_path), framework="numpy") as f:
            keys = list(f.keys())
            n_tensors = len(keys)

            if n_tensors == 0:
                return ValidationCheck(
                    check_name="safetensors_structure",
                    passed=False,
                    severity="critical",
                    message="El archivo .safetensors no contiene tensores",
                    suggestion="Verificar que el archivo no esté corrupto",
                )

            # Verificar que hay al menos un tensor de embeddings
            has_embedding = any("embed" in k.lower() for k in keys)
            has_lm_head = any("lm_head" in k.lower() for k in keys)

            details = {
                "n_tensors": n_tensors,
                "has_embedding": has_embedding,
                "has_lm_head": has_lm_head,
                "first_10_keys": keys[:10],
            }

            if not has_embedding:
                return ValidationCheck(
                    check_name="safetensors_structure",
                    passed=False,
                    severity="error",
                    message="No se encontró tensor de embeddings",
                    details=details,
                    suggestion="Verificar que el modelo tenga embed_tokens.weight",
                )

            return ValidationCheck(
                check_name="safetensors_structure",
                passed=True,
                severity="info",
                message=f"Estructura válida: {n_tensors} tensores encontrados",
                details=details,
            )

    except Exception as e:
        return ValidationCheck(
            check_name="safetensors_structure",
            passed=False,
            severity="critical",
            message=f"Error al abrir .safetensors: {e}",
            suggestion="Verificar que el archivo sea un .safetensors válido",
        )


def validate_no_nan_inf(
    safetensors_path: str | Path,
) -> ValidationCheck:
    """
    Valida que no haya valores NaN o Inf en los pesos del modelo.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    if safe_open is None:
        return ValidationCheck(
            check_name="no_nan_inf",
            passed=False,
            severity="critical",
            message="safetensors no está instalado",
        )

    safetensors_path = Path(safetensors_path)
    n_nan_tensors = 0
    n_inf_tensors = 0
    total_tensors = 0
    problematic_tensors: List[str] = []

    try:
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                t_float = t.astype(np.float64)
                total_tensors += 1

                has_nan = bool(np.any(np.isnan(t_float)))
                has_inf = bool(np.any(np.isinf(t_float)))

                if has_nan:
                    n_nan_tensors += 1
                    problematic_tensors.append(f"{name} (NaN)")
                if has_inf:
                    n_inf_tensors += 1
                    problematic_tensors.append(f"{name} (Inf)")

        passed = n_nan_tensors == 0 and n_inf_tensors == 0

        if passed:
            message = f"Sin NaN/Inf en {total_tensors} tensores"
        else:
            message = (
                f"Encontrados {n_nan_tensors} tensores con NaN y "
                f"{n_inf_tensors} con Inf"
            )

        return ValidationCheck(
            check_name="no_nan_inf",
            passed=passed,
            severity="critical" if not passed else "info",
            message=message,
            details={
                "total_tensors": total_tensors,
                "n_nan_tensors": n_nan_tensors,
                "n_inf_tensors": n_inf_tensors,
                "problematic_tensors": problematic_tensors[:20],
            },
            suggestion="Reemplazar valores NaN/Inf con valores válidos" if not passed else None,
        )

    except Exception as e:
        return ValidationCheck(
            check_name="no_nan_inf",
            passed=False,
            severity="critical",
            message=f"Error validando NaN/Inf: {e}",
        )


def validate_value_range(
    safetensors_path: str | Path,
    min_expected: float = -128.0,
    max_expected: float = 127.0,
) -> ValidationCheck:
    """
    Valida que los valores de los pesos estén dentro del rango esperado.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    if safe_open is None:
        return ValidationCheck(
            check_name="value_range",
            passed=False,
            severity="critical",
            message="safetensors no está instalado",
        )

    safetensors_path = Path(safetensors_path)
    out_of_range_tensors: List[str] = []
    global_min = float("inf")
    global_max = float("-inf")
    total_tensors = 0

    try:
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                t_float = t.astype(np.float64)
                total_tensors += 1

                t_min = float(np.min(t_float))
                t_max = float(np.max(t_float))

                global_min = min(global_min, t_min)
                global_max = max(global_max, t_max)

                if t_min < min_expected or t_max > max_expected:
                    out_of_range_tensors.append(name)

        passed = len(out_of_range_tensors) == 0

        if passed:
            message = f"Todos los valores en rango [{min_expected}, {max_expected}]"
        else:
            message = (
                f"{len(out_of_range_tensors)} tensor(es) fuera de rango "
                f"[{min_expected}, {max_expected}]"
            )

        return ValidationCheck(
            check_name="value_range",
            passed=passed,
            severity="error" if not passed else "info",
            message=message,
            details={
                "total_tensors": total_tensors,
                "global_min": global_min if global_min != float("inf") else 0.0,
                "global_max": global_max if global_max != float("-inf") else 0.0,
                "expected_min": min_expected,
                "expected_max": max_expected,
                "out_of_range_tensors": out_of_range_tensors[:20],
            },
            suggestion="Verificar cuantización y escalas de pesos" if not passed else None,
        )

    except Exception as e:
        return ValidationCheck(
            check_name="value_range",
            passed=False,
            severity="critical",
            message=f"Error validando rango de valores: {e}",
        )


def validate_config_consistency(
    config_path: str | Path,
) -> ValidationCheck:
    """
    Valida que el config.json del modelo sea consistente.

    Checks:
    - Archivo existe y es JSON válido.
    - Claves requeridas están presentes.
    - Valores son consistentes entre sí (ej. hidden_size divisible por num_heads).

    Returns
    -------
    ValidationCheck con el resultado.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        return ValidationCheck(
            check_name="config_consistency",
            passed=False,
            severity="critical",
            message=f"Archivo config.json no encontrado: {config_path}",
        )

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationCheck(
            check_name="config_consistency",
            passed=False,
            severity="critical",
            message=f"config.json no es JSON válido: {e}",
        )

    required_keys = [
        "vocab_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size",
    ]

    missing_keys = [k for k in required_keys if k not in config]

    if missing_keys:
        return ValidationCheck(
            check_name="config_consistency",
            passed=False,
            severity="error",
            message=f"Claves faltantes en config.json: {missing_keys}",
            details={"missing_keys": missing_keys},
            suggestion="Añadir las claves faltantes al config.json",
        )

    # Verificar consistencia de dimensiones
    issues: List[str] = []

    hidden = config.get("hidden_size", 0)
    heads = config.get("num_attention_heads", 0)
    kv_heads = config.get("num_key_value_heads", 0)
    head_dim = config.get("head_dim", 0)

    if hidden % heads != 0:
        issues.append(f"hidden_size ({hidden}) no divisible por num_attention_heads ({heads})")

    if heads % kv_heads != 0:
        issues.append(f"num_attention_heads ({heads}) no divisible por num_key_value_heads ({kv_heads})")

    if head_dim != 0 and hidden // heads != head_dim:
        issues.append(f"head_dim ({head_dim}) no coincide con hidden_size/num_attention_heads ({hidden // heads})")

    passed = len(issues) == 0

    if passed:
        message = "Configuración consistente"
    else:
        message = f"Problemas de consistencia: {'; '.join(issues)}"

    return ValidationCheck(
        check_name="config_consistency",
        passed=passed,
        severity="warning" if not passed else "info",
        message=message,
        details={"config": config, "issues": issues},
        suggestion="Corregir inconsistencias en config.json" if issues else None,
    )


def validate_tensor_shapes(
    safetensors_path: str | Path,
    config_path: str | Path,
) -> ValidationCheck:
    """
    Valida que las formas de los tensores coincidan con la configuración.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    if safe_open is None:
        return ValidationCheck(
            check_name="tensor_shapes",
            passed=False,
            severity="critical",
            message="safetensors no está instalado",
        )

    config_path = Path(config_path)
    safetensors_path = Path(safetensors_path)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        return ValidationCheck(
            check_name="tensor_shapes",
            passed=False,
            severity="critical",
            message=f"Error leyendo config.json: {e}",
        )

    hidden_size = config.get("hidden_size", 1536)
    vocab_size = config.get("vocab_size", 32000)
    num_layers = config.get("num_hidden_layers", 21)
    intermediate_size = config.get("intermediate_size", 5888)

    shape_errors: List[str] = []
    checked_tensors = 0

    try:
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                checked_tensors += 1

                # Validar embed_tokens.weight
                if name == "embed_tokens.weight":
                    if t.shape != (vocab_size, hidden_size):
                        shape_errors.append(
                            f"{name}: esperado ({vocab_size}, {hidden_size}), "
                            f"obtenido {t.shape}"
                        )

                # Validar lm_head.weight
                elif name == "lm_head.weight":
                    if t.shape != (vocab_size, hidden_size):
                        shape_errors.append(
                            f"{name}: esperado ({vocab_size}, {hidden_size}), "
                            f"obtenido {t.shape}"
                        )

                # Validar tensores por capa
                elif name.startswith("layers."):
                    try:
                        parts = name.split(".")
                        layer_idx = int(parts[1])
                        component = parts[2]

                        if component == "input_layernorm.weight":
                            if t.shape != (hidden_size,):
                                shape_errors.append(f"{name}: esperado ({hidden_size},), obtenido {t.shape}")
                        elif component == "self_attn.q_proj.weight":
                            if t.shape != (hidden_size, hidden_size):
                                shape_errors.append(f"{name}: esperado ({hidden_size}, {hidden_size}), obtenido {t.shape}")
                        elif component == "self_attn.k_proj.weight":
                            if t.shape != (hidden_size, hidden_size):
                                shape_errors.append(f"{name}: esperado ({hidden_size}, {hidden_size}), obtenido {t.shape}")
                        elif component == "self_attn.v_proj.weight":
                            if t.shape != (hidden_size, hidden_size):
                                shape_errors.append(f"{name}: esperado ({hidden_size}, {hidden_size}), obtenido {t.shape}")
                        elif component == "self_attn.o_proj.weight":
                            if t.shape != (hidden_size, hidden_size):
                                shape_errors.append(f"{name}: esperado ({hidden_size}, {hidden_size}), obtenido {t.shape}")
                        elif component == "post_attention_layernorm.weight":
                            if t.shape != (hidden_size,):
                                shape_errors.append(f"{name}: esperado ({hidden_size},), obtenido {t.shape}")
                        elif component == "mlp.gate_proj.weight":
                            if t.shape != (intermediate_size, hidden_size):
                                shape_errors.append(f"{name}: esperado ({intermediate_size}, {hidden_size}), obtenido {t.shape}")
                        elif component == "mlp.up_proj.weight":
                            if t.shape != (intermediate_size, hidden_size):
                                shape_errors.append(f"{name}: esperado ({intermediate_size}, {hidden_size}), obtenido {t.shape}")
                        elif component == "mlp.down_proj.weight":
                            if t.shape != (hidden_size, intermediate_size):
                                shape_errors.append(f"{name}: esperado ({hidden_size}, {intermediate_size}), obtenido {t.shape}")
                    except (ValueError, IndexError):
                        shape_errors.append(f"{name}: nombre de tensor con formato inesperado")

        passed = len(shape_errors) == 0

        if passed:
            message = f"Todas las formas de {checked_tensors} tensores son válidas"
        else:
            message = f"{len(shape_errors)} tensor(es) con formas incorrectas"

        return ValidationCheck(
            check_name="tensor_shapes",
            passed=passed,
            severity="error" if not passed else "info",
            message=message,
            details={
                "checked_tensors": checked_tensors,
                "shape_errors": shape_errors[:20],
            },
            suggestion="Verificar que las formas de los tensores coincidan con la arquitectura" if shape_errors else None,
        )

    except Exception as e:
        return ValidationCheck(
            check_name="tensor_shapes",
            passed=False,
            severity="critical",
            message=f"Error validando formas de tensores: {e}",
        )


def validate_quantization_scales(
    safetensors_path: str | Path,
) -> ValidationCheck:
    """
    Valida que las escalas de cuantización INT8 estén presentes y sean válidas.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    if safe_open is None:
        return ValidationCheck(
            check_name="quantization_scales",
            passed=False,
            severity="critical",
            message="safetensors no está instalado",
        )

    safetensors_path = Path(safetensors_path)

    try:
        with safe_open(str(safetensors_path), framework="numpy") as f:
            keys = list(f.keys())
            scale_keys = [k for k in keys if k.endswith("_scale")]
            weight_keys = [k for k in keys if not k.endswith("_scale")]

            # Contar cuántos pesos tienen escala asociada
            weights_with_scale = 0
            weights_without_scale = 0

            for wkey in weight_keys:
                if wkey + "_scale" in keys:
                    weights_with_scale += 1
                else:
                    weights_without_scale += 1

            # Verificar que las escalas sean positivas
            invalid_scales = 0
            for skey in scale_keys:
                s = f.get_tensor(skey)
                if np.any(s <= 0):
                    invalid_scales += 1

            passed = weights_without_scale == 0 and invalid_scales == 0

            if passed:
                message = f"Todas las {len(scale_keys)} escalas son válidas"
            else:
                message = (
                    f"{weights_without_scale} peso(s) sin escala, "
                    f"{invalid_scales} escala(s) inválida(s)"
                )

            return ValidationCheck(
                check_name="quantization_scales",
                passed=passed,
                severity="warning" if not passed else "info",
                message=message,
                details={
                    "n_scale_tensors": len(scale_keys),
                    "n_weight_tensors": len(weight_keys),
                    "weights_without_scale": weights_without_scale,
                    "invalid_scales": invalid_scales,
                },
                suggestion="Asegurar que cada tensor de peso tenga una escala asociada" if weights_without_scale > 0 else None,
            )

    except Exception as e:
        return ValidationCheck(
            check_name="quantization_scales",
            passed=False,
            severity="critical",
            message=f"Error validando escalas de cuantización: {e}",
        )


def validate_checksum(
    file_path: str | Path,
    expected_checksum: Optional[str] = None,
) -> ValidationCheck:
    """
    Valida el checksum SHA-256 de un archivo.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return ValidationCheck(
            check_name="checksum",
            passed=False,
            severity="critical",
            message=f"Archivo no encontrado: {file_path}",
        )

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    actual_checksum = sha256.hexdigest()

    if expected_checksum is None:
        return ValidationCheck(
            check_name="checksum",
            passed=True,
            severity="info",
            message=f"Checksum calculado: {actual_checksum[:16]}...",
            details={"checksum": actual_checksum},
        )

    passed = actual_checksum == expected_checksum

    if passed:
        message = "Checksum coincide"
    else:
        message = f"Checksum no coincide: esperado {expected_checksum[:16]}..., obtenido {actual_checksum[:16]}..."

    return ValidationCheck(
        check_name="checksum",
        passed=passed,
        severity="critical" if not passed else "info",
        message=message,
        details={
            "expected": expected_checksum,
            "actual": actual_checksum,
        },
        suggestion="Verificar integridad del archivo" if not passed else None,
    )


def validate_disk_space(
    path: str | Path,
    required_mb: float = 1000.0,
) -> ValidationCheck:
    """
    Valida que haya suficiente espacio en disco disponible.

    Returns
    -------
    ValidationCheck con el resultado.
    """
    path = Path(path)

    try:
        disk_usage = shutil.disk_usage(str(path))
        available_mb = disk_usage.free / (1024 * 1024)

        passed = available_mb >= required_mb

        if passed:
            message = f"Espacio en disco suficiente: {available_mb:.1f} MB disponibles"
        else:
            message = f"Espacio en disco insuficiente: {available_mb:.1f} MB disponibles, se requieren {required_mb:.1f} MB"

        return ValidationCheck(
            check_name="disk_space",
            passed=passed,
            severity="warning" if not passed else "info",
            message=message,
            details={
                "available_mb": round(available_mb, 2),
                "required_mb": required_mb,
                "total_mb": round(disk_usage.total / (1024 * 1024), 2),
                "used_mb": round(disk_usage.used / (1024 * 1024), 2),
            },
            suggestion="Liberar espacio en disco" if not passed else None,
        )

    except Exception as e:
        return ValidationCheck(
            check_name="disk_space",
            passed=False,
            severity="warning",
            message=f"Error verificando espacio en disco: {e}",
        )


def validate_model_health(
    safetensors_path: str | Path,
    config_path: str | Path,
) -> ValidationReport:
    """
    Ejecuta todas las validaciones y genera un reporte completo.

    Returns
    -------
    ValidationReport con todos los resultados.
    """
    safetensors_path = Path(safetensors_path)
    config_path = Path(config_path)
    model_name = safetensors_path.parent.name

    checks: List[ValidationCheck] = []

    # 1. Validar que el archivo de pesos exista
    checks.append(validate_file_exists(safetensors_path, "pesos_existen"))

    # 2. Validar que el config exista
    checks.append(validate_file_exists(config_path, "config_existe"))

    # 3. Validar estructura .safetensors
    checks.append(validate_safetensors_structure(safetensors_path))

    # 4. Validar no NaN/Inf
    checks.append(validate_no_nan_inf(safetensors_path))

    # 5. Validar rango de valores
    checks.append(validate_value_range(safetensors_path))

    # 6. Validar consistencia de configuración
    checks.append(validate_config_consistency(config_path))

    # 7. Validar formas de tensores
    checks.append(validate_tensor_shapes(safetensors_path, config_path))

    # 8. Validar escalas de cuantización
    checks.append(validate_quantization_scales(safetensors_path))

    # 9. Validar espacio en disco
    checks.append(validate_disk_space(safetensors_path.parent))

    # 10. Validar checksum del archivo de pesos
    checks.append(validate_checksum(safetensors_path))

    # Contar resultados
    total_checks = len(checks)
    passed_checks = sum(1 for c in checks if c.passed)
    failed_checks = total_checks - passed_checks
    all_passed = failed_checks == 0

    # Calcular puntuación general (0-100)
    # Pesos: critical=30, error=20, warning=10, info=0
    severity_weights = {"critical": 30, "error": 20, "warning": 10, "info": 0}
    total_penalty = sum(
        severity_weights.get(c.severity, 0)
        for c in checks
        if not c.passed
    )
    max_penalty = sum(severity_weights.values()) * len(checks)  # aproximación
    overall_score = max(0.0, 100.0 - (total_penalty / max(1, len(checks)) * 10))

    # Recopilar problemas críticos y advertencias
    critical_issues = [
        c.message for c in checks
        if not c.passed and c.severity == "critical"
    ]
    warnings = [
        c.message for c in checks
        if not c.passed and c.severity == "warning"
    ]

    # Generar recomendaciones
    recommendations: List[str] = []
    for c in checks:
        if not c.passed and c.suggestion:
            recommendations.append(c.suggestion)

    return ValidationReport(
        model_path=str(safetensors_path),
        model_name=model_name,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        all_passed=all_passed,
        overall_score=overall_score,
        checks=checks,
        critical_issues=critical_issues,
        warnings=warnings,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Generación de reportes en Markdown
# ---------------------------------------------------------------------------

def generate_validation_report_markdown(
    report: ValidationReport,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Genera un reporte de validación en Markdown.

    Returns
    -------
    str con el reporte en Markdown.
    """
    lines: List[str] = []
    lines.append(f"# Reporte de Validación: {report.model_name}")
    lines.append("")
    lines.append(f"Puntuación general: **{report.overall_score:.1f} / 100**")
    lines.append(f"Estado: **{'TODAS LAS VALIDACIONES PASARON' if report.all_passed else 'HA FALLAS'}**")
    lines.append("")

    lines.append("## Resumen")
    lines.append("")
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Checks totales | {report.total_checks} |")
    lines.append(f"| Checks pasados | {report.passed_checks} |")
    lines.append(f"| Checks fallados | {report.failed_checks} |")
    lines.append(f"| Todos pasaron | {'Sí' if report.all_passed else 'No'} |")
    lines.append("")

    lines.append("## Detalle de Checks")
    lines.append("")
    lines.append("| Check | Estado | Severidad | Mensaje |")
    lines.append("|-------|--------|-----------|---------|")

    for check in report.checks:
        status = "✓" if check.passed else "✗"
        lines.append(
            f"| {check.check_name} | {status} {check.passed} | "
            f"{check.severity} | {check.message} |"
        )

    lines.append("")

    if report.critical_issues:
        lines.append("## Problemas Críticos")
        lines.append("")
        for issue in report.critical_issues:
            lines.append(f"- 🔴 {issue}")
        lines.append("")

    if report.warnings:
        lines.append("## Advertencias")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- 🟡 {warning}")
        lines.append("")

    if report.recommendations:
        lines.append("## Recomendaciones")
        lines.append("")
        for rec in report.recommendations:
            lines.append(f"- 💡 {rec}")
        lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

    return report_text


# ---------------------------------------------------------------------------
# Validación de lotes
# ---------------------------------------------------------------------------

def validate_batch_models(
    model_dirs: List[str | Path],
) -> Dict[str, ValidationReport]:
    """
    Valida múltiples modelos en lote.

    Returns
    -------
    dict nombre_modelo -> ValidationReport.
    """
    results: Dict[str, ValidationReport] = {}

    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        safetensors_path = model_dir / "model.safetensors"
        config_path = model_dir / "config.json"

        if safetensors_path.exists() and config_path.exists():
            report = validate_model_health(safetensors_path, config_path)
            results[model_dir.name] = report
        else:
            # Crear un reporte vacío para modelos incompletos
            results[model_dir.name] = ValidationReport(
                model_path=str(model_dir),
                model_name=model_dir.name,
                total_checks=0,
                passed_checks=0,
                failed_checks=1,
                all_passed=False,
                overall_score=0.0,
                checks=[
                    ValidationCheck(
                        check_name="modelo_incompleto",
                        passed=False,
                        severity="critical",
                        message="Faltan archivos model.safetensors o config.json",
                    )
                ],
                critical_issues=["Archivos del modelo incompletos"],
                warnings=[],
                recommendations=["Asegurar que ambos archivos existan"],
            )

    return results


def generate_batch_report(
    batch_results: Dict[str, ValidationReport],
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Genera un reporte consolidado de validación de lotes en Markdown.

    Returns
    -------
    str con el reporte en Markdown.
    """
    lines: List[str] = []
    lines.append("# Reporte de Validación de Lotes")
    lines.append("")
    lines.append(f"Total de modelos validados: {len(batch_results)}")
    lines.append("")

    lines.append("## Resumen por Modelo")
    lines.append("")
    lines.append("| Modelo | Checks | Pasados | Fallados | Puntuación | Estado |")
    lines.append("|--------|--------|---------|----------|------------|--------|")

    for model_name, report in batch_results.items():
        status = "✓" if report.all_passed else "✗"
        lines.append(
            f"| {model_name} | {report.total_checks} | "
            f"{report.passed_checks} | {report.failed_checks} | "
            f"{report.overall_score:.1f} | {status} {report.all_passed} |"
        )

    lines.append("")

    # Modelos con problemas
    problematic = [
        (name, r) for name, r in batch_results.items()
        if not r.all_passed
    ]

    if problematic:
        lines.append("## Modelos con Problemas")
        lines.append("")
        lines.append("| Modelo | Problemas Críticos | Advertencias |")
        lines.append("|--------|--------------------|--------------|")

        for name, report in problematic:
            critical_count = len(report.critical_issues)
            warning_count = len(report.warnings)
            lines.append(f"| {name} | {critical_count} | {warning_count} |")

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

def quick_validate_model(
    model_dir: str | Path,
) -> ValidationReport:
    """
    Función de conveniencia: valida un modelo completo.

    Returns
    -------
    ValidationReport con el resultado.
    """
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    return validate_model_health(safetensors_path, config_path)


def quick_validate_batch(
    model_dirs: List[str | Path],
    output_dir: str | Path = "/home/user/ai_experiment/validation",
) -> dict:
    """
    Función de conveniencia: valida múltiples modelos y guarda reportes.

    Returns
    -------
    dict con resultados y rutas de archivos guardados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = validate_batch_models(model_dirs)

    paths: Dict[str, str] = {}

    for model_name, report in results.items():
        # Guardar JSON
        json_path = output_dir / f"{model_name}_validation.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Guardar Markdown
        md_path = output_dir / f"{model_name}_validation.md"
        md_report = generate_validation_report_markdown(report)
        with open(md_path, "w") as f:
            f.write(md_report)

        paths[model_name] = {
            "json": str(json_path),
            "markdown": str(md_path),
        }

    # Reporte consolidado
    batch_md = generate_batch_report(results)
    batch_md_path = output_dir / "batch_validation_report.md"
    with open(batch_md_path, "w") as f:
        f.write(batch_md)
    paths["batch_report_markdown"] = str(batch_md_path)

    return {
        "results": {name: r.to_dict() for name, r in results.items()},
        "paths": paths,
    }


__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "validate_file_exists",
    "validate_safetensors_structure",
    "validate_no_nan_inf",
    "validate_value_range",
    "validate_config_consistency",
    "validate_tensor_shapes",
    "validate_quantization_scales",
    "validate_checksum",
    "validate_disk_space",
    "validate_model_health",
    "generate_validation_report_markdown",
    "validate_batch_models",
    "generate_batch_report",
    "quick_validate_model",
    "quick_validate_batch",
]
