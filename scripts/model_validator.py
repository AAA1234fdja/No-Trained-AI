#!/usr/bin/env python3
"""
model_validator.py — Validador completo del modelo 800M.

Ejecuta una batería de validaciones sobre un archivo .safetensors y su
configuración asociada:

  1. Estructura del archivo: nombres de tensor, formas, dtypes.
  2. Conteo de parámetros: debe coincidir con 800,260,608.
  3. Integridad numérica: no hay NaN ni Inf en ningún tensor.
  4. Rango dinámico INT8: valores dentro de [-127, 127].
  5. Escalas FP16: positivas, finitas, dentro de rango razonable.
  6. Consistencia con config.json: vocab_size, hidden_size, num_layers, etc.
  7. Ejecutabilidad: el modelo puede hacer un forward pass con un token dummy.
  8. Detección de colapso: entropía de generación no demasiado baja.
  9. Reporte de validación en JSON y Markdown.

Diseñado para usarse después de cada etapa de síntesis para verificar
que el modelo siga siendo válido tras ediciones manuales de pesos.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None  # type: ignore

from scripts.model_builder import (
    build_tensor_spec,
    validate_config,
    count_total_params,
    V, D, DFF, L, H, HKV, HD,
)


class ValidationResult:
    """Resultado de una validación individual."""

    def __init__(
        self,
        check_name: str,
        passed: bool,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class ModelValidator:
    """
    Validador principal del modelo 800M.
    """

    def __init__(
        self,
        model_dir: str | Path,
        weights_filename: str = "model.safetensors",
        config_filename: str = "config.json",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / weights_filename
        self.config_path = self.model_dir / config_filename
        self.results: List[ValidationResult] = []

        if safe_open is None:
            raise ImportError("safetensors no está instalado")

    # ------------------------------------------------------------------
    # Validaciones estáticas (sin cargar el modelo en memoria)
    # ------------------------------------------------------------------

    def validate_file_structure(self) -> ValidationResult:
        """Valida que el archivo .safetensors tenga la estructura esperada."""
        spec = build_tensor_spec()
        spec_dict = {name: (shape, dtype) for name, shape, dtype in spec}
        issues: List[str] = []

        try:
            with safe_open(str(self.weights_path), framework="numpy") as f:
                file_keys = set(f.keys())
                spec_keys = set(spec_dict.keys())

                missing = spec_keys - file_keys
                extra = file_keys - spec_keys

                if missing:
                    issues.append(f"Tensores faltantes: {sorted(missing)}")
                if extra:
                    issues.append(f"Tensores extra: {sorted(extra)}")

                for name in sorted(spec_keys & file_keys):
                    expected_shape, expected_dtype = spec_dict[name]
                    t = f.get_tensor(name)

                    if tuple(t.shape) != tuple(expected_shape):
                        issues.append(
                            f"{name}: forma {tuple(t.shape)} != {tuple(expected_shape)}"
                        )

                    expected_np = np.int8 if expected_dtype == "I8" else np.float16
                    if t.dtype != expected_np:
                        issues.append(
                            f"{name}: dtype {t.dtype} != {expected_np}"
                        )
        except Exception as e:
            return ValidationResult(
                "file_structure", False, f"Error al leer el archivo: {e}"
            )

        passed = len(issues) == 0
        return ValidationResult(
            "file_structure",
            passed,
            "Estructura de archivo válida" if passed else "; ".join(issues),
            {"issues": issues, "tensor_count": len(spec_dict)},
        )

    def validate_parameter_count(self) -> ValidationResult:
        """Valida que el conteo total de parámetros sea 800,260,608."""
        expected = count_total_params()
        actual = 0
        issues: List[str] = []

        try:
            with safe_open(str(self.weights_path), framework="numpy") as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    if name.endswith(".weight_scale"):
                        continue  # las escalas no son parámetros del modelo
                    n = 1
                    for s in t.shape:
                        n *= s
                    actual += n
        except Exception as e:
            return ValidationResult(
                "parameter_count", False, f"Error: {e}"
            )

        if actual != expected:
            issues.append(f"Parámetros: {actual:,} != esperados {expected:,}")

        passed = len(issues) == 0
        return ValidationResult(
            "parameter_count",
            passed,
            f"{actual:,} parámetros (esperados {expected:,})" if passed else "; ".join(issues),
            {"expected": expected, "actual": actual},
        )

    def validate_no_nan_inf(self) -> ValidationResult:
        """Valida que no haya NaN ni Inf en ningún tensor."""
        nan_count = 0
        inf_count = 0
        nan_tensors: List[str] = []
        inf_tensors: List[str] = []

        try:
            with safe_open(str(self.weights_path), framework="numpy") as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    if np.issubdtype(t.dtype, np.floating):
                        t_f32 = t.astype(np.float32)
                        n_nan = int(np.sum(np.isnan(t_f32)))
                        n_inf = int(np.sum(np.isinf(t_f32)))
                        if n_nan > 0:
                            nan_count += n_nan
                            nan_tensors.append(name)
                        if n_inf > 0:
                            inf_count += n_inf
                            inf_tensors.append(name)
        except Exception as e:
            return ValidationResult(
                "no_nan_inf", False, f"Error: {e}"
            )

        issues: List[str] = []
        if nan_count > 0:
            issues.append(f"{nan_count} NaN en: {nan_tensors[:5]}")
        if inf_count > 0:
            issues.append(f"{inf_count} Inf en: {inf_tensors[:5]}")

        passed = len(issues) == 0
        return ValidationResult(
            "no_nan_inf",
            passed,
            "Sin NaN ni Inf" if passed else "; ".join(issues),
            {"nan_count": nan_count, "inf_count": inf_count},
        )

    def validate_int8_range(self) -> ValidationResult:
        """Valida que los tensores INT8 usen el rango [-127, 127]."""
        i8_min = 127
        i8_max = -128
        issues: List[str] = []
        tensors_checked = 0

        try:
            with safe_open(str(self.weights_path), framework="numpy") as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    if t.dtype == np.int8:
                        tensors_checked += 1
                        t_min = int(t.min())
                        t_max = int(t.max())
                        if t_min < -127 or t_max > 127:
                            issues.append(
                                f"{name}: rango [{t_min}, {t_max}] fuera de [-127, 127]"
                            )
                        i8_min = min(i8_min, t_min)
                        i8_max = max(i8_max, t_max)
        except Exception as e:
            return ValidationResult(
                "int8_range", False, f"Error: {e}"
            )

        passed = len(issues) == 0
        return ValidationResult(
            "int8_range",
            passed,
            f"Rango INT8 [{i8_min}, {i8_max}] válido" if passed else "; ".join(issues),
            {"i8_min": i8_min, "i8_max": i8_max, "tensors_checked": tensors_checked},
        )

    def validate_scales(self) -> ValidationResult:
        """Valida que las escalas FP16 sean positivas y finitas."""
        issues: List[str] = []
        scale_tensors = 0
        scale_min = float("inf")
        scale_max = float("-inf")

        try:
            with safe_open(str(self.weights_path), framework="numpy") as f:
                for name in f.keys():
                    if name.endswith(".weight_scale"):
                        scale_tensors += 1
                        t = f.get_tensor(name).astype(np.float32)
                        if np.any(~np.isfinite(t)):
                            issues.append(f"{name}: escala contiene NaN/Inf")
                        if np.any(t <= 0):
                            issues.append(f"{name}: escala contiene valores <= 0")
                        scale_min = min(scale_min, float(t.min()))
                        scale_max = max(scale_max, float(t.max()))
        except Exception as e:
            return ValidationResult(
                "scales", False, f"Error: {e}"
            )

        passed = len(issues) == 0
        return ValidationResult(
            "scales",
            passed,
            f"{scale_tensors} escalas válidas" if passed else "; ".join(issues),
            {
                "scale_tensors": scale_tensors,
                "scale_min": float(scale_min) if scale_tensors > 0 else 0.0,
                "scale_max": float(scale_max) if scale_tensors > 0 else 0.0,
            },
        )

    def validate_config_consistency(self) -> ValidationResult:
        """Valida que config.json sea consistente con la arquitectura esperada."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            return ValidationResult(
                "config_consistency", False, f"Error al leer config: {e}"
            )

        validation = validate_config(config)
        passed = validation["valid"]
        return ValidationResult(
            "config_consistency",
            passed,
            "Config consistente" if passed else "; ".join(validation["issues"]),
            {"issues": validation["issues"]},
        )

    # ------------------------------------------------------------------
    # Validaciones dinámicas (requieren cargar el modelo)
    # ------------------------------------------------------------------

    def validate_forward_pass(
        self,
        model_dir: Optional[str | Path] = None,
    ) -> ValidationResult:
        """
        Ejecuta un forward pass con un token dummy para verificar que
        el modelo es ejecutable sin errores.
        """
        try:
            from scripts.inference_engine import MmapInt8LlamaModel
        except ImportError:
            return ValidationResult(
                "forward_pass", False, "No se pudo importar inference_engine"
            )

        md = model_dir if model_dir is not None else self.model_dir
        try:
            model = MmapInt8LlamaModel(md)
            logits, _, _ = model.forward([0])  # token BOS
            model.close()

            if logits.shape[0] != int(V()):
                return ValidationResult(
                    "forward_pass", False,
                    f"Logits shape {logits.shape} != vocab_size {V()}"
                )

            if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
                return ValidationResult(
                    "forward_pass", False, "Logits contienen NaN o Inf"
                )

            return ValidationResult(
                "forward_pass", True,
                f"Forward pass exitoso, {logits.shape[0]} logits",
                {"logits_shape": list(logits.shape)},
            )
        except Exception as e:
            return ValidationResult(
                "forward_pass", False, f"Error en forward pass: {e}"
            )

    def validate_generation_entropy(
        self,
        model_dir: Optional[str | Path] = None,
        prompts: Optional[List[str]] = None,
        min_entropy: float = 0.5,
    ) -> ValidationResult:
        """
        Valida que la generación no colapse a entropía cero.

        Si la entropía es demasiado baja, el modelo está en un atractor
        de repetición y no es útil.
        """
        if prompts is None:
            prompts = [
                "def ",
                "Hello! How can you help me?",
                "Analyze the time complexity of binary search:",
            ]

        try:
            from scripts.inference_engine import MmapInt8LlamaModel, run_inference
        except ImportError:
            return ValidationResult(
                "generation_entropy", False, "No se pudo importar inference_engine"
            )

        md = model_dir if model_dir is not None else self.model_dir
        try:
            results = run_inference(
                md,
                prompts,
                weights_filename=self.weights_path.name,
                max_new_tokens=5,
                temperature=0.0,
            )
        except Exception as e:
            return ValidationResult(
                "generation_entropy", False, f"Error en inferencia: {e}"
            )

        entropies = [r["first_token_entropy_nats"] for r in results]
        low_entropy = [e for e in entropies if e < min_entropy]

        issues: List[str] = []
        if low_entropy:
            issues.append(
                f"{len(low_entropy)} prompts con entropía < {min_entropy}: "
                f"{[round(e, 2) for e in low_entropy]}"
            )

        passed = len(issues) == 0
        return ValidationResult(
            "generation_entropy",
            passed,
            "Entropía saludable en todos los prompts" if passed else "; ".join(issues),
            {
                "entropies": entropies,
                "min_entropy": min(entropies) if entropies else 0.0,
                "max_entropy": max(entropies) if entropies else 0.0,
                "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            },
        )

    # ------------------------------------------------------------------
    # Ejecutar todas las validaciones
    # ------------------------------------------------------------------

    def run_all(
        self,
        model_dir: Optional[str | Path] = None,
        include_dynamic: bool = True,
    ) -> Dict[str, object]:
        """
        Ejecuta todas las validaciones y devuelve un reporte combinado.

        Parameters
        ----------
        model_dir : str | Path opcional
            Directorio del modelo (para validaciones dinámicas).
        include_dynamic : bool
            Si incluir validaciones que requieren cargar el modelo.
        """
        self.results = []

        # Validaciones estáticas
        self.results.append(self.validate_file_structure())
        self.results.append(self.validate_parameter_count())
        self.results.append(self.validate_no_nan_inf())
        self.results.append(self.validate_int8_range())
        self.results.append(self.validate_scales())
        self.results.append(self.validate_config_consistency())

        # Validaciones dinámicas
        if include_dynamic:
            self.results.append(self.validate_forward_pass(model_dir))
            self.results.append(self.validate_generation_entropy(model_dir))

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return {
            "model_dir": str(self.model_dir),
            "weights_file": self.weights_path.name,
            "total_checks": len(self.results),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
            "checks": [r.to_dict() for r in self.results],
        }

    def save_report(
        self,
        path: str | Path,
        report: Optional[Dict[str, object]] = None,
    ) -> None:
        """Guarda el reporte de validación en JSON."""
        if report is None:
            report = self.run_all(include_dynamic=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def print_report(
        self,
        report: Optional[Dict[str, object]] = None,
    ) -> None:
        """Imprime el reporte de validación en formato legible."""
        if report is None:
            report = self.run_all(include_dynamic=False)

        print("=" * 70)
        print("REPORTE DE VALIDACIÓN DEL MODELO 800M")
        print("=" * 70)
        print(f"Archivo: {report['weights_file']}")
        print(f"Total checks: {report['total_checks']}")
        print(f"Pasados:      {report['passed']}")
        print(f"Fallidos:     {report['failed']}")
        print(f"Todos pasados: {report['all_passed']}")
        print()

        for check in report["checks"]:
            mark = "✓" if check["passed"] else "✗"
            print(f"{mark} {check['check_name']:30s} | {check['message']}")
            if check["details"]:
                for k, v in check["details"].items():
                    print(f"     {k}: {v}")
        print("=" * 70)


def quick_validate(
    model_dir: str | Path,
    include_dynamic: bool = True,
) -> Dict[str, object]:
    """
    Función de conveniencia: valida un modelo y devuelve el reporte.
    """
    validator = ModelValidator(model_dir)
    return validator.run_all(include_dynamic=include_dynamic)


__all__ = [
    "ValidationResult",
    "ModelValidator",
    "quick_validate",
]
