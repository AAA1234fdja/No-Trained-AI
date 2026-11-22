#!/usr/bin/env python3
"""
weight_editor.py — Editor manual de pesos para el modelo 800M en .safetensors.

Permite editar directamente tensores del modelo sin entrenamiento, mediante:
  - Carga de tensores desde .safetensors con mmap (solo lectura) o carga completa.
  - Edición por fila, por bloque o elemento individual.
  - Aplicación de reglas analíticas (ej. reforzar lm_head para tokens Python).
  - Guardado del modelo editado a un nuevo archivo .safetensors.
  - Validación post-edición (NaN, Inf, rango INT8).
  - Registro de todas las ediciones realizadas (auditoría).

Todas las ediciones se registran en un JSON de auditoría para reproducibilidad.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as save_safetensors
except ImportError:  # pragma: no cover
    safe_open = None  # type: ignore
    save_safetensors = None  # type: ignore


class WeightEditRecord:
    """Registro de una sola edición de peso."""

    def __init__(
        self,
        tensor_name: str,
        operation: str,  # "row", "scalar", "block", "scale"
        indices: List[int],
        old_values: List[float],
        new_values: List[float],
        timestamp: Optional[float] = None,
    ) -> None:
        self.tensor_name = tensor_name
        self.operation = operation
        self.indices = indices
        self.old_values = old_values
        self.new_values = new_values
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.tensor_name,
            "operation": self.operation,
            "indices": self.indices,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "timestamp": self.timestamp,
        }


class WeightEditor:
    """
    Editor de pesos para modelos Llama en formato .safetensors.

    Soporta edición de tensores INT8 (con escala FP16) y FP16/FP32.
    Las ediciones se aplican en memoria y se pueden guardar a disco.
    """

    def __init__(
        self,
        model_dir: str | Path,
        weights_filename: str = "model.safetensors",
        read_only: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / weights_filename
        self.read_only = read_only
        self._tensors: Dict[str, np.ndarray] = {}
        self._scales: Dict[str, np.ndarray] = {}
        self._dtypes: Dict[str, str] = {}
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._edits: List[WeightEditRecord] = []

        self._load_metadata()

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Lee metadados y tensores desde el archivo .safetensors."""
        if safe_open is None:
            raise ImportError("safetensors no está instalado")

        with safe_open(str(self.weights_path), framework="numpy") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name.endswith(".weight_scale"):
                    self._scales[name] = t.copy()
                    self._dtypes[name] = str(t.dtype)
                    self._shapes[name] = tuple(t.shape)
                else:
                    self._tensors[name] = t.copy()
                    self._dtypes[name] = str(t.dtype)
                    self._shapes[name] = tuple(t.shape)

    def list_tensors(self) -> List[str]:
        """Devuelve lista de nombres de tensor disponibles."""
        return sorted(self._tensors.keys())

    def list_scales(self) -> List[str]:
        """Devuelve lista de nombres de escala disponibles."""
        return sorted(self._scales.keys())

    def get_tensor(self, name: str) -> np.ndarray:
        """Devuelve una copia del tensor solicitado."""
        if name not in self._tensors:
            raise KeyError(f"Tensor no encontrado: {name}")
        return self._tensors[name].copy()

    def get_scale(self, name: str) -> np.ndarray:
        """Devuelve una copia de la escala solicitada."""
        if name not in self._scales:
            raise KeyError(f"Escala no encontrada: {name}")
        return self._scales[name].copy()

    def get_shape(self, name: str) -> Tuple[int, ...]:
        """Devuelve la forma de un tensor."""
        if name not in self._shapes:
            raise KeyError(f"Tensor no encontrado: {name}")
        return self._shapes[name]

    # ------------------------------------------------------------------
    # Operaciones de edición
    # ------------------------------------------------------------------

    def edit_scalar(
        self,
        tensor_name: str,
        row: int,
        col: int,
        new_value: float,
    ) -> None:
        """
        Edita un solo elemento de un tensor.

        Para tensores INT8, se desquantiza la fila, se edita el elemento
        y se re-cuantiza.
        """
        if self.read_only:
            raise PermissionError("Editor en modo solo lectura")

        w = self._tensors[tensor_name]
        scale = self._scales.get(tensor_name + "_scale")

        if w.dtype == np.int8 and scale is not None:
            # Desquantizar fila, editar, re-cuantizar
            from scripts.quantization_tools import (
                dequantize_int8_per_row,
                quantize_int8_per_row,
            )
            row_f32 = dequantize_int8_per_row(w, scale)
            old_val = float(row_f32[row, col])
            row_f32[row, col] = new_value
            w_q_new, scale_new = quantize_int8_per_row(row_f32.reshape(1, -1))
            w[row] = w_q_new[0]
            scale[tensor_name + "_scale"][row] = scale_new[0, 0]

            self._record_edit(tensor_name, "scalar", [row, col],
                              [old_val], [new_value])
        else:
            old_val = float(w[row, col])
            w[row, col] = new_value
            self._record_edit(tensor_name, "scalar", [row, col],
                              [old_val], [new_value])

    def edit_row(
        self,
        tensor_name: str,
        row: int,
        new_values: np.ndarray,
    ) -> None:
        """
        Edita una fila completa de un tensor.

        new_values debe tener longitud igual al número de columnas.
        """
        if self.read_only:
            raise PermissionError("Editor en modo solo lectura")

        w = self._tensors[tensor_name]
        scale = self._scales.get(tensor_name + "_scale")

        new_values = np.asarray(new_values, dtype=np.float32)
        if new_values.shape[0] != w.shape[1]:
            raise ValueError(
                f"Longitud {new_values.shape[0]} != columnas {w.shape[1]}"
            )

        if w.dtype == np.int8 and scale is not None:
            from scripts.quantization_tools import (
                dequantize_int8_per_row,
                quantize_int8_per_row,
            )
            row_f32 = dequantize_int8_per_row(w, scale)
            old_row = row_f32[row].copy()
            row_f32[row] = new_values
            w_q_new, scale_new = quantize_int8_per_row(
                row_f32[row].reshape(1, -1)
            )
            w[row] = w_q_new[0]
            scale[tensor_name + "_scale"][row] = scale_new[0, 0]

            self._record_edit(
                tensor_name, "row", [row],
                old_row.tolist(), new_values.tolist(),
            )
        else:
            old_row = w[row].copy()
            w[row] = new_values
            self._record_edit(
                tensor_name, "row", [row],
                old_row.tolist(), new_values.tolist(),
            )

    def edit_block(
        self,
        tensor_name: str,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        new_values: np.ndarray,
    ) -> None:
        """
        Edita un bloque rectangular de un tensor.

        new_values debe tener forma (row_end - row_start, col_end - col_start).
        """
        if self.read_only:
            raise PermissionError("Editor en modo solo lectura")

        w = self._tensors[tensor_name]
        new_values = np.asarray(new_values, dtype=np.float32)
        expected_shape = (row_end - row_start, col_end - col_start)
        if new_values.shape != expected_shape:
            raise ValueError(
                f"Forma {new_values.shape} != esperada {expected_shape}"
            )

        old_block = w[row_start:row_end, col_start:col_end].copy()
        w[row_start:row_end, col_start:col_end] = new_values

        self._record_edit(
            tensor_name, "block",
            [row_start, row_end, col_start, col_end],
            old_block.flatten().tolist(),
            new_values.flatten().tolist(),
        )

    def add_to_row(
        self,
        tensor_name: str,
        row: int,
        delta: np.ndarray,
        col_start: int = 0,
    ) -> None:
        """
        Suma un vector delta a una fila (edición incremental).

        Útil para ajustar finamente pesos sin sobrescribirlos completamente.
        """
        w = self._tensors[tensor_name]
        delta = np.asarray(delta, dtype=np.float32)

        if w.dtype == np.int8:
            from scripts.quantization_tools import (
                dequantize_int8_per_row,
                quantize_int8_per_row,
            )
            row_f32 = dequantize_int8_per_row(w, self._scales[tensor_name + "_scale"])
            old_row = row_f32[row].copy()
            row_f32[row, col_start:col_start + delta.shape[0]] += delta
            w_q_new, scale_new = quantize_int8_per_row(
                row_f32[row].reshape(1, -1)
            )
            w[row] = w_q_new[0]
            self._scales[tensor_name + "_scale"][row] = scale_new[0, 0]
            self._record_edit(
                tensor_name, "add_row", [row],
                old_row.tolist(), row_f32[row].tolist(),
            )
        else:
            old_row = w[row].copy()
            w[row, col_start:col_start + delta.shape[0]] += delta
            self._record_edit(
                tensor_name, "add_row", [row],
                old_row.tolist(), w[row].tolist(),
            )

    def scale_row(
        self,
        tensor_name: str,
        row: int,
        factor: float,
    ) -> None:
        """
        Multiplica una fila completa por un factor escalar.
        """
        w = self._tensors[tensor_name]
        if w.dtype == np.int8:
            from scripts.quantization_tools import (
                dequantize_int8_per_row,
                quantize_int8_per_row,
            )
            scale_arr = self._scales[tensor_name + "_scale"]
            row_f32 = dequantize_int8_per_row(w, scale_arr)
            old_row = row_f32[row].copy()
            row_f32[row] *= factor
            w_q_new, scale_new = quantize_int8_per_row(
                row_f32[row].reshape(1, -1)
            )
            w[row] = w_q_new[0]
            scale_arr[row] = scale_new[0, 0]
            self._record_edit(
                tensor_name, "scale_row", [row],
                old_row.tolist(), row_f32[row].tolist(),
            )
        else:
            old_row = w[row].copy()
            w[row] *= factor
            self._record_edit(
                tensor_name, "scale_row", [row],
                old_row.tolist(), w[row].tolist(),
            )

    # ------------------------------------------------------------------
    # Reglas analíticas predefinidas
    # ------------------------------------------------------------------

    def reinforce_lm_head_tokens(
        self,
        token_ids: List[int],
        feature_col: int,
        value: float,
        tensor_name: str = "lm_head.weight",
    ) -> None:
        """
        Refuerza la conexión entre ciertos tokens y una característica
        en la capa de salida lm_head.

        lm_head[token_id, feature_col] += value
        """
        w = self._tensors[tensor_name]
        old_vals = []
        new_vals = []
        for tid in token_ids:
            old_vals.append(float(w[tid, feature_col]))
            w[tid, feature_col] += value
            new_vals.append(float(w[tid, feature_col]))

        self._record_edit(
            tensor_name, "reinforce_tokens",
            token_ids, old_vals, new_vals,
        )

    def suppress_token(
        self,
        token_id: int,
        feature_col: int,
        value: float,
        tensor_name: str = "lm_head.weight",
    ) -> None:
        """
        Suprime un token restando un valor de su conexión con una característica.
        """
        self.edit_scalar(tensor_name, token_id, feature_col,
                         float(self._tensors[tensor_name][token_id, feature_col]) - value)

    # ------------------------------------------------------------------
    # Auditoría y guardado
    # ------------------------------------------------------------------

    def _record_edit(
        self,
        tensor_name: str,
        operation: str,
        indices: List[int],
        old_values: List[float],
        new_values: List[float],
    ) -> None:
        self._edits.append(
            WeightEditRecord(
                tensor_name=tensor_name,
                operation=operation,
                indices=indices,
                old_values=old_values,
                new_values=new_values,
            )
        )

    def get_edit_log(self) -> List[dict]:
        """Devuelve la lista de ediciones realizadas como diccionarios."""
        return [e.to_dict() for e in self._edits]

    def save_edit_log(self, path: str | Path) -> None:
        """Guarda el log de ediciones en un archivo JSON."""
        with open(path, "w") as f:
            json.dump(self.get_edit_log(), f, indent=2)

    def save_model(
        self,
        output_path: str | Path,
        tensor_order: Optional[List[str]] = None,
    ) -> None:
        """
        Guarda el modelo editado a un archivo .safetensors.

        Si tensor_order se proporciona, los tensores se escriben en ese orden.
        De lo contrario, se usa el orden original de carga.
        """
        if save_safetensors is None:
            raise ImportError("safetensors no está instalado")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tensors_to_save: Dict[str, np.ndarray] = {}

        # Ordenar tensores
        if tensor_order is None:
            tensor_order = self.list_tensors()

        for name in tensor_order:
            if name in self._tensors:
                tensors_to_save[name] = self._tensors[name]
            scale_name = name + "_scale"
            if scale_name in self._scales:
                tensors_to_save[scale_name] = self._scales[scale_name]

        # Metadata
        metadata = {
            "format": "pt",
            "experiment": "AI_experiment_AI_800M_manual_weight_edit",
            "training_steps": "0",
            "total_weight_parameters": "800260608",
            "edit_log_count": str(len(self._edits)),
        }

        save_safetensors(tensors_to_save, str(output_path), metadata=metadata)

    def validate_post_edit(self) -> Dict[str, object]:
        """
        Valida el estado actual de los pesos editados.

        Returns
        -------
        dict con conteo de NaN, Inf, rango INT8, etc.
        """
        issues: List[str] = []
        nan_count = 0
        inf_count = 0
        i8_min = 127
        i8_max = -128
        int8_tensors = 0
        f16_tensors = 0

        for name, t in self._tensors.items():
            if t.dtype == np.int8:
                int8_tensors += 1
                i8_min = min(i8_min, int(t.min()))
                i8_max = max(i8_max, int(t.max()))
            if np.issubdtype(t.dtype, np.floating):
                t_f32 = t.astype(np.float32)
                nan_count += int(np.sum(np.isnan(t_f32)))
                inf_count += int(np.sum(np.isinf(t_f32)))

        for name, s in self._scales.items():
            f16_tensors += 1
            s_f32 = s.astype(np.float32)
            if np.any(~np.isfinite(s_f32)):
                issues.append(f"Escala {name}: NaN/Inf")

        if nan_count > 0:
            issues.append(f"{nan_count} valores NaN en pesos")
        if inf_count > 0:
            issues.append(f"{inf_count} valores Inf en pesos")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "int8_tensors": int8_tensors,
            "f16_tensors": f16_tensors,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "int8_range": [i8_min, i8_max],
            "total_edits": len(self._edits),
        }


def quick_edit_safetensors(
    model_dir: str | Path,
    output_path: str | Path,
    edits: List[dict],
) -> Dict[str, object]:
    """
    Función de conveniencia para aplicar una lista de ediciones y guardar.

    Cada edición es un dict con claves:
        - tensor_name, operation, indices, old_values, new_values
    """
    editor = WeightEditor(model_dir, read_only=False)

    for edit in edits:
        op = edit["operation"]
        if op == "scalar":
            editor.edit_scalar(
                edit["tensor_name"], edit["indices"][0], edit["indices"][1],
                edit["new_values"][0],
            )
        elif op == "row":
            editor.edit_row(
                edit["tensor_name"], edit["indices"][0],
                np.array(edit["new_values"], dtype=np.float32),
            )
        elif op == "scale_row":
            editor.scale_row(
                edit["tensor_name"], edit["indices"][0],
                edit["new_values"][0],
            )

    validation = editor.validate_post_edit()
    editor.save_model(output_path)
    editor.save_edit_log(str(Path(output_path).with_suffix(".edits.json")))

    return {
        "output_path": str(output_path),
        "validation": validation,
        "edit_count": len(edits),
    }


__all__ = [
    "WeightEditRecord",
    "WeightEditor",
    "quick_edit_safetensors",
]
