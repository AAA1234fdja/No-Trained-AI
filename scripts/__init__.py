#!/usr/bin/env python3
"""
scripts — Paquete de herramientas para el experimento AI 800M.

Módulos disponibles:
    rope_utils         — Rotary Position Embedding (RoPE).
    attention_utils    — Atención GQA, softmax, máscara causal, KV-cache.
    mlp_utils          — SwiGLU, RMSNorm, compuertas lógicas analíticas.
    quantization_tools — Cuantización INT8 per-canal, validación, edición.
    tokenizer_utils    — Carga, análisis y construcción de tokenizador BPE 32k.
    weight_editor      — Editor manual de pesos .safetensors.
    weight_analyzer    — Análisis estadístico de pesos y comparaciones.
    model_builder      — Especificación de arquitectura, tensor_spec, config.json.
    inference_engine   — Motor de inferencia mmap con generación.
    evaluation         — Suites de pruebas (Python, análisis, conversación).
    prompt_processor   — Plantillas de prompt, few-shot, sanitización.

Todos los módulos están escritos en NumPy puro para operar sobre pesos
INT8 desquantizados sin depender de frameworks de deep learning pesados.

El objetivo del experimento: construir un modelo de ~800M parámetros
sin entrenamiento, editando directamente los pesos en formato .safetensors,
y evaluar si puede ser bueno en Python, hablar mínimamente y entender
análisis.
"""

__version__ = "0.1.0"
__experiment__ = "AI-Experiment-800M-Analytical"
__total_parameters__ = 800260608

from . import (
    rope_utils,
    attention_utils,
    mlp_utils,
    quantization_tools,
    tokenizer_utils,
    weight_editor,
    weight_analyzer,
    model_builder,
    inference_engine,
    evaluation,
    prompt_processor,
)

__all__ = [
    "rope_utils",
    "attention_utils",
    "mlp_utils",
    "quantization_tools",
    "tokenizer_utils",
    "weight_editor",
    "weight_analyzer",
    "model_builder",
    "inference_engine",
    "evaluation",
    "prompt_processor",
]
