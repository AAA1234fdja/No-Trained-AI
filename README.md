# No-Trained-AI — Espejo del experimento "AI experiment AI"

Este repositorio es la copia en GitHub del trabajo realizado en la carpeta de Google Drive **`AI experiment AI`** (file ID de la carpeta: `1B6gr_qo322gfIMHMCetb8tszMfWW9jCN`).

## ¿De qué trata este experimento?

Objetivo original: crear un modelo de lenguaje de **~800 millones de parámetros** sin entrenarlo (sin corpus, sin backpropagation, sin optimizador), generando y editando directamente sus pesos en formato **`.safetensors`**, para comprobar si una IA puede construir otra IA únicamente editando pesos.

El modelo resultante (`model.safetensors`, etapa 5):

| Propiedad | Valor |
|---|---|
| Arquitectura | LlamaForCausalLM (Decoder-only, RoPE, RMSNorm, SwiGLU, GQA) |
| Parámetros nucleares | 800,260,608 (~800.26M) |
| Cuantización | INT8 simétrica por canal (`.weight` + `.weight_scale` FP16) |
| Tamaño en disco | 801,224,536 bytes (~764.11 MB) |
| Vocabulario | 32,000 tokens BPE (inglés + Python + análisis), construido sin entrenar |
| Pasos de entrenamiento | 0 |
| Capas | 21, hidden 1536, 12 cabezas de atención, 4 KV (GQA 3:1), intermedio 5888 |

## Contenido del repositorio

```
docs/                Documentación técnica original del experimento (EXPERIMENT_DOCUMENTATION.md)
model/               config.json, tokenizer_config.json, tokenizer.json  (tokenizador y configuración; SIN los pesos)
scripts/             build_tokenizer.py, synthesize_weights.py, run_stage5.py, inference_engine.py
results/             experiment_results.json  (métricas, entropías y top-5 logits, etapas 0 a 5)
analysis/ANALYSIS.md Estudio de este agente: ¿sirvió el experimento? Veredicto y evidencia
```

## ⚠️ `model.safetensors` NO está en este repositorio

Pesa ~764 MB, muy por encima del límite de 100 MB por archivo de GitHub. Está excluido mediante `.gitignore`.

**Para ejecutar el modelo localmente:**
1. Descarga `model.safetensors` desde Google Drive (carpeta "AI experiment AI"):
   - File ID: `1Lb_97PHGKeJ1HB6k9Umu0Kpl953fNTAT`
   - O abre la carpeta en Drive y descáralo directamente.
2. Colócalo en `model/model.safetensors` junto a `config.json`, `tokenizer.json` y `tokenizer_config.json`.
3. Ejecuta el motor de inferencia (requiere `numpy`, `safetensors`, `tokenizers`):

```bash
pip install numpy safetensors tokenizers
python3 scripts/inference_engine.py model
```

Los scripts de síntesis (`synthesize_weights.py`, `run_stage5.py`, `build_tokenizer.py`) escriben por defecto en `/home/user/ai_experiment`; ajusta `OUT_DIR` si ejecutas en otro equipo.

## ¿Sirvió el experimento?

Véase `analysis/ANALYSIS.md`. Resumen breve: **funciona como ingeniería (pesos válidos, inferencia mmap <1 GB, circuitos "a mano" que copian en contexto y generan sintaxis Python básica), pero no cumple los objetivos de modelo** (no programa bien, habla mínimamente y colapsa en bucles de repetición sin entrenamiento). Conclusión del experimento mismo: sin optimización sobre datos, la edición analítica de pesos implementa primitivas algorítmicas, no competencia lingüística ni de programación generalista.

## Licencia / Uso

Material experimental. Sin garantías.
