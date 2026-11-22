# No-Trained-AI — Modelo 800M Sin Entrenamiento

Este repositorio documenta un experimento de **edición analítica de pesos** en un modelo de lenguaje de ~800M parámetros, sin corpus de entrenamiento, sin backpropagation y sin optimizador. El objetivo era comprobar si se puede construir competencia lingüística y de programación editando directamente los pesos en formato `.safetensors`.

## ¿De qué trata este experimento?

**Objetivo:** Ajustar manualmente los pesos de un modelo 800M sin entrenamiento para cumplir: bueno en Python, habla mínima, bueno en análisis.

**Método:** Síntesis analítica de pesos + edición manual de tensores .safetensors (lm_head, atención, MLP). Cero pasos de entrenamiento.

**Conclusión honesta:** La edición analítica de pesos puede construir `.safetensors` válidos e implementar primitivas algorítmicas básicas (copia, inducción simple, enrutamiento sintáctico), pero **no logra competencia lingüística ni de programación generalista sin entrenamiento con gradientes**. El repositorio documenta este resultado negativo con evidencia.

## Modelo

| Propiedad | Valor |
|-----------|-------|
| Arquitectura | LlamaForCausalLM (Decoder-only, RoPE, RMSNorm, SwiGLU, GQA) |
| Parámetros totales | 800,260,608 (~800.26M) |
| Cuantización | INT8 simétrica per-canal (`.weight` INT8 + `.weight_scale` FP16) |
| Tamaño en disco | 801,224,536 bytes (~764 MB) |
| Vocabulario | 32,000 tokens |
| Capas | 21, hidden 1536, 12 cabezas atención, 4 KV (GQA 3:1), intermedio 5888 |
| Pasos de entrenamiento | 0 |
| Formato | .safetensors (dividido en 10 partes para GitHub) |

## Contenido del repositorio

```
model/                 config.json, tokenizer.json, tokenizer_config.json,
                       model.safetensors.part.aa..aj (10 partes, <=80 MB cada una),
                       reassemble.sh (reensambla model.safetensors)
scripts/               56 módulos Python (inferencia, edición, evaluación, análisis)
DOCUMENTACION_EXPERIMENTO.md  Documentación completa del experimento
ANTI_LOOP_STRATEGY.md          Estrategias y reglas para eliminar bucles de repetición
REFERENCIA_EDICION_PESOS_SIN_ENTRENAMIENTO.md  Referencia técnica (ROME, MEMIT, ActAdd, etc.)
LICENSE                            Licencia MIT
```

## ⚙️ Cómo usar el modelo

### 1. Reensamblar model.safetensors

El archivo `model.safetensors` está dividido en 10 partes (≤80 MB cada una) para cumplir con el límite de GitHub:

```bash
bash model/reassemble.sh
```

Esto verifica los checksums SHA-256 de cada parte y reensambla `model.safetensors` (~801 MB).

### 2. Ejecutar inferencia

```bash
pip install numpy safetensors tokenizers
python3 scripts/inference_engine.py
```

El motor de inferencia usa mmap (no carga todo el modelo en RAM) y soporta generación autoregresiva con greedy, temperatura, top-k, top-p y repetition penalty.

### 3. Probar estrategias anti-loop

El modelo sin entrenamiento tiende a repetir tokens (`forforfor`, `importimportimport`). El script `scripts/test_idk50m.py` usa un modelo auxiliar de 50M parámetros para probar configuraciones rápidamente:

```bash
python3 scripts/test_idk50m.py
```

**Configuración recomendada para reducir loops:**

| Tipo de loop | Parámetros recomendados |
|--------------|------------------------|
| Single-token (`forforfor`) | `repetition_penalty >= 5.0` |
| Multi-token (`docdocdoc`) | `repetition_penalty=1.8 + frequency_penalty=0.3 + presence_penalty=0.3` |

## Resultados del experimento

### Lo que funcionó ✅
- `.safetensors` válido: 800M parámetros, 341 tensores, 0 NaN/Inf, rango INT8 [-127, 127]
- Inferencia mmap <1 GB, generación autoregresiva funciona
- Circuito de inducción escrito a mano (copia `[A][B]...[A]→[B]` con p≈0.985)
- Cuantización per-canal bien planteada
- Determinismo con `random.Random(seed=42)`

### Lo que NO funcionó ❌
- **No genera código Python válido:** `importimportimport`, `ConnectionAbortedError`, etc.
- **No hace análisis coherente:** `ComplexityComplexityComplexity`, mezcla de errores
- **No tiene habla mínima:** persiste repetición de tokens y caracteres raros
- **Entropía 5-9 nats** (máximo teórico 10.37 con vocab 32k): ruido casi uniforme, no certeza
- **Edición de "columna 0 = Python" es un bias hack**, no conocimiento semántico real

### Métricas reales (no proxy)
El repositorio incluye `scripts/test_idk50m.py` para evaluar con inferencia real:
- Detección de loops de tokens y n-gramas
- Entropía de la distribución de logits
- Tiempo de generación

## Métricas de código

| Métrica | Valor |
|---------|-------|
| Módulos Python en `scripts/` | 56 |
| Líneas de código | 31,774 |
| Todos compilan con `py_compile` | Sí |
| CI con GitHub Actions | Sí (ver `.github/workflows/ci.yml`) |

> **Nota:** Las 31,774 líneas incluyen utilidades, analizadores, exportadores y generadores de plantillas. No son tests ni notas. El requisito de 30,000 líneas se cumple, pero el repositorio admite que muchas son utilidades de ingeniería, no conocimiento del modelo.

## Limitaciones fundamentales

El resultado negativo es correcto y valioso: **sin optimización sobre datos con gradientes, la edición manual de pesos implementa primitivas algorítmicas, no competencia lingüística.** No hay atajo por edición analítica. Para un modelo usable se necesita:

1. **Fine-tuning supervisado** en código Python real, o
2. **Fine-tuning de un modelo abierto** existente, o
3. **Reencuadrar** el experimento como "circuitos escritos a mano" en un modelo tiny (1-10M) donde cada circuito sea auditable al 100%.

## Licencia

MIT License. Véase `LICENSE`.
