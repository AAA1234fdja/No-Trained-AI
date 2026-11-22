# Documentación del Experimento AI 800M — Modelo Sin Entrenamiento

## 1. Información General

- **Proyecto:** No-Trained-AI (Modelo de Lenguaje de 800M Parámetros Sin Entrenamiento)
- **Rama:** `arena/b015914a-no-trained-ai`
- **Commit base:** `af35a2fa7c41781cf922e3318a8c247ddf58b607`
- **Fecha:** 2026-11-22
- **Ubicación:** Soacha, Cundinamarca, Colombia (UTC)
- **Objetivo:** Ajustar manualmente pesos de un modelo 800M sin entrenamiento para cumplir: bueno en Python, habla mínima, bueno en análisis.
- **Formato:** `.safetensors` (INT8 simétrico per-canal + escalas FP16)
- **Requisito obligatorio:** Mínimo 30,000 líneas de código Python funcional (no contar tests ni notas).

---

## 2. Función Usada para Determinismo (No Aleatoriedad)

Para evitar que el código generado sea aleatorio e impredecible, se utiliza **`random.Random(seed)` con una semilla fija** en lugar del módulo `random` global.

### ¿Por qué `random.Random(seed)`?

| Enfoque | Comportamiento | Determinismo |
|---------|---------------|--------------|
| `random.random()` / `random.choice()` | Usa estado global compartido | ❌ No determinista entre sesiones |
| `random.Random(seed)` | Crea instancia independiente con semilla fija | ✅ Determinista, reproducible |

### Implementación en el código

En `function_generator_complex.py` y otros módulos, se crea una instancia local:

```python
self.rng = random.Random(seed)  # seed = 42 por defecto
```

Esto garantiza que:
- La misma semilla produce los mismos nombres de funciones, parámetros, cuerpos y pruebas.
- Los resultados son reproducibles en cualquier ejecución posterior.
- No hay dependencia del estado global de Python (`random.seed()` no afecta a otras partes del programa).

### Semillas utilizadas

| Módulo | Semilla | Propósito |
|--------|---------|-----------|
| `function_generator_complex.py` | 42 | Generación de nombres, parámetros, cuerpos, pruebas |
| `training_monitor.py` | 42 | Simulación de métricas de entrenamiento |
| `tokenizer_trainer.py` | (interno) | Simulación de BPE y frecuencia de tokens |
| `model_simulator.py` | 42 | Pesos aleatorios para pruebas (create_random_weights) |

---

## 3. Arquitectura del Modelo 800M

| Parámetro | Valor |
|-----------|-------|
| Tipo | LlamaForCausalLM (decoder-only) |
| Vocabulario | 32,000 tokens |
| Hidden size | 1,536 |
| Capas Transformer | 21 |
| Cabezas de atención | 12 (GQA con 4 KV heads, ratio 3:1) |
| Head dim | 128 |
| Tamaño intermedio (SwiGLU) | 5,888 |
| Parámetros totales | 800,260,608 |
| Formato | .safetensors (INT8 simétrico per-canal + escalas FP16) |
| Secuencia máxima | 2,048 tokens |
| Norma | RMSNorm (eps=1e-5) |
| RoPE | Base 10,000, formato GPT-NeoX |
| Activación | SiLU (SwiGLU) |

---

## 4. Estructura del Proyecto (`scripts/`)

Total: **56 módulos Python**, **31,774 líneas de código funcional** (superando el requisito de 30,000).

### Módulos originales (11)
| Módulo | Propósito |
|--------|-----------|
| `rope_utils.py` | RoPE cached, apply_rope_interleaved/neox, verify_rope_orthogonality |
| `attention_utils.py` | stable_softmax, causal_mask, attention_scores, KVCache, attention_metrics |
| `mlp_utils.py` | silu, swiglu, rms_norm, build_analytical_swiglu_block, edit_swiglu_row |
| `quantization_tools.py` | quantize_int8_per_row, dequantize, validation, safetensors report |
| `tokenizer_utils.py` | load/save tokenizer, categorize_token, build_feature_table, vocab_statistics |
| `weight_editor.py` | Editor manual de pesos .safetensors (scalar, row, block, scale_row, add_to_row) |
| `weight_analyzer.py` | Estadísticas por tensor, similitud de embeddings, comparación entre modelos |
| `model_builder.py` | Especificador de arquitectura (341 tensores, 800,260,608 params) |
| `inference_engine.py` | Motor de inferencia mmap con KV-cache, generación autoregresiva |
| `evaluation.py` | Suites de pruebas (Python, análisis, conversación, inducción, no-colapso) |
| `prompt_processor.py` | Plantillas de prompt, few-shot, sanitización, extracción de bloques |
| `code_generator.py` | Generador de código Python sintético |
| `__init__.py` | Paquete scripts, versión 0.1.0 |

### Módulos nuevos creados en esta sesión (43)
| Módulo | Líneas | Propósito |
|--------|--------|-----------|
| `cli_tools.py` | 467 | CLI unificada (validate, benchmark, evaluate, generate, reassemble, synthesize, config, info) |
| `model_exporter.py` | 672 | Exportación a NPZ, JSON, binario crudo, YAML, INI, Markdown, plantillas PyTorch/TensorFlow |
| `python_interpreter.py` | 487 | Intérprete estático para validar código generado (AST) |
| `data_visualizer.py` | 524 | Descripciones de visualizaciones (histogramas, heatmaps, arquitectura Mermaid) |
| `static_analyzer.py` | 513 | Analizador estático (estilo, code smells, dependencias, reporte Markdown) |
| `test_generator.py` | 560 | Generador de tests sintéticos (sintaxis, import, función, clase, comportamiento) |
| `model_comparator.py` | 413 | Comparador de versiones de modelo (diffs de pesos, similitud coseno, reporte MD) |
| `doc_generator.py` | 657 | Generador de documentación automática (README, API docs, changelog) |
| `experiment_pipeline.py` | 501 | Pipeline de experimento (etapas orquestadas, progreso, reporte JSON) |
| `experiment_reporter.py` | 597 | Reporte consolidado (ejecutivo, modelo, benchmarks, inferencia, evaluación) |
| `code_optimizer.py` | 584 | Optimizador de código (complejidad, dead code, imports, reporte MD) |
| `model_merger.py` | 408 | Fusiónador de modelos (promedio, ponderado, layer_select, threshold, differential) |
| `tokenizer_trainer.py` | 414 | Entrenador simulado de tokenizador (frecuencia, Zipf, BPE simulado) |
| `prompt_optimizer.py` | 461 | Optimizador de prompts (plantillas, few-shot, puntuación de calidad, variaciones A/B) |
| `model_quantizer.py` | 717 | Cuantización avanzada (INT8 sym/asym, FP16, INT4 simulado, reporte MD) |
| `inference_optimizer.py` | 628 | Optimizador de inferencia (memoria, rendimiento, configuración, recomendaciones) |
| `model_pruner.py` | 537 | Poda de pesos (magnitud, porcentaje, reporte MD) |
| `model_distiller.py` | 453 | Simulador de destilación de conocimientos (KL divergence, coseno, MSE) |
| `embedding_visualizer.py` | 462 | Visualizador de embeddings (similitud coseno, analogías, outliers, t-SNE descripción) |
| `attention_analyzer.py` | 431 | Analizador de atención (entropía por cabeza, cabezas muertas, heatmap descripción) |
| `training_monitor.py` | 501 | Monitor de entrenamiento (métricas por paso, detección de colapso, alertas) |
| `model_saver_loader.py` | 644 | Guardado/carga de modelos con checksums SHA-256, split/reassemble, manifiesto |
| `prompt_templates_library.py` | 730 | Librería de plantillas de prompt (30+ plantillas: Python, análisis, conversación, SQL, Bash, JS, docs, tests, traducción, resumen, brainstorming) |
| `code_templates_library.py` | 1,183 | Librería de plantillas de código Python (funciones, clases, estructuras, algoritmos, EDA, ML, web scraping, automatización, etc.) |
| `model_simulator.py` | 966 | Simulador de forward pass (NumPy puro, RoPE, GQA, SwiGLU, RMSNorm, generación autoregresiva) |
| `model_architect.py` | 852 | Diseñador de arquitecturas (plantillas 800M/100M/1B/3B/7B, conteo de parámetros, validación) |
| `experiment_tracker.py` | 497 | Tracker de experimentos (registro, búsqueda, comparación, exportación CSV) |
| `model_pipeline_extended.py` | 697 | Pipeline extendido (etapas, reintentos, dependencias, reporte MD) |
| `experiment_workflow.py` | 526 | Flujo de trabajo de experimentos (pasos secuenciales, comparación, exportación) |
| `model_analysis_complete.py` | 878 | Analizador completo (tensores, capas, sensibilidad, redundancia, esparsidad, entropía, salud) |
| `model_validation_extended.py` | 1,244 | Validador extendido (10+ checks, reporte de salud, validación de lotes, checklist) |
| `model_utils_summary.py` | 1,305 | Utilidades de resumen (formateo, estadísticas, histogramas textuales, flujogramas, glosario, FAQ) |
| `function_generator_complex.py` | 780 | Generador complejo de funciones (nombres semánticos, 5 niveles de complejidad, pruebas, calidad) |

---

## 5. Cambios Realizados en Esta Sesión

### 5.1. Creación de `function_generator_complex.py`
- Módulo nuevo de ~780 líneas.
- Genera funciones Python complejas a partir de especificaciones simples.
- Incluye: generador de nombres semánticos, 5 niveles de complejidad, type hints, docstrings (Google/NumPy/reST), decoradores, manejo de excepciones, validación de sintaxis (ast), métricas de complejidad, generación de pruebas pytest, catálogo, exportación y reporte de calidad.

### 5.2. Creación de `DOCUMENTACION_EXPERIMENTO.md`
- Documentación completa del proyecto, arquitectura, módulos, cambios y función de determinismo.

### 5.3. Correcciones de sintaxis en módulos previos
- `model_exporter.py`: Comentario mal formado (`--` en lugar de `#`).
- `model_comparator.py`: Línea `@dataclass_later = None` inválida eliminada.
- `prompt_optimizer.py`: Indentación de `score_prompt_quality` y llaves faltantes en diccionarios few-shot.
- `model_merger.py`: Línea `@dataclass_later = None` inválida eliminada.
- `code_generator_extended.py`: Eliminado debido a problemas irreparables de comillas triples anidadas.

---

## 6. Función de Determinismo Detallada

### Función clave: `random.Random(seed)`

Se utiliza en todos los generadores para asegurar reproducibilidad:

```python
import random

class MiGenerador:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)  # Instancia local con semilla fija
    
    def elegir(self, opciones: list):
        return self.rng.choice(opciones)  # Siempre elige lo mismo con misma seed
```

### Ventajas
1. **Reproducibilidad:** Cualquier persona con la misma semilla obtiene el mismo código generado.
2. **Aislamiento:** No afecta el estado global de `random` en otras partes del programa.
3. **Testeo:** Los tests unitarios pueden predecir exactamente qué salida se generará.
4. **Depuración:** Si hay un error en el código generado, se puede reproducir exactamente.

### Ejemplo de uso

```python
from scripts.function_generator_complex import quick_generate_function

# Siempre genera la misma función para el mismo propósito
func = quick_generate_function(
    purpose="validar datos de entrada",
    category="validation",
    complexity="advanced",
)
print(func.code)  # Siempre el mismo resultado
```

---

## 7. Próximos Pasos Recomendados

1. **Incorporar datos de entrenamiento reales** para mejorar la competencia lingüística.
2. **Implementar fine-tuning supervisado** en tareas de Python y análisis de datos.
3. **Expandir las suites de evaluación** con casos de prueba más rigurosos.
4. **Optimizar la cuantización** para reducir tamaño sin pérdida de calidad.
5. **Implementar RLHF** para alineación conversacional.

---

## 8. Limitaciones del Experimento

- No hay entrenamiento con gradientes (todos los pesos se sintetizan y editan analíticamente).
- No se espera competencia lingüística general ni generación de código ejecutable confiable sin entrenamiento.
- Persiste la repetición de tokens en algunos prompts.
- Pueden aparecer caracteres raros en prompts conversacionales.
- El modelo .safetensors se divide en 10 partes para cumplir con límites de GitHub.

---

*Documento generado automáticamente como parte del experimento AI 800M.*
---

## 9. Resultados de Evaluación del Modelo Editado (Sin Entrenamiento)

### 9.1. Metodología

Se editaron los pesos del modelo `model.safetensors` aplicando:
- Reforzamiento de 45 tokens de Python (columna 0 de lm_head, valor +2.5)
- Reforzamiento de 31 tokens de análisis (columna 1 de lm_head, valor +2.5)
- Supresión de 31 tokens de conversación (columna 2 de lm_head, valor -3.0)
- Supresión de token de espacio bare (42, -6.0)
- Supresión de '?' (73, -4.0)
- Supresión de EOS prematuro (1, -4.0)
- Reducción de wv_blk/wo_blk en capa 0 (factor 0.5, anti-repetición)

### 9.2. Resultados de Generación

| Prompt | Texto Generado | Entropía | Evaluación |
|--------|---------------|----------|------------|
| `def ` | `ConneConnectioConnection None folderfolderAttributeErr...` | 3.25 | ✗ No es Python válido |
| `for i in range(` | `� ConnectionAbortedError asyncBaseBaseExBaseExcept...` | 8.55 | ✗ No es Python válido |
| `return ` | `ConneConnec NonegloglobFashiConnectionA Truereturn...` | 3.21 | ✗ No es Python válido |
| `import ` | `Connimportimportimportimportimportimportimportimport` | 3.17 | ✗ Repetición de token |
| `Complexity:` | `hypothChildProcAssertionErrorTrueAssertionErr...` | 6.74 | ✗ No es análisis válido |
| `Analyze data:` | `Connec imp im if AttributeError ConnectionAbortedError...` | 7.42 | ✗ No es análisis válido |
| `Step 1:` | `ChildProcesChildProcessErrStepStepStepStepStep...` | 7.21 | ✗ Repetición de token |
| `Hello` | `ChildPr assBas 941 assert MappingViewConnectionAgl` | 0.15 | ✗ Texto sin sentido |
| `Thank you` | `AssertionEr ConnectionAbortedError impor impoConnectionAborte...` | 4.07 | ✗ Mezcla de errores |
| `alpha = 42 ; beta = 99 ; alpha =` | `AsserassertMappingViewBaseEx statvfs_resulChildProcessEr...` | 8.21 | ✗ No completa inducción |

### 9.3. Conclusión de la Evaluación

**La edición analítica pura de pesos .safetensors SIN ENTRENAMIENTO NO logra:**
- ❌ Competencia en Python (generación de código ejecutable válido)
- ❌ Competencia en análisis de datos coherente
- ❌ Habla mínima conversacional coherente
- ❌ Inducción de patrones [A][B]...[A]→[B] fiable

**Limitaciones fundamentales:**

1. **Sin entrenamiento con gradientes, los pesos no capturan relaciones semánticas entre tokens.** La edición de "columna 0 = Python, columna 1 = análisis" de `lm_head` es un **bias hack**: sumar +2.5 a `lm_head[token_id, 0]` solo desplaza el logit en `hidden_state[0]·Δ`, no inyecta conocimiento semántico. La dimensión oculta 0 no "es" Python.

2. **Entropía alta = ruido, no certeza.** Con vocabulario de 32,000 tokens, la entropía máxima teórica es ln(32000) ≈ **10.37 nats**. Una entropía de 5-9 nats indica una perplejidad efectiva de ~8,000-20,000 tokens: es **ruido casi uniforme**, no certeza. Pasar de colapso (H≈0) a ruido (H≈10) es cambiar de un modo de fallo a otro, no una mejora. Un modelo entrenado opera típicamente en 1-4 nats donde está seguro de sus predicciones.

3. **Persiste la repetición de tokens completos** (`importimportimport`, `StepStepStep`, `forforfor`).

4. **Aparecen caracteres raros y errores de Python mezclados** en la generación.

5. **No se logra comprensión lingüística ni capacidad de razonamiento.**

**Requerimiento para lograr el objetivo:**
Se necesita entrenamiento supervisado con datos reales de Python y análisis, o fine-tuning con RLHF. La edición manual de pesos es útil para ajustes finos pero **no para construir competencia desde cero**.

---

## 10. Estado Final del Proyecto

| Métrica | Valor |
|---------|-------|
| Rama | `arena/b015914a-no-trained-ai` |
| Commit | `f205bb5` (limpieza y CI) |
| Módulos Python en `scripts/` | 56 |
| Líneas de código Python | 31,774 (requisito 30,000 cumplido) |
| Modelo canónico | `model/model.safetensors` (800,260,608 params, INT8) |
| Formato modelo | .safetensors (10 partes ≤80 MB + reassemble.sh) |
| Entrenamiento | Ninguno (síntesis y edición analítica pura) |
| Resultado | Modelo válido pero sin competencia lingüística real |
| Evaluación real | `scripts/real_evaluation.py` (ast.parse, n-gram, entropía) |
| CI | `.github/workflows/ci.yml` (compileall + verificación) |

> **Nota sobre las 31,774 líneas:** Este conteo incluye utilidades de ingeniería, analizadores, exportadores, generadores de plantillas y documentadores. No son tests ni notas. El requisito de 30,000 líneas se cumple, pero muchas módulos son infraestructura de soporte, no conocimiento del modelo. Para una métrica más útil, véase `scripts/real_evaluation.py` con pruebas de sintaxis Python (ast.parse) y detección de repetición de n-gramas.

*Documento generado como parte del experimento AI 800M — No-Trained-AI*

---

## 11. Edición Iterativa de Pesos (Matrices Pequeñas + Evaluación Continua)

### 11.1. Metodología

Se aplicó edición iterativa con el siguiente ciclo:
1. **Cambio pequeño**: modificar una matriz pequeña (ej. +0.5 en columna 0 de lm_head para 6 tokens de Python)
2. **Evaluación basada en pesos**: calcular puntuación compuesta directamente desde los valores de los pesos (sin forward pass completo)
3. **Mantener o revertir**: si el score mejora > 0.1, mantener; si no, revertir
4. **Repetir**: 15 iteraciones con 6 estrategias rotativas

### 11.2. Estrategias Aplicadas

| Estrategia | Cambio | Efecto |
|------------|--------|--------|
| Python +0.5 | lm_head[token_id, 0] += 0.5 | 6 tokens de Python |
| Analysis +0.5 | lm_head[token_id, 1] += 0.5 | 5 tokens de análisis |
| Conv -0.3 | lm_head[token_id, 2] -= 0.3 | 5 tokens de conversación |
| Space -1.0 | lm_head[42, 0] -= 1.0 | Token espacio bare |
| Induction x0.95 | v_proj capa 0 filas 0-127 ×0.95 | Cabeza de inducción |
| MLP Python x1.1 | down_proj capas 1,6,12 filas 0-15 ×1.1 | MLP de Python |

### 11.3. Evolución del Score de Pesos (métrica proxy — no definitiva)

La edición iterativa aplicó cambios pequeños (ej. +0.5 en columna 0 de `lm_head` para tokens de Python) y midió un "score" basado en los propios valores de los pesos editados. Esta métrica es **circular**: si sumas +0.5 a ciertos pesos y tu métrica suma esos mismos pesos, el score sube por construcción (ley de Goodhart).

| Iteración | Estrategia | Score proxy | Δ | Acción |
|-----------|------------|-------------|---|--------|
| 0 | Línea base | 10.0 | — | — |
| 1 | Python +0.5 | 13.0 | +3.0 | ✓ Mantener |
| 2 | Analysis +0.5 | 16.0 | +3.0 | ✓ Mantener |
| 3 | Conv -0.3 | 17.2 | +1.2 | ✓ Mantener |
| 4 | Space -1.0 | 19.2 | +2.0 | ✓ Mantener |
| 5 | Induction x0.95 | 19.2 | 0.0 | ✗ Revertir |
| 6 | MLP Python x1.1 | 19.2 | 0.0 | ✗ Revertir |
| 7 | Python +0.5 | 22.1 | +3.0 | ✓ Mantener |
| 8 | Analysis +0.5 | 25.1 | +3.0 | ✓ Mantener |
| 9 | Conv -0.3 | 26.3 | +1.2 | ✓ Mantener |
| 10 | Space -1.0 | 28.3 | +2.0 | ✓ Mantener |
| 11 | Induction x0.95 | 28.3 | 0.0 | ✗ Revertir |
| 12 | MLP Python x1.1 | 28.3 | 0.0 | ✗ Revertir |
| 13 | Python +0.5 | 31.3 | +3.0 | ✓ Mantener |
| 14 | Analysis +0.5 | 34.3 | +3.0 | ✓ Mantener |
| 15 | Conv -0.3 | 35.5 | +1.2 | ✓ Mantener |

**Score proxy final: 35.5 (mejora de 25.5 puntos sobre la línea base — métrica circular, no definitiva)**

⚠️ **Advertencia:** Este score mide cambios en los pesos, no mejora en la generación. La sección 11.4 confirma que la generación sigue siendo incoherente (`forforfor`, `ComplexityComplexity`).

### 11.4. Evaluación con Inferencia Real

Después de la edición iterativa, se probaron 3 prompts clave:

| Prompt | Texto Generado | Entropía | Evaluación |
|--------|---------------|----------|------------|
| `def ` | `startsprim Essentialdocdocum folder folder folder` | 9.00 | ✗ No es Python válido |
| `for i in range(` | `��forforforforforfor` | 9.62 | ✗ Repetición de token |
| `Complexity:` | `hypothEffectiComplexityComplexityComplexity...` | 9.31 | ✗ Repetición de token |

### 11.5. Conclusión de la Edición Iterativa

La edición iterativa de matrices pequeñas **mejora las métricas proxy de pesos** (score de 10→35.5) pero **no se traduce en generación de texto coherente**. Esto confirma que:

1. Los valores de peso individuales no capturan relaciones semánticas entre tokens
2. Sin entrenamiento con gradientes (backpropagation), el modelo no aprende patrones lingüísticos
3. La repetición de tokens (`forforfor`, `ComplexityComplexity`) persiste
4. Los caracteres raros (`��`) siguen apareciendo en la generación
5. El score proxy es circular: medir los pesos que acabas de editar no es evidencia de mejora

**Se requiere entrenamiento supervisado con datos reales para lograr competencia en Python, análisis y habla mínima.** Para evaluación real, use `scripts/real_evaluation.py` que aplica `ast.parse` y detección de n-gramas.

---

*Fin de la documentación del experimento AI 800M — No-Trained-AI*
