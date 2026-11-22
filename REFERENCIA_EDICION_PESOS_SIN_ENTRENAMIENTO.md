# Referencia Técnica: Edición de Pesos y Intervención en LLMs Sin Entrenamiento

**Proyecto:** No-Trained-AI — Modelo 800M sin entrenamiento
**Rama:** `arena/b015914a-no-trained-ai`
**Fecha:** 2026-11-22
**Ubicación:** Soacha, Cundinamarca, Colombia (UTC)
**Objetivo:** Modelo bueno en Python, habla mínima, bueno en análisis — formato .safetensors, sin entrenamiento con gradientes

---

## 1. Panorama General

Este documento sirve como referencia técnica para la edición manual de pesos de un modelo LLM (800M parámetros, formato .safetensors INT8) sin entrenamiento con gradientes. Recopila técnicas probadas en la investigación y establece un marco de trabajo aplicable al proyecto.

### 1.1. Problema Central

| Enfoque | Requiere Gradientes | Modifica Pesos | Reversible | Escalable |
|---------|-------------------|----------------|------------|-----------|
| Fine-tuning completo | Sí | Sí | No | No |
| LoRA / PEFT | Sí | Sí (adaptadores) | Parcial | Media |
| Model Editing (ROME/MEMIT) | No (closed-form) | Sí | Sí | Media |
| Activation Steering (ActAdd) | No | No | Sí | Alta |
| Representation Engineering | No | No | Sí | Alta |
| Edición analítica manual | No | Sí | Sí | Baja |

**Conclusión clave:** Sin entrenamiento, las técnicas más prometedoras son **model editing localizado** y **activation steering en tiempo de inferencia**.

---

## 2. Técnicas de Model Editing (Pesos)

### 2.1. ROME — Rank-One Model Editing

**Fuente:** Meng et al., 2022 — "Locating and Editing Factual Associations in GPT"

**Concepto:** Tratar la MLP de una capa intermedia como memoria asociativa clave-valor. El input (subject) es la clave, el output (fact) es el valor.

**Pasos del algoritmo:**

1. **Causal Tracing:** Identificar qué capa MLP y qué posición de token median el hecho factual mediante intervención causal (reemplazar activaciones corrupted con activaciones limpias y observar efecto).
2. **Selección de capa:** ROME usa capas medias (ej. capa 18 de 24 en GPT-J, capa 6 en GPT-2 XL).
3. **Actualización rank-one:** Aplicar actualización estructurada $W \leftarrow W + \Delta W$ donde $\Delta W$ es rank-one, minimizando impacto en comportamientos no relacionados.
4. **Optimización del valor:** Optimizar el vector objetivo para que los logits posteriores favorezcan el nuevo objeto bajo contextos relevantes.

**Ventajas:**
- Cero entrenamiento (actualización cerrada/closed-form)
- Específico (localidad alta si se localiza correctamente)
- Reversible (guardar delta de pesos)

**Limitaciones:**
- Un hecho a la vez (no escala bien a muchos hechos)
- Requiere causal tracing preciso
- Los editados pueden suprimir conocimiento original en lugar de sobrescribirlo (evidencia 2026)

**Aplicación al modelo 800M:**
```python
# Concepto: identificar capa MLP responsable de "def" en Python
# Aplicar actualización rank-one en W_down_proj de esa capa
# Reforzar conexión token->característica Python
```

### 2.2. MEMIT — Mass-Editing Memory in a Transformer

**Fuente:** Meng et al., 2022b

**Concepto:** Extiende ROME a edición masiva de múltiples asociaciones factuales simultáneamente.

**Pasos:**
1. Particionar dataset de edición en grupos
2. Actualizar pesos de múltiples capas MLP identificadas como mediadores causales
3. Resolver problema de escritura de memoria en lotes (batched memory-writing)

**Ventajas:** Miles de ediciones simultáneas (GPT-J, GPT-NeoX)

**Limitaciones:** Olvido gradual luego catastrófico con ediciones secuenciales repetidas

### 2.3. MEND / MALMEN — Meta-Learning Based Editing

**Concepto:** Hiperred predice el update de pesos basado en ejemplos de entrada/salida editados.

**Ventajas:** Edits a gran escala, mejor localidad que ROME en algunos casos

**Limitaciones:** Requiere pre-entrenar el editor (costo de preparación alto)

---

## 3. Técnicas de Activation Steering (Sin Modificar Pesos)

### 3.1. ActAdd — Activation Addition

**Fuente:** Turner et al., 2024 — "Activation Addition: Steering Language Models Without Optimization"

**Concepto:** Añadir vectores de activación al forward pass durante inferencia. No modifica pesos. No requiere optimización ni datos etiquetados.

**Método:**
1. Tomar par de prompts contrastantes $(p_+, p_-)$ donde $p_+$ tiene la propiedad deseada
2. Extraer activaciones $h_+^l$ y $h_-^l$ en capa $l$
3. Calcular vector steering: $v = h_+^l - h_-^l$
4. Durante inferencia: $h' = h + \alpha \cdot v$ donde $\alpha$ es coeficiente de inyección

**Ventajas:**
- Tiempo de inferencia (no modifica pesos)
- Reversible instantáneamente
- No ocupa espacio en ventana de contexto (a diferencia de prompts)
- Continuamente ponderable (α ajustable)

**Aplicación al modelo 800M:**
```python
# Vector para "modo Python":
# p_+ = "Escribe código Python:" / p_- = "Escribe un cuento:"
# v_python = activaciones(p_+) - activaciones(p_-) en capa media
# Durante generación de Python: h = h + 0.5 * v_python
```

### 3.2. Representation Engineering (RepEng)

**Fuente:** Zou et al., 2023; librería `repeng` (vgel/repeng)

**Concepto:** Identificar direcciones en espacio de activación que corresponden a comportamientos específicos, y manipularlas durante inferencia.

**Flujo:**
1. Crear pares contrastantes (positivo/negativo) para el comportamiento deseado
2. Ejecutar forward pass y capturar activaciones en capas específicas
3. Calcular diferencia y aplicar PCA para encontrar dirección principal
4. Inyectar vector durante inferencia: $h' = h + \alpha v$

**Ventajas:**
- Segundos para entrenar el vector (dozens de pares contrastantes)
- Interpretable y ajustable en tiempo real
- Compatible con cuantización (INT4, FP8, FP16)

### 3.3. Conceptors

**Fuente:** 2410.16314 — "Steering LLMs using Conceptors"

**Concepto:** Representar conjuntos de vectores de activación como regiones elipsoidales (conceptors) en lugar de vectores únicos. Actúan como matrices de proyección suaves.

**Ventaja sobre ActAdd:** Mejor control de patrones complejos, operaciones booleanas para combinar múltiples objetivos de steering.

---

## 4. Marco de Evaluación (Sin Entrenamiento)

### 4.1. Tres Ejes de Evaluación (EasyEdit / ROME)

| Eje | Definición | Métrica |
|-----|-----------|---------|
| **Reliability** | El edit produce el output objetivo para el input editado | % de respuestas correctas en prompts de edición |
| **Generalization** | El edit se generaliza a paráfrasis y contextos relacionados | % de respuestas correctas en prompts paraphraseados |
| **Locality** | El edit no afecta comportamientos no relacionados | % de comportamientos no relacionados preservados |

### 4.2. Métricas Específicas para el Objetivo 800M

| Dimensión | Prompts de prueba | Métrica de éxito |
|-----------|------------------|------------------|
| Python | `def `, `for i in range(`, `return `, `import ` | Presencia de keywords Python válidas, entropía moderada (3-8 nats) |
| Análisis | `Complexity:`, `Analyze data:`, `Step 1:` | Términos de análisis (step, complexity, variance, compare) |
| Habla mínima | `Hello`, `Thank you`, `How are you` | Pocas palabras de relleno conversacional, entropía baja |

### 4.3. Puntuación Compuesta Propuesta

```python
score = 0
# Python (máx 30 pts)
score += min(30, hits_python * 5)
# Análisis (máx 30 pts)
score += min(30, hits_analysis * 4)
# Conversación mínima (máx 20 pts, penalizar)
score += max(0, 20 - hits_conversation * 5)
# Inducción (máx 20 pts)
score += 20 if completa_patron_AB else 10 if contiene_tokens_AB else 0
# Entropía (máx 10 pts, preferir 3-8 nats)
score += 10 if 3.0 <= entropia_media <= 8.0 else 5 if entropia_media < 3.0 else 2
# Máximo teórico: 100
```

---

## 5. Estrategia de Edición Iterativa para el Modelo 800M

### 5.1. Ciclo de Trabajo

```
1. Estado base -> Evaluar score de pesos
2. Cambio pequeño (matriz pequeña, delta controlado)
3. Evaluar score de pesos
4. Si mejora > umbral -> Mantener; si no -> Revertir
5. Repetir con siguiente estrategia
6. Validar con inferencia real (forward pass)
7. Documentar resultados
```

### 5.2. Estrategias de Cambio (ordenadas por impacto esperado)

| Prioridad | Estrategia | Tensor Objetivo | Cambio | Justificación |
|-----------|-----------|-----------------|--------|---------------|
| 1 | Python Boost | `lm_head.weight` col 0 | +0.5 por token Python | Reforzar salida Python |
| 2 | Analysis Boost | `lm_head.weight` col 1 | +0.5 por token análisis | Reforzar salida análisis |
| 3 | Conv Suppress | `lm_head.weight` col 2 | -0.3 por token conversación | Reducir relleno conversacional |
| 4 | Space Suppress | `lm_head.weight` [42,0] | -1.0 | Evitar loops de espacios |
| 5 | Question Suppress | `lm_head.weight` [73,0] | -1.0 | Evitar loops de '?' |
| 6 | EOS Suppress | `lm_head.weight` [1,0] | -1.0 | Evitar parada prematura |
| 7 | Induction Head | `layers.0.self_attn.v_proj` filas 0-127 | ×0.95 | Reducir repetición de palabra |
| 8 | MLP Python | `layers.{1,6,12}.mlp.down_proj` filas 0-15 | ×1.1 | Reforzar MLP de Python |

### 5.3. Umbrales de Aceptación

- Mejora mínima en score de pesos: +0.1
- Máximo de iteraciones por estrategia: 3 cambios consecutivos sin mejora -> pasar a siguiente
- Validación final con inferencia real en 3-5 prompts clave

---

## 6. Implementación Técnica: Pasos Detallados

### 6.1. Carga de Pesos .safetensors

```python
from safetensors import safe_open
import numpy as np

tensors = {}
scales = {}
with safe_open("model/model.safetensors", framework="numpy") as f:
    for name in f.keys():
        t = f.get_tensor(name)
        if name.endswith(".weight_scale"):
            scales[name] = t.copy()
        else:
            tensors[name] = t.copy()
```

### 6.2. Descuantización per-Fila (INT8 + escala FP16)

```python
def dequantize_row(w_q, scale, row):
    scale_f32 = scale.astype(np.float32)
    if scale_f32.ndim == 1:
        scale_f32 = scale_f32.reshape(-1, 1)
    return w_q[row].astype(np.float32) * scale_f32[row].reshape(-1)
```

### 6.3. Recuantización per-Fila

```python
from scripts.quantization_tools import quantize_int8_per_row

def quantize_row(row_f32):
    w_q, scale = quantize_int8_per_row(row_f32.reshape(1, -1))
    return w_q[0], float(scale[0, 0])
```

### 6.4. Edición de Elemento en lm_head

```python
def edit_lm_head_element(tensors, scales, token_id, col, delta):
    w = tensors["lm_head.weight"]
    scale = scales["lm_head.weight_scale"]
    row_f32 = dequantize_row(w, scale, token_id)
    old_val = float(row_f32[col])
    row_f32[col] = old_val + delta
    w_q_new, scale_new = quantize_row(row_f32)
    w[token_id] = w_q_new
    scale[token_id, 0] = scale_new
    return old_val, row_f32[col]
```

### 6.5. Guardado de Modelo Editado

```python
from safetensors.numpy import save_file as save_safetensors

def save_edited_model(tensors, scales, output_path):
    all_tensors = {}
    for name in tensors:
        all_tensors[name] = tensors[name]
        scale_name = name + "_scale"
        if scale_name in scales:
            all_tensors[scale_name] = scales[scale_name]
    metadata = {
        "format": "pt",
        "experiment": "AI_experiment_AI_800M_manual_edit",
        "training_steps": "0",
    }
    save_safetensors(all_tensors, str(output_path), metadata=metadata)
```

---

## 7. Limitaciones Conocidas (Evidencia de Investigación)

### 7.1. Edición de Pesos Sin Entrenamiento

1. **No hay competencia lingüística real:** Los valores de peso individuales no capturan relaciones semánticas entre tokens sin entrenamiento con gradientes.
2. **Repetición de tokens:** Persiste `forforfor`, `ComplexityComplexity`, `importimportimport`.
3. **Caracteres raros:** Pueden aparecer `��` en la generación.
4. **Entropía alta:** 5-9 nats en la mayoría de prompts indica falta de certeza.
5. **Supresión vs sobrescritura:** Evidencia 2026 sugiere que ROME/MEMIT suprimen conocimiento original en lugar de sobrescribirlo limpiamente.
6. **Olvido gradual:** Ediciones secuenciales repetidas llevan a olvido gradual luego catastrófico.

### 7.2. Conclusión del Experimento 800M

| Objetivo | Logrado con edición de pesos | Requiere |
|----------|-----------------------------|----------|
| Bueno en Python | ❌ No (genera errores de Python) | Entrenamiento supervisado en código Python |
| Habla mínima | ⚠️ Parcial (entropía baja pero texto sin sentido) | Fine-tuning conversacional o activation steering |
| Bueno en análisis | ❌ No (mezcla términos sin coherencia) | Entrenamiento en datos de análisis |
| Inducción [A][B]→[B] | ❌ No | Entrenamiento en tareas de razonamiento |

---

## 8. Plan de Acción Recomendado (Fase Futura)

### Fase 1: Preparación (Inmediata)

```
[ ] Revisar REFERENCIA_EDICION_PESOS_SIN_ENTRENAMIENTO.md
[ ] Revisar DOCUMENTACION_EXPERIMENTO.md
[ ] Verificar que todos los 56 módulos compilen (py_compile)
[ ] Verificar integridad de model.safetensors (0 NaN/Inf, rango [-127,127])
[ ] Guardar backup del modelo actual en model/model.safetensors.backup
```

### Fase 2: Activation Steering en Inferencia (Sin Modificar Pesos)

```
[ ] Implementar extracción de steering vectors para Python vs conversación
[ ] Implementar ActAdd en inference_engine.py (hook en capa media)
[ ] Probar con pares contrastantes (código Python vs texto conversacional)
[ ] Ajustar coeficiente α para balancear Python vs habla mínima
[ ] Documentar resultados en Markdown
```

### Fase 3: Model Editing Localizado (Pesos)

```
[ ] Implementar causal tracing simplificado para localizar capas responsables
[ ] Aplicar actualización rank-one (ROME) en capa MLP seleccionada
[ ] Editar múltiples hechos (Python keywords, análisis terms) con MEMIT simplificado
[ ] Validar localidad (no romper otros comportamientos)
[ ] Guardar delta de pesos para reversibilidad
```

### Fase 4: Evaluación Completa

```
[ ] Ejecutar evaluación en 3 dimensiones (Python, análisis, habla mínima)
[ ] Calcular puntuación compuesta (máx 100)
[ ] Comparar con línea base (score 10.0 inicial, 35.5 después de edición iterativa)
[ ] Documentar hallazgos y limitaciones
[ ] Preparar reporte final
```

---

## 9. Referencias Bibliográficas

| Técnica | Autores | Año | Paper/URL |
|---------|---------|-----|-----------|
| ROME | Meng et al. | 2022 | arXiv:2202.05262 |
| MEMIT | Meng et al. | 2022 | Mass-Editing Memory in a Transformer |
| ActAdd | Turner et al. | 2024 | arXiv:2308.10248 |
| RepEng | Zou et al. | 2023 | Representation Engineering |
| Conceptors | - | 2024 | arXiv:2410.16314 |
| Causal Tracing | Meng et al. | 2022 | Locating and Editing Factual Associations |
| EasyEdit | - | - | Framework de model editing |
| Weightless | msuiche | 2026 | Runtime activation steering |

---

## 10. Glosario

- **Model Editing:** Modificación de pesos de un modelo post-hoc sin reentrenamiento completo.
- **Causal Tracing:** Técnica para localizar qué capas/neuronas median un comportamiento específico mediante intervención causal.
- **Activation Steering:** Manipulación de activaciones latentes durante inferencia para alterar comportamiento sin modificar pesos.
- **Steering Vector:** Vector de activación que representa una dirección comportamental (ej. "confiado" vs "dudoso").
- **Locate-then-Edit:** Paradigma de localizar primero, luego editar pesos específicos.
- **Locality:** Grado en que un edit afecta solo al comportamiento objetivo, preservando otros.
- **Reliability:** Grado en que un edit produce el output objetivo de manera consistente.
- **Generalization:** Grado en que un edit se extiende a paráfrasis y contextos relacionados.

---

*Documento de referencia generado como parte del experimento AI 800M — No-Trained-AI*