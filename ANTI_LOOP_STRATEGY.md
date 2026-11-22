# Estrategia Anti-Loop — Modelo 800M Sin Entrenamiento

**Proyecto:** No-Trained-AI
**Rama:** `arena/b015914a-no-trained-ai`
**Fecha:** 2026-11-22
**Objetivo:** Eliminar bucles de repetición (forforfor, ComplexityComplexity, importimportimport) sin entrenamiento con gradientes

---

## 1. Problema Observado

Durante la evaluación del modelo 800M editado, se detectaron bucles de repetición en la generación:

| Prompt | Texto generado (loop) | Tipo de loop |
|--------|----------------------|--------------|
| `def ` | `Folde folderdocumdocumAsdocdocdoc Essentialpr` | Multi-token (doc, As) |
| `for i in range(` | `�forforforforforforforforfor` | Single-token (for ×9) |
| `Complexity:` | `hypothEffectiComplexityComplexityComplexity...` | Multi-token (Complexity) |
| `import ` | `Connimportimportimportimportimportimportimportimport` | Single-token (import) |

---

## 2. Análisis de Causa Raíz

### 2.1. Pesos del Modelo (NO son la causa principal)

El análisis de pesos (`anti_loop_weight_analyzer.py`) reveló:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Auto-similitud media de atención | -0.0 | Muy baja → las cabezas NO causan loops |
| Auto-similitud máxima de atención | 0.0876 | Muy baja → no hay auto-refuerzo en atención |
| Riesgo de repetición Python (lm_head col 0) | 1.489 | Moderado |
| Riesgo de repetición Análisis (lm_head col 1) | 1.495 | Moderado |
| Riesgo de repetición Conversación (lm_head col 2) | -0.903 | Suprimido |
| Token espacio bare (42, col 0) | -1.984 | Suprimido |
| Norma v_proj capa 0 (induction head) | 1.0 | Normal |

**Conclusión:** Las cabezas de atención ya tienen baja auto-similitud. Los loops NO vienen de los pesos de atención, sino del **proceso de muestreo durante la generación**.

### 2.2. Muestreo (SÍ es la causa)

El token `for` tiene una probabilidad muy alta en `lm_head`, y la penalidad de repetición estándar (1.5) no es suficiente para superar su probabilidad base. El modelo selecciona `for` una y otra vez porque:

1. `for` tiene un logit muy alto en lm_head
2. La penalidad de repetición del 1.5 reduce el logit pero no lo suficiente
3. Una vez seleccionado `for`, el contexto refuerza la selección de `for` nuevamente

---

## 3. Estrategias Anti-Loop Evaluadas

### 3.1. Configuraciones Probadas

| Configuración | Parámetros | Loop `forforfor` | Loop `docdocdoc` | Nota |
|--------------|-----------|------------------|------------------|------|
| baseline | temp=0.3, top_k=6, rep=1.5 | 🔴 Sí | 🔴 Sí | Configuración por defecto |
| rep_pen=5.0 | temp=0.3, top_k=6, rep=5.0 | 🟢 No | - | ✅ Efectivo |
| rep_pen=10.0 | temp=0.3, top_k=6, rep=10.0 | 🟢 No | - | ✅ Efectivo |
| freq_pen=5.0 | temp=0.3, top_k=6, rep=1.5, freq=5.0 | 🟢 No | - | ✅ Efectivo |
| presence_pen=2.0 | temp=0.3, top_k=6, rep=1.5, pres=2.0 | 🔴 Sí (9 reps) | - | ❌ Insuficiente |
| presence_pen=5.0 | temp=0.3, top_k=6, rep=1.5, pres=5.0 | 🟢 No | - | ✅ Efectivo |
| combined | temp=0.3, top_k=6, rep=1.8, freq=0.3, pres=0.3 | - | 🟢 No | ✅ Efectivo multi-token |
| all_extreme | temp=0.2, top_k=4, rep=5.0, freq=2.0, pres=2.0 | 🟢 No | - | ✅ Muy efectivo |
| n_gram_block=3 | temp=0.3, top_k=6, rep=1.5, ngram=3 | 🔴 Sí | 🔴 Sí | ❌ No ayuda con single-token |

### 3.2. Reglas Definitivas

#### Para bucles de un solo token (forforfor, importimportimport):

**UNA de las siguientes es necesaria:**

| Parámetro | Valor mínimo | Efecto |
|-----------|-------------|--------|
| `repetition_penalty` | >= 5.0 | Divide/multiplica logits de tokens repetidos |
| `frequency_penalty` | >= 5.0 | Resta `penalty × frecuencia` del logit |
| `presence_penalty` | >= 5.0 | Resta `penalty` si el token ha aparecido |

#### Para bucles multi-token (docdocdoc, ComplexityComplexity):

| Parámetro | Valor recomendado | Efecto |
|-----------|------------------|--------|
| `repetition_penalty` | 1.8 - 2.5 | Penalización moderada |
| `frequency_penalty` | 0.3 - 0.5 | Penalización por frecuencia |
| `presence_penalty` | 0.3 - 0.5 | Penalización por presencia |

#### Configuración recomendada (equilibrada):

```python
{
    "temperature": 0.3,
    "top_k": 6,
    "top_p": 1.0,
    "repetition_penalty": 2.0,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
}
```

#### Configuración agresiva (si persisten loops):

```python
{
    "temperature": 0.2,
    "top_k": 4,
    "top_p": 1.0,
    "repetition_penalty": 5.0,
    "frequency_penalty": 2.0,
    "presence_penalty": 2.0,
}
```

---

## 4. Implementación en el Código

### 4.1. Módulo: `scripts/anti_loop_inference.py`

Implementa todas las estrategias anti-loop como una envoltura sobre `MmapInt8LlamaModel`:

```python
from scripts.anti_loop_inference import AntiLoopInferenceEngine

engine = AntiLoopInferenceEngine(Path("/home/user/No-Trained-AI/model"))

result = engine.generate_anti_loop(
    prompt="for i in range(",
    max_new_tokens=10,
    temperature=0.3,
    top_k=6,
    repetition_penalty=5.0,      # ← clave para bucles single-token
    frequency_penalty=0.5,       # ← ayuda con bucles multi-token
    presence_penalty=0.5,        # ← ayuda con bucles multi-token
    n_gram_block=3,              # ← bloquea n-grams repetidos (útil para secuencias)
    dynamic_rep_pen=True,        # ← aumenta rep_pen si se detecta loop
    temp_anneal=True,            # ← reduce temperatura si hay loop
    dynamic_top_k=True,          # ← reduce top_k si hay loop
)
```

### 4.2. Funciones de utilidad incluidas

| Función | Descripción |
|---------|-------------|
| `detect_token_loop(text, min_repeats)` | Detecta repetición de token completo |
| `detect_ngram_loop(tokens, n, min_repeats)` | Detecta n-grams repetidos |
| `loop_score(text)` | Puntuación de severidad de loop (0-1) |
| `apply_repetition_penalty(logits, ids, penalty)` | Penalidad clásica de repetición |
| `apply_frequency_penalty(logits, ids, penalty)` | Penalidad por frecuencia |
| `apply_presence_penalty(logits, ids, penalty)` | Penalidad por presencia |
| `apply_ngram_blocking(logits, ids, n, penalty)` | Bloquea n-grams ya generados |

---

## 5. Limitaciones

| Limitación | Explicación |
|------------|-------------|
| No elimina todos los loops | Algunas configuraciones siguen produciendo repetición |
| Penalías altas reducen diversidad | rep_pen=10.0 puede hacer el texto muy predecible |
| No funciona para todos los prompts | Algunos prompts son más propensos a loops que otros |
| Requiere ajuste por caso | No hay configuración única que funcione para todo |

---

## 6. Plan de Acción

1. **Inmediato:** Usar `repetition_penalty=2.0` + `frequency_penalty=0.5` + `presence_penalty=0.5` como configuración por defecto
2. **Si persisten loops:** Aplicar `repetition_penalty=5.0` o `presence_penalty=5.0`
3. **Futuro:** Implementar `anti_loop_inference.py` en `inference_engine.py` como método nativo del modelo
4. **Futuro:** Combinar con activation steering para empujar activaciones lejos de direcciones de repetición

---

## 7. Archivos Relacionados

| Archivo | Descripción |
|---------|-------------|
| `scripts/anti_loop_inference.py` | Motor de inferencia con capacidades anti-loop |
| `scripts/anti_loop_weight_analyzer.py` | Analizador de loops basado en pesos (sin forward pass) |
| `scripts/anti_loop_evaluator.py` | Evaluador de estrategias anti-loop (inferencia completa) |
| `scripts/evaluate_quick.py` | Evaluación rápida del modelo editado |
| `scripts/edit_model_intelligent.py` | Edición inteligente de pesos |

---

*Documento generado como parte del experimento AI 800M — No-Trained-AI*