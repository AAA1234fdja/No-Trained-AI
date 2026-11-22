# Análisis: ¿sirvió el experimento "AI experiment AI"?

**Autor del análisis:** agente de Arena.ai (rama `arena/d55ec3df-no-trained-ai`)
**Fecha:** 22 de noviembre de 2026
**Fuente:** carpeta de Google Drive `AI experiment AI` (file ID `1B6gr_qo322gfIMHMCetb8tszMfWW9jCN`), copiada íntegramente a este repositorio excepto `model.safetensors` (~764 MB, excluido por `.gitignore` por el límite de 100 MB/archivo de GitHub).

---

## 1. Resumen ejecutivo

**¿Sirvió?** Parcialmente, y con matices.

- **Como ingeniería:** SÍ. Se construyó un archivo `.safetensors` válido, ejecutable y sin `NaN`/`Inf`, de 800,260,608 parámetros en INT8 por canal, con inferencia en memoria (`mmap`) que cabe en <1 GB de RAM, un tokenizador BPE de 32,000 tokens construido sin corpus, y cuatro etapas de edición analítica de pesos documentadas con código reproducible.
- **Como modelo de IA:** NO cumple sus objetivos declarados. No programa bien en Python, habla mínimamente, no entiende análisis, y sin entrenamiento colapsa en bucles de repetición, eco del prompt y entropía cercana a cero.

El documento del propio experimento llega a la misma conclusión: la edición analítica de pesos puede implementar **primitivas algorítmicas** (memoria de trabajo, cabeza de inducción `[A][B] ... [A] → [B]`, enrutamiento sintáctico tras `def `), pero no puede producir competencia lingüística o de programación generalista sin optimización (SGD/Adam) sobre datos.

---

## 2. Inventario de lo que se trajo de Drive

| Archivo | Tamaño | En este repo | Notas |
|---|---:|---|---|
| `model.safetensors` | 801,224,536 B (~764 MB) | **NO** (`.gitignore`) | Pesos INT8 finales, etapa 5. Drive file ID: `1Lb_97PHGKeJ1HB6k9Umu0Kpl953fNTAT` |
| `config.json` | 870 B | `model/config.json` | LlamaForCausalLM, V=32000, D=1536, L=21, H=12, KV=4, DFF=5888, 0 training steps |
| `tokenizer.json` | 2,316,783 B | `model/tokenizer.json` | BPE 32k tokens, programático, sin entrenamiento |
| `tokenizer_config.json` | 264 B | `model/tokenizer_config.json` | Mapeo de tokens especiales |
| `build_tokenizer.py` | 41,865 B | `scripts/build_tokenizer.py` | Constructor del tokenizador |
| `synthesize_weights.py` | 36,202 B | `scripts/synthesize_weights.py` | Síntesis analítica de pesos, etapas 0-4 |
| `run_stage5.py` | 17,408 B | `scripts/run_stage5.py` | Etapa final (fix del oscilador antipodal) + inferencia + verificación |
| `inference_engine.py` | 10,134 B | `scripts/inference_engine.py` | Motor mmap INT8: RoPE, GQA, SwiGLU, KV-cache |
| `EXPERIMENT_DOCUMENTATION.md` | 18,051 B | `docs/EXPERIMENT_DOCUMENTATION.md` | Memoria técnica completa: matemáticas, 7 fallos, resultados |
| `experiment_results.json` | 40,554 B | `results/experiment_results.json` | Top-5 logits, entropías y textos generados, etapas 0-5 |

Todos los archivos de texto fueron descargados de Drive y verificados: JSONs válidos, scripts Python completos (terminan con su bloque `if __name__ == "__main__"`).

---

## 3. Arquitectura y matemáticas (según la documentación)

- Subespacios ortogonales de los 1536 canales residuales: léxico `[0:512]`, categorías `[512:640]`, Python `[640:896]`, lógica/análisis `[896:1152]`, conversación `[1152:1280]`, buffer previo-token `[1280:1408]`, buffer de inducción `[1408:1536]`.
- Capa 0, cabeza 0: cabeza RoPE de "token anterior" ($\Delta = t-s = 1$) que copia identidad léxica y flags categóricos.
- Capa 2: cabeza de inducción sin entrenamiento: cuando reaparece el token `[A]`, copia la firma de `[B]` al buffer de memoria.
- Capas 1, 6, 12, 16: 11 neuronas SwiGLU editadas a mano como compuertas lógicas suaves (`PY_EXP_IDENT`, `PY_EXP_PAREN`, `PY_EXP_RETURN`, `CV_GREETING`, etc.); el resto es capacidad determinista en INT8.
- `lm_head` editado directamente para impulsar tokens de Python, análisis y conversación, e inhibir el espacio vacío.
- Cuantización: $s_r = \max_c\|W_{r,c}\|/127$, $W^{\text{int8}} = \text{round}(\text{clip}(W/s_r, -127, 127))$, reconstrucción $y = (x \cdot (W^{\text{int8}})^T) \odot s^T$.

**Conteo verificado:** 800,260,608 parámetros núcleos + 429,568 escalares de escala; 341 tensores; rango INT8 usado `[-127, 127]`; 0 tensores con `NaN`/`Inf`. Verificado con `safetensors.safe_open`.

---

## 4. Evidencia de resultados (etapas 0 a 5)

Extraído de `results/experiment_results.json` y `docs/EXPERIMENT_DOCUMENTATION.md`. Cuatro prompts de prueba:

1. `alpha = 42 ; beta = 99 ; alpha =` (copia en contexto / inducción)
2. `def ` (sintaxis Python)
3. `Analyze the time complexity of binary search:` (análisis)
4. `Hello! How can you help me?` (conversación mínima)

| Etapa | `alpha = ... alpha =` | `def ` | Analiza complejidad | `Hello! ...` | Entropía típica | Qué pasó |
|---|---|---|---|---|---|---|
| 0 (base aleatoria ortogonal) | `finisOrder346yeses Thousand` | `Validat eigenvectores...` | `newsRuntimeErrbisect_right...` | `Virtue Advi studio heatDeprecation` | ~9.87 nats | Ruido máximo, sin estructura |
| 1 (embeddings analíticos + `lm_head` semántico positivo) | ` = = = = =` | `     ` (espacios) | `:::::` | `?????` | ≈ 0 | **Fallo 4**: atractor de auto-repetición |
| 2 (cabeza de inducción activa) | ` 99 ; beta` ✓ (copia real) | `     ` (espacios) | `AnalyzeAnalyzeAnalyze...` | `! How can you help` | ≈ 0 | Inducción funciona; resto colapsa |
| 3 (calibración SwiGLU 0.055) | ` 99 ; beta` ✓ | `process` / `analyze` / `solve` / `compute` (top-5) | … | `! How can you help me?! How can you help` | bajo | Python sintáctico arranca; aparecen osciladores antipodales (`writtens ↔ =`, `discip ↔ :`, `Fait ↔ ?`) — **Fallo 6** |
| 4 (calibrada final) | inducción real con p=0.985 para ` 99` | identificadores válidos tras `def ` | … | … | H ∈ [1.5, 4.5] | Mejor etapa, pero con trastornos |
| 5 (etapa final del archivo `.safetensors`) | ` 99 ; beta = 99 ; be` | `          ` (espacios, colapso) | `the time complexity of binary search: the` | `! How can you help me?! How can` | 0.02–0.69 nats | **Fallo 7**: inducción domina; eco y repetición |

**Observaciones clave de la etapa 5 (el `.safetensors` que se distribuiría):**

- Inducción: `alpha = 42 ; beta = 99 ; alpha =` → ` 99 ; beta = 99 ; be`. Copia parcialmente, luego se repite/degrada (H = 0.076).
- Python: `def ` → solo espacios (H = 0.057). El cabeza de inducción secuestra la generación: repite espacios del prompt.
- Análisis: `Analyze the time complexity of binary search:` → repite el prompt `the time complexity of binary search: the` (H = 0.69). No razona, eco.
- Conversación: `Hello! How can you help me?` → `! How can you help me?! How can` (H = 0.11). Bucle de repetición del signo de exclamación y la frase de ayuda.
- `for i in range(10):\n    ` → ` i in range(10):\n     i in range(10` — repite el prefijo del bucle (H = 0.02).
- Datos: `Compare variance and outliers in numerical data analysis:` → ` and outliers in numerical data analysis: and outliers` — eco puro (H ≈ 0).

---

## 5. Los 7 fallos documentados en el experimento (y su estado final)

1. **Merges BPE del tokenizador** — corregido (`roundtrip_ok=True`).
2. **Cuello de CPU/RAM generando pesos en float64** — corregido (síntesis directa en INT8, ~6–10 s para el modelo completo).
3. **Desalineación de 3,072 bytes en el streaming `.safetensors`** — corregido (orden de tensores sincronizado, detección de `NaN/Inf`, contador = 0).
4. **Etapa 1: atractor de auto-repetición** (`= = = = =`, `:::::`, `?????`) — mitigado en etapas posteriores.
5. **Etapa 3: explosión de escala en RMSNorm y secuestro por inducción** — mitigado con ganancia SwiGLU 0.055 y escala `sg = 0.35`.
6. **Etapa 4: osciladores antipodales de ciclo 2** (`writtens ↔ =`, `discip ↔ :`, `Fait ↔ ?`) por la penalización `-E E^T` — corregido en etapa 5 (`lm_head[:, 0:128] = 0`, sin ruido gaussiano).
7. **Etapa 5: dominio del atractor de inducción** — **no corregido**: sin inhibición léxica ni estadística entrenada, la cabeza de inducción domina y genera eco/repetición.

---

## 6. Veredicto: ¿sirvió el experimento?

| Criterio objetivo del usuario | Resultado |
|---|---|
| Modelo de ~800M parámetros | ✅ Sí: 800,260,608 parámetros verificados |
| Formato `.safetensors` | ✅ Sí: válido, 341 tensores, sin `NaN`/`Inf` |
| Bueno en Python | ❌ No: genera espacios, identificadores sueltos, no código ejecutable |
| Hable mínimamente | ⚠️ Parcial: saluda, pero cae en bucles (`! How can you help?! How can`) |
| Entienda / haga análisis | ❌ No: eco del prompt, sin razonamiento |
| Creado sin entrenamiento, solo editando pesos | ✅ Sí: 0 pasos de entrenamiento, sintaxis y matemáticas escritas a mano |
| Documentado en md (matemáticas, fallos, pasos) | ✅ Sí: `docs/EXPERIMENT_DOCUMENTATION.md` + este análisis |

**Conclusión:** El experimento **sirvió como demostración de principio** ("una IA puede editar directamente los pesos de un modelo de 800M para implementar circuitos algorítmicos reales: copia en contexto, inducción, enrutamiento sintáctico") y como estudio de ingeniería de pesos (cuantización INT8, `mmap`, tokenizador sin corpus). **No sirvió como modelo usable:** no es bueno en Python, no entiende análisis, y no mantiene una conversación sin colapsar en repeticiones. Eso era predecible por la documentación misma: sin un proceso iterativo de optimización sobre datos, no se pueden comprimir las estadísticas del lenguaje en 1,536 ejes lineales mediante edición manual.

Para que "sirva" como modelo, haría falta: (1) entrenamiento real con gradientes sobre código Python y texto de análisis, o (2) al menos ajuste fino de `lm_head` y capas superiores con retropropagación ligera, lo cual rompe la premisa de "cero entrenamiento".

---

## 7. Cómo reproducir / usar este repositorio

```bash
# 1. Obtener los pesos (excluidos de Git por tamaño)
# Google Drive, carpeta "AI experiment AI", file ID:
#   1Lb_97PHGKeJ1HB6k9Umu0Kpl953fNTAT
# Colocar el archivo en model/model.safetensors

# 2. Ejecutar inferencia
pip install numpy safetensors tokenizers
python3 scripts/inference_engine.py model

# 3. Regenerar tokenizador y pesos (escriben en /home/user/ai_experiment por defecto)
python3 scripts/build_tokenizer.py
python3 scripts/synthesize_weights.py   # etapas 0-4
python3 scripts/run_stage5.py           # etapa 5 + experiment_results.json
```

Nota: los scripts de síntesis importan entre sí (`run_stage5.py` importa `synthesize_weights` e `inference_engine`); ejecutarlos desde el directorio `scripts/` o ajustar `PYTHONPATH`. `OUT_DIR` está hardcodeado a `/home/user/ai_experiment` en los scripts originales; cámbialo si usas otra ruta.
