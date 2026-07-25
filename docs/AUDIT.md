# Auditoría del repositorio — Credit Risk Engine

**Fecha:** 2026-07-24
**Commit auditado:** `2395a37` (Fix error AWS ECS service name), rama `main`
**Alcance:** código fuente, notebooks, tests, Docker, CI/CD, despliegue AWS Fargate, artefactos de modelo

Todos los hallazgos están verificados ejecutando el código, no solo leyéndolo. Las mediciones
concretas se incluyen junto a cada punto.

---

## Índice

- [🔴 Crítico](#-crítico) — 6 hallazgos: rompen producción o invalidan métricas
- [🟡 Recomendable](#-recomendable) — 14 hallazgos: corrección, deuda técnica, seguridad
- [🟢 Opcional](#-opcional) — 9 hallazgos: consistencia y limpieza
- [⏳ Pendiente importante: alinear los notebooks](#-pendiente-importante-alinear-los-notebooks-con-el-pipeline-limpio)
- [Orden de trabajo propuesto](#orden-de-trabajo-propuesto)
- [Anexo A: SMOTE vs pesos de clase](#anexo-a-smote-vs-pesos-de-clase)
- [Anexo B: comandos de verificación](#anexo-b-comandos-de-verificación)

---

## 🔴 Crítico

### C1. `loan_int_rate` está en escala equivocada en la API

**Dónde:** `src/api/schemas.py:70`
**Estado:** corregido (2026-07-25) — schema en escala porcentual, suelo en 1.0 para rechazar la escala fraccionaria; tests de regresion en `tests/test_schemas.py`

El schema impone `ge=0, le=1`, pero el modelo se entrenó con el tipo de interés en **porcentaje**:

```
loan_int_rate (datos de entrenamiento):
  min 5.42 | media 11.03 | std 3.07 | max 23.22
```

Consecuencia: un tipo real del 8% enviado como `0.08` se escala a **−3.57σ**, muy fuera del rango
visto en entrenamiento. Y enviarlo como `8.0` devuelve HTTP 422. En la práctica, **la API no admite
ningún tipo de interés válido**.

Medición end-to-end con el modelo y preprocessor reales:

| input | valor escalado | P(default) |
|-------|----------------|------------|
| 0.08  | −3.568         | 0.0014     |
| 8.0   | −0.989         | 0.0004     |
| 16.0  | 1.617          | 0.0008     |
| 23.0  | 3.897          | 0.0639     |

**Agravante:** `tests/test_schemas.py:125` afirma explícitamente que `loan_int_rate = 1.5` debe
rechazarse, de modo que el bug está fijado por un test. Corregirlo exige tocar también el test y los
fixtures de `tests/conftest.py` (`0.08` aparece en 5 sitios).

---

### C2. `models/` está en `.gitignore` → la imagen desplegada no lleva modelo

**Dónde:** `.gitignore` (sección "Modelos entrenados"), `.github/workflows/cd.yml:37`
**Estado:** corregido (2026-07-25) — artefactos versionados via excepcion en `.gitignore` (504 KB); paso de verificacion del CD reescrito para arrancar el contenedor y llamar a `/health` y `/predict`

`.dockerignore` **no** excluye `models/`, así que el build local funciona porque los `.joblib` están
en el disco del desarrollador. Pero GitHub Actions parte de un checkout limpio: no hay artefactos →
`get_predictor()` lanza excepción → todo `POST /predict` devuelve HTTP 500.

El paso de verificación del pipeline no lo detecta:

```yaml
- name: Verify Image Functionality
  run: docker run --rm credit-risk-engine:latest python -c "import src; print('Import successful')"
```

`import src` pasa igualmente con los artefactos ausentes.

**Opciones de solución** (decisión pendiente):
1. Versionar los artefactos con Git LFS (~860 KB en total, perfectamente asumible).
2. Descargarlos de S3 / MLflow Model Registry en el build.
3. Publicarlos como release assets de GitHub y hacer `curl` en el Dockerfile.

Sea cual sea la elegida, el paso de verificación debe pasar a arrancar el contenedor y llamar a
`/health` y `/predict` de verdad.

---

### C3. El arranque se bloquea ~247 s intentando conectar a MLflow

**Dónde:** `src/predictor.py:51-53`
**Estado:** corregido (2026-07-25) — MLflow es opt-in via `MLFLOW_TRACKING_URI`; suite de tests de 251.68s a 4.30s

```python
mlflow.set_tracking_uri("http://localhost:5000")   # URI hardcodeada
self.model = mlflow.pyfunc.load_model("models:/CreditScorer/Staging")
```

**Medición aislada:**

```
FAILED after 246.7s: MlflowException
  Retrying (total=1, connect=1) ... [Errno 111] Connection refused
  Retrying (total=0, connect=0) ... [Errno 111] Connection refused
```

Coincide con el perfilado de la suite de tests: el test más lento tarda **247.76 s** y es
simplemente el primero que construye el predictor; todo lo demás está mockeado y tarda 0.01 s.

```
247.76s call     tests/test_api.py::TestHealthCheck::test_health_check
  0.03s setup    tests/test_api.py::TestHealthCheck::test_health_check
  0.02s call     tests/test_api.py::TestPredictEndpoint::test_predict_valid_application
...
73 passed in 251.68s
```

**Detalle importante:** el error es `Connection refused`, es decir, rechazo TCP **instantáneo**. Los
~4 minutos son enteramente *retry backoff* de `urllib3` dentro del cliente MLflow, no un timeout de
red. En Fargate, donde tampoco hay nada escuchando en `localhost:5000`, el comportamiento será
idéntico. **No se arregla bajando un timeout**: hay que desactivar el intento o hacerlo configurable
por entorno (`MLFLOW_TRACKING_URI` ausente ⇒ ir directo a joblib).

**Impacto en producción:** el contenedor tarda ~4 min en arrancar (el `lifespan` llama a
`get_predictor()`), mientras el `HEALTHCHECK` del Dockerfile (`start-period=10s`, `interval=30s`,
`retries=3`) lo marca *unhealthy* a los ~100 s. Posible bucle de reinicio en ECS.

---

### C4. `/health` devuelve HTTP 200 aunque el modelo no cargue

**Dónde:** `src/api/main.py:99-114`
**Estado:** corregido (2026-07-25) — `/health` devuelve 503 sin modelo; test de regresion en `tests/test_api.py`

El endpoint devuelve `status: "unhealthy"` y `model_loaded: false` en el body, pero el código HTTP
sigue siendo 200. Un health check de ALB o de ECS da por sano un servicio que no puede predecir.

Combinado con C2, esto hace que un despliegue completamente roto aparezca en verde. Debe devolver
**503** cuando el modelo no está cargado.

---

### C5. Data leakage en el pipeline de entrenamiento

**Dónde:** `notebooks/feature_engineering.ipynb` (celdas 23, 30, 36, 40)
**Estado:** corregido (2026-07-25) — pipeline sin leakage en `scripts/train.py`; promovido el brazo sin pesos de clase. Ver [Resolución](#resolución) abajo

El notebook ejecuta sobre las **31.679 filas completas**:

1. `preprocessor.fit_transform(X)` → el `StandardScaler` calcula media y desviación con el test set.
2. Eliminación por correlación >0.85 (41 → 35 features).
3. Eliminación por importancia de un `RandomForestClassifier` entrenado sobre todo (35 → 17).
4. Eliminación por varianza (17 → 17).

El train/test split ocurre **después**, en `model_selection.ipynb` celda 14, sobre el CSV ya escalado
y ya reducido a 17 features. Es decir: el escalador y, sobre todo, el selector de features vieron el
conjunto de test. **Las métricas publicadas están sesgadas al alza.**

**Detalle relevante:** `feature_engineering_sagemaker.py:134` **sí lo hace correctamente**:

```python
# CRITICAL: Split data BEFORE feature selection to prevent data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(...)
```

Existe la versión corregida del pipeline, pero los artefactos que sirve la API vienen del camino con
leakage.

### Resolución

Sustituido por `scripts/train.py`, un pipeline donde todo `fit` ocurre solo sobre train, el umbral se
elige en un split de **validación** (elegirlo en test era el mismo error de fondo que C5) y el test se
toca una única vez, al final. Split 64/16/20 estratificado.

Se ejecutaron tres brazos con hiperparámetros idénticos para aislar cada variable:

| brazo | ROC-AUC | PR-AUC | Brier ↓ | media pred. | Precision | Recall | F1 | umbral | features |
|---|---|---|---|---|---|---|---|---|---|
| `leaky-weighted` (pipeline viejo) | 0.9447 | 0.9001 | 0.0696 | 0.3063 | 0.9503 | 0.7421 | 0.8334 | 0.71 | 17 |
| `clean-weighted` | 0.9455 | 0.9013 | 0.0683 | 0.3046 | 0.9427 | 0.7473 | 0.8337 | 0.69 | 18 |
| **`clean-unweighted`** (promovido) | **0.9460** | **0.9019** | **0.0520** | **0.2159** | 0.9313 | **0.7553** | **0.8341** | 0.39 | 18 |

*(tasa real de positivos en test: 0.2154)*

**Hallazgo 1 — el leakage apenas sesgaba las métricas.** El brazo con leakage puntúa incluso algo
*peor* (0.9447 vs 0.9455 de ROC-AUC). Con 31.679 filas y un filtro de selección poco agresivo, el
escalador y el selector no extraían ventaja apreciable del test set. El fallo metodológico era real y
había que corregirlo, pero **no inflaba los números publicados**. Conviene decirlo así en el README en
lugar de dramatizar la corrección.

Nota: la selección limpia se queda con **18** features en lugar de 17 (mismo recorte por correlación,
41→35, y luego umbral de importancia sobre train), así que las dos listas no son idénticas.

**Hallazgo 2 — `scale_pos_weight` no aporta y descalibra.** Confirma sobre el pipeline correcto lo
que ya apuntaba el Anexo A:

- ROC-AUC y PR-AUC: sin pesos gana por márgenes mínimos (0.9460 vs 0.9455; 0.9019 vs 0.9013).
- F1 prácticamente idéntico (0.8341 vs 0.8337), pero con **mejor recall** (0.7553 vs 0.7473), que es
  la métrica que importa en riesgo de crédito.
- **Brier 0.0520 vs 0.0683**, un 24% mejor sin pesos.
- Media de P(default) predicha: **0.2159** frente a una tasa real de **0.2154**. Con pesos: 0.3046,
  inflada un 41%.

Calibración por deciles del modelo promovido sobre el test set:

```
 decil      n   P media predicha   tasa real observada    error
     1    634             0.0021                0.0032  -0.0011
     5    633             0.0490                0.0600  -0.0111
     8    634             0.1912                0.1814  +0.0098
     9    633             0.6777                0.6935  -0.0158
    10    634             0.9921                1.0000  -0.0079

Error de calibracion medio (ECE por deciles): 0.0076
```

Ningún decil se desvía más de 1,6 puntos porcentuales. Las probabilidades ya son interpretables como
*probability of default* real, lo que habilita usarlas para pricing o expected loss, y hace que elegir
el umbral sea una decisión de negocio con significado en lugar de un ajuste ciego.

**Decisión:** promovido `clean-unweighted`. El umbral óptimo baja de 0.55 a **0.39** — consecuencia
directa de la recalibración, no un cambio de criterio: al no estar infladas las probabilidades, el
corte que maximiza F1 cae más abajo.

**Consecuencia:** `results/model_comparison.csv`, `results/cross_validation_results.csv` y
`results/tuned_models_comparison.csv` provienen del pipeline con leakage y quedan obsoletos. El README
debe citar `results/leakage_and_weighting_comparison.csv`.

**Reproducir:**

```bash
uv run python scripts/train.py                          # comparar los tres brazos
uv run python scripts/train.py --save clean-unweighted  # promover a models/
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # experimento "credit-risk-leakage-and-weighting"
```

---

### C6. `.dockerignore` incompleto → 1,1 GB de ficheros locales en la imagen de producción

**Dónde:** `.dockerignore`, `Dockerfile:34` (`COPY . .`)
**Estado:** corregido (2026-07-25) — `.dockerignore` ampliado; imagen de 4,08 GB a 2,94 GB
**Nota:** hallazgo descubierto al verificar C2, no estaba en la auditoría inicial.

`COPY . .` copia todo lo que `.dockerignore` no excluya. El fichero no cubría varios directorios
locales pesados, así que acababan dentro de la imagen desplegada en Fargate:

```
1.4G  /app/.venv        <- dependencias reales (ver R9)
916M  /app/.conda       <- instalación local de conda del desarrollador
233M  /app/mlartifacts  <- artefactos locales de MLflow
940K  /app/mlflow.db    <- base de datos local de tracking
472K  /app/tests
 52K  /app/.coverage
```

Además de peso, es un problema de higiene: `mlflow.db` y `mlartifacts/` son estado local del
desarrollador que no pinta nada en un contenedor de producción.

Tras ampliar `.dockerignore` (`.conda/`, `mlartifacts/`, `mlruns/`, `mlflow.db*`, `tests/`, `docs/`,
`.github/`, `.coverage`, `coverage.xml`, `.ruff_cache`, `.claude`, `CLAUDE.md`,
`requirements.txt.bak`): **4,08 GB → 2,94 GB**, verificado que la imagen sigue sirviendo
predicciones correctamente.

**Pendiente:** los 2,94 GB restantes están dominados por el `.venv` de 1,4 GB, que es R9
(`mlflow` + `evidently` + `seaborn` + `nvidia-nccl-cu12` como dependencias runtime). Sacarlos a
grupos opcionales debería dejar la imagen en pocos cientos de MB.

---

## 🟡 Recomendable

### R1. Fallback de nombres de features: código muerto **y** incorrecto

**Dónde:** `src/preprocessing.py:139-206`

El preprocessor real sí expone `get_feature_names_out()` (sklearn 1.7.2), así que las ~65 líneas del
fallback nunca se ejecutan. Y si se ejecutaran, mapearían mal las columnas **en silencio**:

| hardcodeado en el fallback | orden real del preprocessor |
|----------------------------|------------------------------|
| `RENT, OWN, MORTGAGE, OTHER` | `MORTGAGE, OTHER, OWN, RENT` (alfabético) |
| `cb_person_default_on_file_0` / `_1` | `cb_person_default_on_file_N` / `_Y` |

Además el docstring (líneas 120-122) dice "11 numeric / 29 categorical" cuando en realidad son
**10 / 31** (el total, 41, sí cuadra).

**Acción:** eliminar el fallback y fallar de forma explícita si `get_feature_names_out()` no existe.
Un mapeo silenciosamente incorrecto es peor que un error.

---

### R2. `cb_person_default_on_file`: `int` en la API vs `str` en el preprocessor

El preprocessor se ajustó con valores `'Y'`/`'N'` (ver `get_feature_names_out()`), pero la API acepta
`int` 0/1 (`src/api/schemas.py:74`). Con `handle_unknown='ignore'`, ambas dummies quedan a 0 sin
avisar.

Hoy **no afecta al resultado** porque `cb_person_default_on_file_N/_Y` no están entre las 17 features
seleccionadas (se usa `default_flag`, que es el cast numérico). Pero es una inconsistencia latente
que explotará si cambia la selección de features.

---

### R3. `age_bucket` excluye la edad 18

**Dónde:** `src/preprocessing.py:231`
**Estado:** corregido (2026-07-25) — `AGE_BINS` empieza en 17, la edad 18 cae en su bucket
**Estado:** corregido (2026-07-25) — `create_derived_features` mapea `Y`/`N` y `0`/`1` al mismo `default_flag`

```python
pd.cut(df_fe["person_age"], bins=[18, 25, 35, 45, 55, 65, 120], ...)
```

`pd.cut` es abierto por la izquierda por defecto, así que `person_age == 18` → `NaN` → imputado a la
moda. Y `MIN_AGE = 18` en `src/config.py:23` lo hace perfectamente alcanzable.

**Fix:** `bins=[17, 25, ...]` o `include_lowest=True`.

---

### R4. Lógica de features derivadas duplicada y ya divergente

| | `src/preprocessing.py:208` | `feature_engineering_sagemaker.py:20` |
|---|---|---|
| división por cero | `.replace(0, 0.01)` | `.replace(0, np.nan)` |
| buckets | `Categorical` | `.astype(str)` |

Mismo concepto, dos comportamientos. Debe existir una única implementación compartida.

---

### R5. Los tests no prueban prácticamente nada real

**Dónde:** `tests/conftest.py:10-135`
**Estado:** corregido (2026-07-25) — implementacion unica en `src/preprocessing.create_derived_features`, usada por entrenamiento y serving

Un fixture con `autouse=True` mockea `joblib.load` de forma global, y `mock_transform` devuelve
`np.random.randn(n, 41)`. Consecuencias:

- Ningún test toca el preprocessor real ni el modelo real.
- Nada verifica que las 17 features salgan en el orden correcto.
- Las aserciones sobre probabilidades pasan con datos aleatorios.

Cobertura alta, señal nula. Es exactamente la razón por la que C1 y C2 pasan CI sin ruido.

**Falta el test que importa:** un caso conocido → probabilidad esperada, con los artefactos de verdad
(test de regresión del modelo).

---

### R6. `feature_engineering_sagemaker.py` sin commitear en la raíz

Aparece como untracked en `git status`. Decidir: mover a `scripts/` o `sagemaker/` y commitear, o
eliminar. Contiene la versión correcta del pipeline (ver C5), así que probablemente conviene
conservarlo.

---

### R7. El pipeline de CD se dispara dos veces por push

**Dónde:** `.github/workflows/cd.yml:4-9`

```yaml
on:
  push:
    branches: [ main ]
  workflow_run:
    workflows: [ "CI - Lint and Test" ]
    types: [ completed ]
    branches: [ main ]
```

Ambos disparadores se activan con un push a `main` → dos despliegues concurrentes al mismo servicio
ECS. Falta un bloque `concurrency:` y, muy probablemente, eliminar uno de los dos triggers.

---

### R8. La imagen se construye dos veces

El job `build-and-test` la construye con buildx y caché `gha`; después `deploy-to-aws` hace un
`docker build` a pelo (`cd.yml:85`). Se paga el build dos veces **y se despliega una imagen que no es
la que se verificó**.

---

### R9. Dependencias de producción infladas

**Dónde:** `pyproject.toml:28-64`

Son dependencias runtime: `mlflow`, `evidently`, `seaborn` y `nvidia-nccl-cu12==2.29.7`. El último
son ~200 MB de CUDA completamente inútiles en Fargate CPU. `seaborn` solo se usa en notebooks.

**Acción:** mover a grupos opcionales (`[dependency-groups]`) y dejar en runtime únicamente lo que
necesita la API.

---

### R10. `ModelInfo` descarta `model_source` silenciosamente

`get_model_info()` devuelve `model_source` (`src/predictor.py:190`), pero el schema no lo declara
(`src/api/schemas.py:133`), así que FastAPI lo filtra. Es justo el campo que dice si se está sirviendo
desde MLflow o desde joblib — información operativa valiosa.

---

### R11. CORS abierto con TODO pendiente

**Dónde:** `src/api/main.py:64-71`

```python
# CORS - TODO Configure according to security requirements
allow_origins=["*"], allow_credentials=True, ...
```

`allow_origins=["*"]` junto a `allow_credentials=True` es además una combinación **inválida** según la
especificación CORS (los navegadores la rechazan).

---

### R12. El informe de monitoring no mide nada

**Dónde:** `src/model_monitoring.py:217-218`

Compara `data/credit_risk_fe.csv` (referencia) contra `data/test_samples.csv`, que son **3 filas
inventadas a mano** con valores redondos:

```csv
loan_percent_income,person_income,loan_int_rate,...
-0.85,1.2,-0.7,-0.5,0,1,1.1,1.5,...
```

Peor aún: `_resolve_target_column` acaba eligiendo `default_flag`, que en el CSV de referencia es una
*feature escalada* (`-0.4647...`), no la etiqueta. El `.astype(int)` la trunca a 0. `loan_status`, la
etiqueta real, ni siquiera existe en `test_samples.csv`.

El "target drift" del HTML generado es ruido sin significado.

---

### R13. `_load_artifacts()` nunca se llama

**Dónde:** `src/model_monitoring.py:33`. Código muerto.

---

### R14. `requirements.txt.bak` commiteado y desactualizado

Contiene `numpy==2.2.6` frente a `numpy>=1.26,<2.1` en `pyproject.toml`. El README actual lo ofrece
como vía de instalación alternativa (`pip install -r requirements.txt.bak`), lo que instalaría una
versión incompatible.

---

## 🟢 Opcional

### O1. Cinco versiones de Python distintas conviviendo

| Sitio | Versión |
|---|---|
| `pyproject.toml` `requires-python` | `>=3.10` |
| `[tool.ruff] target-version` | `py311` |
| `[tool.mypy] python_version` | `3.10` |
| `Dockerfile` / CI | `3.11` |
| venv local | `3.12.3` |

### O2. Configuración muerta y contradictoria

`[tool.black]` e `[tool.isort]` con `line-length = 100`, cuando el proyecto usa ruff con
`line-length = 88`. Ni black ni isort están en las dependencias. `[tool.uv.sources]` está vacío, solo
con comentarios de ejemplo.

### O3. Detalles del Dockerfile

- `RUN chown -R appuser:appgroup /app` (línea 32) va **antes** del `COPY . .` (línea 34), así que lo
  copiado queda propiedad de root.
- `FROM python:3.11.10-slim as builder` — `as` en minúsculas genera warning de BuildKit; debe ser `AS`.
- `ENV PORT=8000` declarado pero el `CMD` hardcodea `--port 8000`.

### O4. `.env.example` desalineado con el CD real

Dice `ECS_SERVICE=credit-risk-service`; `cd.yml` usa `credit-risk-task-service-d1oa9itj`. Además esas
variables no las lee nadie: el workflow las tiene hardcodeadas en su bloque `env:`.

### O5. `daemon.json` en la raíz del repo

Es configuración de `dockerd`, no del proyecto. No pinta nada en el repositorio (o necesita
documentarse por qué está).

### O6. `optimal_idx = 9` hardcodeado

**Dónde:** `notebooks/model_selection.ipynb` celda 53. El umbral "data-driven" es en realidad un
índice escrito a mano. Funciona y el razonamiento de la celda 51 es correcto, pero no se re-deriva si
cambian los datos.

### O7. `models/best_model_xgboost.joblib` sin usar

El modelo sin tunear sigue en el repo y nadie lo carga.

### O8. `src/config.py` sin overrides por entorno

Todas las constantes son literales. Nada configurable en despliegue sin rebuild de la imagen.

### O9. README actual desactualizado

- "Python 3.8+" (el proyecto pide 3.10+).
- `src/ (To be populated with API and deployment code)` y `tests/ (To be populated with test suite)`
  — ambos existen desde hace muchos commits.
- "50+ engineered features" — son 41 tras encoding, 17 seleccionadas.
- Sección "📚 Detailed Implementation" completamente vacía.
- Emoji roto en la línea 189 (`## � Model Insights`).
- Cero mención de Docker, AWS Fargate, MLflow, CI/CD o monitoring.
- Afirma *"Well-calibrated probability estimates"*, lo cual es **falso**: `scale_pos_weight` distorsiona
  la calibración por construcción.
- Métricas repetidas tres veces con cifras distintas entre secciones.

---

## ⏳ Pendiente importante: alinear los notebooks con el pipeline limpio

**Estado:** pendiente
**Depende de:** C5 (resuelto)

`scripts/train.py` ya entrena sin leakage y sin pesos de clase, y es la fuente de los artefactos que
sirve la API. Pero los notebooks siguen documentando el método antiguo, así que hoy el repo cuenta dos
historias distintas. Cualquiera que lea `notebooks/` para entender el modelo aprende el procedimiento
equivocado.

**Aclaración de alcance:** *no hay SMOTE que quitar*. Se revisaron los cuatro notebooks y ninguno usa
SMOTE, `imblearn` ni ningún sobremuestreo. La única coincidencia es "unskewed" en
`exploratory_data_analysis.ipynb`, que se refiere a la **asimetría de distribuciones numéricas**
(regla IQR, `series.skew()`) — un tema distinto del desbalanceo de clases. El desbalanceo se trataba
con `scale_pos_weight` y `class_weight="balanced"`, que es lo que hay que retirar.

### Qué hay que cambiar

**`notebooks/feature_engineering.ipynb`** — es el origen de C5:
- Mover el `train_test_split` **antes** de `preprocessor.fit_transform` y de la selección de features
  (celdas 23, 30, 36, 40).
- Importar `create_derived_features` desde `src.preprocessing` en lugar de reimplementarla, para no
  reabrir la divergencia que documentaba R4.

**`notebooks/model_selection.ipynb`**:
- Quitar `class_weight="balanced"` de Logistic Regression, Random Forest y SVM, y `scale_pos_weight`
  de XGBoost (celda 17). Justificación medida en el Anexo A y en la Resolución de C5: no mejora el
  ranking y descalibra las probabilidades.
- Elegir el umbral sobre un split de **validación**, no sobre test (celdas 48-53).
- Sustituir el `optimal_idx = 9` hardcodeado (O6) por la selección derivada de los datos.

### Entregable que falta

La tabla comparativa de los cuatro modelos base (Logistic Regression / Random Forest / XGBoost / SVM)
**reentrenada con el pipeline limpio y sin pesos**. Hoy `results/model_comparison.csv` y
`results/cross_validation_results.csv` vienen del camino con leakage y con pesos, así que quedan
obsoletos y el README no debe citarlos.

Dos caminos posibles, a decidir:
1. Extender `scripts/train.py` con los cuatro modelos y regenerar los CSV desde ahí (reproducible en
   CI, es la opción que recomiendo).
2. Corregir los notebooks y regenerar los CSV desde ellos (mantiene la narrativa exploratoria, pero
   no es reproducible automáticamente).

No son excluyentes: lo natural es (1) para los números publicados y (2) para que el notebook cuente la
historia correcta.

---

## Orden de trabajo propuesto

### Bloque 1 — Rompe producción
`C2` → `C4` → `C3` → `C1`

En ese orden porque C2 y C4 son los que hacen que un despliegue roto parezca sano; sin arreglarlos no
se puede verificar el resto.

### Bloque 2 — Integridad del ML
`R5` (un test real end-to-end primero, que es lo que da red de seguridad) → `R1` → `R12` → `C5`

### Bloque 3 — Limpieza
`R7`-`R9`, `R13`-`R14`, y todo el bloque opcional en un barrido único.

---

## Anexo A: SMOTE vs pesos de clase

**Pregunta:** con el desbalanceo presente en este dataset (21,5% de positivos, ratio 3,64:1),
¿aporta algo SMOTE frente a `scale_pos_weight`?

**Respuesta corta:** no. SMOTE empeora ligeramente el ranking y no aporta nada que el umbral no dé
ya. Pero el experimento revela algo más útil: **`scale_pos_weight` tampoco aporta**, y además
descalibra las probabilidades.

### Metodología

Pipeline **sin leakage** (split primero, `fit` del preprocessor solo sobre train), 80/20
estratificado, `random_state=42`, las 41 features tras encoding, e hiperparámetros idénticos en todos
los brazos (los del modelo tuneado guardado: `n_estimators=300, max_depth=4, learning_rate=0.1,
subsample=0.8`). Test set intacto de 6.336 filas.

### Resultados

| brazo | n_train | ROC-AUC | PR-AUC | Brier ↓ | media pred. | F1@0.50 | umbral óptimo | F1@óptimo |
|---|---|---|---|---|---|---|---|---|
| A. sin tratamiento | 25.343 | 0.9510 | **0.9084** | **0.0505** | **0.2144** | 0.8395 | 0.53 | 0.8409 |
| B. `scale_pos_weight` (actual) | 25.343 | **0.9515** | **0.9086** | 0.0672 | 0.3063 | 0.8142 | 0.73 | 0.8370 |
| C. SMOTE naive | 39.766 | 0.9451 | 0.9006 | 0.0525 | 0.2353 | 0.8374 | 0.47 | 0.8402 |
| D. SMOTENC | 39.766 | 0.9417 | 0.8967 | 0.0539 | 0.2474 | 0.8433 | 0.54 | **0.8447** |
| E. SMOTE + weight | 39.766 | 0.9457 | 0.9005 | 0.0823 | 0.3455 | 0.7763 | 0.74 | 0.8401 |

*(tasa real de positivos en test: 0.2154)*

### Conclusiones

**1. SMOTE no mejora el ranking; lo empeora.** Los tres brazos con SMOTE tienen ROC-AUC y PR-AUC
**por debajo** tanto del baseline como de `scale_pos_weight`. Diferencia pequeña (−0.005 a −0.010 de
ROC-AUC) pero consistente en las dos métricas y en las tres variantes. Duplicar el tamaño del train
con datos sintéticos no añade información: la añade *interpolada de la que ya había*.

**2. Una vez optimizas el umbral, da todo igual.** F1 en el umbral óptimo: 0.8370–0.8447 entre los
cinco brazos, un rango de 0.008. Es decir: **el beneficio que SMOTE promete es exactamente el
problema que este proyecto ya resuelve con la optimización de umbral.** (Es la reproducción sobre
este dataset del resultado de Elor & Averbuch-Elor, *"To SMOTE or not to SMOTE"*, 2022: para
learners fuertes, el balanceo solo ayuda si te quedas anclado al umbral 0.5.)

**3. El hallazgo colateral: `scale_pos_weight` descalibra las probabilidades.**

```
tasa real de default en test  : 0.2154
media de P(default) predicha  : 0.2144  (sin tratamiento)  ← casi perfecta
                                0.3063  (scale_pos_weight) ← inflada un 42%
                                0.3455  (SMOTE + weight)   ← inflada un 60%
```

El Brier score lo confirma: 0.0505 sin tratamiento vs 0.0672 con `scale_pos_weight` (peor). Y se ve
en el desplazamiento del umbral óptimo: 0.53 sin tratamiento, **0.73** con `scale_pos_weight` — el
modelo necesita un corte mucho más alto justo porque sus probabilidades están infladas.

Esto tiene dos implicaciones directas:

- Invalida la afirmación *"Well-calibrated probability estimates"* del README actual (ver O9). Es
  falsa, y es falsa **por culpa de** `scale_pos_weight`.
- Si en algún momento se quiere usar la probabilidad como *probability of default* real (pricing,
  expected loss, capital regulatorio), el modelo actual no sirve sin recalibrar.

**4. SMOTE además rompe el one-hot encoding.** Al interpolar sobre la matriz ya codificada:

```
celdas one-hot con valor fraccionario (filas sintéticas)      : 4.87%
filas sintéticas con bloque loan_grade que NO es one-hot válido: 15.00%
```

Un 15% de las filas sintéticas tienen un "loan_grade" que no existe: repartido entre varias
categorías en lugar de una sola. `SMOTENC` está diseñado para evitarlo, pero en este pipeline el
one-hot se aplica *antes*, así que habría que reordenar el pipeline entero para usarlo bien — y aun
haciéndolo (brazo D) el ROC-AUC es el peor de los cinco.

### Recomendación

**No introducir SMOTE.** Razones, por orden de peso:

1. 21,5% de positivos **no es desbalanceo severo**. SMOTE está pensado para <5%, típicamente <1%.
   Aquí hay 6.825 ejemplos positivos: señal de sobra.
2. Empeora el ranking (ROC-AUC y PR-AUC) en las tres variantes probadas.
3. Añade una dependencia (`imbalanced-learn`), complejidad en el pipeline y un punto más donde meter
   leakage (SMOTE debe aplicarse **solo** sobre train, dentro de cada fold de CV; es un error
   clásico aplicarlo antes de la validación cruzada).
4. Lo que promete ya lo da la optimización de umbral, que además es interpretable y ajustable a
   negocio sin reentrenar.

**Consideración adicional sobre `scale_pos_weight`:** los datos sugieren que también se podría
quitar. No mejora el ranking (+0.0005 de ROC-AUC frente al baseline, dentro del ruido) y a cambio
descalibra. La combinación *sin tratamiento de desbalanceo + umbral optimizado* da el mejor Brier
(0.0505), probabilidades interpretables como PD real, y prácticamente el mismo F1 (0.8409 vs 0.8370).

Es una decisión de diseño, no un bug: si el umbral se va a optimizar de todas formas —y aquí se
optimiza— cargar la función de pérdida solo sirve para desplazar el umbral óptimo hacia arriba.
Merece la pena evaluarlo cuando se aborde C5.

---

## Anexo B: comandos de verificación

```bash
# C1 — escala de loan_int_rate
.venv/bin/python -c "
import pandas as pd
print(pd.read_csv('data/credit_risk_cleaned.csv').loan_int_rate.describe())"

# C2 — comprobar que models/ no está versionado
git ls-files models/          # no devuelve nada
git check-ignore -v models/best_tuned_model_xgboost.joblib

# C3 — medir el bloqueo de MLflow
.venv/bin/python -m pytest tests/ -q --durations=5

# C5 — orden de operaciones en el notebook
.venv/bin/python -c "
import json; nb=json.load(open('notebooks/feature_engineering.ipynb'))
print(''.join(nb['cells'][23]['source']))"

# Estado general
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
```
