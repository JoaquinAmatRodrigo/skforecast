# Handoff: revisión del módulo de deep learning (`ForecasterRnn`)

**Fecha:** 2026-10-14.
**Rama:** `review/deep-learning-module`, creada desde `origin/0.26.x` en `92b97f2`. Contiene solo este documento y los scripts de `dev/deep_learning_review/`; no hay cambios en `skforecast/`.
**Para:** quien implemente las tareas en otra sesión. Borrar este documento y la carpeta de scripts antes de fusionar cualquier PR que salga de esta rama.

> **Cómo usar este documento**
> 1. Pide a Claude, por ejemplo: «Lee `dev/handoff_deep_learning.md` y haz la tarea T1».
> 2. Las tareas son independientes salvo donde se indica. Cada una trae la reproducción, la causa, el arreglo propuesto, los tests y la nota de versión.
> 3. Antes de empezar, vuelve a ejecutar la reproducción sobre el `0.26.x` del momento: `python dev/deep_learning_review/repro_bugs.py`.
> 4. Las tareas marcadas **decisión** esperan a que el autor elija una opción; la recomendada va primero.

---

## 1. Objetivo y decisiones

**Objetivo.** El módulo de deep learning no es el foco de skforecast (lo son machine learning y foundation), pero debe servir al menos **para comparar**: que un usuario pueda preguntarse si una red entrenada con sus datos mejora a LightGBM o a un modelo foundation y obtener una respuesta fiable. Hoy no la obtiene: si la red sale peor, no sabe si es por el modelo o por el módulo.

**Decisión: arreglar `ForecasterRnn` y documentar un baseline neuronal; no crear un forecaster nuevo.**

| Opción | Estado |
|:--|:--|
| Nuevo forecaster en PyTorch nativo (modelo MLP tipo TiDE, API de `ForecasterRecursiveMultiSeries`) | **Descartada.** Crear un forecaster es de los cambios más grandes en skforecast y abre muchos frentes (API, persistencia, búsqueda, tests, docs, skills). |
| Rediseñar la arquitectura de `create_and_compile_model` | **Descartada por ahora.** El arreglo mínimo (posición del paso en el decodificador) no mejora en todos los casos (§3.2). |
| **Arreglar los bugs de `ForecasterRnn`, mejorar el entrenamiento por defecto y documentar el MLP de scikit-learn como baseline neuronal** | **Elegida.** Cambios acotados a `_forecaster_rnn.py`, `utils.py`, sus tests y la documentación; sin API nueva. |

El argumento clave es la medida del §3.3: un `MLPRegressor` de scikit-learn dentro de `ForecasterRecursiveMultiSeries` o `ForecasterRecursive` rinde como un MLP específico de series temporales (tipo TiDE) escrito en PyTorch, y ya tiene toda la API de skforecast. La red simple está cubierta sin código; `ForecasterRnn` solo tiene que cubrir los modelos secuenciales (LSTM/GRU) y hacerlo de forma correcta y estable.

**Hipótesis inicial no confirmada.** Se sospechaba que con muchos `levels` la red no aprendía. En `store_sales` (series homogéneas) la arquitectura actual **mejora** con más series. Lo que falla es el entrenamiento por defecto, que se detiene en una meseta (§3.1). Con series heterogéneas no se ha medido.

**Opcional, más adelante:** `architecture="mlp"` en `create_and_compile_model` para tener también un MLP multi-output dentro de `ForecasterRnn`. No es necesario para el objetivo.

---

## 2. Hecho en esta sesión

- Revisión de `skforecast/deep_learning/_forecaster_rnn.py`, `skforecast/deep_learning/utils.py`, el uso de `ForecasterRnn` en `model_selection` y `deepcopy_forecaster`, y las guías `forecasting-with-deep-learning-rnn-lstm.ipynb` y `comparing-forecasting-models.ipynb`.
- Experimentos y reproducciones en `dev/deep_learning_review/` (§6). No hay cambios en el paquete, ni tests nuevos, ni notas de versión.
- Origen de los bugs: T1 y T3 salieron de la revisión del código; T4 lo detectó el autor (la salida publicada de la guía dice `device: mps` aunque Keras entrenó en CPU) y se confirmó en el código de skforecast y de Keras; T2 apareció al preparar la reproducción de `lags_grid`.

**Estado de la rama:**
- Subida a `origin/review/deep-learning-module`, primer commit `a33c955`. **Sin PR abierto.**
- El autor del commit es la identidad que trae configurada el entorno de la nube (`Claude <noreply@anthropic.com>`); no se cambió porque las reglas del repo piden no sobrescribirla. Si se quiere a nombre del autor, hay que rehacer el commit en una rama nueva (el force push está bloqueado).
- Los scripts de `dev/deep_learning_review/` dan avisos de ruff que se dejaron a propósito: imports después de fijar `KERAS_BACKEND` (E402, necesario) y varias sentencias por línea. CI no pasa ruff sobre `dev/`, y los ficheros que ya había en `dev/` también los dan. Solo se aplicaron los arreglos automáticos seguros (`ruff check --fix`).

---

## 3. Resultados de los experimentos

Entorno: Python 3.12.3, torch 2.14.1 (build de PyPI, ejecutado en CPU), keras 3.15.1 (backend torch), numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.7.0; CPU de 4 núcleos, sin GPU. Los tiempos se midieron con varios procesos en paralelo: sirven para comparar modelos, no como valores absolutos.

"Entrenamiento por defecto" es `EarlyStopping(patience=5, restore_best_weights=True)` con 40 épocas máximo (lo habitual en las guías). "Robusto" es `EarlyStopping(patience=15, restore_best_weights=True)` + `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)` con 100 épocas máximo. Siempre Adam con lr 1e-3, MSE y `MinMaxScaler` ajustado en train.

### 3.1 Muchos levels (`exp_levels.py`)

`store_sales`, N series elegidas al azar, `lags=28`, `steps=14`. Train hasta 2016-09-30, validación hasta 2016-12-31 (solo para early stopping), test 2017 en 26 bloques de 14 días sin reentrenar. Métrica: media por serie de MAE / MAE del naive estacional (repetir la última semana). 3 semillas; entre paréntesis, mínimo y máximo.

| Modelo | N=10 | N=50 | N=200 |
|:--|:--|:--|:--|
| Arquitectura actual (`create_and_compile_model`, LSTM 100 + Dense 64), por defecto | 1.01 (1.01 a 1.02) | 0.91 (0.76 a 0.99) | 0.84 (0.77 a 0.96) |
| Arquitectura actual, robusto | 0.95 (0.91 a 0.97) | 0.81 (0.76 a 0.89) | **0.75** (0.75 a 0.75) |
| Misma LSTM, global (una muestra por serie, pesos compartidos), robusto | 1.00 | 0.80 (1 semilla, por defecto) | no medido |
| MLP global (lags aplanados, 128x128, salto lineal), robusto | **0.82** (0.81 a 0.82) | **0.79** (0.78 a 0.79) | 0.78 (0.77 a 0.78) |
| LightGBM, `ForecasterRecursiveMultiSeries`, `lags=28` | 0.82 | 0.81 | 0.81 |

Conclusiones:
- Con el entrenamiento por defecto, la LSTM se detiene en la época 9 a 12 sin haber aprendido: con N=10 es peor que el naive en las 3 semillas, y con N=50 la misma configuración da 0.76 o 0.99 según la semilla. La guía `comparing-forecasting-models.ipynb` observa lo mismo («la precisión de la LSTM varía con la semilla»).
- El entrenamiento robusto arregla casi todo y, con muchas series homogéneas, la arquitectura actual es la mejor.
- La LSTM global no mejora a la multivariante y es mucho más lenta; el MLP es estable entre semillas.

### 3.2 Exógenas (`exp_exog.py`)

`bike_sharing_extended_features`, target `users`, `lags=48`, `steps=24`. Train hasta 2012-04-30, validación hasta 2012-08-31, test hasta 2012-12-30 en bloques de 24 h. MAE medio de 2 semillas; el naive de 24 h da 80.4. Por velocidad, este experimento usa una **réplica en PyTorch nativo** de la arquitectura de `create_and_compile_model` (misma inicialización que Keras: Glorot, ortogonal y sesgo de la puerta de olvido a 1).

| Modelo | sin exog | `temp` | 6 exog | 88 exog | 6 exog + 50 de ruido |
|:--|:--|:--|:--|:--|:--|
| Arquitectura actual, por defecto | 73.8 | **155.6** | 74.9 | 94.1 | 90.3 |
| Arquitectura actual, robusto | 70.8 | **155.5** | 73.7 | 94.1 | 90.6 |
| Actual + one-hot del paso en el decodificador | | 80.6 | 80.2 | 105.9 | 91.2 |
| Encoder con exog pasadas + one-hot del paso | | 88.5 | 70.6 | 94.4 | 80.8 |
| MLP tipo TiDE (lags aplanados + exog futuras proyectadas a 4 dimensiones con una capa compartida por paso, salto lineal) | **62.5** | **68.6** | **57.5** | **66.1** | **60.1** |
| LightGBM, `ForecasterRecursive`, `lags=48`, sin ajustar | 75.3 | 73.5 | 66.0 | 74.1 | 70.1 |

Las 6 exog son `temp`, `holiday`, `hour_day_sin/cos` y `week_day_sin/cos`; las 88 son todas las columnas numéricas del dataset.

Causa del 155: en el camino con exógenas, el estado final de la LSTM se repite con `RepeatVector(steps)` y se concatena con la exógena de cada paso antes de un `TimeDistributed(Dense)`. Todos los pasos reciben lo mismo salvo la exógena, así que **el decodificador no sabe qué paso del horizonte predice**. Con una exógena que no codifica la hora, la red no puede reproducir el perfil diario. Con muchas exógenas, todas las variantes LSTM sobreajustan (paran hacia la época 26). La guía de comparación lo refleja: con la temperatura como exógena, la LSTM queda al nivel del baseline.

Tiempo medio por modelo: MLP unos 13 s; variantes LSTM de 90 a 160 s.

### 3.3 Baseline neuronal sin código nuevo (`exp_sklearn_mlp.py`)

`MLPRegressor(hidden_layer_sizes=(128, 128), early_stopping=True, n_iter_no_change=15, max_iter=300)` dentro de los forecasters existentes, con `MinMaxScaler` y las mismas particiones:

| Caso | MLP de scikit-learn | MLP tipo TiDE (§3.1 y §3.2) | LSTM actual, robusto | LightGBM |
|:--|:--|:--|:--|:--|
| `store_sales`, N=50 (`ForecasterRecursiveMultiSeries`, `encoding="onehot"`) | **0.79** (0.791 a 0.792) | 0.79 | 0.81 | 0.81 |
| `bike`, sin exog (`ForecasterRecursive`) | 64.7 | 62.5 | 70.8 | 75.3 |
| `bike`, 6 exog | **55.2** | 57.5 | 73.7 | 66.0 |
| `bike`, 88 exog | 72.2 (64.7 y 79.6) | 66.1 | 94.1 | 74.1 |

Tarda de 5 a 25 s por backtest. Con 88 exógenas varía más entre semillas. No usa GPU.

### 3.4 Otras medidas

- **Velocidad** (`bench_lstm.py`): una época de LSTM 100 sobre 11 000 ventanas de 48 pasos, CPU de 4 hilos y máquina libre: **5.8 s con Keras (backend torch) frente a 2.0 s con `torch.nn.LSTM`**.
- **Intervalos** (`exp_intervals.py`): `ForecasterRnn` con las 6 exog, conformal al 80 %: con residuos in-sample, **66.3 % de cobertura** (ancho medio 142); con residuos de un backtest sobre validación, 74.8 % (ancho 197). Ojo: los dos backtests dieron MAE distintos (64.0 y 65.8) por el bug T1.
- **Warm start en backtesting** (`exp_warm_start.py`): `bike`, `users`, LSTM 64, 30 épocas, `lags=24`, `steps=24`, backtest sin refit desde 2012-09-01. MAE con un forecaster nuevo: 85.2 y 88.0 (2 semillas). Con un forecaster ya entrenado con todo el histórico: 71.7 y 69.9. Control, ya entrenado solo con train: 73.9 y 75.8. La mayor parte de la diferencia es warm start (el doble de épocas acumuladas); la fuga de datos existe pero es menor (2 a 6 puntos).

---

## 4. Tareas pendientes

**Para el autor, antes de empezar:**
- Decidir T2 (qué se puede buscar), T5, T6, T7 y la pregunta de T8 sobre la guía de comparación.
- Decidir si la rama se queda solo como registro o si las tareas salen de ella. Si salen de ella, borrar este documento y `dev/deep_learning_review/` en el último commit de cada PR, antes de fusionar.
- Para la autoría de los commits, ver el estado de la rama en el §2.

**Para la siguiente sesión:**

Orden recomendado: T1, T2, T3, T4 (bugs; cada uno, un commit con su test) y después T5 a T8. Nota de versión con el skill `release-note`, sección 0.26.0, apartado Fixed (o Changed) de `ForecasterRnn`. Si cambia algo público, `ai-context-sync`.

### T1. `fit` no reinicia los pesos (warm start)

- **Reproducción:** `repro_bugs.py`, caso 1: el segundo `fit` empieza con la pérdida donde acabó el primero.
- **Causa:** `fit` llama a `self.estimator.fit` sobre el modelo ya entrenado (`_forecaster_rnn.py`, alrededor de la línea 1079). `deepcopy_forecaster` copia el modelo Keras con sus pesos (`utils/utils.py`, alrededor de la línea 4997), así que el backtesting entrena partiendo de los pesos que tuviera el forecaster.
- **Impacto:** el resultado de un backtest depende del estado previo del forecaster; con refit, cada fold sigue entrenando el anterior; y si el forecaster se entrenó antes con datos del test, hay fuga (§3.4).
- **Arreglo propuesto:** guardar los pesos iniciales en `__init__` (`self.estimator.get_weights()`) y, al empezar `fit`, restaurarlos y reiniciar el estado del optimizador (recompilar con la configuración de compilación actual). Comprobar que `deepcopy_forecaster`, `save_forecaster` y `load_forecaster` conservan el atributo.
- **Tests:** dos `fit` seguidos dan el mismo `history_`; un backtest da el mismo resultado con un forecaster nuevo y con uno ya entrenado.

### T2. La búsqueda de hiperparámetros no funciona nunca

- **Reproducción:** `repro_bugs.py`, caso 4: `grid_search_forecaster_multiseries` devuelve un DataFrame vacío con el warning «Parameters skipped ... Cannot clone object '<Functional ...>'» para cada combinación, y `set_params` lanza `TypeError`.
- **Causa:** `ForecasterRnn.set_params` (línea 1991) usa `sklearn.base.clone`, que no admite modelos Keras. El método tiene un `# TODO create testing`. Además, `set_lags` no hace nada (línea 2048), así que un `lags_grid` se ignoraría aunque la búsqueda funcionara.
- **Agravante:** la skill `skills/deep-learning-forecasting/SKILL.md` remite a `hyperparameter-optimization` para afinar la RNN, y `skills/complete-api-reference/references/model-selection-signatures.md` lista `ForecasterRnn` en las funciones de búsqueda.
- **Arreglo propuesto:** `keras.models.clone_model` + `compile` con la configuración de compilación actual actualizada con `params`. Que `set_lags` lance un error si los lags no coinciden con los del modelo (la arquitectura los fija). Test de `set_params` y de una búsqueda pequeña.
- **Decisión:** ¿qué se puede buscar? (a) Solo parámetros de compilación, documentándolo en la skill y en la guía (recomendado); (b) parámetros de `create_and_compile_model` (unidades, capas), que requiere que el forecaster sepa reconstruir el modelo.

### T3. `exog` sin las primeras `window_size` filas

- **Reproducción:** `repro_bugs.py`, caso 2: `fit` falla con «Data cardinality is ambiguous ... 'x' sizes: 288, 278».
- **Causa:** `_create_train_X_y` acepta esa longitud (línea 829), pero luego aplica `_create_lags` a la exógena corta (línea 886), que vuelve a descontar `window_size`: salen menos filas y desplazadas.
- **Arreglo propuesto:** tras transformar la exógena, reindexarla al índice de `series` (las primeras `window_size` filas nunca se usan como exógena futura) o desplazar el inicio de la ventana. Test con las dos longitudes que comprueba que las matrices coinciden.

### T4. Dispositivo de torch elegido por skforecast, no por Keras

- **Reproducción:** `repro_bugs.py`, caso 3 (simula una GPU): skforecast mueve los datos a `cuda` aunque Keras use `cpu`. En la práctica: `comparing-forecasting-models.ipynb` fija `KERAS_TORCH_DEVICE="cpu"` y su salida publicada dice `Using 'torch' backend with device: mps`.
- **Causa:** `fit` (línea 1048) elige cuda, luego mps, luego cpu, lo imprime y mueve todo el conjunto de entrenamiento a ese dispositivo. Keras usa `KERAS_TORCH_DEVICE` si está definido, y si no mps, cuda, xpu, cpu (`keras/src/backend/torch/core.py`), y en `convert_to_tensor` vuelve a mover cada tensor a su dispositivo.
- **Impacto:** mensaje falso, una copia de ida y vuelta, y el conjunto entero en la memoria de la GPU en lugar de por lotes.
- **Arreglo propuesto:** no mover los datos (pasar arrays de numpy y dejar que Keras los mueva por lotes) y, si se quiere informar, leer el dispositivo del modelo (`self.estimator.weights[0].value.device`). Quitar el `print` o ponerlo detrás de un `verbose`. Re-ejecutar las dos guías para actualizar las salidas.

### T5. Entrenamiento robusto por defecto (**decisión**)

- **Medida:** §3.1. Con paciencia 5 la LSTM se para en una meseta.
- **Opciones:** (a) si el usuario pasa `series_val` y ningún callback, añadir `EarlyStopping(patience=15, restore_best_weights=True)` y `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)` (recomendado); (b) solo recomendarlo en la guía y en la skill.

### T6. Intervalos con residuos de validación (**decisión**)

- **Medida:** §3.4. Con residuos in-sample, 66 % de cobertura para un 80 % nominal.
- **Opciones:** (a) si hay `series_val`, calcular en `fit` los residuos sobre validación y usarlos como out-of-sample (recomendado); (b) documentar que con redes hay que usar `set_out_sample_residuals`.

### T7. Exógenas en la arquitectura recurrente (**decisión**)

- **Medida:** §3.2. Añadir la posición del paso arregla `temp` (155 a 81) pero empeora con 6 y 88 exógenas.
- **Opciones:** (a) documentar la limitación en la guía y en la skill y remitir al baseline de T8 cuando haya exógenas (recomendado); (b) probar antes otras variantes (proyección de las exógenas, codificación del paso con embedding) y medirlas con `exp_exog.py`.

### T8. Documentación: baseline neuronal y guías

- Sección en `forecasting-with-deep-learning-rnn-lstm.ipynb` con el MLP de scikit-learn en `ForecasterRecursiveMultiSeries` como baseline neuronal (§3.3), y cuándo usar cada uno.
- Actualizar `skills/deep-learning-forecasting/` (y la mención a `hyperparameter-optimization` según T2) y `tools/ai/llms-base.txt`; regenerar con `ai-context-sync`.
- **Decisión:** ¿añadir el MLP a `comparing-forecasting-models.ipynb`? Respondería directamente a «¿una red entrenada con mis datos mejora a LightGBM?».

### T9. Detalles menores

- El docstring de `ForecasterRnn` (atributo `estimator`, líneas 120 y 121) dice que se entrena una instancia por paso y se guardan en `self.estimators_`: es un texto copiado de los Direct.
- `create_and_compile_model` imprime versiones de keras y del backend en cada llamada: pasar a un parámetro `verbose`.
- `compile_kwargs` tiene por defecto una instancia de `Adam()` creada al importar y compartida entre llamadas. Hoy no causa errores porque `ForecasterRnn` hace `deepcopy` del modelo, pero conviene usar `None` y crearla dentro.

### Fuera de alcance (anotado)

Limitaciones sin tarea: no hay window features ni differentiation (TODO en el código); `series` debe ser un DataFrame alineado (sin dict, sin series de distinta longitud, sin NaN); un modelo propio debe usar los nombres de capa `series_input`, `output_dense_td_layer` y `exog_input`; sin pérdidas por cuantiles.

---

## 5. Entorno para las tareas

- Los tests de deep learning necesitan torch y keras. En la nube, el índice CPU de PyTorch (`download-r2.pytorch.org`) está bloqueado por la política de red, así que `SKFORECAST_CLOUD_DL=1` y `--torch-backend cpu` fallan. Alternativas: añadir `download-r2.pytorch.org` a los dominios permitidos del entorno, o instalar desde PyPI: `uv pip install --python "$(which python)" torch "keras>=3.3,<4.0"` (build CUDA, funciona en CPU).
- Sin TensorFlow, los tests de `test_fit_tensorflow.py` fallan; es previo.

---

## 6. Verificación y scripts

Scripts en `dev/deep_learning_review/` (no forman parte del paquete; borrarlos con este documento):

| Script | Qué hace |
|:--|:--|
| `get_data.py` | Descarga `store_sales` y `bike_sharing_extended_features` en `data/` (ignorada por git). Ejecutar primero. |
| `repro_bugs.py` | Reproduce T1 a T4 en unos segundos. |
| `exp_levels.py` | §3.1. Argumento: lista de N, p. ej. `python exp_levels.py 10,50,200`. |
| `exp_exog.py` | §3.2. Variables de entorno `SEEDS` y `THREADS`. |
| `exp_sklearn_mlp.py` | §3.3. |
| `exp_intervals.py` | §3.4, intervalos. |
| `exp_warm_start.py` | §3.4, warm start y fuga. |
| `bench_lstm.py` | §3.4, velocidad Keras frente a torch nativo. |
| `common.py` | Ventanas, entrenamiento y modelos Keras compartidos. |

Ejecutar desde la carpeta, con un hilo de torch por proceso si se lanzan varios a la vez (`THREADS=2`); si no, la sobresuscripción de CPU multiplica los tiempos. Los experimentos largos tardan de 30 a 60 minutos cada uno en 4 núcleos.

Estado de la base (`92b97f2`) al terminar esta sesión: `python dev/deep_learning_review/repro_bugs.py` reproduce los cuatro bugs. Tests de `skforecast/deep_learning/tests`: 199 pasan y 3 fallan, los de `test_fit_tensorflow.py` por falta de TensorFlow (previo). Ninguno cubre los bugs T1 a T4.
