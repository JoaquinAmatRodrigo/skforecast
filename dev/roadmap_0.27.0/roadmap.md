# Roadmap de skforecast 0.27.0

Fecha: 2026-10-18. Base: rama `0.26.x` (`__version__ = "0.26.0"`, publicada el 2026-10-08).
Fuentes: inventario de fase 1 (`work/phase1_open_work.md`, `work/phase1_product_context.md`),
7 lentes (`work/lens_*.md`), triaje (`work/triage.md`), 26 verificaciones (`work/verify/*.md`)
y scripts de reproducción (`repro/*.py`). Las rutas son relativas a la raíz del repositorio
salvo que se indique otra cosa.

Entorno de medida: Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1,
lightgbm 4.7.0, numba 0.68.0, scipy 1.18.1, optuna 5.0.0, torch 2.14.1+cpu,
chronos-forecasting 2.3.2, tabicl 2.2.0. Máquina de 4 núcleos compartida con otros agentes
(los tiempos son medianas de 3 repeticiones).

Esfuerzo: S (horas a 1 día, con tests y docs), M (2 a 5 días), L (más de una semana).

---

## 0. Revisión: lo que queremos hacer en 0.27.0

Revisión del 2026-10-18 sobre este informe. **Esta sección manda sobre la sección 1**, que
mete en 0.27.0 unos 40 elementos porque el análisis se hizo sin límite de tiempo. Las
secciones 1 a 7 quedan como referencia: evidencia, antes y después, y esbozos de cada arreglo.

**Comprobado en local** (Python 3.14.3, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1,
numba 0.68.0, statsmodels 0.14.6): BUG-03, BUG-01 y BUG-stats-interval-ignored se
reproducen con sus scripts. QW-03 se confirma en el código: el `ValueError` de
`skforecast/model_selection/_utils.py:270-288` exige repetir `differentiation` en el cv,
pero `_validation.py:513-518` la sobrescribe después con la del forecaster. El resto de
cifras son las de la sesión en la nube y no se han vuelto a medir.

### Alcance

| Tema | Elementos | Notas |
|---|---|---|
| Resultados erróneos silenciosos | BUG-03, BUG-01, BUG-stats-interval-ignored, BUG-04, BUG-10, PR-1407 | Todos S. BUG-01 solo como aviso silenciable: el backtesting sobre datos de entrenamiento es un uso documentado. BUG-stats-interval-ignored implementado en local (2026-10-19), pendiente de PR; DEP-stats-alpha descartado (decisión 5) |
| Backtesting probabilístico (titular de la release) | QW-03, QW-09, QW-04, QW-08, BUG-02, QW-10 | Es lo que se pide desde 2024 (#885, #902, #1253, #1409, D1119). Tres cambios de salida independientes que se fusionan por separado y se anuncian juntos en `releases.md`: QW-09 (`coverage` e `interval_width`, #1409) va primero porque no depende de QW-04; QW-04 (`y_true`) y QW-10 (`step`) después |
| Foundation para producción | QW-05, BUG-06, QW-02 | QW-02 pendiente de decisión |
| Paneles multiserie | QW-01, BUG-07, ISSUE-1410 | PR-1408 (IDs enteros, #1357) está resuelto: se cerró y lo sustituyó #1412, ya fusionado en `0.27.x` |
| Stats: corrección y rendimiento | BUG-08, PERF-02, PR-1400 | PR-1400 ya está fusionado en `0.27.x`. Volver a medir PERF-02 después de BUG-08 |
| PRs externos | PR-1402, PR-1387, PR-1388, PR-1389 | Inventariados, no revisados línea a línea |
| Pequeños | QW-07, DEP-python310 (anunciar la retirada en 0.28), DEP-arima-dictstyle | |

El orden dentro de cada tema sigue las dependencias de la sección 1. En particular:

- PR-1407 va antes que QW-09 y QW-04: toca el mismo bloque de
  `_backtesting_forecaster_multiseries` y decide qué series entran en los agregados.
- BUG-stats-interval-ignored va antes que QW-09. Corrección (2026-10-19): `backtesting_stats`
  no estaba afectado (allí `alpha=None` por defecto y `interval` se respetaba); el bug solo
  afectaba a `ForecasterStats.predict_interval` llamado directamente. Lo que aporta a QW-09 es
  que, tras el arreglo, `backtesting_stats` traduce `alpha` a `interval` al principio, así que
  el intervalo nominal está siempre en `interval`.
- ISSUE-1410 conviene antes (mismo fichero, dos líneas), pero no bloquea.
- BUG-02 va antes que QW-10.

Orden de QW-09: PR-1407 → BUG-stats-interval-ignored → (ISSUE-1410) → QW-09 → QW-08 (validar
también los nombres reservados `coverage` e `interval_width`) → QW-04 → BUG-02 + QW-10 →
FE-04 → foundation con `interval` (tras BUG-06).

### Decisiones tomadas (2026-10-18)

1. **QW-09 separado de QW-04 y de QW-10.** #1409 no necesita la columna `y_true`: el valor real
   ya está en el cálculo de métricas (`_validation.py:669`, `_utils.py:1160-1166`).
2. **`coverage` e `interval_width` sin flag:** se añaden siempre que `interval` tenga dos cotas.
3. **Aviso de infracobertura descartado definitivamente** (sección 7).
4. **QW-10 (columna `step`) se hará:** es barata y no depende de QW-04 ni de QW-09.
5. **DEP-stats-alpha descartado (2026-10-19).** `alpha` se mantiene como alias de `interval`
   en `ForecasterStats.predict_interval` y `backtesting_stats`: es el nombre habitual para
   quien viene de estadística (statsmodels, pmdarima) y los estimadores `Arima` y `Sarimax`
   también lo aceptan. El bug lo causaba su valor por defecto (0,05), no el alias. En su lugar,
   pasar `alpha` e `interval` a la vez da `ValueError`, como ya hace `Arima.predict_interval`
   con `alpha` y `level`, y desaparece la regla de precedencia.

### Decisiones pendientes (cerrarlas antes de implementar)

1. **BUG-02** (`fold_stride < steps`): quedarse con la última predicción de cada fecha es
   deliberado. Propuesta: parámetro nuevo en `TimeSeriesFold` con el comportamiento actual
   por defecto y aviso cuando `fold_stride < steps`. Cambiar el defecto, solo anunciado y en
   otra versión.
2. **QW-04** (columna `y_true` sin flag): cambia la salida del backtesting y obliga a
   actualizar unas 160 aserciones. Propuesta: hacerlo sin flag, como `fold` en 0.18.0, y
   anunciarlo en las notas de la release.
3. **QW-02** (aviso en foundation cuando se ajusta con exog y `predict` no la recibe):
   invierte un silencio documentado. Propuesta: hacerlo; en ML el mismo olvido ya da error.
4. **HANDOFF-T2/T3/T4** (`dev/handoff_utils_pendiente.md`): pendientes de decisión de Joaquín.
5. **PR-1341** (TiRex-2): aceptarlo o no; la revisión es grande (L).

### Para después de 0.27.0

FE-02, FE-03, FE-04, PERF-01 (gana 3,4 veces con una serie y nada con 50), QW-06, BUG-09,
BUG-11, GAP-foundation-backtesting-interval (depende de BUG-06), ISSUE-1131 y todo lo de la
sección 5.

### Huecos del análisis a cubrir al implementar

- **Foundation**: solo se ejecutaron Chronos-2 y TabICL con pesos reales. BUG-04, QW-05 y
  BUG-06 se extienden al resto de adaptadores leyendo el código: comprobarlos con cada
  adaptador al arreglarlos.
- **ML**: `ForecasterDirectMultiVariate` apenas se revisó y las exógenas categóricas no se
  revisaron.
- **Repositorio**: antes de publicar 0.27.0, decidir si `dev/roadmap_0.27.0/` (sobre todo
  `work/`) se queda, porque `0.27.x` se fusiona en `main`.

---

## 1. Propuesta para 0.27.0

**Temas**

1. **Backtesting en el que se puede confiar.** Hoy hay cuatro maneras de obtener en silencio
   una métrica o un modelo erróneos en el flujo central (evaluar in-sample sin aviso,
   `fold_stride < steps`, métricas "mayor es mejor" en las búsquedas, agregados multiserie
   sin una serie), y evaluar intervalos exige reconstruir a mano valor real, paso del
   horizonte y cobertura. La 0.27 corrige lo primero e integra lo segundo con piezas que ya
   existen (`calculate_coverage`, columna `fold`).
2. **Paneles multiserie reales sin fricción:** series que terminan antes (desaparecen de
   `predict`), que empiezan tarde, con IDs enteros, en formato largo con `set_index`, con
   `differentiation`. Tres PR externos ya cubren parte.
3. **Foundation listo para producción:** el modelo ajustado no se pierde, el punto es el
   mismo en `predict` y en los intervalos, las covariables se usan como dice la
   documentación, el fichero guardado pesa KB y el backtesting es hasta 3,4 veces más rápido.

**Orden propuesto** (valor para el usuario frente a esfuerzo, sin recorte por calendario)

| Bloque | Elementos, en orden |
|---|---|
| A. Resultados silenciosamente erróneos, esfuerzo S | BUG-03, BUG-01, BUG-02, PR-1407 (#1406), BUG-04, BUG-10, BUG-stats-interval-ignored (implementado, pendiente de PR) |
| B. Quick wins diarios y PR listos | QW-03, QW-01, QW-09, QW-04, QW-10 (tres cambios de salida, un solo anuncio), PR-1389, PR-1388, ISSUE-1410, QW-02, QW-05, ISSUE-1385, PR-1400, PR-1402, PR-1387 |
| C. Corrección de más alcance, STATS y quick wins secundarios | BUG-06, BUG-08, PERF-02, QW-07 (+ DOC-skills-percentiles-stale), BUG-07, QW-08, QW-06, BUG-11, BUG-09 |
| D. Features y gaps | FE-02, PERF-01, GAP-foundation-backtesting-interval, FE-04, FE-03 |
| E. Mantenimiento y decisiones | DEP-python310 (anunciar), DEP-arima-dictstyle, HANDOFF-T2/T3/T4, PR-1341 (TiRex-2), ISSUE-1131 (alcance) |

**Por qué esta selección y este orden**

- Primero, todo resultado silenciosamente erróneo en un flujo común: los siete elementos del
  bloque A están reproducidos, son S y casi todos se arreglan en menos de 20 líneas.
  BUG-03 reentrena con el peor modelo con `return_best=True` (valor por defecto); BUG-01 da
  métricas 2,5 veces optimistas sin aviso; BUG-02 hace que las búsquedas elijan por el error
  a 1 paso; BUG-04 y BUG-10 hacen que foundation prediga con otro modelo u otros datos.
- Después, la fricción semanal con arreglos mínimos: QW-03 son 2 líneas y quita un
  `ValueError` (o un DataFrame vacío) de todo backtesting con `differentiation`;
  QW-09, QW-04 y QW-10 son el quick win de calibración que se pide desde 2024 (#885, #902,
  #1253, #1409, D1119) y solo reutilizan funciones existentes.
- Dependencias que fijan el orden interno: BUG-10 antes de FE-02; BUG-06 antes de aceptar
  `interval` en `backtesting_foundation`; PR-1407 y BUG-stats-interval-ignored antes de QW-09;
  QW-04, QW-09, PR-1407 e ISSUE-1410 antes de FE-04; BUG-02 antes de QW-10 (columna `step`)
  y de BUG-09 (este reutiliza la misma
  deduplicación); BUG-04 y QW-05 comparten la declaración del atributo de backend.
- STATS entra solo por corrección y rendimiento medidos (intervalo ignorado; BUG-08, de 5 a
  15 veces más rápido; PERF-02, de 3 a 70 veces con carga); BUG-09 va al final por ser nicho.
  Deep learning queda fuera por diseño; de los módulos restantes solo entran PR-1402 y PR-1387.
- Las features van detrás porque aportan menos por unidad de esfuerzo o tienen más riesgo:
  FE-03 mejora el MAE un 2% y solo con árboles; FE-04 no tiene peticiones explícitas, pero
  corrige un sesgo real al comparar con el baseline.
- Cambios de comportamiento que hay que anunciar en `releases.md`: columnas nuevas `y_true` y
  `step` y métricas `coverage`/`interval_width` sin flag (precedente: `fold` en 0.18.0); `alpha`
  deprecado en stats; resultados distintos con `freeze_params=True` (Fixed); aviso nuevo en
  foundation sin exog futura; `save_forecaster` deja de guardar los pesos foundation; columna
  `mean_pinball_loss_q_{alpha}`.

---

## 2. Quick wins

| Item | Área | Valor para el usuario | A cuántos usuarios afecta | Esfuerzo | Evidencia | Esbozo del cambio / API |
|---|---|---|---|---|---|---|
| **QW-03** Heredar `differentiation` en `TimeSeriesFold` y `OneStepAheadFold` | model_selection | No hay que repetir `differentiation` en el cv; deja de fallar el código generado con la propia referencia IA; desaparece el peor caso (grid devuelve DataFrame vacío y forecaster sin ajustar) | Flujo habitual: todo backtesting o búsqueda con `differentiation` (recomendada con árboles), una o varias series | S (2 líneas, docstrings, 2 tests) | `skforecast/model_selection/_utils.py:274-288,602-620` lanza el error y `_validation.py:513-518,1286-1291` sobrescribe después `cv.differentiation` con `forecaster.differentiation_max`; `tools/ai/llms-base.txt:340,370` ya dice "set automatically"; prototipo: predicciones idénticas (`DataFrame.equals`), 220 tests pasan | Comprobar solo si `cv.differentiation is not None`; mantener el `ValueError` si el usuario fija un valor distinto. Quitar el comentario "must match" de 5 celdas en 3 guías |
| **QW-01** Aviso cuando `predict()` con `levels=None` excluye series | recursive (multiseries) | Las series cuya última observación no llega a la fecha máxima dejan de desaparecer en silencio del forecast de producción | Flujo habitual en paneles de retail o demanda construidos desde formato largo o con NaN al final | S (~15 líneas) | `skforecast/utils/utils.py:4680` (`if input_levels_is_list and ...`); medido: 6 series entrenadas, 4 predichas, 0 avisos; store_sales 50 entrenadas, 47 predichas. Misma familia que #1406 / PR-1407 (aviso en backtesting), pero en `predict` y con código distinto | `IgnoredArgumentWarning` también con `levels=None`, con lista ordenada y truncada a 10, la fecha máxima y cómo predecirlas; un solo aviso en `predict_interval(method='bootstrapping')` |
| **QW-09** Cobertura y anchura del intervalo en las métricas de backtesting (#1409) | model_selection / metrics | `coverage` e `interval_width` en la tabla de métricas cuando `interval` tiene dos cotas, por nivel y en los agregados (`average`, `weighted_average`, `pooling`), sin flag | Flujo habitual: todo backtesting probabilístico de ML y stats (las búsquedas pasan `interval=None`, `_search.py:454,1454,2778,3033`, y no cambian) | S (~1 día con tests y docs) | https://github.com/skforecast/skforecast/issues/1409; las métricas solo usan `pred` (`_validation.py:688`, `_utils.py:1067`); `skforecast/metrics/metrics.py:459` (`calculate_coverage`, no ignora NaN); medido en bike_sharing (80% nominal): cobertura de LGBM con residuos in-sample 0,53 (bootstrapping) y 0,26 (conformal) frente a 0,80 y 0,62 con residuos out-of-sample, sin ninguna señal (`work/verify/FE-01-value.md`) | Helper privado que calcula las dos métricas si la salida tiene `lower_bound` y `upper_bound`, sobre las mismas filas que el resto de métricas (tras `keep='last'`) y sin las filas con `y_true` NaN (en multiserie solo `pred` se pone a NaN, `_validation.py:1485`); columnas detrás de las métricas del usuario. En multiserie, pasar las cotas a `_calculate_metrics_backtesting_multiseries` (hoy solo `[['pred']]`, `_validation.py:1497`). Cubre `backtesting_forecaster`, `_multiseries` (incluido `ForecasterRnn`) y `_stats` (uno o varios estimadores); foundation cuando acepte `interval`. `ValueError` si una métrica del usuario se llama `coverage` o `interval_width`. Sin aviso de infracobertura (descartado, sección 7) |
| **QW-10** Columna `step` en las predicciones de backtesting (D1119) | model_selection | Paso del horizonte de cada predicción, para analizar error y cobertura por horizonte sin reconstruirlo a mano | Flujo habitual: todo informe por horizonte | S | https://github.com/skforecast/skforecast/discussions/1119; `_validation.py:660,1471,2140,2574` (solo `fold`); `cumcount()+1` por fold falla con `gap>0` (con `gap=3` los pasos reales son 4..13, `work/verify/FE-01-risk.md`) | Columna `step` (de `gap + 1` a `gap + steps`) junto a `fold` en los cuatro backtesting; receta documentada por `step` y `level`. Después de BUG-02; independiente de QW-04 y QW-09. Cambio de salida anunciado |
| **QW-04** Columna `y_true` en las predicciones de backtesting | model_selection | Valor real junto a la predicción en los cuatro backtesting: residuos, gráficos y métricas propias sin merge (que hoy cambia según dict, ancho o largo, y falla con el snippet habitual en DataFrame ancho) | Flujo habitual: casi todo usuario que inspecciona o grafica el backtesting | S de código (~40 líneas); ~1 día con la actualización mecánica de ~160 aserciones | `y_true` se calcula y se descarta en `_validation.py:669,2167,2180,2681` y `_utils.py:1149-1166`; coste medido 0,1 a 0,4% del tiempo (20 a 97 ms sobre 9 a 36 s); 21 usos de `data.loc[pred.index, col]` en 8 guías | Columna `y_true` justo antes de `pred`, sin flag. Rompe código que compara el DataFrame entero o indexa por posición. En multiserie ML, poner a NaN todas las columnas de predicción sin observación (hoy solo `pred`: la cobertura sale 0,63 frente a 0,73) |
| **QW-02** Aviso en foundation si se ajustó con exog y `predict` no la recibe | foundation | Hoy el efecto de promociones o festivos desaparece sin señal con Chronos-2 (adaptador por defecto), TimesFM 3.0 y TS-ICL; en ML el mismo olvido da `ValueError` | Flujo habitual de producción con foundation y exógenas conocidas a futuro (no afecta al backtesting) | S (~20 líneas) | `skforecast/foundation/_foundation_model.py:973-993` (solo avisa si `supports_past_only_covariates=False`); `_adapters.py:170,1154,4028`; medido con Chronos-2 small: días con promo 10,08/10,03 sin exog frente a 15,08/15,27 con exog | `MissingExogWarning` (ya existe) en `FoundationModel._check_exog_columns` cuando no hay exog futura y las columnas se usan solo como pasadas. Requiere visto bueno: invierte un silencio documentado (`skills/foundation-forecasting/references/adapter-parameters.md:226`) y hay que reescribir 6 parametrizaciones de `test_check_exog_columns.py:182-201` |
| **QW-05** `save_forecaster` sin los pesos foundation cargados tras `predict` | foundation / utils | El fichero guardado tras validar pasa de cientos de MB a KB, no queda atado a torch ni al dispositivo y deja de fallar con TimesFM 2.5 | Flujo habitual de paso a producción, cualquier adaptador | S (~30 líneas, 2 a 4 h) | Medido con chronos-2-small: 0,01 MB antes de `predict` y 111,81 MB después; carga 4,25 s frente a 0,33 s; TimesFM 2.5: `PicklingError` por clase local (`_adapters.py:970-975`); no hay `__getstate__` en `skforecast/foundation/` | `__getstate__` en `_AdapterBase` que vacía el backend cargado de forma perezosa (flag en los 9 `_load_*`) y las cachés `_forecast_obj`/`_resolved_device`, conservando el objeto que pasó el usuario (BUG-04). Documentar que el primer `predict` tras cargar recarga los pesos |
| **QW-07** Mensajes de error accionables en cuatro puntos | utils / model_selection | Frecuencia inferida, intervalos en percentiles, residuos ausentes en multiserie y función equivocada para el tipo de forecaster pasan a decir cómo resolverse (hoy grid/random con la familia equivocada devuelven un DataFrame vacío sin excepción) | Usuarios nuevos, quien migra desde 0.23/0.24 y quien pasa de ML a foundation; una vez por usuario, muy frecuente | S (+142/-3 en el prototipo) | `utils.py:2046-2050`, `:1222-1229`, `:1875-1884,1936-1947`; `_search.py:474,519`; `_validation.py:866-871,1695-1700,2318-2323,2825-2830`; prototipo: 615 + 365 tests pasan sin cambios. (b) amplía DOC-skills-percentiles-stale | (a) `pd.infer_freq` y receta `.asfreq`; (b) frase de migración si el límite está en (1, 100]; (c) un dict con todos los valores None cuenta como vacío y reutiliza el mensaje accionable; (d) `check_forecaster_type_model_selection` al inicio de cada backtesting y búsqueda, nombrando las funciones correctas. Corregir `skills/troubleshooting-common-errors/SKILL.md:72,184` |
| **QW-08** RMSE como string y nombres de métrica únicos | metrics | `metric='root_mean_squared_error'` sin importar sklearn; nombres de pinball con el cuantil; error claro en backtesting con dos métricas del mismo nombre | RMSE: flujo habitual (ahorro diario pequeño); pinball con varios cuantiles: nicho | S (+7/-1 y validación) | `skforecast/metrics/metrics.py:45-62` (solo la variante escalada, `:52`) y `:555-558`; medido: dos pinball dan columnas duplicadas, y en multiserie `TypeError` (`_utils.py:1247`); las búsquedas ya validan nombres únicos (`_search.py:394-400`) | Añadir RMSE a `allowed_metrics`; validar unicidad en los backtesting; `__name__ = f"mean_pinball_loss_q_{alpha}"` (Changed: renombra una columna). Completar los docstrings de `metric` (MedAE, sMAPE) |
| **QW-06** Alinear `exog` por índice en `predict` de Recursive y Direct | recursive / direct | Pasar un único DataFrame de exog (pasado y futuro) sin recortarlo, como ya hacen MultiSeries y Foundation | Flujo habitual de producción con calendario o meteo; coherencia entre forecasters | S (+36/-1 en el prototipo) | `utils.py:1656` (`align_by_index` solo en MultiSeries) y `:1738-1745`; prototipo: mismos resultados que el recorte manual, 871 tests pasan (1 a adaptar); con RangeIndex reseteado daría `[-1.331, ...]` frente a `[1.005, ...]` en silencio | Solo con `DatetimeIndex`: si el índice contiene el primer paso esperado, `exog.iloc[pos:]` antes del chequeo de NaN y de `transform_dataframe`; también Classifier y DirectMultiVariate; RangeIndex mantiene el chequeo estricto; `fit` no cambia. Toca el mismo camino de `predict` que HANDOFF-DOUBLE-WARN (sección 5): resolver el doble aviso a la vez si sale barato |

### QW-03: antes y después

```python
# Antes (0.26.0)
forecaster = ForecasterRecursive(LGBMRegressor(), lags=24, differentiation=1)
cv = TimeSeriesFold(steps=24, initial_train_size=n_train)
backtesting_forecaster(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# ValueError: The differentiation included in the forecaster (1) differs from ...
# grid_search_forecaster con el mismo cv: DataFrame vacío y forecaster sin ajustar
cv = TimeSeriesFold(steps=24, initial_train_size=n_train, differentiation=1)  # hay que repetirlo
```

```python
# Después
cv = TimeSeriesFold(steps=24, initial_train_size=n_train)  # se hereda del forecaster
backtesting_forecaster(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# Mismo resultado que con differentiation=1 explícito; un valor explícito distinto sigue dando ValueError
```

### QW-01: antes y después

```python
# Antes
forecaster.fit(series=series_dict)   # 2 artículos sin registro el último día
p = forecaster.predict(steps=7)        # sin ningún aviso
faltan = set(forecaster.series_names_in_) - set(p["level"])  # hay que saber que esto pasa
# {'item_3', 'item_5'}: para predecirlas hay que llamar aparte con levels=[...]
```

```python
# Después
p = forecaster.predict(steps=7)
# IgnoredArgumentWarning: Only series whose last window ends at the same index can be
# predicted together. 2 series do not reach the maximum index, '2024-07-18 00:00:00',
# and are excluded from prediction: ['item_3', 'item_5']. To predict them, pass their
# names in `levels` (their predictions start after their own last observation) or
# provide `last_window`.
# Se silencia con suppress_warnings=True o pasando levels.
```

### QW-09: antes y después

```python
# Antes
metric, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
# metric: levels, mean_absolute_error (las cotas del intervalo no se evalúan)
y_long = series.stack().rename("y_true")                         # solo DataFrame ancho
y_long.index.names = ["date", "level"]
preds = preds.rename_axis("date").reset_index().merge(y_long.reset_index(), on=["date", "level"])
preds = preds.dropna(subset=["y_true"])  # si no, calculate_coverage cuenta NaN como no cubierto
# ... bucle por level con calculate_coverage y (upper_bound - lower_bound).mean()
# Ninguna señal de que la cobertura in-sample con LightGBM sea 0,53 frente al 0,80 nominal
```

```python
# Después
metric, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
# metric: levels, mean_absolute_error, coverage, interval_width
#         (una fila por nivel y average, weighted_average, pooling)
# Sin `interval`, o con cuantiles, bootstrapping o una distribución, la tabla no cambia.
```

### QW-10: antes y después

```python
# Antes
preds["step"] = preds.groupby(["fold", "level"]).cumcount() + 1   # incorrecto si gap > 0
```

```python
# Después
# preds: level, fold, step, pred, lower_bound, upper_bound   (+ y_true con QW-04)
by_step = preds.groupby(["level", "step"]).apply(lambda g: pd.Series({
    "mae": mean_absolute_error(g.y_true, g.pred),
    "coverage": calculate_coverage(g.y_true, g.lower_bound, g.upper_bound),
}), include_groups=False)   # receta documentada, sin función nueva
```

### QW-04: antes y después

```python
# Antes
_, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
# series dict:
y = pd.concat(series, names=["level", "date"]).rename("y_true").reset_index()
# series DataFrame ancho (pd.concat falla con TypeError):
# y = series.stack().rename("y_true").rename_axis(["date", "level"]).reset_index()
preds = preds.rename_axis("date").reset_index().merge(y, on=["level", "date"], how="left").set_index("date")
preds = preds.dropna(subset=["y_true"])  # si no, calculate_coverage cuenta NaN como no cubierto
```

```python
# Después (backtesting_forecaster, _multiseries, _stats y _foundation)
_, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
preds[["level", "fold", "y_true", "pred", "lower_bound", "upper_bound"]]
residuals = preds["y_true"] - preds["pred"]
# Las filas sin valor observado tienen NaN en y_true y en todas las columnas de predicción.
```

### QW-02: antes y después

```python
# Antes
forecaster = ForecasterFoundation(FoundationModel("autogluon/chronos-2-small", context_length=256))
forecaster.fit(series=y, exog=exog_train)   # exog_train con la columna 'promo'
predictions = forecaster.predict(steps=7)   # sin aviso: 'promo' solo se usa como covariable pasada
```

```python
# Después
predictions = forecaster.predict(steps=7)
# MissingExogWarning: `exog` is None, so the historical exog columns are used only as
# past covariates by ChronosAdapter for series {'y': ['promo']}. If their future values
# are known (calendar, promotions, holidays...), pass them in `exog`. If they are
# past-only covariates on purpose, ignore this warning.
```

### QW-05: antes y después

```python
# Antes
forecaster = ForecasterFoundation(FoundationModel("autogluon/chronos-2-small"))
forecaster.fit(series=y)
backtesting_foundation(forecaster, series=y, cv=cv, metric="mean_absolute_error")  # carga los pesos
save_forecaster(forecaster, "forecaster.joblib")  # 112 MB
# Con "google/timesfm-2.5-200m-pytorch": PicklingError y fichero corrupto de 0 a 2 bytes
```

```python
# Después
save_forecaster(forecaster, "forecaster.joblib")  # 0,01 MB con cualquier adaptador
forecaster = load_forecaster("forecaster.joblib")
forecaster.predict(steps=7)  # recarga los pesos desde la caché de HF (0,33 s), mismas predicciones
# Un objeto pasado por el usuario (FoundationModel(..., pipeline=pipe_ajustado)) se conserva.
```

### QW-07: antes y después

```python
# Antes
y = pd.read_csv("data.csv", parse_dates=["date"], index_col="date")["y"]
forecaster.fit(y)
# ValueError: `y` has a pandas DatetimeIndex without a frequency. To avoid this error, set the frequency of the DatetimeIndex.
forecaster.predict_interval(steps=12, interval=[10, 90])
# ValueError: Lower interval bound (10) must be >= 0 and < 1.
forecaster_ms.fit(series)                 # store_in_sample_residuals=False por defecto
forecaster_ms.predict_interval(steps=12)
# ValueError: Residuals for level 'x1' are None or empty. (con encoding=None: '_unknown_level')
grid_search_forecaster(forecaster_ms, y, cv, param_grid, metric="mean_absolute_error")
# un RuntimeWarning por combinación y DataFrame vacío, sin excepción
grid_search_forecaster(forecaster_foundation, y, cv, {"context_length": [256, 512]}, metric="mean_absolute_error")
# AttributeError: 'ForecasterFoundation' object has no attribute 'lags'
```

```python
# Después
# ValueError: ... Inferred frequency: 'MS'. Use `.asfreq('MS')` to set it.
# ValueError: Lower interval bound (10) must be >= 0 and < 1. Since version 0.23.0, `interval`
#   and `quantiles` are expressed as quantiles in [0, 1], not percentiles: e.g. `interval = [0.1, 0.9]`.
# ValueError: `forecaster.in_sample_residuals_by_bin_` is either None or empty. Use
#   `store_in_sample_residuals = True` when fitting the forecaster or `set_in_sample_residuals()`.
# TypeError: ... Got ForecasterRecursiveMultiSeries. For `ForecasterRecursiveMultiSeries` use:
#   `backtesting_forecaster_multiseries`, `grid_search_forecaster_multiseries`, ...
# TypeError: ... Got ForecasterFoundation. For `ForecasterFoundation` use:
#   `backtesting_foundation`, `bayesian_search_foundation`.
```

### QW-08: antes y después

```python
# Antes
from sklearn.metrics import root_mean_squared_error
metric, preds = backtesting_forecaster(forecaster, y=y, cv=cv, metric=[root_mean_squared_error, "mean_absolute_error"])
# metric="root_mean_squared_error" -> ValueError: Allowed metrics are: [...]
q1, q9 = create_mean_pinball_loss(0.1), create_mean_pinball_loss(0.9)
backtesting_forecaster(..., metric=[q1, q9])              # columnas duplicadas 'mean_pinball_loss_q'
backtesting_forecaster_multiseries(..., metric=[q1, q9])  # TypeError: Axis must be specified ...
```

```python
# Después
metric, preds = backtesting_forecaster(forecaster, y=y, cv=cv, metric=["root_mean_squared_error", "mean_absolute_error"])
create_mean_pinball_loss(0.1).__name__  # 'mean_pinball_loss_q_0.1': columnas distintas
# Dos métricas con el mismo __name__ (lambdas, partials) -> ValueError claro, igual que en las búsquedas
```

### QW-06: antes y después

```python
# Antes
forecaster.fit(y=y, exog=exog_all.loc[y.index])
start = forecaster.last_window_.index[-1] + forecaster.last_window_.index.freq
predictions = forecaster.predict(steps=24, exog=exog_all.loc[start:])
# predict(steps=24, exog=exog_all) -> ValueError: `exog` must start one step ahead of `last_window`
```

```python
# Después (solo DatetimeIndex; RangeIndex mantiene el chequeo estricto)
forecaster.fit(y=y, exog=exog_all.loc[y.index])
predictions = forecaster.predict(steps=24, exog=exog_all)
```

---

## 3. Features y gaps

| Item | Área | Valor para el usuario | A cuántos usuarios afecta | Esfuerzo | Evidencia | Esbozo del cambio / API | Riesgo | Rompe la API |
|---|---|---|---|---|---|---|---|---|
| **FE-02** Covariables solo pasadas en `backtesting_foundation` (`past_only_exog`). Amplía ISSUE-1131 (solo la parte foundation) | foundation / model_selection | Evaluar de forma honesta variables observadas a posteriori o indicadores adelantados, y ajustar `context_length` con ellas, sin bucle manual | Nicho dentro de foundation (Chronos-2, TS-ICL, TimesFM 3.0), que es foco principal | S (30 a 40 líneas, ~4 tests) | `skforecast/model_selection/_validation.py:2521,2596,2606` (todas las columnas de exog van como futuras); `_search.py:2328-2339`; medido con Chronos-2 small: MAE 1,186 con la columna como futura frente a 2,919 honesto; como solo pasada mejora un 6% frente a descartarla (3,104) | `past_only_exog: str or list[str] or None = None` en `backtesting_foundation` y `bayesian_search_foundation`; se quitan de `exog_test` y se conservan en `context_exog`; `ValueError` si no existen; `IgnoredArgumentWarning` con adaptadores sin soporte | Bajo si BUG-10 se corrige antes; sin él, TabICL y TabPFN-TS corrompen la predicción en silencio. Presentarlo como hueco, no como fuga (el contrato de exog es "conocida a futuro") | No |
| **FE-04** `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries` | model_selection | Baseline en una llamada con métricas por serie y agregadas y, sobre todo, con los mismos pliegues que el modelo global (el bucle manual compara periodos distintos sin avisar) | Flujo habitual en multiserie para quien valida contra un baseline; sin peticiones explícitas | M (~120 líneas, 12 a 15 tests, docs) | `TypeError` en `_validation.py:1687-1701`; medido: con una serie que empieza 60 días tarde, el bucle evalúa 40 predicciones desde 2022-12-27 y el modelo global 100 desde 2022-10-28; prototipo: filas (idx, level, fold) idénticas, 50 series en 0,63 s | Función privada `_backtesting_equivalent_date_multiseries` con pliegues sobre el índice común y un clon por nivel; solo conformal; errores claros con bootstrapping, `return_predictors` y `use_in_sample_residuals=False`; arreglo de `_split.py:985` (DateOffset con dict). Ya en docs: `initial_train_size` como fecha en el bucle manual | Corrección: pliegues, series cortas (misma frontera `window_size` que #1410, `_forecaster_equivalent_date.py:433`), DateOffset con dict. Conflictos con PR-1407, ISSUE-1410, QW-04 y QW-09: hacerlo después. Las dos lentes la ven de prioridad media-baja | No |
| **FE-03** Ventanas estacionales en `RollingFeatures` (`season_length`) | preprocessing | Medias estacionales (misma hora de los últimos días, mismo día de las últimas semanas) sin clases propias frágiles que además no se guardan | Series horarias o diarias con estacionalidad múltiple y modelos de árboles | M (código S, ~40 líneas; el resto es docs y contexto IA) | `skforecast/preprocessing/_preprocessing.py:1604,1657,1673` (solo ventanas contiguas); `skforecast/utils/utils.py:3599-3611` (las clases propias no se guardan); medido con LGBM: -2,2% de MAE en bike_sharing y -1,9% en vic_electricity frente a lags estacionales; con Ridge, 0%; prototipo: diferencia transform/transform_batch 3e-14 | `season_length=None` como último parámetro; `window_sizes` sigue siendo el tramo en pasos; validar `window_sizes % season_length == 0`; atributo de clase `season_length = None` para pickles de 0.26; nombres `roll_mean_168_s24`. Documentar ya los lags estacionales múltiples (`skills/feature-engineering/SKILL.md:250-257`) | Bajo con la semántica de tramo (la API propuesta al principio rompía `window_size` de los forecasters). En Direct la ventana se alinea con el origen; `window_size` grande excluye series cortas. Duda: mejora modesta y solo con árboles | No |

### FE-02: antes y después

```python
# Antes: bucle manual (x_observed solo pasada, weekend conocida a futuro)
preds = []
for k, end in enumerate(range(initial_train_size, len(y) - steps + 1, steps)):
    p = forecaster.predict(
        steps=steps, context=y.iloc[:end], context_exog=exog.iloc[:end],
        exog=exog[["weekend"]].iloc[end:end + steps],
    )
    preds.append(p.assign(fold=k))
# Multiserie, gap, cuantiles o bayesian_search_foundation: sin alternativa
```

```python
# Después
metric, preds = backtesting_foundation(
    forecaster, series=y, exog=exog, cv=cv, metric="mean_absolute_error",
    past_only_exog=["x_observed"],
)
results, best = bayesian_search_foundation(
    forecaster, series=y, exog=exog, cv=cv, search_space=search_space,
    metric="mean_absolute_error", past_only_exog=["x_observed"],
)
```

### FE-04: antes y después

```python
# Antes: bucle por serie; con initial_train_size entero, las series que empiezan tarde
# se evalúan en otro periodo que el modelo global (sin aviso)
preds = []
for level in series.columns:
    _, p = backtesting_forecaster(ForecasterEquivalentDate(offset=7), y=series[level], cv=cv,
                                  metric="mean_absolute_error", show_progress=False)
    preds.append(p.assign(level=level))
# métricas agregadas (average, weighted_average, pooling): a mano
```

```python
# Después
metrics, preds = backtesting_forecaster_multiseries(
    forecaster=ForecasterEquivalentDate(offset=7),
    series=series, cv=cv,                      # mismo cv que el modelo global -> mismos pliegues
    metric=["mean_absolute_error", "mean_absolute_scaled_error"],
    interval=[0.1, 0.9],                       # solo conformal
)
```

### FE-03: antes y después

```python
# Antes: clase propia (no se guarda con save_forecaster y es fácil desalinear transform y transform_batch)
window_features = [RollingFeatures(["mean"], [24]), SeasonalRolling(24, 7), SeasonalRolling(168, 4)]
```

```python
# Después
window_features = RollingFeatures(
    stats         = ["mean", "mean", "mean"],
    window_sizes  = [24, 168, 672],    # tramo en pasos de tiempo (mismo significado que hoy)
    season_length = [None, 24, 168],   # None = ventana contigua
)
# roll_mean_168_s24 = media de y[t-24], y[t-48], ..., y[t-168] (7 puntos)
```

---

## 4. Bugs a corregir

Todos reproducidos con un script en `repro/` (vueltos a ejecutar en la crítica: BUG-03,
BUG-07, BUG-10, BUG-11 y BUG-stats-interval-ignored). BUG-10, BUG-11 y
BUG-stats-interval-ignored son hallazgos laterales (verificación de FE-02, de FE-04 y fase 1).

| ID | Descripción | Área | Esfuerzo | Script de reproducción | Esbozo del arreglo |
|---|---|---|---|---|---|
| **BUG-03** | Las búsquedas eligen el peor modelo con métricas en las que mayor es mejor (`r2_score`): el orden y la dirección de Optuna dependen del tipo de tarea, no de la métrica, y `return_best=True` reentrena con el peor | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-03.py` | (1) En las 3 búsquedas bayesianas, ordenar por `study.direction` (`_search.py:916,2049,2427`). (2) Helper `_infer_metric_direction` con la convención de sklearn (`*_score` se maximiza; `*_error`, `*_loss`, `*_deviance` se minimizan) usado en todas las ordenaciones (`_search.py:514,916,1537,2049,2427,2817,3077`) y como `direction` por defecto (`:846,1953,2369`). (3) Docstrings de `metric` y `kwargs_create_study` |
| **BUG-01** | `backtesting` con `initial_train_size=None` sobre un forecaster ya ajustado evalúa in-sample sin ningún aviso | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-01.py` | En la rama `else` de `check_backtesting_input` (`skforecast/model_selection/_utils.py:370-380`), comparar la primera fecha de test (`index[window_size + gap]`) con `forecaster.training_range_` (dict por serie en multiserie) y emitir un aviso silenciable con `suppress_warnings` con el número y rango de predicciones in-sample. Que el `NotFittedError` sugiera primero `initial_train_size` |
| **BUG-02** | Con `fold_stride < steps` la métrica solo mide los horizontes `1..fold_stride` (con 1, el error a 1 paso) y todas las búsquedas seleccionan por ese error | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-02.py` | Parámetro en `TimeSeriesFold`, p. ej. `overlapping_metric="last"` o `"all"`, aplicado en los 5 puntos de deduplicación (`_validation.py:662-667,1488-1493,2159-2164,2643-2648,2674-2679`); aviso mientras el defecto sea `"last"` (como `OneStepAheadValidationWarning`, `_utils.py:653-658`); no cambiar el defecto sin anunciarlo (la receta de meses naturales depende de `"last"`); corregir `skills/backtesting-configuration/SKILL.md:202-208`, la guía y `llms-base.txt`. Coordinar con BUG-09 |
| **BUG-04** | Un pipeline o modelo precargado (p. ej. Chronos-2 con fine-tuning) se descarta en silencio al crear `ForecasterFoundation`: `clone` lo pierde y se recarga el checkpoint base | foundation | S | `dev/roadmap_0.27.0/repro/BUG-04.py` | `FoundationModel.__sklearn_clone__`: clon estándar y, si el usuario pasó el objeto, copiar la referencia al clon sin copiar pesos. Así lo heredan `ForecasterFoundation.__init__` (`_forecaster_foundation.py:156`), `deepcopy_forecaster` (`utils.py:4999`) y `bayesian_search_foundation`. Cada adaptador declara su atributo de backend (compartido con QW-05). Requisito de PLAN-CKPT (`dev/PLAN_UnifiedFinetunedCheckpoint.md`) |
| **BUG-10** | TabICL (medido) y TabPFN-TS (mismo código) no ignoran las covariables solo pasadas aunque `IgnoredArgumentWarning` diga que sí: predicciones absurdas en silencio al ajustar con exog y predecir sin ella o con menos columnas | foundation | S | `dev/roadmap_0.27.0/repro/BUG-10.py` (requiere `tabicl`) | En `FoundationModel.predict`, si `adapter.supports_past_only_covariates` es False, quitar de `context_exog` las columnas sin futuro antes de llamar al adaptador (arreglo centralizado). Causa: `_adapters.py:2679-2685` (TabICL) y `:3347-3353` (TabPFN) concatenan toda la exog de contexto; T0 (`:3869-3878`) y Nori (`:4795-4799`) ya filtran bien |
| **BUG-stats-interval-ignored** | `ForecasterStats.predict_interval` ignora `interval` porque `alpha=0.05` por defecto tiene precedencia: `interval=[0.1, 0.9]` devuelve el intervalo del 95%. La skill lo usa como ejemplo de intervalo del 80% | stats | S | `dev/roadmap_0.27.0/repro/BUG-stats-interval-ignored.py` | **Implementado (2026-10-19), pendiente de PR.** `alpha=None` por defecto; se usa `interval` si se pasa y el 95% si no se pasa ninguno; `alpha` sigue como alias (sin deprecar, decisión 5) y pasar los dos da `ValueError`, en `predict_interval` y en `backtesting_stats`. `backtesting_stats` traduce `alpha` a `interval` al principio. `interval` pasa a ir antes que `alpha` en las dos firmas (una llamada posicional antigua falla con `TypeError`, no en silencio; no hay ninguna en el repo). Ahora también se valida `alpha` (entre 0 y 1) en `predict_interval`. Corregidos `skills/statistical-models/SKILL.md`, `skills/prediction-intervals/` y `skills/complete-api-reference/`, la guía `forecasting-statistical-models.ipynb` (re-ejecutada, mismas predicciones) y `releases.md` (Fixed, Changed, highlight y "Before upgrading") |
| **BUG-06** | En TabICL y Nori (`point_estimate='mean'` por defecto) y TabPFN con `point_estimate` distinto de `'median'`, `predict()` da la media pero `predict_interval()['pred']` y la métrica de `backtesting_foundation` con `quantiles` usan `q_0.5`; además la salida pierde `pred` | foundation | M | `dev/roadmap_0.27.0/repro/BUG-06.py` | `FoundationModel.predict(quantiles=..., include_point=True)` devuelve `pred` junto a `q_*` (reutiliza `q_0.5` si `point_is_median`; TabICL y TabPFN-TS ya calculan el punto en la misma llamada; Nori necesita otra). `predict_interval` (`_forecaster_foundation.py:946-960`) y `_backtesting_foundation` (`_validation.py:2590-2599,2650-2653,2683-2686`) usan ese `pred`. Corregir el docstring (`_foundation_model.py:1041,1058`) y la tabla de la guía (TimesFM 2.5 usa la mediana) |
| **BUG-08** | `freeze_params=True` (por defecto) no congela nada en `backtesting_stats` ni en `grid/random_search_stats`: la búsqueda automática de Arima/Ets se repite en cada fold y en cada candidato | model_selection (stats) | S | `dev/roadmap_0.27.0/repro/BUG-08.py` | En `_backtesting_stats`, tras el primer ajuste y antes de `Parallel`, propagar `best_params_` a las plantillas: `for est, est_ in zip(forecaster.estimators, forecaster.estimators_): if getattr(est_, "best_params_", None): est._set_params(**est_.best_params_)`. Causa: `_validation.py:2097-2100` congela `estimators_` y `_forecaster_stats.py:521` los vuelve a clonar desde las plantillas. Regenerar valores esperados de tests automáticos con `refit=True`; nota Fixed |
| **BUG-07** | El formato largo con MultiIndex que documenta la guía solo funciona si el MultiIndex sale de `reshape_series_wide_to_long`; con `df.set_index(["series_id", "date"])` fallan `fit`, backtesting, búsquedas multiserie y foundation | recursive (multiseries) / utils | S | `dev/roadmap_0.27.0/repro/BUG-07.py` | En la rama MultiIndex de `check_preprocess_series` (`utils.py:4199-4202`), si el grupo no tiene `freq`, inferirla con `pd.infer_freq` (al menos 3 observaciones) y aplicar `asfreq`; si no se puede, `ValueError` propio del MultiIndex que sugiera `reshape_series_long_to_dict(..., freq=...)`. Corregir la errata "If it a" (`:4271,4285`) y alinear `skills/forecasting-multiple-series/SKILL.md:52` con la guía. PR-1408 tocaba `check_preprocess_series`, pero se cerró y #1412, que lo sustituye, no toca `utils.py`: sin conflicto |
| **BUG-11** | `backtesting_forecaster(ForecasterEquivalentDate, interval=...)` con el `interval_method='bootstrapping'` por defecto lanza `AttributeError: ... no attribute 'predict_quantiles'` en lugar de un error claro (la guía lo esquiva fijando `conformal`) | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-11.py` | `ForecasterEquivalentDate` está en `forecasters_boot_conformal` (`skforecast/model_selection/_utils.py:222-228`) pero solo admite conformal: lista propia de forecasters solo conformal y `ValueError` que indique `interval_method="conformal"` (o usar conformal por defecto para este forecaster) |
| **BUG-09** | `backtesting_stats` falla con `fold_stride != steps` con varios estimadores, y con Sarimax y `refit=False` (también con `refit` entero >= 3 y `fold_stride` por defecto); `grid_search_stats` devuelve 0 filas. Nicho: prioridad baja dentro de STATS | model_selection (stats) | S (~10 líneas) | `dev/roadmap_0.27.0/repro/BUG-09.py` | (a) Deduplicar por (fecha, `estimator_id`) con varios estimadores (`_validation.py:2160-2164`). (b) `last_window` desde el origen del fold anterior hasta el del actual (`_validation.py:1825-1830`). Validado: diferencia 0,0 frente a la referencia; 52 tests de stats pasan con el parche |

**Rendimiento (reproducido)**

| ID | Descripción | Área | Esfuerzo | Script de reproducción | Esbozo del arreglo |
|---|---|---|---|---|---|
| **PERF-02** | El ajuste de Ets y Sarimax (L-BFGS-B de scipy >= 1.15) se vuelve de 3 a 70 veces más lento con cualquier otra carga en la máquina, y `n_jobs='auto'` deja en serie todo backtesting de Arima/Ets/Arar (refit obligatorio) | stats / model_selection (stats) | S | `dev/roadmap_0.27.0/repro/PERF-02.py` | (a) Envolver `Ets.fit`, `Sarimax.fit` (y `Arima.fit`) en `threadpoolctl.threadpool_limits(1, user_api="blas")` (dependencia de scikit-learn; 1,45 ms por llamada, 0,007 ms con un `ThreadpoolController` reutilizado). (b) `select_n_jobs_backtesting` (`_utils.py:706`): `max(1, cpu_count() - 1)` para `ForecasterStats` con `refit` True o 1. Volver a medir tras BUG-08 |
| **PERF-01** | `backtesting_foundation` llama al modelo una vez por fold aunque es zero-shot, y prepara los datos con 0,87 a 1,33 ms por serie y fold | foundation / model_selection | M | `dev/roadmap_0.27.0/repro/PERF-01.py` | (1) Posiciones de fold con `searchsorted` y sumas acumuladas de `notna` una vez por serie, en lugar de `_extract_data_folds_multiseries`. (2) Pseudo-series (serie, fold) en trozos de ~1024 por llamada; una llamada por fold con `cross_learning=True` (mezclar folds filtraría futuro). (3) Cachear `expand_index` y comprobar NaN con numpy en `_prepare_future_exog` (`_foundation_model.py:870,877,1261`). Sin cambios de API |

### Observado frente a esperado

- **BUG-03.** Observado (h2o, Ridge, lags=12): `grid_search_forecaster(metric=[r2_score, ...])`
  pone en la fila 0 `alpha=100` (R2 -0,806) y deja `forecaster.estimator.alpha=100`; la
  bayesiana minimiza R2; con `direction="maximize"`, `study.best_value` es 0,857 pero se
  reentrena con `alpha=766,6` (R2 -1,084); el clasificador con `zero_one_loss` reentrena con
  `C=0.0001` (0,333 frente a 0,317). Esperado: maximizar R2 (`alpha=0.01`, R2 0,857) y
  respetar `direction`. Evidencia: `_search.py:514` (`ascending=True if is_regression`).
- **BUG-01.** Observado (h2o, LGBM, lags=12): `fit(y)` y luego `TimeSeriesFold(steps=12)` da
  MAE 0,0291 con 192 de 192 predicciones dentro de `training_range_` y 0 avisos; en el mismo
  periodo, 0,0384 in-sample frente a 0,0966 fuera de muestra (2,5 veces); multiserie 0,0378
  frente a 0,1047. Esperado: aviso con el número y rango de predicciones in-sample que
  recomiende `initial_train_size`; el uso intencionado (`backtesting.ipynb`, "Backtesting on
  training data") sigue funcionando con `suppress_warnings=True`.
- **BUG-02.** Observado (bike_sharing horario, LGBM, steps=24): `fold_stride=1` devuelve MAE
  35,37 (igual al del paso 1) frente a 72,62 sobre todos los pares origen-fecha; solo usa 720
  de 17004 predicciones; `grid_search` elige `lags=24` (real 72,62) en lugar de los lags
  estacionales (real 58,97, un 23% mejor); multiserie igual. Esperado: métrica del horizonte
  configurado (72 a 76) sea cual sea `fold_stride`, o al menos un aviso. La skill pública
  recomienda `fold_stride=1` (`skills/backtesting-configuration/SKILL.md:202-208`).
  Origen de `fold_stride`: https://github.com/skforecast/skforecast/issues/764.
- **BUG-04.** Observado (Chronos-2 small real ajustado 30 pasos): `FoundationModel` ajustado
  da `[10.231 9.79 9.795 ...]`; `ForecasterFoundation` con ese modelo da
  `[12.083 12.797 11.061 ...]`, idéntico al base (diferencia máxima 3,0 en una serie de nivel
  ~10); `adapter._pipeline is None`, 0 avisos; `deepcopy_forecaster` lo vuelve a perder.
  Esperado: predecir con el objeto del usuario (o, como mínimo, avisar). Evidencia:
  `skforecast/foundation/tests/tests_forecaster_foundation/fixtures_forecaster_foundation.py:196-211`
  ya reconoce el problema.
- **BUG-10.** Observado (TabICL real, y alrededor de -16): `predict(context=y, context_exog=exog)`
  sin exog futura da `[5.21 ... 5.06]` frente a `[-17.29 ... -18.22]` sin exog; backtest de 5
  folds MAE 15,5 frente a 3,6; el aviso dice que las columnas se ignoran. Esperado: mismo
  resultado que sin exog, como promete `_foundation_model.py:973-993`.
- **BUG-stats-interval-ignored.** Observado (h2o, `Arima(order=(1, 1, 0))`, medido hoy):
  sin argumentos y con `interval=[0.1, 0.9]` los límites del primer paso son idénticos
  (0,4631 / 1,0723); solo con `alpha=None` sale el 80% (0,5685 / 0,9668). Esperado: el
  intervalo pedido en `interval`. `backtesting_stats` no se ve afectado (`alpha=None` allí).
- **BUG-06.** Observado: Nori falso con serie lognormal, `predict` 1,438 frente a
  `predict_interval` 1,003; MAE de backtesting 1,1693 sin cuantiles frente a 1,0834 con
  cuantiles y sin columna `pred`; TabICL real: el MAE cambia +1,5% (h2o), -4,7%
  (bike_sharing), +3,9% (items_sales), +3,5% (Poisson). Esperado, como en ML:
  `predict_interval()["pred"] == predict()["pred"]` y pedir cuantiles no cambia la métrica.
  TimesFM 2.5 no está afectado.
- **BUG-08.** Observado (800 puntos, m=7, 20 folds, `refit=True`): 20 llamadas a `auto_arima`
  con `freeze_params` True y False (44,8 s y 43,4 s, MAE idéntico 1,559485; 15 órdenes
  distintos en 20 folds); Ets automático 13,1 s y 11,4 s; `grid_search_stats` con 2
  candidatos hace 40 búsquedas (76,4 s). Esperado: una búsqueda; con el arreglo, Arima 8,5 s
  (5,3x) y Ets 0,9 s (15x). Presente desde la 0.20.0.
- **BUG-07.** Observado: MultiIndex creado con `set_index` o `reset_index().set_index()` sobre
  los datos de la guía tiene `levels[1].freq=None` y `fit` lanza "If `series` is a
  dictionary, all series must have ... Found series with no frequency or step"; los rodeos
  dan las mismas predicciones. Esperado: inferir la frecuencia por serie. La guía usa el
  MultiIndex como entrada principal (`independent-multi-time-series-forecasting.ipynb`,
  celdas 5 y 11). No es silencioso, de ahí su posición.
- **BUG-11.** Observado (h2o, `ForecasterEquivalentDate(offset=12)`, medido hoy):
  `interval=[0.1, 0.9]` da `AttributeError: 'ForecasterEquivalentDate' object has no attribute
  'predict_quantiles'`; con `interval_method="conformal"` funciona. Esperado: `ValueError`
  claro o conformal automático.
- **BUG-09.** Observado: `ForecasterStats([Arima, Ets])` con `fold_stride=2` o 1 lanza
  `ValueError: Length of values (2) does not match length of index (1)` tras hacer todos los
  ajustes; Sarimax con `refit=False` y `fold_stride` 1, 3 o 10, o con `refit=3`, lanza
  "`last_window` has to start at the end of the index seen by the forecaster". Esperado: una
  fila por estimador (Arima 6,2311, Ets 8,3975) y predicciones iguales a la referencia
  (diferencia 0,0 con el arreglo).
- **PERF-02.** Observado (1000 puntos, 30 folds): `n_jobs` 1 frente a 3: Arima 6,59 frente a
  2,59 s, Ets automático 23,3 frente a 8,37 s, Sarimax 20,4 frente a 6,73 s, predicciones
  idénticas; el mismo backtest de Ets tardó 194 s con carga externa frente a 24,1 s en
  reposo; `multiprocessing.Pool(4)` con 8 series de Ets 73,6 s frente a 2,56 s con el
  límite BLAS; `setulb` ocupa el 84% del tiempo. Esperado: sin degradación por carga externa
  y paralelismo por defecto donde cada fold es independiente. Precedente aguas arriba:
  https://github.com/scipy/scipy/issues/22438 (optuna 5.0 ya lo parchea).
- **PERF-01.** Observado (chronos-2-small, CPU): 1 serie y 200 folds de 24 pasos, 5,68 s
  frente a 1,68 s agrupado (3,4x), diferencia máxima 3,8e-06 y MAE idéntico; 5 series 1,4x;
  50 series 1,1x (dentro del ruido); con 2000 series simuladas la preparación baja de 17,45 s
  a 2,70 s sin exog y de 26,69 s a 8,13 s con exog. Esperado: una o pocas llamadas
  agrupadas. Solo medido Chronos-2 en CPU; TabICL, TabPFN-TS y Nori solo ganarían con (1)
  y (3).

---

## 5. Para versiones posteriores

| Item | Estado | Motivo y condición para retomarlo |
|---|---|---|
| BUG-05: grid/random con todas las combinaciones fallidas devuelven un DataFrame vacío y el forecaster sin ajustar | Verificado (`later`), `repro/BUG-05.py` | No es silencioso: cada aviso lleva la causa, y el comportamiento es deliberado y testeado desde la 0.21 (commit c66130afa). Pulido barato: `ValueError(...) from first_exc`, como sklearn y `bayesian_search`. QW-03 y QW-07 (d) quitan los disparadores más frecuentes. Requiere acuerdo de mantenedores |
| FE-01 completo: función pública `evaluate_backtesting` / `calculate_backtesting_metrics` por fold, paso, serie y grupos | Verificado con veredicto partido | Lente de valor: tras QW-04, QW-09 y QW-10 aporta 2 o 3 líneas por uso y basta una receta. Lente de riesgo: versión reducida M (sin `y`, sin aviso, sin MASE, total distinto del backtesting con folds solapados). Decidir en 0.28 según demanda (única petición: https://github.com/skforecast/skforecast/issues/720); requiere BUG-02 |
| WAPE (de QW-08) | Verificado (`later`) | Función nueva con tests y docs; sMAPE ya cubre las series con ceros |
| `suppress_warnings` en `ForecasterFoundation.predict*` | Hallazgo de QW-02 | Hueco de consistencia con ML |
| Validar el cv una vez antes del bucle de grid | Hallazgo de QW-03 | Junto con BUG-05 |
| MultiSeries con exog de RangeIndex reseteado da un resultado erróneo en silencio | Hallazgo de QW-06, medido | Poco frecuente; anotar y tratarlo con QW-06 si el diseño lo permite |
| skops en `ForecasterFoundation` | Hallazgo de QW-05 | Posible cuando el backend quede fuera del estado |
| PLAN-CKPT (`finetuned_checkpoint`) y PLAN-TTM (adaptador TinyTimeMixer) | Plan sin código (`dev/PLAN_*.md`), no verificado | BUG-04 es requisito de PLAN-CKPT |
| `grid_search_equivalent_date` multiserie | Fuera del alcance de FE-04 | Después de FE-04 |
| Kalman de Arima estacional más rápido | `work/lens_stats.md` | 14 ms por evaluación, 729 evaluaciones con n=5000 y m=24; esfuerzo L |
| HANDOFF-STATS-LW: `fit(store_last_window=False)` y `predict(last_window=...)` dan `AttributeError` | No verificado | `skforecast/recursive/_forecaster_stats.py:626`; sin decisión |
| HANDOFF-DOUBLE-WARN: doble aviso de NaN en la exog de `predict` de Recursive/Direct | No verificado | Mismo patrón que #1375 en multiserie; junto con QW-06 (mismo camino de `predict`) o con los avisos agregados |
| DISC-1153: asimetría en `RollingFeatures` y errores difíciles de diagnosticar en window features propias | No verificado | https://github.com/skforecast/skforecast/discussions/1153 (14 comentarios); FE-03 reduce parte de la necesidad |
| ISSUE-995: pruning de Optuna en la búsqueda bayesiana | No verificado | https://github.com/skforecast/skforecast/issues/995; pregunta abierta desde 2025-07 |
| TODO-multivariate-gap: formato largo con varios niveles y `gap>0` en multivariante | No verificado | `skforecast/model_selection/_validation.py:1099`; comprobar antes si es un caso real |
| BR-synthetic-data: generador sintético de paneles | WIP, 288 commits por detrás | Rebase y decidir si va en `experimental` o `datasets`; útil para docs y benchmarks |
| ISSUE-974: skforecast como motor de reducción de sktime | No verificado | https://github.com/skforecast/skforecast/issues/974; estratégico, sin actividad |
| Elementos aplazados en el triaje | Ver sección 7 | Avisos agregados, columna de tiempo, calibración conformal foundation, `search_space='auto'`, `static_features`, decaimiento temporal, compilación de Arima |

---

## 6. Ya en curso o identificado

| ID | Referencia | Estado | Acción en 0.27.0 |
|---|---|---|---|
| PR-1400 (rama `perf/ets-numba-compile`) | https://github.com/skforecast/skforecast/pull/1400 | Terminada, etiquetada 0.27.0 | Fusionar: primera compilación de Ets de 12 s a 7 s, mismas estimaciones. Crea la sección 0.27.0 de `releases.md` (conflicto trivial con #1402) |
| PR-1402 (rama `fix/fetch-dataset-website-visits-m5`) | https://github.com/skforecast/skforecast/pull/1402 | Terminada, con tests, etiquetada 0.27.0 | Fusionar: `website_visits` y `m5` sin `DatetimeIndex`; m5 de más de 15 GB a ~10 GB |
| PR-1407 / ISSUE-1406 | https://github.com/skforecast/skforecast/pull/1407, https://github.com/skforecast/skforecast/issues/1406 | PR externo, `mergeable=blocked` | Revisar y fusionar (bloque A): los agregados se calculan sin la serie y sin aviso. Toca el mismo código que QW-04 |
| PR-1408 / ISSUE-1357 | https://github.com/skforecast/skforecast/pull/1408, https://github.com/skforecast/skforecast/pull/1412, https://github.com/skforecast/skforecast/issues/1357 | Resuelto: PR-1408 se cerró sin fusionar el 2026-10-18 y lo sustituyó #1412, fusionado en `0.27.x` el mismo día (#1357 cerrada) | Nada que hacer. Los nombres enteros se conservan como enteros: QW-04 debe usar la serie ya preprocesada |
| PR-1389 / ISSUE-1354 | https://github.com/skforecast/skforecast/pull/1389, https://github.com/skforecast/skforecast/issues/1354 | PR externo | Revisar: `set_in_sample_residuals` con CatBoost y exog categórica |
| PR-1388 / ISSUE-1355 | https://github.com/skforecast/skforecast/pull/1388, https://github.com/skforecast/skforecast/issues/1355 | PR externo | Revisar: búsquedas con `lags=None` y solo window features |
| PR-1387 / ISSUE-1386 | https://github.com/skforecast/skforecast/pull/1387, https://github.com/skforecast/skforecast/issues/1386 | PR externo (+10/-5) | Fusionar: `plot_residuals` ignora `fig` |
| ISSUE-1410 | https://github.com/skforecast/skforecast/issues/1410; `skforecast/model_selection/_utils.py:1006,1027` | Sin PR | Corregir la comparación `<`/`>=` con `window_size` (S). Misma frontera que en FE-04 |
| ISSUE-1385 (T0) | https://github.com/skforecast/skforecast/issues/1385; `skforecast/foundation/_adapters.py:3724-3729` | Sin PR; no reproducido aquí (sin `tfc-t0`) | Adaptar a `tfc-t0>=0.4` (`predict` sin `context` ni `quantiles`; torch como extra desde 0.6.0) |
| ISSUE-1409 + DISC-1119 + GAP-backtesting-interval-metrics | https://github.com/skforecast/skforecast/issues/1409, https://github.com/skforecast/skforecast/discussions/1119 | Sin PR | Se implementan por separado como QW-09 (`coverage` e `interval_width`, #1409) y QW-10 (columna `step`, D1119) (sección 2). Sin aviso de infracobertura (sección 7) |
| GAP-foundation-backtesting-interval | `skforecast/model_selection/_validation.py:2712-2720` (`backtesting_foundation` solo tiene `quantiles`); `skforecast/model_selection/_search.py:2091` (única búsqueda foundation) | Asimetría de API, comprobada en las firmas; sin medir | Aceptar `interval` en `backtesting_foundation` después de BUG-06; `grid_search_foundation` opcional |
| PR-1341 / ISSUE-1190 (TiRex-2) | https://github.com/skforecast/skforecast/pull/1341, https://github.com/skforecast/skforecast/issues/1190 | PR externo +1376 líneas, blocked; no revisado | Decidir: foco foundation y demanda recurrente (TabPFN, T0 y Nori ya entraron así); revisión L |
| ISSUE-1131 (exógenas as-of) | https://github.com/skforecast/skforecast/issues/1131 | Aplazado en marzo "a la release siguiente"; no verificado | Decidir el alcance: FE-02 cubre las variables nunca conocidas a futuro en foundation; las exógenas que son previsiones publicadas (propuesta `exog_builder`) siguen abiertas (L) |
| DEP-stats-alpha | `skforecast/recursive/_forecaster_stats.py:1047-1055`; `backtesting_stats` | Descartado el 2026-10-19 (decisión 5) | Nada: `alpha` queda como alias de `interval` y pasar los dos da `ValueError` (hecho con BUG-stats-interval-ignored) |
| DEP-python310 | `pyproject.toml:26,65`; `.github/workflows/unit-tests-min-deps.yml:30` | Python 3.10 llega a fin de vida en octubre de 2026 (PEP 619) | Anunciar en 0.27 y retirar en 0.28 (precedente: 3.9 en la 0.19.0) |
| DEP-arima-dictstyle | `skforecast/stats/arima/_arima_base.py:271-290` | `FutureWarning` sin versión desde la 0.22; única deprecación activa | Retirar en 0.27 o fijar la versión; valorar usar `runtime_deprecated` (`skforecast/exceptions/exceptions.py:21`), que no se usa |
| DOC-skills-percentiles-stale | `skills/prediction-intervals/SKILL.md:55-57,70,265`; `references/interval-compatibility.md:23,205`; `skills/complete-api-reference/references/forecaster-methods.md:432,455` (`set_params` marcado como ausente en EquivalentDate) | Desactualizado (percentiles dan `ValueError` desde 0.25) | Corregir junto a QW-07 (b) y regenerar el contexto IA |
| HANDOFF-T2 | `dev/handoff_utils_pendiente.md` (T2) | Pendiente, marcado "para la 0.27"; no reverificado | Implementar: categóricas nativas no llegan al estimador dentro de `CalibratedClassifierCV` o `TransformedTargetRegressor` |
| HANDOFF-T3 / HANDOFF-T4 | `dev/handoff_utils_pendiente.md` (T3, T4) | Pendiente de decisión de Joaquín; no reverificado | Decidir; T3 (Pipeline con LightGBM o CatBoost y exog categórica: `fit` falla) es un error claro en un flujo común |
| ISSUE-1356 | https://github.com/skforecast/skforecast/issues/1356 | Sin PR | Baja prioridad: el orden de `X_train_series_names_in_` depende de `encoding`; nada se calcula mal |
| Mantenimiento | rama `ci/minimum-versions-job`; https://github.com/skforecast/skforecast/pull/1404 | Fusionada; PR de dependabot contra `main` | Borrar la rama; fusionar #1404 o reabrirla contra `0.26.x` |

---

## 7. Descartado

**Descartes de la verificación y de fase 1**

- Aviso de infracobertura en backtesting (`ResidualsUsageWarning` si la cobertura queda más de
  0,10 por debajo de la nominal; propuesto en FE-01 y en la primera versión de QW-09):
  descartado definitivamente el 2026-10-18. Con `coverage` en la tabla de métricas no aporta
  información, el umbral y el mínimo de filas son arbitrarios, no está claro si saltaría por
  serie o por agregado, y su consejo (residuos out-of-sample) no basta: LGBM con conformal y
  residuos out-of-sample cubre 0,62 frente al 0,80 nominal.
- DEP-stats-alpha (deprecar `alpha` en favor de `interval` en `ForecasterStats` y
  `backtesting_stats`): descartado el 2026-10-19. Se mantiene como alias para quien viene de
  estadística; la ambigüedad se resuelve con un `ValueError` si se pasan los dos (decisión 5).
- Duplicado "grid/random devuelven un DataFrame vacío" (lente ml-multi): fusionado en BUG-05.
- Duplicado "propagar `differentiation` a `TimeSeriesFold`" (ml-multi): fusionado en QW-03.
- Duplicado "aviso de exog en foundation" (consistency): fusionado en QW-02.
- TODO-reset-forecaster (`_forecaster_recursive.py:1289`, `_forecaster_recursive_multiseries.py:2345`): refactor interno sin valor para el usuario.
- TODO-search-exog-wide (`_search.py:1320,1722`): limpieza interna sin efecto visible.
- TODO-stats-aggregate (`_forecaster_stats.py:219`): funcionalidad nueva en STATS, fuera de foco.
- Vía de modelos locales para `ForecasterStats` en backtesting multiserie (lente de FE-04): funcionalidad nueva en STATS, fuera de foco.
- ISSUE-1084 (https://github.com/skforecast/skforecast/issues/1084, `cudf.pandas`): sin actividad desde abril y sin demanda.
- BR-review-deep-learning e ISSUE-1065: deep learning fuera de alcance por diseño.
- Categorías nuevas de una exógena en `predict` codificadas como NaN sin aviso (lente ml-multi): nicho.
- Fricciones de consistencia en foundation (formato largo con `level` en una sola serie, intervalo por defecto del 80% frente al 90% de ML, nombres `fit(series)`/`context`): cambiarlas rompe código sin ganancia clara.
- Ensembles de forecasters: medido -0,7% de MAE con dos LightGBM; no justifica API nueva.
- Reconciliación jerárquica, diferenciación estacional y ventanas expanding: sin demanda en issues o bajo valor medido (`work/lens_competitors.md`).
- Ya existen (lente competitors): calendario y festivos, `ewm`, Box-Cox y escaladores por serie, diferenciación por serie, estrategias de refit, importancia por horizonte, series nuevas en modelos globales, covariables en foundation.

**Descartados o aplazados en el triaje**

- Agregar los avisos por serie en paneles grandes: esfuerzo M y los avisos responden a errores reales; abordar junto a HANDOFF-DOUBLE-WARN y HANDOFF-T4.
- Columna de tiempo por combinación en grid/random: valor bajo; `bayesian_search` ya lo da en `study.trials_dataframe()["duration"]`.
- Calibración conformal de los cuantiles nativos foundation: esfuerzo M con diseño abierto; primero medir cobertura con QW-09.
- `search_space="auto"`: comodidad de esfuerzo M, rangos opinables y mantenimiento continuo.
- `static_features` en modelos globales: esfuerzo M a L y existe la alternativa de exog en dict.
- Pesos de decaimiento temporal para `weight_func`: nicho; la ganancia depende de la semivida (-2,8% con 365 días, +3,2% con 90) y la alternativa son 3 líneas.
- Compilación numba de Arima en el primer uso (11,5 s): `njit(cache=True)` (`_arima_base.py:578`), se paga una vez por entorno; seguimiento natural de PR-1400.

**Afirmaciones corregidas en la verificación**

- BUG-01: el "23 veces más optimista" no se reproduce; en el mismo periodo son 2,5 veces.
- BUG-06: TimesFM 2.5 no está afectado (timesfm devuelve `q0.5` como punto); lo erróneo es la tabla de `docs/user_guides/foundation-forecasting-models.ipynb:466` ("Mean").
- FE-03: el -4,1% se midió sin lags estacionales y con una clase propia con un bug; frente a la base justa es -2%.
- FE-02: las cifras originales (0,484 / 2,992) no mostraban valor de la covariable; las nuevas son 1,186 / 2,919 / 3,104.
- PERF-01: 3,4x (no 3,6x) con una serie; en CPU solo 1,1x con 50 series.
- PERF-02: `threadpool_limits` cuesta 1,45 ms por llamada, no 0,1 ms.
- QW-07 (d): el caso grave es el DataFrame vacío sin excepción; `backtesting_forecaster` con Foundation ya da `TypeError`.
- BUG-07: la exog larga creada con `set_index` ya funciona; el mensaje sí menciona el MultiIndex.
- FE-04: iterar `backtesting_forecaster` por nivel con el mismo cv reproduciría el desalineamiento de pliegues.

### Cobertura del análisis (lo que no se pudo comprobar)

- **Deep learning** (`deep_learning`, `ForecasterRnn`): fuera de alcance por diseño; no se analizó la rama `review/deep-learning-module`.
- **GitHub**: GraphQL y la API de búsqueda (`gh search`, `/search/issues`) bloqueadas en la sesión. Issues y PR se listaron por REST y MCP; las discusiones se leyeron por REST (34 en total, solo 3 en los últimos 9 meses: D1119, D1153, D1191). La búsqueda de duplicados y de demanda se hizo con la búsqueda semántica del MCP y con los títulos de 1376 issues y PR; puede faltar demanda no reportada con esos términos.
- **Foundation**: solo se ejecutaron con pesos reales `autogluon/chronos-2-small` (CPU) y TabICL 2.2.0 (instalado con `uv pip install --no-deps`). TimesFM 2.5/3.0, Moirai-2, TabPFN-TS, T0, Nori y TS-ICL no se probaron con backend real: BUG-04, QW-05 y BUG-06 se extienden a ellos leyendo el código (mismo patrón), y TimesFM 2.5 en QW-05 se probó con un paquete simulado. BUG-10 en TabPFN-TS no está medido. ISSUE-1385 (T0) no se reprodujo. Sin GPU: las ganancias de PERF-01 en GPU y la carga en CPU de un pickle hecho en GPU no están medidas.
- **Rendimiento**: 4 núcleos compartidos con otros agentes; se repitieron las medidas contaminadas y se descartaron las que no se pudieron repetir (Sarimax con m=24 y n=5000). El problema BLAS solo se midió con scipy 1.18.1 (no con scipy < 1.15).
- **ML**: no hay cifras de cobertura con RandomForest (escenario interrumpido por CPU); `random_search` multiserie no se ejecutó (comparte el patrón de grid); `ForecasterDirectMultiVariate` se revisó poco (lags dict, intervalos, NaN al inicio, `differentiation`). Categóricas en exog no se revisaron (cubiertas por HANDOFF-T2/T3/T4, que no se reverificaron).
- **Competidores**: funciones de mlforecast y darts comprobadas en la web; las de autogluon, sktime y neuralforecast, de memoria. Los experimentos usan un solo dataset por caso (bike_sharing o vic_electricity), así que las mejoras de FE-03 son indicativas.
- **PR externos** (#1341, #1388, #1389, #1407, #1408): inventariados, no revisados línea a línea.
