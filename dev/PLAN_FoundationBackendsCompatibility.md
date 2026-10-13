# Plan: workflow de compatibilidad de los backends de foundation models

## Contexto

Cada adapter de `skforecast/foundation/_adapters.py` envuelve una librería externa
(`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`, `tabpfn-time-series`,
`tfc-t0`, `synthefy-nori`, `tsicl`) que publica versiones con frecuencia y cambia su
API sin aviso. Hoy no hay nada que detecte esas roturas:

- Los tests unitarios usan fakes que copian la API del backend en el momento de
  escribir el adapter (`FakeT0Forecaster.predict(self, context, horizon, quantiles,
  ...)` en `tests_foundation_models/fixtures_adapters.py:447`). Si el backend cambia,
  los tests siguen en verde.
- La user guide `foundation-forecasting-models.ipynb` solo se ejecuta en las releases,
  necesita varios entornos (Moirai no convive con el resto) y es lenta en CPU.

Caso real (2026-10-09): `tfc-t0` 0.4.0 (2026-09-15) renombró `context` a
`model_input` y 0.5.0 renombró `quantiles` a `quantile_levels`; `T0Adapter` de
skforecast 0.26.0 falla con `TypeError` desde entonces y nadie lo supo hasta ejecutar
la guía. La 0.6.0 (2026-10-13) mantiene ambos nombres nuevos. Borrador de la issue en
`dev/issue_t0_tfc_t0_compat.md`.

## Objetivo

Un único workflow de GitHub Actions que, para cada adapter y en su propio entorno,
compruebe que el adapter funciona con la versión del backend, y muestre el resultado
en una sola tabla por ejecución.

Decisiones ya tomadas:

- **Un solo workflow** (`.github/workflows/foundation-backends.yml`) con una
  **matriz**: un job por adapter (y variante), cada uno con su venv, más un job
  `summary` que agrega todo en una tabla en la página de la ejecución.
  Motivos frente a un único job en bucle: paralelismo, un fallo (OOM, cuelgue) no
  tumba al resto (`fail-fast: false`), se puede relanzar solo el job fallido y los
  logs quedan separados.
- **Dos niveles de test**: contrato (sin pesos, segundos) y smoke (pesos reales,
  minutos).
- **Dos variantes de entorno**: `latest` (última versión del backend, detecta
  roturas nuevas) y `pinned` (versiones conocidas como buenas, para saber si el fallo
  es nuestro o del backend).
- **Los tests de backends viven fuera del paquete**, en `tools/foundation_compat/tests/`.
  Quedan fuera de `testpaths = ["skforecast"]`, así que la suite normal nunca los
  recoge y no hace falta ninguna lógica de saltos.
- **Sin Dependabot por ahora**: los pins de `pinned` se suben a mano cuando la tabla
  muestra `latest` en verde con una versión más nueva (ver "Fuera de alcance").
- La user guide deja de ser el test; queda como documentación que se ejecuta en las
  releases con los mismos entornos.

## Estructura de ficheros

```
tools/foundation_compat/
├── adapters.toml                    # Fuente única: adapter -> paquete, pins, overrides, secretos
├── run.py                           # Crea el venv, instala, lanza pytest y escribe results/<job>.json
├── summarize.py                     # Lee results/*.json y escribe la tabla Markdown
└── tests/
    ├── conftest.py                  # Opción --backend <adapter> (obligatoria) y --level
    ├── contracts.py                 # BACKEND_CONTRACTS: qué importa y llama cada adapter
    ├── test_contract.py             # Nivel 1
    └── test_smoke.py                # Nivel 2

skforecast/foundation/tests/tests_foundation_models/
└── test_backends_compat_config.py   # Unitario: adapters.toml cubre el registro

.github/workflows/foundation-backends.yml
```

Principio: **el workflow es fino**. Toda la lógica está en `run.py`, de modo que el
mismo comando reproduce en local lo que hace CI:

```bash
python tools/foundation_compat/run.py --adapter t0 --variant latest
python tools/foundation_compat/run.py --adapter moirai --variant pinned --level contract
```

Salvedad: `t0` y `timesfm` (3.0) traen una variante MLX con firmas distintas (por
ejemplo, el `from_pretrained` de `TimesFM3Forecaster` en MLX no tiene `device`). En
macOS el resultado local puede no coincidir con el de CI (Linux, torch); la
referencia es CI.

## Paso 1: `adapters.toml` (fuente única)

Una entrada por adapter, con todo lo que `run.py` necesita **sin importar
skforecast** (se ejecuta antes de que exista el venv, y `_ADAPTER_REGISTRY` va por
prefijo de `model_id`, no por adapter).

```toml
[chronos]
adapter = "ChronosAdapter"
backend_package = "chronos-forecasting"       # Igual que ChronosAdapter.backend_package
model_id = "autogluon/chronos-2-small"
pinned = ["chronos-forecasting==2.3.2", "torch==2.9.1"]
exclude_newer = "2026-10-09"                  # Congela las dependencias transitivas

[moirai]
adapter = "MoiraiAdapter"
backend_package = "uni2ts"
model_id = "Salesforce/moirai-2.0-R-small"
overrides = ["scipy>=1.12"]                   # uni2ts fija scipy<1.12; uv --override
pinned = ["uni2ts==2.0.0", "torch==2.4.1", "numpy==1.26.4"]
exclude_newer = "2026-10-09"

[tabpfn]
adapter = "TabPFNAdapter"
backend_package = "tabpfn-time-series>=1.3"
model_id = "priorlabs/tabpfn-ts"
secrets = ["TABPFN_TOKEN"]                    # Sin el secreto: smoke "skipped (no token)"
init_kwargs = { mode = "local" }
pinned = ["tabpfn-time-series==1.3.0", "tabpfn==9.1.0", "torch==2.9.1"]
exclude_newer = "2026-10-09"

# timesfm25, timesfm3, tabicl, t0, nori, tsicl ...
```

Test unitario normal (`test_backends_compat_config.py`, sin backends, corre con la
suite de siempre):

- Cada clase de `_ADAPTER_REGISTRY` tiene al menos una entrada en `adapters.toml`.
- `backend_package` de la entrada coincide con el atributo `backend_package` de la
  clase.
- `_resolve_adapter(model_id)` devuelve la clase indicada en `adapter`.

Así un adapter nuevo, o un cambio de `backend_package`, no puede quedarse fuera del
workflow. Leer el TOML con `tomllib` (o `tomli` en Python 3.10, ya está en el extra
`test`).

## Paso 2: `run.py`

Para un `--adapter` y `--variant`:

1. `uv venv` en un directorio temporal con Python 3.12.
2. Instalar **en una sola resolución** (dos pasos permiten que el segundo cambie en
   silencio numpy, pandas o scipy):

   ```bash
   uv pip install -e . pytest <backend> --torch-backend cpu [--override ...] [--exclude-newer ...]
   ```

   - `-e .` y `pytest`, **no** `.[test]`: ese extra trae lightgbm, xgboost, catboost,
     keras y statsmodels, que alargan la instalación y añaden restricciones que
     pueden retener el backend en una versión antigua.
   - `--torch-backend cpu` evita las ruedas CUDA (varios GB).
   - `latest`: `<backend>` es `backend_package`.
   - `pinned`: `<backend>` son las líneas de `pinned`, más
     `--exclude-newer <exclude_newer>`.
   - Si hay `overrides`, se escriben en un fichero temporal y se pasa `--override`.
3. Guardar en el JSON las versiones instaladas del backend, torch, numpy, scipy y
   huggingface_hub (`importlib.metadata`, ejecutado con el Python del venv).
4. **Comprobar que `latest` es la última**: consultar
   `https://pypi.org/pypi/<paquete>/json` y guardar la última versión publicada. Si
   la instalada es menor, el resolver la ha retenido: se marca como `held_back` en
   el JSON y cuenta como fallo en el cron (un job verde que prueba una versión vieja
   es justo lo que este workflow debe evitar). El nombre del paquete se saca de
   `backend_package` quitando extras y especificadores.
5. Lanzar el contrato siempre. Si falta un secreto requerido, el smoke se escribe
   como `skipped` con el motivo y no cuenta como fallo.
6. `pytest tools/foundation_compat/tests --backend <adapter>
   --level <contract|smoke|all> --junitxml ...`.
7. Escribir `results/<adapter>-<variant>.json`: adapter, variante, versiones
   (instalada y última en PyPI), estado de la instalación y de cada nivel
   (`passed`/`failed`/`skipped`), primera línea del error de cada test fallido y
   duración.
8. Código de salida: distinto de cero si falla la instalación, pytest o hay
   `held_back`.

A tener en cuenta:

- Si una instalación no resuelve (como `uni2ts` sin override), el JSON debe recoger
  `install: failed` y el mensaje de uv, no solo un job en rojo sin explicación.
- Fijar `HF_HUB_DISABLE_PROGRESS_BARS=1` y `TQDM_DISABLE=1` para que los logs se
  puedan leer.

## Paso 3: tests de contrato (nivel 1)

`contracts.py` describe, por adapter, cada símbolo del backend que usa el adapter y
los argumentos con nombre que le pasa. Estado en 0.26.0, contrastado con las firmas
de las últimas versiones en PyPI (2026-10-13):

| Adapter | Símbolo | Llamada y argumentos con nombre | Contrato |
|---|---|---|---|
| Chronos | `chronos.Chronos2Pipeline` | `predict_quantiles(inputs, prediction_length, quantile_levels)` | Débil: `cross_learning` va por `**predict_kwargs` |
| TimesFM 2.5 | `timesfm.TimesFM_2p5_200M_torch`, `timesfm.ForecastConfig` | `forecast(horizon, inputs)`; `ForecastConfig(max_context, max_horizon)` | Fuerte |
| TimesFM 3.0 | `timesfm.TimesFM3Forecaster` | `predict_batch(contexts, horizon, past_only_covariates, past_future_covariates, return_quantiles, padding_mode)` | Fuerte |
| Moirai | `uni2ts.model.moirai2.Moirai2Forecast`, `Moirai2Module` | `Moirai2Forecast(module, prediction_length, context_length, target_dim, feat_dynamic_real_dim, past_feat_dynamic_real_dim)`; `hparams_context(prediction_length)`; `predict(past_target)` | Fuerte |
| TabICL | `tabicl.forecast.TabICLForecaster` | `__init__(max_context_length, temporal_features, point_estimate, tabicl_config)`; `predict_df(context_df, future_df, quantiles)` | Fuerte |
| TabPFN-TS | `tabpfn_time_series.TabPFNTSPipeline`, `TabPFNMode` | `__init__(max_context_length, tabpfn_mode, tabpfn_output_selection, tabpfn_model_config, temporal_features)`; `predict_df(context_df, future_df, quantiles)`; `TabPFNMode.LOCAL`, `TabPFNMode.CLIENT` | Fuerte |
| T0 | `t0.T0Forecaster` | `predict(context, horizon, quantiles, future_covariates)` | Fuerte (hoy en rojo) |
| Nori | `synthefy_nori.NoriRegressor` | `__init__(model)`; `fit(X, y)`; `predict(X, output_type, quantiles)` | Fuerte |
| TS-ICL | `tsicl.TSICL` | `__init__(checkpoint_version, allow_auto_download)`; `forecast(inputs, prediction_length, quantile_levels, context_length, device, denormalize, squeeze_output)` | Fuerte |

Notas sobre la tabla:

- **Sin `from_pretrained`**: los cuatro que se usan (Chronos, TimesFM 2.5, Moirai,
  T0) son genéricos (`**kwargs` de `huggingface_hub`) y se llaman con argumentos
  posicionales; no aportan nada al contrato. Solo se comprueba que el atributo
  existe. La excepción es `TimesFM3Forecaster.from_pretrained(device)`, que sí
  declara `device`.
- **Chronos** se comprueba contra `Chronos2Pipeline` (la clase que devuelve
  `BaseChronosPipeline.from_pretrained` para los checkpoints registrados), no contra
  la clase base. `cross_learning` no aparece en ninguna firma, así que un renombrado
  solo lo detecta el smoke.
- El contrato no ve la **estructura de lo que devuelve** el backend
  (`forecast.quantiles` en T0, `model.config.quantiles` en TimesFM 3.0, las columnas
  de `predict_df`, la forma `(n_quantiles, steps)` de Nori). Eso es del smoke.

`test_contract.py`, parametrizado sobre esa tabla:

- Importar el símbolo (falla si se movió o renombró).
- `inspect.signature(fn).bind_partial(**{k: None for k in kwargs})` para cada llamada:
  falla con `TypeError` si un argumento ya no existe. Es exactamente el fallo de T0.
- Si la firma tiene `**kwargs`, `bind_partial` siempre pasa: comprobar además que
  cada argumento está declarado de forma explícita, salvo los marcados como débiles
  en `contracts.py` (hoy solo `cross_learning`).
- Un test más comprueba que el adapter de `--backend` tiene entrada en
  `BACKEND_CONTRACTS` (corre en todos los jobs, así que un adapter sin contrato
  deja su job en rojo).
- Revisar la tabla contra el código antes de darla por buena
  (`grep -n "self._model\.\|self._pipeline\.\|self._forecast_obj\." _adapters.py`).

No descarga pesos ni necesita secretos: corre en segundos.

## Paso 4: smoke tests (nivel 2)

`test_smoke.py`, un único fichero parametrizado por adapter (no un fichero por
adapter), usando solo la API pública (`FoundationModel` + `ForecasterFoundation` +
`backtesting_foundation`), para recorrer el mismo camino que el usuario:

- Datos sintéticos: 2 series horarias de ~300 observaciones (tendencia + estacionalidad
  diaria + ruido, semilla fija) y una exógena numérica.
- `context_length=64`, `steps=12`.
- Casos, activados según `get_model_info(model_id)` (`allow_exog`, etc.):
  - `fit` + `predict` de una serie: forma, índice esperado, valores finitos.
  - `predict_interval([0.1, 0.9])`: `lower_bound <= pred <= upper_bound`.
  - Multi-serie (`levels=None`).
  - Con `exog` si el adapter lo admite.
  - `backtesting_foundation` con 2 folds.
- No se comparan valores numéricos con referencias: cambian con cada versión y
  hardware. Solo invariantes.

Coste esperado: predicciones en CPU de 5 a 25 s por llamada en la ejecución del
2026-10-09 (4 núcleos), más la descarga de pesos. Con 2 folds, unos pocos minutos por
adapter. `timeout-minutes: 30` por job.

## Paso 5: el workflow

```yaml
name: Foundation backends compatibility

on:
  schedule:
    - cron: '0 6 * * 1'            # Lunes
  workflow_dispatch:
    inputs:
      variant: { type: choice, options: [both, latest, pinned], default: both }
  pull_request:
    branches: [main, '*.x']
    paths:
      - 'skforecast/foundation/**'
      - 'tools/foundation_compat/**'
      - '.github/workflows/foundation-backends.yml'

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  backend:
    name: ${{ matrix.adapter }} (${{ matrix.variant }})
    runs-on: ubuntu-latest
    timeout-minutes: 30
    # En PRs, `latest` es informativo: una release de un backend no bloquea el PR
    continue-on-error: ${{ github.event_name == 'pull_request' && matrix.variant == 'latest' }}
    strategy:
      fail-fast: false
      matrix:
        adapter: [chronos, timesfm25, timesfm3, moirai, tabicl, tabpfn, t0, nori, tsicl]
        variant: [latest, pinned]
    steps:
      # checkout, setup-uv (Python 3.12), cache de ~/.cache/huggingface
      # (clave: adapter + model_id), y:
      - run: python tools/foundation_compat/run.py --adapter ${{ matrix.adapter }} --variant ${{ matrix.variant }}
        env:
          TABPFN_TOKEN: ${{ secrets.TABPFN_TOKEN }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
      - uses: actions/upload-artifact@...   # results/*.json, if: always()

  summary:
    needs: backend
    if: always()
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write                          # Solo para el paso de la issue
    steps:
      # download-artifact (todos los results), y:
      - run: python tools/foundation_compat/summarize.py results >> "$GITHUB_STEP_SUMMARY"
      # Si event_name == 'schedule' y hay fallos: crear o actualizar la issue
```

Detalles:

- **Qué rama prueba cada disparador**:
  - `schedule` solo corre en la rama por defecto (`main`), y solo desde que el
    workflow esté en `main`, es decir, tras la release que lo incluya. Prueba el
    código publicado contra los backends nuevos, que es lo que ven los usuarios.
  - `pull_request` corre **las dos variantes** sobre la rama de desarrollo:
    `pinned` es bloqueante (detecta regresiones nuestras) y `latest` es informativo
    (`continue-on-error`; el job sale en verde, el fallo se ve en la tabla del
    resumen). Así un arreglo de un adapter en `X.Y.x` se valida contra `latest`
    antes de la release.
  - `workflow_dispatch` permite lanzarlo a mano sobre cualquier rama.
- La fase 1 no tiene `pinned`: la matriz empieza solo con `latest` y sin
  `continue-on-error`.
- Versiones de las actions: las mismas que ya usa `foundation-models-metadata.yml`
  (`actions/checkout@v7`, `astral-sh/setup-uv@v10.2.0`), `persist-credentials: false`.
- **Secretos**: `TABPFN_TOKEN` y `HF_TOKEN` como secretos del repo. Los jobs
  instalan paquetes de terceros sin fijar con esos secretos en el entorno, así que
  deben ser tokens de **solo lectura de una cuenta dedicada**, con acceso únicamente
  a los repositorios de los modelos. En PRs desde forks no hay secretos: el smoke de
  TabPFN sale como "skipped (no token)", no como fallo.
- **Issue automática** (solo `schedule`): buscar una issue abierta con la etiqueta
  `backend-compat`; si no existe, crearla. Si existe, comentar **solo cuando cambie
  el conjunto de fallos** (guardar en el cuerpo de la issue una huella de
  adapter + variante + nivel fallido y compararla), no cada lunes. La issue indica
  la rama y el commit probados. Cerrarla a mano cuando se arregle (o automáticamente
  cuando el cron salga verde, opcional).

## Paso 6: `summarize.py`

Lee `results/*.json` y escribe una tabla Markdown:

| Adapter | Variante | Backend | Instalada | Última en PyPI | torch | Instalación | Contrato | Smoke | Detalle |
|---|---|---|---|---|---|---|---|---|---|
| T0 | latest | tfc-t0 | 0.6.0 | 0.6.0 | 2.9.1 | ✓ | ✗ | n/a | `predict()`: argumento `context` no existe |
| T0 | pinned | tfc-t0 | 0.3.2 | 0.6.0 | 2.9.1 | ✓ | ✓ | ✓ | |
| Nori | latest | synthefy-nori | 0.18.2 | 0.21.0 | 2.9.1 | retenida | ✓ | ✓ | El resolver no instala la última versión |
| Moirai | latest | uni2ts | 2.0.0 | 2.0.0 | 2.4.1 | ✓ (override scipy) | ✓ | ✓ | |
| TabPFN-TS | latest | tabpfn-time-series | 1.3.0 | 1.3.0 | 2.9.1 | ✓ | ✓ | skipped | sin `TABPFN_TOKEN` |

- Una fila `pinned` en verde con "Instalada" menor que una fila `latest` en verde es
  la señal para subir el pin a mano.
- Debe funcionar aunque falten JSON (job cancelado o caído): esa fila sale como
  "sin resultado", con el enlace al job.

## Paso 7: variante `pinned`

Sin ficheros de requisitos: los pins están en `adapters.toml` (`pinned` y
`exclude_newer`). Fijar solo el backend y torch no es determinista, porque las
dependencias transitivas flotan y ya han roto adapters (`_adapters.py` tiene dos
parches por cambios de `huggingface_hub`, en TimesFM 2.5 y en T0). `exclude_newer`
congela todo lo demás a una fecha con una sola línea.

A verificar al implementarlo: que `uv --exclude-newer` funciona junto con
`--torch-backend cpu` (el índice de PyTorch puede no publicar la fecha de subida de
las ruedas). Si no funciona, alternativa: generar un lock completo por adapter con
`uv pip compile` y guardarlo en `tools/foundation_compat/envs/<adapter>.txt`.

Versiones conocidas como buenas de la ejecución de la guía del 2026-10-09
(skforecast 0.26.0, Python 3.12, CPU), todas con `exclude_newer = "2026-10-09"`:

| Adapter | `pinned` |
|---|---|
| chronos | `chronos-forecasting==2.3.2`, `torch==2.9.1` |
| timesfm25, timesfm3 | `timesfm[torch]==3.0.2`, `torch==2.9.1` |
| moirai | `uni2ts==2.0.0`, `torch==2.4.1`, `numpy==1.26.4`; override `scipy>=1.12` |
| tabicl | `tabicl[forecast]==2.2.0`, `torch==2.9.1` |
| tabpfn | `tabpfn-time-series==1.3.0`, `tabpfn==9.1.0`, `torch==2.9.1` (sin validar: no había token) |
| t0 | `tfc-t0==0.3.2`, `torch==2.9.1` |
| nori | `synthefy-nori==0.18.2`, `torch==2.9.1` |
| tsicl | `tsicl==0.2.1`, `torch==2.9.1` (`tsicl` limita torch a 2.9.x y pide Python >=3.12) |

Pendiente de aclarar: el entorno del 2026-10-09 instaló `synthefy-nori` 0.18.2, pero
en PyPI la última es la 0.21.0 (subida el 2026-09-14). Averiguar qué la retuvo
(`uv pip install synthefy-nori==0.21.0` en el entorno de Nori y leer el conflicto)
antes de decidir el pin; es el primer caso real de la comprobación `held_back`.

Subir un pin: cambiar la versión y la fecha de `exclude_newer` en `adapters.toml` en
un PR normal; el propio workflow lo valida.

## Fases (cada una se puede mergear por separado)

1. **Contrato + workflow + resumen, solo `latest`**: `adapters.toml`, `run.py` (con
   la comprobación de la última versión en PyPI), `contracts.py`,
   `test_contract.py`, `summarize.py`, el workflow y el test unitario de
   `adapters.toml`. Es lo más barato y ya habría detectado T0 el 15 de septiembre.
2. **Smoke tests**: `test_smoke.py`, caché de Hugging Face y secretos.
3. **Variante `pinned`** (pins y `exclude_newer` en `adapters.toml`), `latest`
   informativo en PRs e **issue automática**.
4. Opcional: script para ejecutar la user guide por segmentos con estos mismos
   entornos en cada release (lo que se hizo a mano el 2026-10-09: ejecutar cada
   bloque de modelo en su venv, fusionar las salidas y renumerar `execution_count`).

## Verificación en local antes de subir

- Cada fase: `ruff check` de los ficheros nuevos y `pytest` del test unitario nuevo
  (`test_backends_compat_config.py`) con el entorno normal.
- `run.py` en local para al menos tres adapters representativos:
  - `--adapter t0 --variant latest` → contrato en rojo con el mensaje de `context`.
  - `--adapter t0 --variant pinned` → todo en verde.
  - `--adapter moirai --variant latest` → instalación con override y verde.
  - `--adapter nori --variant latest` → comprobar si sale `held_back`.
- `summarize.py` sobre esos JSON y revisar la tabla.
- `pytest skforecast/foundation/tests -q` no recoge nada de
  `tools/foundation_compat/tests` (no debe afectar a la suite normal ni descargar
  pesos).
- El workflow solo se puede probar de verdad en GitHub: lanzarlo con
  `workflow_dispatch` desde la rama antes de abrir el PR, o con
  [`act`](https://github.com/nektos/act) para la sintaxis.

## Fuera de alcance

- Arreglar `T0Adapter` para `tfc-t0>=0.4` (issue aparte, `dev/issue_t0_tfc_t0_compat.md`).
- Cambiar los fakes de los tests unitarios: siguen siendo útiles para la lógica del
  adapter; el contrato es lo que vigila la API del backend.
- **Dependabot para los pins**: el cron `latest` más la issue ya avisan cada semana
  de las versiones nuevas. Dependabot añadiría un fichero de requisitos por adapter,
  secretos propios (los PRs de Dependabot no ven los secretos normales) y PRs contra
  `main` con mucha frecuencia (`tfc-t0` publicó cuatro versiones en un mes).
  Reconsiderarlo solo si subir los pins a mano resulta molesto.
- GPU: todo en CPU. Los runners con GPU son de pago y no aportan para compatibilidad.
