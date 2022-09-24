![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
![PyPI](https://img.shields.io/pypi/v/skforecast)
[![codecov](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast)
[![Build status](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
![Licence](https://img.shields.io/badge/Licence-MIT-green)
[![Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/skforecast)


# skforecast

<p><img src="./images/logo_skforecast_no_background.png" alt="logo-skforecast" title="logo-skforecast" width="200" align="right"></p>

**Time series forecasting with scikit-learn regressors.**

**Skforecast** is a python library that eases using scikit-learn regressors as multi-step forecasters. It also works with any regressor compatible with the scikit-learn API (pipelines, CatBoost, LightGBM, XGBoost, Ranger...).

**Documentation: https://joaquinamatrodrigo.github.io/skforecast/**


# Installation

```bash
pip install skforecast
```

Specific version:

```bash
pip install skforecast==0.4.3
```

Latest (unstable):

```bash
pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```


# Dependencies

+ numpy>=1.20, <=1.23
+ pandas>=1.2, <=1.4
+ tqdm>=4.57.0, <=4.64
+ scikit-learn>=1.0, <=1.1.2
+ statsmodels>=0.12, <=0.13.2
+ matplotlib>=3.3, <=3.5
+ seaborn==0.11.2
+ optuna==2.10.0
+ scikit-optimize==0.9.0
+ joblib>=1.1.0, <=1.2.0


# Features

+ Create recursive autoregressive forecasters from any regressor that follows the scikit-learn API
+ Create direct autoregressive forecasters from any regressor that follows the scikit-learn API
+ Create multi-series autoregressive forecasters from any regressor that follows the scikit-learn API
+ Include exogenous variables as predictors
+ Include custom predictors (rolling mean, rolling variance ...)
+ Multiple backtesting methods for model validation
+ Grid search, random search and bayesian search to find optimal lags (predictors) and best hyperparameters
+ Include custom metrics for model validation and grid search
+ Prediction interval estimated by bootstrapping and quantile regression
+ Get predictor importance
+ Forecaster in production

## What is new in skforecast 0.5.0?

- [x] Random search and bayesian search (using [**optuna**](https://optuna.org/) or [**skopt**](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html)) for hyperparameter optimization. Try `random_search_forecaster()` and `bayesian_search_forecaster()`.
- [x] `ForecasterAutoregMultiOutput` has been renamed to `ForecasterAutoregDirect`.
- [x] Modeling multiple time series simultaneously. Try `ForecasterAutoregMultiSeries` and the new functions available in `model_selection_multiseries`.
- [x] Allow different transformations for each predictor (lags and exogenous).
- [x] Multiple metrics in `backtesting_forecaster`, `grid_search_forecaster`, `random_search_forecaster`, and `backtesting_forecaster_multiseries`.
- [x] Bug fixes and performance improvements.

Try it:

```bash
pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```

Visit [changelog](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/changelog.md) to view all notable changes.


# Documentation

The documentation for the latest release is at [skforecast docs
](https://joaquinamatrodrigo.github.io/skforecast/).

Recent improvements are highlighted in the [release notes](https://joaquinamatrodrigo.github.io/skforecast/latest/releases/releases.html).

+ [Introduction to time series and forecasting](https://joaquinamatrodrigo.github.io/skforecast/latest/quick-start/introduction-forecasting.html)

+ [Recursive multi-step forecasting](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/autoregresive-forecaster.html)

+ [Multi-time series forecasting](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/multi-time-series-forecasting.html)

+ [Backtesting (validation) of forecasting models](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/backtesting.html)

+ [Hyperparameter tuning and lags selection of forecasting models](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/hyperparameter-tuning-and-lags-selection.html)

+ [Prediction intervals](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/prediction-intervals.html)

+ [Using forecaster in production](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/forecaster-in-production.html)


# Examples and tutorials 

**English**

+ [**Skforecast: time series forecasting with Python and Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html)      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Ftime-series-forecasting-python-scikitlearn.ipynb)

+ [**Forecasting electricity demand with Python**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-electricity-power-demand-python.ipynb)

+ [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-web-traffic-machine-learning.ipynb)

+ [**Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.ipynb)

+ [**Bitcoin price prediction with Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-cryptocurrency-bitcoin-machine-learning-python.html) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-cryptocurrency-bitcoin-machine-learning-python.ipynb) 

+ [**Prediction intervals in forecasting models**](https://www.cienciadedatos.net/documentos/py42-forecasting-prediction-intervals-machine-learning.html) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-prediction-intervals-machine-learning.ipynb)

**Español**

+ [**Skforecast: forecasting series temporales con Python y Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html)      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-series-temporales-python-scikitlearn.ipynb)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html)      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-demanda-energia-electrica-python.ipynb)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html)      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-visitas-web-machine-learning.ipynb)

+ [**Forecasting series temporales con gradient boosting: Skforecast, XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html)      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.ipynb)

+ [**Predicción del precio de Bitcoin con Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-criptomoneda-bitcoin-machine-learning-python.html)      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fforecasting-criptomoneda-bitcoin-machine-learning-python.ipynb)

+ [**Workshop predicción de series temporales con machine learning Universidad de Deusto / Deustuko Unibertsitatea**](https://youtu.be/MlktVhReO0E)

+ [**Intervalos de predicción en modelos de forecasting**](https://www.cienciadedatos.net/documentos/py42-intervalos-prediccion-modelos-forecasting-machine-learning.html)      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoaquinAmatRodrigo/skforecast/master?labpath=examples%2Fintervalos-prediccion-modelos-forecasting-machine-learning.ipynb)


# Donating

If you found skforecast useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks!

[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)


# Citation

If you use this software, please cite it using the following metadata.

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. skforecast (Version 0.5.0) [Computer software]
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
license = {MIT},
month = {9},
title = {{skforecast}},
version = {0.5.0},
year = {2022}
}
```

View [citation file](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CITATION.cff).


# License

**joaquinAmatRodrigo/skforecast** is licensed under the **MIT License**, a short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.
