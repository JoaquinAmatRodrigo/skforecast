[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![Licence](https://img.shields.io/badge/Licence-MIT-green)
[![Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/skforecast)
![PyPI](https://img.shields.io/pypi/v/skforecast)

![Build status](https://github.com/JoaquinAmatRodrigo/skforecast/workflows/unit-tests.yml/badge.svg)

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

+ numpy>=1.20, <=1.22
+ pandas>=1.2, <=1.4
+ tqdm>=4.57.0, <=4.62
+ scikit-learn>=1.0, <=1.1.1
+ statsmodels>=0.12, <=0.13.2
+ matplotlib==3.5.2
+ seaborn==0.11.2
+ optuna==2.10.0
+ scikit-optimize==0.9.0


# Features

+ Create recursive autoregressive forecasters from any regressor that follows the scikit-learn API
+ Create multi-output autoregressive forecasters from any regressor that follows the scikit-learn API
+ Grid search to find optimal hyperparameters
+ Grid search to find optimal lags (predictors)
+ Include exogenous variables as predictors
+ Include custom predictors (rolling mean, rolling variance ...)
+ Multiple backtesting methods for model validation
+ Include custom metrics for model validation
+ Prediction interval estimated by bootstrapping
+ Get predictor importance

## What is coming in the new release?

- [x] Random search and bayesian search (using [**optuna**](https://optuna.org/) or [**skopt**](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html)) for hyperparameter optimization. Try `random_search_forecaster()` and `bayesian_search_forecaster()`.
- [x] `ForecasterAutoregMultiOutput` has been renamed to `ForecasterAutoregDirect`.
- [x] Modeling multiple time series simultaneously. Try `ForecasterAutoregMultiSeries` and the new functions availables in `model_selection_multiseries`.
- [ ] Allow different transformations for each predictor (lags and exogenous).
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

+ [Introduction to time series and forecasting](https://joaquinamatrodrigo.github.io/skforecast/0.4.3/quick-start/introduction-forecasting.html)

+ [Recursive multi-step forecasting](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/autoregresive-forecaster.html)

+ [Backtesting (validation) of forecasting models](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/backtesting.html)

+ [Grid search of forecasting models](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/grid-search-forecaster.html)

+ [Prediction intervals](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/prediction-intervals.html)

+ [Using forecaster in production](https://joaquinamatrodrigo.github.io/skforecast/latest/user_guides/forecaster-in-production.html)


# Examples and tutorials 

**English**

+ [**Skforecast: time series forecasting with Python and Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html)

+ [**Forecasting electricity demand with Python**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html)

+ [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html)

+ [**Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)

+ [**Bitcoin price prediction with Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-cryptocurrency-bitcoin-machine-learning-python.html)

+ [**Prediction intervals in forecasting models**](https://www.cienciadedatos.net/documentos/py42-forecasting-prediction-intervals-machine-learning.html)

**Español**

+ [**Skforecast: forecasting series temporales con Python y Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html)

+ [**Forecasting series temporales con gradient boosting: Skforecast, XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html)

+ [**Predicción del precio de Bitcoin con Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-criptomoneda-bitcoin-machine-learning-python.html)

+ [**Workshop predicción de series temporales con machine learning Universidad de Deusto / Deustuko Unibertsitatea**](https://youtu.be/MlktVhReO0E)

+ [**Intervalos de predicción en modelos de forecasting**](https://www.cienciadedatos.net/documentos/py42-intervalos-prediccion-modelos-forecasting-machine-learning.html)


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
title = {{skforecast}},
version = {0.5.0}
}
```

View [citation file](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CITATION.cff).


# License

**joaquinAmatRodrigo/skforecast** is licensed under the **MIT License**, a short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.
