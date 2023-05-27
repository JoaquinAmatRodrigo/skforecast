<script src="https://kit.fontawesome.com/d20edc211b.js" crossorigin="anonymous"></script>

<img src="img/banner-landing-page-skforecast.png#only-light" align="left"  style="
    margin-bottom: 30px;
    margin-top: 0px;">

<img src="img/banner-landing-page-dark-mode-skforecast.png#only-dark" align="left" style="
    margin-bottom: 30px;
    margin-top: 0px;">

![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/)
[![codecov](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast)
[![Build status](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/graphs/commit-activity)
[![License](https://img.shields.io/github/license/JoaquinAmatRodrigo/skforecast)](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/skforecast)


**Skforecast** is a Python library that eases using scikit-learn regressors as single and multi-step forecasters. It also works with any regressor compatible with the scikit-learn API (pipelines, CatBoost, LightGBM, XGBoost, Ranger...).

**Why use skforecast?**

The fields of statistics and machine learning have developed many excellent regression algorithms that can be useful for forecasting, but applying them effectively to time series analysis can still be a challenge. To address this issue, the skforecast library provides a comprehensive set of tools for training, validation and prediction in a variety of scenarios commonly encountered when working with time series. The library is built using the widely used scikit-learn API, making it easy to integrate into existing workflows. With skforecast, users have access to a wide range of functionalities such as feature engineering, model selection, hyperparameter tuning and many others. This allows users to focus on the essential aspects of their projects and leave the intricacies of time series analysis to skforecast. In addition, skforecast is developed according to the following priorities:

+ Fast and robust prototyping. :zap:
+ Validation and backtesting methods to have a realistic assessment of model performance. :mag:
+ Models must be deployed in production. :hammer:
+ Models must be interpretable. :crystal_ball:


## Installation

The default installation of skforecast only installs hard dependencies.

```bash
pip install skforecast
```

Specific version:

```bash
pip install skforecast==0.8.1
```

Latest (unstable):

```bash
pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```

Install the full version (all dependencies):

```bash
pip install skforecast[full]
```

Install optional dependencies:

```bash
pip install skforecast[sarimax]
```

```bash
pip install skforecast[plotting]
```


## Dependencies

+ Python >= 3.8

### Hard dependencies

+ numpy>=1.20, <1.25
+ pandas>=1.2, <2.1
+ tqdm>=4.57.0, <4.65
+ scikit-learn>=1.0, <1.3
+ optuna>=2.10.0, <3.2
+ joblib>=1.1.0, <1.3.0

### Optional dependencies

+ matplotlib>=3.3, <3.8
+ seaborn>=0.11, <0.13
+ statsmodels>=0.12, <0.14
+ pmdarima>=2.0, <2.1


## Features

+ Create recursive autoregressive forecasters from any regressor that follows the scikit-learn API
+ Create direct autoregressive forecasters from any regressor that follows the scikit-learn API
+ Create multi-time series autoregressive forecasters from any regressor that follows the scikit-learn API
+ Create multivariate autoregressive forecasters from any regressor that follows the scikit-learn API
+ Include exogenous variables as predictors
+ Include custom predictors (rolling mean, rolling variance ...)
+ Multiple backtesting methods for model validation
+ Grid search, random search and Bayesian search to find optimal lags (predictors) and best hyperparameters
+ Include custom metrics for model validation and grid search
+ Prediction interval estimated by bootstrapping and quantile regression
+ Get predictor importance
+ Forecaster in production


## Examples and tutorials

### English

<i class="fa-duotone fa-chart-line fa" style="font-size: 25px; color:#1DA1F2;"></i>  [**Skforecast: time series forecasting with Python and Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html)

<i class="fa-duotone fa-lightbulb fa" style="font-size: 25px; color:#fcea2b;"></i> [**Forecasting electricity demand with Python**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html)

<i class="fa-duotone fa-rss fa" style="font-size: 25px; color:#666666;"></i> [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html)

<i class="fa-solid fa-bicycle fa" style="font-size: 25px; color:#00cc99;"></i> [**Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)

<i class="fa-brands fa-bitcoin fa" style="font-size: 25px; color:#f7931a;"></i> [**Bitcoin price prediction with Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-cryptocurrency-bitcoin-machine-learning-python.html)

<i class="fa-light fa-chart-line fa" style="font-size: 25px; color:#f26e1d;"></i>  [**Prediction intervals in forecasting models**](https://www.cienciadedatos.net/documentos/py42-forecasting-prediction-intervals-machine-learning.html)

<i class="fa-duotone fa-water fa" style="font-size: 25px; color:teal;"></i> [**Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast.html)

<i class="fa-solid fa-virus-covid" style="font-size: 25px; color:red;"></i> [**Reducing the influence of Covid-19 on time series forecasting models**](https://www.cienciadedatos.net/documentos/py45-weighted-time-series-forecasting.html)

<i class="fa-solid fa-magnifying-glass" style="font-size: 25px; color:purple;"></i> [**Forecasting time series with missing values**](https://www.cienciadedatos.net/documentos/py46-forecasting-time-series-missing-values.html)

<i class="fa-solid fa-wave-square" style="color: #fbbb09;"></i> [**Intermittent demand forecasting**](https://www.cienciadedatos.net/documentos/py48-intermittent-demand-forecasting.html)


### Español

<i class="fa-duotone fa-chart-line fa" style="font-size: 25px; color:#1DA1F2;"></i> [**Skforecast: forecasting series temporales con Python y Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html)

<i class="fa-duotone fa-lightbulb fa" style="font-size: 25px; color:#fcea2b;"></i> [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html)

<i class="fa-duotone fa-rss fa" style="font-size: 25px; color:#666666;"></i>  [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html)

<i class="fa-solid fa-bicycle fa" style="font-size: 25px; color:#00cc99;"></i> [**Forecasting series temporales con gradient boosting: Skforecast, XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html)

<i class="fa-brands fa-bitcoin fa" style="font-size: 25px; color:#f7931a;"></i> [**Predicción del precio de Bitcoin con Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-criptomoneda-bitcoin-machine-learning-python.html)

<i class="fa-brands fa-youtube" style="font-size: 25px; color:#c4302b;"></i> [**Workshop predicción de series temporales con machine learning 
Universidad de Deusto / Deustuko Unibertsitatea**](https://youtu.be/MlktVhReO0E)

<i class="fa-light fa-chart-line fa" style="font-size: 25px; color:#f26e1d;"></i>  [**Intervalos de predicción en modelos de forecasting**](https://www.cienciadedatos.net/documentos/py42-intervalos-prediccion-modelos-forecasting-machine-learning.html)

<i class="fa-duotone fa-water fa" style="font-size: 25px; color:teal;"></i> [**Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast-español.html)

<i class="fa-solid fa-wave-square" style="color: #fbbb09;"></i> [**Predicción demanda intermitente**](https://www.cienciadedatos.net/documentos/py48-forecasting-demanda-intermitente.html)


## How to contribute

For more information on how to contribute to skforecast, see our [Contribution Guide](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CONTRIBUTING.md).


## Donating

If you found skforecast useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks! :hugging_face: :heart_eyes:

[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)


## Citation

If you use this software, please cite it using the following metadata.

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. skforecast (Version 0.8.1) [Computer software]
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
license = {MIT},
month = {5},
title = {{skforecast}},
version = {0.8.1},
year = {2023}
}
```

View the [citation file](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CITATION.cff).


## License

[BSD 3-Clause License](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE)