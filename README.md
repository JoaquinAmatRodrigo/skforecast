<h1 align="left">
<img src="https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/images/banner-landing-page-skforecast.png?raw=true#only-light" style= margin-top: 0px;>
</h1>

![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/)
[![codecov](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast)
[![Build status](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/graphs/commit-activity)
[![License](https://img.shields.io/github/license/JoaquinAmatRodrigo/skforecast)](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/skforecast)


**Skforecast** is a Python library that eases using scikit-learn regressors as single and multi-step forecasters. It also works with any regressor compatible with the scikit-learn API (LightGBM, XGBoost, CatBoost, ...).

**Why use skforecast?**

The fields of statistics and machine learning have developed many excellent regression algorithms that can be useful for forecasting, but applying them effectively to time series analysis can still be a challenge. To address this issue, the skforecast library provides a comprehensive set of tools for training, validation and prediction in a variety of scenarios commonly encountered when working with time series. The library is built using the widely used scikit-learn API, making it easy to integrate into existing workflows. With skforecast, users have access to a wide range of functionalities such as feature engineering, model selection, hyperparameter tuning and many others. This allows users to focus on the essential aspects of their projects and leave the intricacies of time series analysis to skforecast. In addition, skforecast is developed according to the following priorities:

+ Fast and robust prototyping. :zap:
+ Validation and backtesting methods to have a realistic assessment of model performance. :mag:
+ Models must be deployed in production. :hammer:
+ Models must be interpretable. :crystal_ball:

**Documentation: https://skforecast.org** :books:


# Installation

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

# Dependencies

+ Python >= 3.8

## Hard dependencies

+ numpy>=1.20, <1.25
+ pandas>=1.2, <2.1
+ tqdm>=4.57.0, <4.65
+ scikit-learn>=1.0, <1.3
+ optuna>=2.10.0, <3.2
+ joblib>=1.1.0, <1.3.0

## Optional dependencies

+ matplotlib>=3.3, <3.8
+ seaborn>=0.11, <0.13
+ statsmodels>=0.12, <0.14
+ pmdarima>=2.0, <2.1

## What is new in skforecast 0.8.1?

Visit the [release notes](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/changelog.md) to view all notable changes.

- [x] Support for `pandas 2.0.x`.
- [x] New user guide on how to include **categorical variables** in the Forecasters.
- [x] New user guide on how to use **GPU in Google Colab** with XGBoost and LightGBM regressors.
- [x] Include custom kwargs during fit.
- [x] The dtypes of exogenous variables are maintained when generating the training matrices with the `create_train_X_y` method in all the Forecasters.
- [x] Include `gap` argument in backtesting functions to omit observations between training and prediction.
- [x] Bug fixes and performance improvements.


# Forecasters

A Forecaster object in the skforecast library is a comprehensive container that provides essential functionality and methods for training a forecasting model and generating predictions for future points in time.

The skforecast library offers a variety of forecaster types, each tailored to specific requirements such as single or multiple time series, direct or recursive strategies, or custom predictors. Regardless of the specific forecaster type, all instances share the same API.

| Forecaster | Single series | Multiple series | Recursive strategy | Direct strategy | Probabilistic prediction | Exogenous features | Custom features |
|:-----------|:-------------:|:---------------:|:------------------:|:---------------:|:------------------------:|:------------------:|:---------------:|
|[ForecasterAutoreg](https://skforecast.org/latest/user_guides/autoregresive-forecaster.html)|✔️||✔️||✔️|✔️||
|[ForecasterAutoregCustom](https://skforecast.org/latest/user_guides/custom-predictors.html)|✔️||✔️||✔️|✔️|✔️|✔️|
|[ForecasterAutoregDirect](https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html)|✔️|||✔️|✔️|✔️||
|[ForecasterMultiSeries](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html)||✔️|✔️||✔️|✔️||
|[ForecasterMultiSeriesCustom](https://skforecast.org/latest/user_guides/custom-predictors.html)||✔️|✔️||✔️|✔️|✔️|✔️|
|[ForecasterMultiVariate](https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting.html)||✔️||✔️|✔️|✔️||
|[ForecasterSarimax](https://skforecast.org/latest/user_guides/forecasting-sarimax-arima.html)|✔️||✔️||✔️|✔️||


# Documentation

The documentation for the latest release is at [skforecast docs](https://skforecast.org).

+ [Introduction to time series and forecasting](https://skforecast.org/latest/user_guides/quick-start-skforecast.html)

+ [Recursive multi-step forecasting](https://skforecast.org/latest/user_guides/autoregresive-forecaster.html)

+ [Direct multi-step forecasting](https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html)

+ [Independent multi-series forecasting](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html)

+ [Dependent multi-series forecasting (Multivariate forecasting)](https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting.html)

+ [Backtesting (validation) of forecasting models](https://skforecast.org/latest/user_guides/backtesting.html)

+ [Hyperparameter tuning and lags selection of forecasting models](https://skforecast.org/latest/user_guides/hyperparameter-tuning-and-lags-selection.html)

+ [Probabilistic forecasting](https://skforecast.org/latest/user_guides/probabilistic-forecasting.html)

+ [Using forecasters in production](https://skforecast.org/latest/user_guides/forecaster-in-production.html)


# Examples and tutorials

**English**

+ [**Skforecast: time series forecasting with Python and Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X1DJF4pZlklIt5srQnyTYoyFVLunr_OQ)

+ [**Forecasting electricity demand with Python**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1efCKQtuHOlw7MLojIwqi2zrU2NZbG-FP)

+ [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QhLkJAAEfvgYoVkQXy58-T_sloNFCV1o)

+ [**Forecasting with gradient boosting: skforecast, XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Imy8ZM3DqPXg7UllRDH9gqWb_XSrqzzh)

+ [**Bitcoin price prediction with Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-cryptocurrency-bitcoin-machine-learning-python.html)

+ [**Prediction intervals in forecasting models**](https://www.cienciadedatos.net/documentos/py42-forecasting-prediction-intervals-machine-learning.html)

+ [**Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast.html)

+ [**Reducing the influence of Covid-19 on time series forecasting models**](https://www.cienciadedatos.net/documentos/py45-weighted-time-series-forecasting.html)

+ [**Forecasting time series with missing values**](https://www.cienciadedatos.net/documentos/py46-forecasting-time-series-missing-values.html)

+ [**Intermittent demand forecasting**](https://www.cienciadedatos.net/documentos/py48-intermittent-demand-forecasting.html)


**Español**

+ [**Skforecast: forecasting series temporales con Python y Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mjmccrMA-XxOVXm-3wKSIQ9__oo9dJ5a)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15kQpANRBCLfNf77nmNcV6GjGPoYdOmmF)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uw2nyjA9XMcstfkpbWC4zCULN7Qp7MWV)

+ [**Forecasting con gradient boosting: skforecast, XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UAjX8vUKDoY0XJtq5WtHlJ4qwPvSgLrD)

+ [**Predicción del precio de Bitcoin con Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-criptomoneda-bitcoin-machine-learning-python.html)

+ [**Workshop predicción de series temporales con machine learning Universidad de Deusto / Deustuko Unibertsitatea**](https://youtu.be/MlktVhReO0E)

+ [**Intervalos de predicción en modelos de forecasting**](https://www.cienciadedatos.net/documentos/py42-intervalos-prediccion-modelos-forecasting-machine-learning.html)

+ [**Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast-español.html)

+ [**Predicción de demanda intermitente**](https://www.cienciadedatos.net/documentos/py48-forecasting-demanda-intermitente.html)


# Donating

If you found skforecast useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks!

[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)


# How to contribute

For more information on how to contribute to skforecast, see our [Contribution Guide](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CONTRIBUTING.md).


# Citation

If you use this software, please cite it using the following metadata.

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. skforecast (Version 0.8.1) [Computer software]
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
license = {BSD 3-Clause License},
month = {5},
title = {{skforecast}},
version = {0.8.1},
year = {2023}
}
```

View the [citation file](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CITATION.cff).


# License

[BSD 3-Clause License](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE)