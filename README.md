<h1 align="left">
<img src="https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/images/banner-landing-page-skforecast.png?raw=true#only-light" style= margin-top: 0px;>
</h1>


| | |
| --- | --- |
| Package | ![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue) [![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/) [![Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/skforecast) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/graphs/commit-activity) [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) |
| Meta | [![License](https://img.shields.io/github/license/JoaquinAmatRodrigo/skforecast)](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE) [![DOI](https://zenodo.org/badge/337705968.svg)](https://zenodo.org/doi/10.5281/zenodo.8382787) |
| Testing | [![Build status](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg) [![codecov](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast) |
|Donation | [![paypal](https://img.shields.io/static/v1?style=social&amp;label=Donate&amp;message=%E2%9D%A4&amp;logo=Paypal&amp;color&amp;link=%3curl%3e)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6) [![buymeacoffee](https://img.shields.io/badge/-Buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/skforecast) ![GitHub Sponsors](https://img.shields.io/github/sponsors/joaquinamatrodrigo?logo=github&label=Github%20sponsors&link=https%3A%2F%2Fgithub.com%2Fsponsors%2FJoaquinAmatRodrigo) |
|Affiliation | [![NumFOCUS Affiliated](https://img.shields.io/badge/affiliated-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)



# About The Project

**Skforecast** is a Python library that eases using scikit-learn regressors as single and multi-step forecasters. It also works with any regressor compatible with the scikit-learn API (LightGBM, XGBoost, CatBoost, ...).

**Why use skforecast?**

The fields of statistics and machine learning have developed many excellent regression algorithms that can be useful for forecasting, but applying them effectively to time series analysis can still be a challenge. To address this issue, the skforecast library provides a comprehensive set of tools for training, validation and prediction in a variety of scenarios commonly encountered when working with time series. The library is built using the widely used scikit-learn API, making it easy to integrate into existing workflows. With skforecast, users have access to a wide range of functionalities such as feature engineering, model selection, hyperparameter tuning and many others. This allows users to focus on the essential aspects of their projects and leave the intricacies of time series analysis to skforecast. In addition, skforecast is developed according to the following priorities:

+ Fast and robust prototyping. :zap:
+ Validation and backtesting methods to have a realistic assessment of model performance. :mag:
+ Models must be deployed in production. :hammer:
+ Models must be interpretable. :crystal_ball:

**Share Your Thoughts with Us**

Thank you for choosing skforecast! We value your suggestions, bug reports and recommendations as they help us identify areas for improvement and ensure that skforecast meets the needs of the community. Please consider sharing your experiences, reporting bugs, making suggestions or even contributing to the codebase on GitHub. Together, let's make time series forecasting more accessible and accurate for everyone.


# Documentation

For detailed information on how to use and leverage the full potential of **skforecast** please refer to the comprehensive documentation available at:

**https://skforecast.org** :books:


# Installation

The default installation of skforecast only installs hard dependencies.

```bash
pip install skforecast
```

Specific version:

```bash
pip install skforecast==0.11.0
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

```bash
pip install skforecast[deeplearning]
```


# Dependencies

+ Python >= 3.8

## Hard dependencies

+ numpy>=1.20, <1.27
+ pandas>=1.2, <2.3
+ tqdm>=4.57.0, <4.67
+ scikit-learn>=1.2, <1.5
+ optuna>=2.10.0, <3.7
+ joblib>=1.1.0, <1.5

## Optional dependencies

+ matplotlib>=3.3, <3.9
+ seaborn>=0.11, <0.14
+ statsmodels>=0.12, <0.15
+ pmdarima>=2.0, <2.1
+ tensorflow>=2.15, <2.17


# What is new in skforecast 0.12?

Visit the [release notes](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/changelog.md) to view all notable changes.

- [x] Bayesian hyperparameter search for the `ForecasterAutoregMultiSeries`, `ForecasterAutoregMultiSeriesCustom`, and `ForecasterAutoregMultiVariate` using `optuna` as the search engine.
- [x] Added `select_features` function to the `model_selection` and `model_selection_multiseries` modules to perform feature selection using scikit-learn selectors.
- [x] Allow different exogenous variables per series in the `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiSeriesCustom`.
- [x] New encoding options for the `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiSeriesCustom`.
- [x] Bug fixes and performance improvements.


# Forecasters

A **Forecaster** object in the skforecast library is a comprehensive container that provides essential functionality and methods for training a forecasting model and generating predictions for future points in time.

The **skforecast** library offers a variety of forecaster types, each tailored to specific requirements such as single or multiple time series, direct or recursive strategies, or custom predictors. Regardless of the specific forecaster type, all instances share the same API.

| Forecaster | Single series | Multiple series | Recursive strategy | Direct strategy | Probabilistic prediction | Time series differentiation | Exogenous features | Custom features |
|:-----------|:-------------:|:---------------:|:------------------:|:---------------:|:------------------------:|:---------------------------:|:------------------:|:---------------:|
|[ForecasterAutoreg](https://skforecast.org/latest/user_guides/autoregresive-forecaster.html)|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||
|[ForecasterAutoregCustom](https://skforecast.org/latest/user_guides/custom-predictors.html)|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterAutoregDirect](https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html)|:heavy_check_mark:|||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:||
|[ForecasterMultiSeries](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html)||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:||
|[ForecasterMultiSeriesCustom](https://skforecast.org/latest/user_guides/custom-predictors.html)||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterMultiVariate](https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting.html)||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:||
|[ForecasterSarimax](https://skforecast.org/latest/user_guides/forecasting-sarimax-arima.html)|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||


# Main User Guides

+ [Introduction to time series and forecasting](https://skforecast.org/latest/introduction-forecasting/introduction-forecasting.html)

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

+ [**ARIMA and SARIMAX models**](https://www.cienciadedatos.net/documentos/py51-arima-sarimax-models-python.html)

+ [**Forecasting with gradient boosting: skforecast, XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Imy8ZM3DqPXg7UllRDH9gqWb_XSrqzzh)

+ [**Modelling time series trend with tree based models**](https://www.cienciadedatos.net/documentos/py49-modelling-time-series-trend-with-tree-based-models.html)

+ [**Forecasting energy demand with machine learning**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1efCKQtuHOlw7MLojIwqi2zrU2NZbG-FP)

+ [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QhLkJAAEfvgYoVkQXy58-T_sloNFCV1o)

+ [**Bitcoin price prediction with Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-cryptocurrency-bitcoin-machine-learning-python.html)

+ [**Prediction intervals in forecasting models**](https://www.cienciadedatos.net/documentos/py42-forecasting-prediction-intervals-machine-learning.html)

+ [**Global Forecasting Models: Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast.html)

+ [**Global Forecasting Models: Comparative Analysis of Single and Multi-Series Forecasting Modeling**](https://www.cienciadedatos.net/documentos/py53-global-forecasting-models.html)

+ [**Reducing the influence of Covid-19 on time series forecasting models**](https://www.cienciadedatos.net/documentos/py45-weighted-time-series-forecasting.html)

+ [**Forecasting time series with missing values**](https://www.cienciadedatos.net/documentos/py46-forecasting-time-series-missing-values.html)

+ [**Intermittent demand forecasting**](https://www.cienciadedatos.net/documentos/py48-intermittent-demand-forecasting.html)

+ [**Stacking ensemble of machine learning models to improve forecasting**](https://cienciadedatos.net/documentos/py52-stacking-ensemble-models-forecasting.html)


**Español**

+ [**Skforecast: forecasting series temporales con Python y Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mjmccrMA-XxOVXm-3wKSIQ9__oo9dJ5a)

+ [**Modelos ARIMA y SARIMAX**](https://cienciadedatos.net/documentos/py51-modelos-arima-sarimax-python.html)

+ [**Forecasting con gradient boosting: skforecast, XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UAjX8vUKDoY0XJtq5WtHlJ4qwPvSgLrD)

+ [**Modelar series temporales con tendencia utilizando modelos de árboles**](https://cienciadedatos.net/documentos/py49-modelar-tendencia-en-series-temporales-modelos-de-arboles.html)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15kQpANRBCLfNf77nmNcV6GjGPoYdOmmF)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uw2nyjA9XMcstfkpbWC4zCULN7Qp7MWV)

+ [**Predicción del precio de Bitcoin con Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-criptomoneda-bitcoin-machine-learning-python.html)

+ [**Workshop predicción de series temporales con machine learning Universidad de Deusto / Deustuko Unibertsitatea**](https://youtu.be/MlktVhReO0E)

+ [**Intervalos de predicción en modelos de forecasting**](https://www.cienciadedatos.net/documentos/py42-intervalos-prediccion-modelos-forecasting-machine-learning.html)

+ [**Global Forecasting Models: Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast-español.html)

+ [**Predicción de demanda intermitente**](https://www.cienciadedatos.net/documentos/py48-forecasting-demanda-intermitente.html)


# How to contribute

Primarily, skforecast development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/JoaquinAmatRodrigo/skforecast/issues).
- Contribute a Jupyter notebook to our [examples](https://skforecast.org/latest/examples/examples).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

For more information on how to contribute to skforecast, see our [Contribution Guide](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CONTRIBUTING.md).

Visit our [authors section](https://skforecast.org/latest/authors/authors) to meet all the contributors to skforecast.


# Citation

If you use skforecast for a scientific publication, we would appreciate citations to the published software.

**Zenodo**

```
Amat Rodrigo, Joaquin, & Escobar Ortiz, Javier. (2023). skforecast (v0.11.0). Zenodo. https://doi.org/10.5281/zenodo.8382788
```

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. (2023). skforecast (Version 0.11.0) [Computer software]. https://doi.org/10.5281/zenodo.8382788
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
title = {skforecast},
version = {0.11.0},
month = {11},
year = {2023},
license = {BSD-3-Clause},
url = {https://skforecast.org/},
doi = {10.5281/zenodo.8382788}
}
```

View the [citation file](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/CITATION.cff).


# Donating

If you found skforecast useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks!

<a href="https://www.buymeacoffee.com/skforecast"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=skforecast&button_colour=f79939&font_colour=000000&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>
<br>


[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)


# License

[BSD-3-Clause License](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE)
