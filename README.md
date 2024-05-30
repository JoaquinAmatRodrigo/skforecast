<h1 align="left">
<img src="https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/images/banner-landing-page-skforecast.png?raw=true#only-light" style= margin-top: 0px;>
</h1>


| | |
| --- | --- |
| Package | ![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue) [![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/) [![Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/skforecast) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/graphs/commit-activity) [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) |
| Meta | [![License](https://img.shields.io/github/license/JoaquinAmatRodrigo/skforecast)](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE) [![DOI](https://zenodo.org/badge/337705968.svg)](https://zenodo.org/doi/10.5281/zenodo.8382787) |
| Testing | [![Build status](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg) [![codecov](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast) |
|Donation | [![paypal](https://img.shields.io/static/v1?style=social&amp;label=Donate&amp;message=%E2%9D%A4&amp;logo=Paypal&amp;color&amp;link=%3curl%3e)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6) [![buymeacoffee](https://img.shields.io/badge/-Buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/skforecast) ![GitHub Sponsors](https://img.shields.io/github/sponsors/joaquinamatrodrigo?logo=github&label=Github%20sponsors&link=https%3A%2F%2Fgithub.com%2Fsponsors%2FJoaquinAmatRodrigo) |
|Community | [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/skforecast/)
|Affiliation | [![NumFOCUS Affiliated](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)


# Table of Contents

- :information_source: [About The Project](#about-the-project)
- :books: [Documentation](#documentation)
- :computer: [Installation & Dependencies](#installation--dependencies)
- :sparkles: [What is new in skforecast 0.12?](#what-is-new-in-skforecast-012)
- :crystal_ball: [Forecasters](#forecasters)
- :mortar_board: [Examples and tutorials](#examples-and-tutorials)
- :handshake: [How to contribute](#how-to-contribute)
- :memo: [Citation](#citation)
- :money_with_wings: [Donating](#donating)
- :scroll: [License](#license)


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

| Documentation                           |     |
|:----------------------------------------|:----|
| :book: [Introduction to forecasting]    | Basics of forecasting concepts and methodologies |
| :rocket: [Quick start]                  | Get started quickly with skforecast |
| :hammer_and_wrench: [User guides]       | Detailed guides on skforecast features and functionalities |
| :mortar_board: [Examples and tutorials] | Learn through practical examples and tutorials to master skforecast |
| :question: [FAQ and tips]               | Find answers and tips about forecasting |
| :books: [API Reference]                 | Comprehensive reference for skforecast functions and classes |
| :black_nib: [Authors]                   | Meet the authors and contributors of skforecast |

[Introduction to forecasting]: https://skforecast.org/latest/introduction-forecasting/introduction-forecasting
[Quick start]: https://skforecast.org/latest/quick-start/quick-start-skforecast
[User guides]: https://skforecast.org/latest/user_guides/user-guides
[Examples and tutorials]: https://skforecast.org/latest/examples/examples
[FAQ and tips]: https://skforecast.org/latest/faq/faq
[API Reference]: https://skforecast.org/latest/api/forecasterautoreg
[Authors]: https://skforecast.org/latest/authors/authors


# Installation & Dependencies

To install the basic version of `skforecast` with its core dependencies, run:

```bash
pip install skforecast
```

If you want to learn more about the installation process, dependencies and optional features, please refer to the [Installation Guide](https://skforecast.org/latest/quick-start/how-to-install.html).


# What is new in skforecast 0.12?

Visit the [release notes](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/changelog.md) to view all notable changes.

- [x] Multiseries forecaster (Global Models) can be trained using [series of different lengths and with different exogenous variables](https://skforecast.org/latest/user_guides/multi-series-with-different-length-and-different_exog) per series.
- [x] [Bayesian hyperparameter search](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting#hyperparameter-tuning-and-lags-selection-multi-series) is now available for all multiseries forecasters using `optuna` as the search engine.
- [x] New functionality to [select features](https://skforecast.org/latest/user_guides/feature-selection) using scikit-learn selectors (`select_features` and `select_features_multiseries`).
- [x] Added new forecaster `ForecasterRnn` to create forecasting models based on [deep learning](https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm) (RNN and LSTM).
- [x] New method to [predict intervals conditioned on the range of the predicted values](https://skforecast.org/latest/user_guides/probabilistic-forecasting#intervals-conditioned-on-predicted-values-binned-residuals). This is can help to improve the interval coverage when the residuals are not homoscedastic (`ForecasterAutoreg`).
- [x] All Recursive Forecasters are now able to [differentiate the time series](https://skforecast.org/latest/faq/time-series-differentiation) before modeling it.
- [x] Bug fixes and performance improvements.


# Forecasters

A **Forecaster** object in the skforecast library is a comprehensive container that provides essential functionality and methods for training a forecasting model and generating predictions for future points in time.

The **skforecast** library offers a variety of forecaster types, each tailored to specific requirements such as single or multiple time series, direct or recursive strategies, or custom predictors. Regardless of the specific forecaster type, all instances share the same API.

| Forecaster | Single series | Multiple series | Recursive strategy | Direct strategy | Probabilistic prediction | Time series differentiation | Exogenous features | Custom features |
|:-----------|:-------------:|:---------------:|:------------------:|:---------------:|:------------------------:|:---------------------------:|:------------------:|:---------------:|
|[ForecasterAutoreg]|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||
|[ForecasterAutoregCustom]|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterAutoregDirect]|:heavy_check_mark:|||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:||
|[ForecasterMultiSeries]||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||
|[ForecasterMultiSeriesCustom]||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterMultiVariate]||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:||
|[ForecasterRNN]||:heavy_check_mark:||:heavy_check_mark:|||||
|[ForecasterSarimax]|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||

[ForecasterAutoreg]: https://skforecast.org/latest/user_guides/autoregresive-forecaster.html
[ForecasterAutoregCustom]: https://skforecast.org/latest/user_guides/custom-predictors.html
[ForecasterAutoregDirect]: https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html
[ForecasterMultiSeries]: https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html
[ForecasterMultiSeriesCustom]: https://skforecast.org/latest/user_guides/custom-predictors.html
[ForecasterMultiVariate]: https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting.html
[ForecasterRNN]: https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm
[ForecasterSarimax]: https://skforecast.org/latest/user_guides/forecasting-sarimax-arima.html


# Examples and tutorials

**English**

+ [**Skforecast: time series forecasting with Machine Learning**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X1DJF4pZlklIt5srQnyTYoyFVLunr_OQ)

+ [**ARIMA and SARIMAX models**](https://www.cienciadedatos.net/documentos/py51-arima-sarimax-models-python.html)

+ [**Forecasting with gradient boosting: XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Imy8ZM3DqPXg7UllRDH9gqWb_XSrqzzh)

+ [**Global Forecasting Models: Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast.html)

+ [**Probabilistic forecasting**](https://www.cienciadedatos.net/documentos/py42-probabilistic-forecasting.html)

+ [**Forecasting with Deep Learning**](https://cienciadedatos.net/documentos/py54-forecasting-with-deep-learning)

+ [**Modelling time series trend with tree based models**](https://www.cienciadedatos.net/documentos/py49-modelling-time-series-trend-with-tree-based-models.html)

+ [**Reducing the influence of Covid-19 on time series forecasting models**](https://www.cienciadedatos.net/documentos/py45-weighted-time-series-forecasting.html)

+ [**Forecasting time series with missing values**](https://www.cienciadedatos.net/documentos/py46-forecasting-time-series-missing-values.html)

+ [**Intermittent demand forecasting**](https://www.cienciadedatos.net/documentos/py48-intermittent-demand-forecasting.html)

+ [**Stacking ensemble of machine learning models to improve forecasting**](https://cienciadedatos.net/documentos/py52-stacking-ensemble-models-forecasting.html)

+ [**Forecasting energy demand with machine learning**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1efCKQtuHOlw7MLojIwqi2zrU2NZbG-FP)

+ [**Global Forecasting Models: Comparative Analysis of Single and Multi-Series Forecasting Modeling**](https://www.cienciadedatos.net/documentos/py53-global-forecasting-models.html)

+ [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QhLkJAAEfvgYoVkQXy58-T_sloNFCV1o)

+ [**Bitcoin price prediction with Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-cryptocurrency-bitcoin-machine-learning-python.html)


**Español**

+ [**Skforecast: forecasting series temporales con Machine Learning**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mjmccrMA-XxOVXm-3wKSIQ9__oo9dJ5a)

+ [**Modelos ARIMA y SARIMAX**](https://cienciadedatos.net/documentos/py51-modelos-arima-sarimax-python.html)

+ [**Forecasting con gradient boosting: XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UAjX8vUKDoY0XJtq5WtHlJ4qwPvSgLrD)

+ [**Global Forecasting Models: Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast-español.html)

+ [**Forecasting Probabilístico**](https://www.cienciadedatos.net/documentos/py42-intervalos-prediccion-modelos-forecasting-machine-learning.html)

+ [**Forecasting con Deep Learning**](https://cienciadedatos.net/documentos/py54-forecasting-con-deep-learning)

+ [**Modelar series temporales con tendencia utilizando modelos de árboles**](https://cienciadedatos.net/documentos/py49-modelar-tendencia-en-series-temporales-modelos-de-arboles.html)

+ [**Predicción de demanda intermitente**](https://www.cienciadedatos.net/documentos/py48-forecasting-demanda-intermitente.html)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15kQpANRBCLfNf77nmNcV6GjGPoYdOmmF)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uw2nyjA9XMcstfkpbWC4zCULN7Qp7MWV)

+ [**Predicción del precio de Bitcoin con Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-criptomoneda-bitcoin-machine-learning-python.html)

+ [**Workshop predicción de series temporales con machine learning Universidad de Deusto / Deustuko Unibertsitatea**](https://youtu.be/MlktVhReO0E)


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
Amat Rodrigo, Joaquin, & Escobar Ortiz, Javier. (2024). skforecast (v0.12.1). Zenodo. https://doi.org/10.5281/zenodo.8382788
```

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. (2024). skforecast (Version 0.12.1) [Computer software]. https://doi.org/10.5281/zenodo.8382788
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
title = {skforecast},
version = {0.12.1},
month = {5},
year = {2024},
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
