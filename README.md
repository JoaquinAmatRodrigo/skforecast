<h1 align="left">
<img src="https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/images/banner-landing-page-skforecast.png?raw=true#only-light" style= margin-top: 0px;>
</h1>


| | |
| --- | --- |
| Package | ![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue) [![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/) [![Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/skforecast) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/graphs/commit-activity) [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) |
| Meta | [![License](https://img.shields.io/github/license/JoaquinAmatRodrigo/skforecast)](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/LICENSE) [![DOI](https://zenodo.org/badge/337705968.svg)](https://zenodo.org/doi/10.5281/zenodo.8382787) |
| Testing | [![Build status](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/JoaquinAmatRodrigo/skforecast/actions/workflows/unit-tests.yml/badge.svg) [![codecov](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/JoaquinAmatRodrigo/skforecast) |
|Donation | [![paypal](https://img.shields.io/static/v1?style=social&amp;label=Donate&amp;message=%E2%9D%A4&amp;logo=Paypal&amp;color&amp;link=%3curl%3e)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6) [![buymeacoffee](https://img.shields.io/badge/-Buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/skforecast) ![GitHub Sponsors](https://img.shields.io/github/sponsors/joaquinamatrodrigo?logo=github&label=Github%20sponsors&link=https%3A%2F%2Fgithub.com%2Fsponsors%2FJoaquinAmatRodrigo) |
|Community | [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/skforecast/)
|Affiliation | [![NumFOCUS Affiliated](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)


# Table of Contents

- :information_source: [About The Project](#about-the-project)
- :books: [Documentation](#documentation)
- :computer: [Installation & Dependencies](#installation--dependencies)
- :sparkles: [What is new in skforecast 0.13?](#what-is-new-in-skforecast-013)
- :crystal_ball: [Forecasters](#forecasters)
- :mortar_board: [Examples and tutorials](#examples-and-tutorials)
- :handshake: [How to contribute](#how-to-contribute)
- :memo: [Citation](#citation)
- :money_with_wings: [Donating](#donating)
- :scroll: [License](#license)


# About The Project

**Skforecast** is a Python library for time series forecasting using machine learning models. It works with any regressor compatible with the scikit-learn API, including popular options like LightGBM, XGBoost, CatBoost, Keras, and many others.

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


# What is new in skforecast 0.13?

Visit the [release notes](https://github.com/JoaquinAmatRodrigo/skforecast/blob/master/changelog.md) to view all notable changes.

- [x] Support for `python 3.12`, `python 3.8` is no longer supported.
- [x] Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> are able to [predict series not seen during training](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html#forecasting-unknown-series). This is useful when the user wants to predict a new series that was not included in the training data.
- [x] `encoding` can be set to `None` in Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code>. This option does [not add the encoded series ids](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting#series-encoding-in-multi-series) to the regressor training matrix.
- [x] New `create_predict_X` method in all recursive and direct Forecasters to allow the user to inspect the matrix passed to the predict method of the regressor.
- [x] New module <code>[metrics]</code> with functions to calculate metrics for time series forecasting such as <code>[mean_absolute_scaled_error]</code> and <code>[root_mean_squared_scaled_error]</code>. Visit [Time Series Forecasting Metrics](https://skforecast.org/latest/user_guides/metrics.html) for more information.
- [x] New argument `add_aggregated_metric` in <code>[backtesting_forecaster_multiseries]</code> to include, in addition to the metrics for each level, the aggregated metric of all levels  using the average (arithmetic mean), weighted average (weighted by the number of predicted values of each level) or pooling (the values of all levels are pooled and then the metric is calculated).
- [x] New argument `skip_folds` in <code>[model_selection]</code> and <code>[model_selection_multiseries]</code> functions. It allows the user to [skip some folds during backtesting](https://skforecast.org/latest/user_guides/backtesting#backtesting-with-skip-folds), which can be useful to speed up the backtesting process and thus the hyperparameter search.
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
[ForecasterAutoregCustom]: https://skforecast.org/latest/user_guides/window-features-and-custom-features.html
[ForecasterAutoregDirect]: https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html
[ForecasterMultiSeries]: https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html
[ForecasterMultiSeriesCustom]: https://skforecast.org/latest/user_guides/window-features-and-custom-features.html#forecasterautoregmultiseriescustom
[ForecasterMultiVariate]: https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting.html
[ForecasterRNN]: https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm
[ForecasterSarimax]: https://skforecast.org/latest/user_guides/forecasting-sarimax-arima.html


# Examples and tutorials

**English**

+ [**Skforecast: time series forecasting with Machine Learning**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html)

+ [**ARIMA and SARIMAX models**](https://www.cienciadedatos.net/documentos/py51-arima-sarimax-models-python.html)

+ [**Forecasting with gradient boosting: XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)

+ [**Forecasting with XGBoost**](https://www.cienciadedatos.net/documentos/py56-forecasting-time-series-with-xgboost.html)

+ [**Global Forecasting Models: Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast.html)

+ [**Probabilistic forecasting**](https://www.cienciadedatos.net/documentos/py42-probabilistic-forecasting.html)

+ [**Forecasting with Deep Learning**](https://cienciadedatos.net/documentos/py54-forecasting-with-deep-learning)

+ [**Modelling time series trend with tree based models**](https://www.cienciadedatos.net/documentos/py49-modelling-time-series-trend-with-tree-based-models.html)

+ [**Mitigating the impact of covid on forecasting models**](https://www.cienciadedatos.net/documentos/py45-weighted-time-series-forecasting.html)

+ [**Forecasting time series with missing values**](https://www.cienciadedatos.net/documentos/py46-forecasting-time-series-missing-values.html)

+ [**Intermittent demand forecasting**](https://www.cienciadedatos.net/documentos/py48-intermittent-demand-forecasting.html)

+ [**Stacking ensemble of machine learning models to improve forecasting**](https://cienciadedatos.net/documentos/py52-stacking-ensemble-models-forecasting.html)

+ [**Forecasting energy demand with machine learning**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html)

+ [**Global Forecasting Models: Comparative Analysis of Single and Multi-Series Forecasting Modeling**](https://www.cienciadedatos.net/documentos/py53-global-forecasting-models.html)

+ [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html)

+ [**Interpretable forecasting models**](https://www.cienciadedatos.net/documentos/py57-interpretable-forecasting-models.html)

+ [**Bitcoin price prediction with Python**](https://www.cienciadedatos.net/documentos/py41-forecasting-cryptocurrency-bitcoin-machine-learning-python.html)


**Español**

+ [**Skforecast: forecasting series temporales con Machine Learning**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html)

+ [**Modelos ARIMA y SARIMAX**](https://cienciadedatos.net/documentos/py51-modelos-arima-sarimax-python.html)

+ [**Forecasting con gradient boosting: XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html)

+ [**Forecasting con XGBoost**](https://www.cienciadedatos.net/documentos/py56-forecasting-series-temporales-con-xgboost.html)

+ [**Global Forecasting Models: Multi-series forecasting**](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast-español.html)

+ [**Modelos de forecasting globales: Análisis comparativo de modelos de una y múltiples series**](https://www.cienciadedatos.net/documentos/py53-modelos-forecasting-globales.html)

+ [**Forecasting Probabilístico**](https://www.cienciadedatos.net/documentos/py42-intervalos-prediccion-modelos-forecasting-machine-learning.html)

+ [**Forecasting con Deep Learning**](https://cienciadedatos.net/documentos/py54-forecasting-con-deep-learning)

+ [**Modelar series temporales con tendencia utilizando modelos de árboles**](https://cienciadedatos.net/documentos/py49-modelar-tendencia-en-series-temporales-modelos-de-arboles.html)

+ **Forecasting de series incompletas con valores faltantes**(https://www.cienciadedatos.net/documentos/py46-forecasting-series-temporales-incompletas.html)

+ [**Predicción de demanda intermitente**](https://www.cienciadedatos.net/documentos/py48-forecasting-demanda-intermitente.html)

+ [**Reducir el impacto del Covid en modelos de forecasting**](https://cienciadedatos.net/documentos/py45-weighted-time-series-forecasting-es.html)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html)

+ [**Interpretabilidad en modelos de forecasting**](https://www.cienciadedatos.net/documentos/py57-modelos-forecasting-interpretables.html)

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
Amat Rodrigo, Joaquin, & Escobar Ortiz, Javier. (2024). skforecast (v0.13.0). Zenodo. https://doi.org/10.5281/zenodo.8382788
```

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. (2024). skforecast (Version 0.13.0) [Computer software]. https://doi.org/10.5281/zenodo.8382788
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
title = {skforecast},
version = {0.13.0},
month = {8},
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



<!-- Links to API Reference -->
<!-- Forecasters -->
<!-- [ForecasterAutoreg]: https://skforecast.org/latest/api/forecasterautoreg
[ForecasterAutoregCustom]: https://skforecast.org/latest/api/forecasterautoregcustom
[ForecasterAutoregDirect]: https://skforecast.org/latest/api/forecasterautoregdirect
[ForecasterRNN]: https://skforecast.org/latest/api/forecasterrnn
[ForecasterSarimax]: https://skforecast.org/latest/api/forecastersarimax
[Sarimax]: https://skforecast.org/latest/api/sarimax
[ForecasterEquivalentDate]: https://skforecast.org/latest/api/forecasterbaseline#skforecast.ForecasterBaseline.ForecasterEquivalentDate -->
[ForecasterAutoregMultiSeries]: https://skforecast.org/latest/api/forecastermultiseries
[ForecasterAutoregMultiSeriesCustom]: https://skforecast.org/latest/api/forecastermultiseriescustom
[ForecasterAutoregMultiVariate]: https://skforecast.org/latest/api/forecastermultivariate

<!-- metrics -->
[metrics]: https://skforecast.org/latest/api/metrics
[add_y_train_argument]: https://skforecast.org/latest/api/metrics#skforecast.metrics.metrics.add_y_train_argument
[mean_absolute_scaled_error]: https://skforecast.org/latest/api/metrics#skforecast.metrics.metrics.mean_absolute_scaled_error
[root_mean_squared_scaled_error]: https://skforecast.org/latest/api/metrics#skforecast.metrics.metrics.root_mean_squared_scaled_error

<!-- model_selection -->
[model_selection]: https://skforecast.org/latest/api/model_selection
[backtesting_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.backtesting_forecaster
[grid_search_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.grid_search_forecaster
[random_search_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.random_search_forecaster
[bayesian_search_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.bayesian_search_forecaster
[select_features]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.select_features

<!-- model_selection_multiseries -->
[model_selection_multiseries]: https://skforecast.org/latest/api/model_selection_multiseries
[backtesting_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection_multiseries#skforecast.model_selection_multiseries.model_selection_multiseries.backtesting_forecaster_multiseries
[grid_search_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection_multiseries#skforecast.model_selection_multiseries.model_selection_multiseries.grid_search_forecaster_multiseries
[random_search_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection_multiseries#skforecast.model_selection_multiseries.model_selection_multiseries.random_search_forecaster_multiseries
[bayesian_search_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection_multiseries#skforecast.model_selection_multiseries.model_selection_multiseries.bayesian_search_forecaster_multiseries
[select_features_multiseries]: https://skforecast.org/latest/api/model_selection_multiseries#skforecast.model_selection_multiseries.model_selection_multiseries.select_features_multiseries

<!-- model_selection_sarimax -->
[model_selection_sarimax]: https://skforecast.org/latest/api/model_selection_sarimax
[backtesting_sarimax]: https://skforecast.org/latest/api/model_selection_sarimax#skforecast.model_selection_sarimax.model_selection_sarimax.backtesting_sarimax
[grid_search_sarimax]: https://skforecast.org/latest/api/model_selection_sarimax#skforecast.model_selection_sarimax.model_selection_sarimax.grid_search_sarimax
[random_search_sarimax]: https://skforecast.org/latest/api/model_selection_sarimax#skforecast.model_selection_sarimax.model_selection_sarimax.random_search_sarimax

<!-- preprocessing -->
[preprocessing]: https://skforecast.org/latest/api/preprocessing
[TimeSeriesDifferentiator]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.TimeSeriesDifferentiator
[series_long_to_dict]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.series_long_to_dict
[exog_long_to_dict]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.exog_long_to_dict
[DateTimeFeatureTransformer]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.DateTimeFeatureTransformer
[create_datetime_features]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.create_datetime_features

<!-- plot -->
[plot]: https://skforecast.org/latest/api/plot
[set_dark_theme]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.set_dark_theme
[plot_residuals]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_residuals
[plot_multivariate_time_series_corr]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_multivariate_time_series_corr
[plot_prediction_distribution]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_prediction_distribution
[plot_prediction_intervals]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_prediction_intervals

<!-- datasets -->
[datasets]: https://skforecast.org/latest/api/datasets
[fetch_dataset]: https://skforecast.org/latest/api/datasets#skforecast.datasets.fetch_dataset
[load_demo_dataset]: https://skforecast.org/latest/api/datasets#skforecast.datasets.load_demo_dataset

<!-- utils -->
[utils]: https://skforecast.org/latest/api/utils
