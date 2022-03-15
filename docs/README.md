<script src="https://kit.fontawesome.com/d20edc211b.js" crossorigin="anonymous"></script>

<img src="img/logo_skforecast_no_background.png" width=185 height=185 align="right">

# **skforecast**

**Time series forecasting with scikit-learn regressors.**

**Skforecast** is a python library that eases using scikit-learn regressors as multi-step forecasters. It also works with any regressor compatible with the scikit-learn API (pipelines, CatBoost, LightGBM, XGBoost, Ranger...).

!!! info

     **Version 0.4** has undergone a huge code refactoring. Main changes are related to input-output formats (only pandas series and dataframes are allowed although internally numpy arrays are used for performance) and model validation methods (unified into backtesting with and without refit). All notable changes are listed in [Releases](./releases/releases.md).

## Installation

```
pip install skforecast
```

Specific version:

```
pip install skforecast==0.4.2
```

Latest (unstable):

```
pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```

## Dependencies

```
numpy>=1.20, <=1.22
pandas>=1.2, <=1.4
tqdm>=4.57.0, <=4.62
scikit-learn>=1.0
statsmodels>=0.12, <=0.13
```

## Features

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


## Tutorials 

### English

<i class="fa-duotone fa-chart-line fa" style="font-size: 25px; color:#1DA1F2;"></i>  [**Skforecast: time series forecasting with Python and Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html)

<i class="fa-duotone fa-lightbulb fa" style="font-size: 25px; color:#fcea2b;"></i> [**Forecasting electricity demand with Python**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html)

<i class="fa-solid fa-bicycle fa" style="font-size: 25px; color:#00cc99;"></i> [**Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)

<i class="fa-duotone fa-rss fa" style="font-size: 25px; color:#666666;"></i> [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html)


### Español

<i class="fa-duotone fa-chart-line fa" style="font-size: 25px; color:#1DA1F2;"></i> [**Skforecast: forecasting series temporales con Python y Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html)

<i class="fa-duotone fa-lightbulb fa" style="font-size: 25px; color:#fcea2b;"></i> [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html)

<i class="fa-solid fa-bicycle fa" style="font-size: 25px; color:#00cc99;"></i>  [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html)

<i class="fa-duotone fa-rss fa" style="font-size: 25px; color:#666666;"></i> [**Forecasting series temporales con gradient boosting: Skforecast, XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html)


## Donating

If you found skforecast useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks! :hugging_face: :heart_eyes:

[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)


## Licence

**joaquinAmatRodrigo/skforecast** is licensed under the **MIT License**, a short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.