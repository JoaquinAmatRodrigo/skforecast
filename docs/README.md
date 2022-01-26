<img src="img/logo_skforecast_no_background.png" width=185 height=185 align="right">

# **skforecast**

**Time series forecasting with scikit-learn regressors.**

**Skforecast** is a python library that eases using scikit-learn regressors as multi-step forecasters. It also works with any regressor compatible with the scikit-learn API (pipelines, CatBoost, LightGBM, XGBoost, Ranger...).

**Version 0.4** has undergone a huge code refactoring. Main changes are related to input-output formats (only pandas series and dataframes are allowed although internally numpy arrays are used for performance) and model validation methods (unified into backtesting with and without refit). All notable changes are listed in [Releases](./releases/releases.md).

## Installation

```
pip install skforecast
```

Specific version:

```
pip install skforecast==0.4.1
```

Latest (unstable):

```
pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```

## Dependencies

```
python>=3.7.1
numpy>=1.20.1
pandas>=1.2.2
tqdm>=4.57.0
scikit-learn>=1.0.1
statsmodels>=0.12.2
```

## Features

+ Create recursive autoregressive forecasters from any scikit-learn regressor
+ Create multi-output autoregressive forecasters from any scikit-learn regressor
+ Grid search to find optimal hyperparameters
+ Grid search to find optimal lags (predictors)
+ Include exogenous variables as predictors
+ Include custom predictors (rolling mean, rolling variance ...)
+ Backtesting
+ Prediction interval estimated by bootstrapping
+ Get predictor importance


## Tutorials 

**English**

+ [**Skforecast: time series forecasting with Python and Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html)

+ [**Forecasting electricity demand with Python**](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html)

+ [**Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM and CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)

+ [**Forecasting web traffic with machine learning and Python**](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning.html)

**Español**

+ [**Skforecast: forecasting series temporales con Python y Scikit-learn**](https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html)

+ [**Forecasting series temporales con gradient boosting: Skforecast, XGBoost, LightGBM y CatBoost**](https://www.cienciadedatos.net/documentos/py39-forecasting-series-temporales-con-skforecast-xgboost-lightgbm-catboost.html)


## References

+ Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia

+ Time Series Analysis and Forecasting with ADAM Ivan Svetunkov

+ Python for Finance: Mastering Data-Driven Finance


## Licence

**joaquinAmatRodrigo/skforecast** is licensed under the **MIT License**, a short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.