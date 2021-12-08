<img src="img/logo_skforecast_no_background.png" width=185 height=185 align="right">

# **skforecast**

**Time series forecasting with scikit-learn regressors.**

**Skforecast** is a python library that eases using scikit-learn regressors as multi-step forecasters. It also works with any regressor compatible with the scikit-learn API (pipelines, CatBoost, LightGBM, XGBoost, Ranger...).

**Documentation: https://joaquinamatrodrigo.github.io/skforecast/**

## Installation

```
pip install skforecast
```

Specific version:

```
pip install skforecast==0.3
```

Latest (unstable):

```
pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```

The most common error when importing the library is:

 `'cannot import name 'mean_absolute_percentage_error' from 'sklearn.metrics'`.
 
 This is because the scikit-learn installation is lower than 1.0.1. Try to upgrade scikit-learn with
 
 `pip install scikit-learn==1.0.1`

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

+ [**Time series forecasting with Python and Scikit-learn**](https://joaquinamatrodrigo.github.io/skforecast/0.4/html/py27-time-series-forecasting-python-scikitlearn.html)


**Español**

+ [**Forecasting series temporales con Python y Scikit-learn**](https://joaquinamatrodrigo.github.io/skforecast/0.4/html/py27-forecasting-series-temporales-python-scikitlearn.html)

+ [**Forecasting de la demanda eléctrica**](https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html)

+ [**Forecasting de las visitas a una página web**](https://www.cienciadedatos.net/documentos/py37-forecasting-visitas-web-machine-learning.html)


## References

+ Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia

+ Time Series Analysis and Forecasting with ADAM Ivan Svetunkov

+ Python for Finance: Mastering Data-Driven Finance


## Licence

**joaquinAmatRodrigo/skforecast** is licensed under the **MIT License**, a short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.