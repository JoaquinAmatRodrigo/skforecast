# skforecast

**Time series forecasting with scikit-learn regressors.**

Python package that eases using scikit-learn regressors as multi-step forecaster.

<br>

Joaquin Amat Rodrigo

<br>

## Installation

```bash
$ pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```

## Dependencies

+ python>=3.7.1
+ numpy>=1.20.1
+ pandas>=1.2.2
+ tqdm>=4.57.0
+ scikit-learn>=0.24

## Features

+ Create autoregressive forecasters from any scikit-learn regressor.
+ Grid search to find optimal hyperparameters.
+ Grid search to find optimal lags (predictors).
+ Include exogenous variables as predictors.

## Introducción
<br>

A [**time series**](https://en.wikipedia.org/wiki/Time_series) is a sequence of data arranged chronologically, in principle, equally spaced in time. **Time series forecasting** is the use of a model to predict future values based on previously observed values, with the option of also including other external variables.

When working with time series, it is seldom needed to predict only the next element in the series $`(t+1)`$. Instead, the most common goal is to predict a whole future interval $`(t+1, ..., t+n`$  or a far point in time $`t+n`$ .

Since the value of $`t + 1`$ is required to predict the point $`t + 2`$, and $`t + 1`$ is unknown, it is necessary to make recursive predictions in which, each new prediction , is based on the previous one. This process is known as recursive forecasting or multi-step forecasting and it is the main difference with respect to conventional regression problems.

<p><img src="./images/forecasting_multi-step.gif" alt="forecasting-python" title="forecasting-python"></p>

The main challenge when using scikit learn models for forecasting is transforming the time series in an matrix where, each value of the series, is related to the time window (lags) that precede it.

<p><img src="./images/transform_timeseries.gif" alt="forecasting-python" title="forecasting-python"></p>

<center><font size="2.5"> <i>Tranformación de una serie temporal en una matriz de 5 lags y un vector con el valor de la serie que sigue a cada fila de la matriz.</i></font></center>

**Skforecast** is a python library that eases using scikit-learn regressors as multi-step forecasters.

## Examples
