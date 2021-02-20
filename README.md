# skforecast

**Time series forecasting with scikit-learn regressors.**


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



## Introduction


A time series is a sequence of data arranged chronologically, in principle, equally spaced in time. Time series forecasting is the use of a model to predict future values based on previously observed values, with the option of also including other external variables.

When working with time series, it is seldom needed to predict only the next element in the series $`(t+1)`$. Instead, the most common goal is to predict a whole future interval $`(t+1, ..., t+n`$  or a far point in time $`t+n`$ .

Since the value of $`t + 1`$ is required to predict the point $`t + 2`$, and $`t + 1`$ is unknown, it is necessary to make recursive predictions in which, each new prediction , is based on the previous one. This process is known as recursive forecasting or multi-step forecasting and it is the main difference with respect to conventional regression problems.

<p><img src="./images/forecasting_multi-step.gif" alt="forecasting-python" title="forecasting-python"></p>

<br>

The main challenge when using scikit learn models for forecasting is transforming the time series in an matrix where, each value of the series, is related to the time window (lags) that precede it.

<p><img src="./images/transform_timeseries.gif" alt="forecasting-python" title="forecasting-python"></p>

<center><font size="2.5"> <i>Time series  transformation into a matrix of 5 lags and a vector with the value of the series that follows each row of the matrix.</i></font></center>

<br><br>

**Skforecast** is a python library that eases using scikit-learn regressors as multi-step forecasters.

## Examples



## Author

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work by  Joaquín Amat Rodrigo is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
