
# **Introduction to forecasting**


## **Time series and forecasting**

A time series is a sequence of data arranged chronologically, in principle, equally spaced in time. Forecasting is the process of predicting future values of a time series based on its previously observed patterns (autoregressive), or including external variables.

<p align="center"><img src="../img/forecasting_multi-step.gif" style="width: 500px"></p>


## **Machine learning for forecasting**

The main adaptation that needs to be done to apply machine learning models to forecasting problems is to transform the time series into a matrix in which each value is related to the time window (lags) that precedes it.

<p align="center"><img src="../img/transform_timeseries.gif" style="width: 500px;"></p>

<center><font size="2.5"> <i>Time series transformation into a matrix of 5 lags and a vector with the value of the series that follows each row of the matrix.</i></font></center>

This type of transformation also allows to include additional variables.

<p align="center"><img src="../img/matrix_transformation_with_exog_variable.png" style="width: 600px;"></p>

<center><font size="2.5"> <i>Time series transformation including an exogenous variable.</i></font></center>

Once data have been rearranged into the new shape, any regression model can be trained to predict the next value (step) of the series.

<p align="center"><img src="../img/diagram-trainig-forecaster.png" style="width: 700px;"></p>


## **Single-step forecasting**

Single-step prediction is used when the goal is to predict only the next value of the series.

<p align="center"><img src="../img/diagram-single-step-forecasting.png" style="width: 700px;"></p>


## **Multi-step forecasting**

When working with time series, it is seldom needed to predict only the next element in the series (*t+1*). Instead, the most common goal is to predict a whole future interval (*t+1, ..., t+n*)  or a far point in time (*t+n*). Several strategies allow generating this type of prediction.


### **Recursive multi-step forecasting**

Since the value *t(n-1)* is required to predict *t(n)*, and *t(n-1)* is unknown, it is necessary to make recursive predictions in which, each new prediction, is based on the previous one. This process is known as recursive forecasting or recursive multi-step forecasting.

<p align="center"><img src="../img/diagram-recursive-mutistep-forecasting.png" style="width: 700px"></p>


### **Direct multi-step forecasting**

Direct multi-step forecasting consists of training a different model for each step. For example, to predict the next 5 values of a time series, 5 different models are trained, one for each step. As a result, the predictions are independent of each other.

<p align="center"><img src="../img/diagram-direct-multi-step-forecasting.png" style="width: 700px"></p>


### **Multiple output forecasting**

Some machine learning models, such as long short-term memory (LSTM) neural network, can predict simultaneously several values of a sequence (*one-shot*). This strategy is not currently implemented in skforecast library.
