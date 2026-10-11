# Probabilistic forecasting

When trying to predict future values, most forecasting models estimate a single value, usually the expected (mean) value of the series at each future step. This is called point forecasting. Although knowing the expected value of a time series in advance is useful in almost any business case, this type of prediction does not provide any information about the confidence of the model or the uncertainty of the prediction.

Probabilistic forecasting, as opposed to point forecasting, is a family of techniques that estimate the distribution of the outcome rather than a single future value. This type of forecasting provides much richer information because it allows the creation of prediction intervals, the range of likely values where the true value may fall. More formally, a prediction interval defines the interval within which the true value of the response variable is expected to be found with a given probability.

Skforecast implements three families of methods for probabilistic forecasting, each summarized below and explained in detail in its own user guide:

+ [**Bootstrapped residuals**](probabilistic-forecasting-bootstrapped-residuals.ipynb): simulate many possible future paths by adding resampled past errors to the predictions.

+ [**Conformal prediction**](probabilistic-forecasting-conformal-prediction.ipynb): build the interval by adding and subtracting a quantile of the past errors to the point forecast. Conformal methods can also be used to [calibrate intervals](probabilistic-forecasting-conformal-calibration.ipynb) obtained with other techniques.

+ [**Quantile regression**](probabilistic-forecasting-quantile-regression.ipynb): train models that directly predict the quantiles that define the bounds of the interval.

The last section gives guidelines to choose between them. How to evaluate the resulting intervals is explained in [Metrics in probabilistic forecasting](probabilistic-forecasting-metrics.ipynb).


### Bootstrapped residuals

Bootstrapping is a statistical technique that allows for estimating the distribution of a statistic by resampling the data with replacement. Applied to forecasting, the idea is to simulate many possible futures. In each simulation (bootstrapping iteration), a residual (a past error of the model) is drawn at random and added to the prediction of each step. In recursive forecasters, this noisy prediction is also used as a lag to predict the next step, so the errors accumulate along the forecast horizon, as they would in reality. Repeating this process hundreds of times produces a collection of plausible trajectories, whose spread reflects the uncertainty of the forecast.

+ [`predict_bootstrapping()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_bootstrapping): returns the simulated trajectories themselves, a pandas `DataFrame` with one row per predicted step and one column per bootstrapping iteration. For example, with `steps=24` and `n_boot=500`, the output has shape `(24, 500)`: each column is one possible future, and each row contains 500 plausible values for that step.

The other three methods call `predict_bootstrapping()` internally and summarize, row by row, the distribution of the simulated values:

+ [`predict_interval(method='bootstrapping')`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_interval): the lower and upper bounds of the interval, computed as quantiles of the simulated values (for example, the 0.1 and 0.9 quantiles for an 80% interval).

+ [`predict_quantiles()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_quantiles): any list of quantiles of the simulated values.

+ [`predict_dist()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_dist): the parameters of a parametric distribution fitted to the simulated values. Any of the continuous distributions available in [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions) can be used.

These methods are available in [`ForecasterRecursive`](https://skforecast.org/latest/api/forecasterrecursive), [`ForecasterDirect`](https://skforecast.org/latest/api/forecasterdirect), [`ForecasterRecursiveMultiSeries`](https://skforecast.org/latest/api/forecasterrecursivemultiseries) and [`ForecasterDirectMultiVariate`](https://skforecast.org/latest/api/forecasterdirectmultivariate).

The four methods can use two types of residuals, selected with the `use_in_sample_residuals` argument:

+ **In-sample residuals** (`use_in_sample_residuals=True`, the default): the errors of the model on the same data used to train it. They are always available, but they tend to underestimate the real error, since the model has already seen these observations.

+ **Out-of-sample residuals** (`use_in_sample_residuals=False`): the errors of the model on data not used for training, for example a validation set. They must be calculated by the user and stored in the forecaster with [`set_out_sample_residuals()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.set_out_sample_residuals).

In both cases, the residuals can be conditioned on the predicted value (`use_binned_residuals=True`) to account for a possible relationship between the magnitude of the predictions and the size of the errors.

Discover how to use these methods in [Probabilistic forecasting with bootstrapped residuals](probabilistic-forecasting-bootstrapped-residuals.ipynb).


### Conformal prediction

Conformal prediction is a framework for constructing prediction intervals that are guaranteed to contain the true value with at least a specified probability (coverage probability), on average over new observations (marginal coverage), under the assumption that the data are exchangeable. In time series this assumption holds only approximately, so the empirical coverage should always be validated.

It works by combining the predictions of a point forecasting model with its past residuals (differences between previous predictions and actual values). In the split conformal approach, the absolute residuals are computed on a calibration set (data not used to train the model), and their quantile at the nominal coverage level, $\hat{q}$, is calculated. The prediction interval is then built around the point forecast as $\hat{y} \pm \hat{q}$. Skforecast implements [Split Conformal Prediction (SCP)](https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/#2-the-split-method) through the [`predict_interval(method='conformal')`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_interval) method. Conformal prediction is available in the same forecasters as bootstrapping, and it is the only probabilistic method available in [`ForecasterRnn`](https://skforecast.org/latest/api/forecasterrnn) and [`ForecasterEquivalentDate`](https://skforecast.org/latest/api/forecasterequivalentdate).

Discover how to use conformal methods in [Probabilistic forecasting with conformal prediction](probabilistic-forecasting-conformal-prediction.ipynb).

Conformal methods can also calibrate prediction intervals generated by other techniques, such as quantile regression or bootstrapped residuals. In this case, the conformal method widens or narrows the intervals so that their empirical coverage on a calibration set matches the nominal coverage. Skforecast provides this functionality through the [`ConformalIntervalCalibrator`](https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing._preprocessing.ConformalIntervalCalibrator) transformer.

Discover how to perform this calibration in [Probabilistic forecasting, conformal calibration](probabilistic-forecasting-conformal-calibration.ipynb).

!!! warning

    There are several well-established [methods for conformal prediction](https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/), each with its own characteristics and assumptions. However, when applied to time series forecasting, their coverage guarantees are only valid for one-step-ahead predictions. For multi-step-ahead predictions, the coverage probability is not guaranteed, so it is advisable to validate the empirical coverage with backtesting. Skforecast implements Split Conformal Prediction (SCP) due to its balance between complexity and performance.


### Quantile regression

Quantile regression is a technique for estimating the conditional quantiles of a response variable. Instead of minimizing the squared error, the model minimizes the quantile loss (also known as pinball loss), which penalizes underestimation and overestimation asymmetrically, so that the predictions converge to the requested quantile. By combining the predictions of two quantile estimators, an interval can be constructed, with each model estimating one of the bounds of the interval. For example, models trained on $Q = 0.1$ and $Q = 0.9$ produce a nominal 80% prediction interval ($90\% - 10\% = 80\%$).

If a machine learning algorithm capable of modeling quantiles is used as the `estimator` in a forecaster, the `predict` method returns predictions for the quantile the estimator was trained on. Examples of such estimators are LightGBM (`LGBMRegressor(objective='quantile', alpha=0.9)`) and scikit-learn's `GradientBoostingRegressor(loss='quantile', alpha=0.9)` or `HistGradientBoostingRegressor(loss='quantile', quantile=0.9)`. By creating two forecasters, each configured with a different quantile, their predictions can be combined to generate a prediction interval. As explained in the next section, a direct forecaster such as [`ForecasterDirect`](https://skforecast.org/latest/api/forecasterdirect) is preferred for this method.

Discover how to use this method in [Probabilistic forecasting with quantile regression](probabilistic-forecasting-quantile-regression.ipynb).


### Which method to use?

There is no definitive answer to this question, as the resulting coverage may vary depending on the dataset and estimator. However, some general guidelines can be followed:

+ **Bootstrapped residuals**: This method achieves good results in most cases, especially when using residuals conditioned on the predicted value (`use_binned_residuals=True`), since the width of the interval then adapts to the predicted value. For example, when the error grows with the level of the series, this produces narrow intervals for low predictions and wide intervals for high ones. However, it is computationally expensive, especially when using a large number of bootstrapping iterations, and may not scale well to large datasets or multiple time series.

+ **Conformal prediction**: Similar results to bootstrapping, often with narrower intervals, and it is the fastest method, since no bootstrapping is needed. With `use_binned_residuals=True`, the width of the interval also adapts to the predicted value.

+ **Quantile regression**: An appropriate choice when the estimator is capable of modeling quantiles. However, it has three drawbacks. First, one forecaster must be trained for each quantile, which can be computationally expensive. Second, since each model is trained independently, nothing prevents the predicted quantiles from crossing (for example, a lower bound above the upper bound). Third, in recursive forecasters the quantile predictions are fed back as lags, so the predictions of later steps are built on quantile values instead of expected values. For this reason, a direct strategy ([`ForecasterDirect`](https://skforecast.org/latest/api/forecasterdirect)) is preferred.

A comparison of these methods on the same dataset can be found in the article [Probabilistic forecasting with machine learning](https://cienciadedatos.net/documentos/py42-probabilistic-forecasting).


### Other probabilistic forecasting options

Besides the methods above, skforecast offers probabilistic forecasts in other contexts:

+ **Global models**: all the methods described above can be applied to models trained on multiple series. See [Probabilistic forecasting: global models](probabilistic-forecasting-global-models.ipynb).

+ **Statistical models**: [`ForecasterStats`](https://skforecast.org/latest/api/forecasterstats) returns the prediction intervals calculated by the underlying statistical model (ARIMA, SARIMAX, ETS, ARAR) through its `predict_interval()` method.

+ **Foundation models**: [`ForecasterFoundation`](https://skforecast.org/latest/api/forecasterfoundation) returns the quantiles predicted natively by pre-trained foundation models through its `predict_interval()` and `predict_quantiles()` methods, with no residuals or additional training required.

!!! warning

    As Rob J Hyndman explains in his [blog](https://robjhyndman.com/hyndsight/narrow-pi/), in real-world problems, almost all prediction intervals are too narrow. For example, nominal 95% intervals may only provide coverage between 71% and 87%. This is a well-known phenomenon that arises because they do not account for all sources of uncertainty. With forecasting models, there are at least four sources of uncertainty:

    + The random error term
    + The parameter estimates
    + The choice of model for the historical data
    + The continuation of the historical data generating process into the future

    When producing prediction intervals for time series models, generally only the first of these sources is taken into account. Therefore, it is advisable to use test data to validate the empirical coverage of the interval and not only rely on the expected one.

!!! tip

    For more examples on how to use probabilistic forecasting, check out the following articles:

    + [Probabilistic forecasting with machine learning](https://cienciadedatos.net/documentos/py42-probabilistic-forecasting)
    + [Probabilistic forecasting: prediction intervals for multi-step time series forecasting](https://cienciadedatos.net/documentos/py60-probabilistic-forecasting-prediction-intervals-multi-step-forecasting)
    + [Continuous Ranked Probability Score (CRPS) in probabilistic forecasting](../faq/probabilistic-forecasting-crps-score.ipynb)
    + [Calibration of probabilistic forecasting intervals](../faq/probabilistic-forecasting-calibrate-intervals.ipynb)
