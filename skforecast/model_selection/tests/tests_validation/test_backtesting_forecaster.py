# Unit test backtesting_forecaster
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from ..fixtures_model_selection import y


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    estimator = Ridge(random_state=123),
                    lags      = 2
                 )
    forecaters_allowed = [
        'ForecasterRecursive', 
        'ForecasterDirect',
        'ForecasterEquivalentDate',
        'ForecasterRecursiveClassifier'
    ]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            refit                 = False,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    err_msg = re.escape(
        f"`forecaster` must be of type {forecaters_allowed}. For all other "
        f"types of forecasters use the other functions available in the "
        f"`model_selection` module."
    )
    
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            cv                    = cv,
            metric                = 'mean_absolute_error',
            exog                  = None,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize(
    "interval",
    [None, [0.1, 0.3, 0.9], [0.05, 0.1, 0.9, 0.95], 'bootstrapping', norm],
    ids = lambda interval: f'interval: {interval}'
)
def test_backtesting_forecaster_no_interval_metrics_when_interval_has_not_two_bounds(interval):
    """
    Test that the interval metrics (`coverage`, `interval_width`) are not added 
    to the metrics when `interval` is None, a list of quantiles that is not 2
    quantiles (optionally with the median), 'bootstrapping' or a scipy.stats
    distribution.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    cv = TimeSeriesFold(
             steps              = 3,
             initial_train_size = len(y[:-12]),
             refit              = False
         )

    metric, _ = backtesting_forecaster(
                    forecaster    = forecaster,
                    y             = y,
                    cv            = cv,
                    metric        = 'mean_absolute_error',
                    interval      = interval,
                    n_boot        = 10,
                    show_progress = False
                )

    assert list(metric.columns) == ['mean_absolute_error']


def test_backtesting_forecaster_cv_differentiation_is_deprecated_and_ignored():
    """
    Test that `differentiation` of the cv is deprecated and has no effect: a
    value different from the one of the forecaster raises a FutureWarning and
    gives the same predictions as `None`. Backtesting does not raise the
    warning, the same cv object works with forecasters with different
    `differentiation` and it is not modified.
    """
    forecaster_diff = ForecasterRecursive(estimator=Ridge(), lags=3, differentiation=1)
    forecaster_no_diff = ForecasterRecursive(estimator=Ridge(), lags=3)
    cv = TimeSeriesFold(steps=4, initial_train_size=30, refit=True, verbose=False)
    warn_msg = re.escape("`differentiation` is deprecated since version 0.27.0")
    with pytest.warns(FutureWarning, match=warn_msg):
        cv_explicit = TimeSeriesFold(
            steps=4, initial_train_size=30, refit=True, differentiation=2,
            verbose=False
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=warn_msg, category=FutureWarning)
        metric, predictions = backtesting_forecaster(
            forecaster=forecaster_diff, y=y, cv=cv, metric='mean_absolute_error',
            show_progress=False
        )
        metric_no_diff, _ = backtesting_forecaster(
            forecaster=forecaster_no_diff, y=y, cv=cv, metric='mean_absolute_error',
            show_progress=False
        )
    metric_explicit, predictions_explicit = backtesting_forecaster(
        forecaster=forecaster_diff, y=y, cv=cv_explicit, metric='mean_absolute_error',
        show_progress=False
    )

    pd.testing.assert_frame_equal(metric, metric_explicit)
    pd.testing.assert_frame_equal(predictions, predictions_explicit)
    assert not metric.equals(metric_no_diff)
    assert cv.differentiation is None
    assert cv.window_size is None


class RidgeNoInputValidation(RegressorMixin, BaseEstimator):
    """
    Ridge regression whose `predict` does not validate its input, so a NaN
    predictor gives a NaN prediction instead of an error.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator_ = Ridge(random_state=self.random_state).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(X) @ self.estimator_.coef_ + self.estimator_.intercept_


@pytest.mark.parametrize(
    "estimator, err_msg",
    [(Ridge(random_state=123),
      "Input X contains NaN in the predictors ['x'] at step 6. Ridge does not "
      "accept missing values encoded as NaN, so its predictions would be NaN."),
     (RidgeNoInputValidation(random_state=123),
      "5 backtesting predictions are NaN at dates with observed values "
      "(first: 2020-04-05 00:00:00), so the metrics cannot be calculated.")],
    ids=lambda x: f"{type(x).__name__}"
)
def test_backtesting_forecaster_ValueError_when_exog_has_NaN_in_test_fold(
    estimator, err_msg
):
    """
    Test backtesting_forecaster raises a ValueError when `exog` has a missing
    value in the test set of a fold. A linear model, which does not accept
    missing values, raises when predicting, naming the predictor and the step.
    An estimator that returns NaN predictions raises before calculating the
    metrics, with the number of NaN predictions and the first date. Before,
    the metric raised 'Input contains NaN.'.
    """
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(100, 5, 120), index=idx, name="y")
    exog = pd.Series(rng.normal(0, 1, 120), index=idx, name="x")
    exog.iloc[95] = np.nan
    forecaster = ForecasterRecursive(estimator, lags=3)
    cv = TimeSeriesFold(steps=10, initial_train_size=60, refit=False)

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        backtesting_forecaster(
            forecaster        = forecaster,
            y                 = y,
            exog              = exog,
            cv                = cv,
            metric            = 'mean_absolute_error',
            show_progress     = False,
            suppress_warnings = True
        )
