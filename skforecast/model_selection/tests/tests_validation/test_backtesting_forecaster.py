# Unit test backtesting_forecaster
# ==============================================================================
import re
import pytest
import pandas as pd
from scipy.stats import norm
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
            differentiation       = None,
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


def test_backtesting_forecaster_cv_differentiation_None_uses_forecaster_differentiation():
    """
    Test that when `differentiation` of the cv is `None`, backtesting_forecaster
    uses the `differentiation` of the forecaster: the predictions are the same as
    with the explicit value. The same cv object works with forecasters with
    different `differentiation` and it is not modified.
    """
    forecaster_diff = ForecasterRecursive(estimator=Ridge(), lags=3, differentiation=1)
    forecaster_no_diff = ForecasterRecursive(estimator=Ridge(), lags=3)
    cv = TimeSeriesFold(steps=4, initial_train_size=30, refit=True, verbose=False)
    cv_explicit = TimeSeriesFold(
        steps=4, initial_train_size=30, refit=True, differentiation=1, verbose=False
    )

    metric, predictions = backtesting_forecaster(
        forecaster=forecaster_diff, y=y, cv=cv, metric='mean_absolute_error',
        show_progress=False
    )
    metric_explicit, predictions_explicit = backtesting_forecaster(
        forecaster=forecaster_diff, y=y, cv=cv_explicit, metric='mean_absolute_error',
        show_progress=False
    )
    metric_no_diff, _ = backtesting_forecaster(
        forecaster=forecaster_no_diff, y=y, cv=cv, metric='mean_absolute_error',
        show_progress=False
    )

    pd.testing.assert_frame_equal(metric, metric_explicit)
    pd.testing.assert_frame_equal(predictions, predictions_explicit)
    assert not metric.equals(metric_no_diff)
    assert cv.differentiation is None
    assert cv.window_size is None
