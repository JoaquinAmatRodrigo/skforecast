# Unit test _calculate_metrics_one_step_ahead
# ==============================================================================
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection.model_selection import _calculate_metrics_one_step_ahead
from skforecast.metrics import add_y_train_argument

# Fixtures
from .fixtures_model_selection import y
from .fixtures_model_selection import exog


def test_calculate_metrics_one_step_ahead_when_forecaster_autoreg():
    """
    Testing _calculate_metrics_one_step_ahead when forecaster is of type ForecasterAutoreg.
    """

    forecaster = ForecasterAutoreg(regressor = LinearRegression(), lags = 5)
    metrics = [mean_absolute_error, mean_absolute_percentage_error, mean_absolute_scaled_error]
    metrics = [add_y_train_argument(metric) for metric in metrics]
    X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(y=y, exog=exog, initial_train_size=10)
    results = _calculate_metrics_one_step_ahead(forecaster=forecaster, metrics=metrics, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    results = [float(result) for result in results]
    expected = [0.4270805279719919, 1.649832582914358, 1.4902865958485683]

    assert results == expected


def test_calculate_metrics_one_step_ahead_when_forecaster_autoreg_direct():
    """
    Testing _calculate_metrics_one_step_ahead when forecaster is of type ForecasterAutoregDirect.
    """

    forecaster = ForecasterAutoregDirect(regressor = LinearRegression(), lags = 5, steps=3)
    metrics = [mean_absolute_error, mean_absolute_percentage_error, mean_absolute_scaled_error]
    metrics = [add_y_train_argument(metric) for metric in metrics]
    X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(y=y, exog=exog, initial_train_size=10)
    results = _calculate_metrics_one_step_ahead(forecaster=forecaster, metrics=metrics, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    results = [float(result) for result in results]
    expected = [0.4345264892604896, 1.710997093692879, 1.5162690876145781]

    assert results == expected
