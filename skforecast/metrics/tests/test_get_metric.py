# Unit test _get_metric
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.metrics import _get_metric
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error


def test_get_metric_ValueError_when_metric_not_in_metrics_allowed():
    """
    Test ValueError is raised when metric is not in metrics allowed.
    """
    metric = 'not_a_metric'
    allowed_metrics = [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
        "mean_absolute_scaled_error",
        "root_mean_squared_scaled_error",
    ]
    
    err_msg = re.escape(
                (f"Allowed metrics are: {allowed_metrics}. Got {metric}.")
              )
    with pytest.raises(ValueError, match = err_msg):
        _get_metric(metric)


def test_get_metric_import_and_calculate_mean_squared_error_correctly():
    """
    Test get_metric import and calculate mean_squared_error correctly.
    """
    metric_str = 'mean_squared_error'
    metric = _get_metric(metric_str)
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    expected = mean_squared_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected


def test_get_metric_import_and_calculate_mean_absolute_error_correctly():
    """
    Test get_metric import and calculate mean_absolute_error correctly.
    """
    metric_str = 'mean_absolute_error'
    metric = _get_metric(metric_str)
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    expected = mean_absolute_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected


def test_get_metric_import_and_calculate_mean_absolute_percentage_error_correctly():
    """
    Test get_metric import and calculate mean_absolute_percentage_error correctly.
    """
    metric_str = 'mean_absolute_percentage_error'
    metric = _get_metric(metric_str)
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    expected = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected


def test_get_metric_import_and_calculate_mean_squared_log_error_correctly():
    """
    Test get_metric import and calculate mean_squared_log_error correctly.
    """
    metric_str = 'mean_squared_log_error'
    metric = _get_metric(metric_str)
    y_true = np.array([3, 5, 2.5, 7])
    y_pred = np.array([2.5, 5, 4, 8])
    expected = mean_squared_log_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected


def test_get_metric_import_and_calculate_mean_absolute_scaled_error_correctly():
    """
    Test get_metric import and calculate mean_absolute_scaled_error correctly.
    """
    metric_str = 'mean_absolute_scaled_error'
    metric = _get_metric(metric_str)
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    y_train = np.array([2, 3, 4, 5])
    expected = mean_absolute_scaled_error(y_true=y_true, y_pred=y_pred, y_train=y_train)

    assert metric(y_true=y_true, y_pred=y_pred, y_train=y_train) == expected


def test_get_metric_import_and_calculate_root_mean_squared_scaled_error_correctly():
    """
    Test get_metric import and calculate root_mean_squared_scaled_error correctly.
    """
    metric_str = 'root_mean_squared_scaled_error'
    metric = _get_metric(metric_str)
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    y_train = np.array([2, 3, 4, 5])
    expected = root_mean_squared_scaled_error(y_true=y_true, y_pred=y_pred, y_train=y_train)

    assert metric(y_true=y_true, y_pred=y_pred, y_train=y_train) == expected