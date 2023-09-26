# Unit test _get_metric
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.model_selection.model_selection import _get_metric
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error


def test_get_metric_ValueError_when_metric_not_in_metrics_allowed():
    """
    Test ValueError is raised when metric is not in metrics allowed.
    """
    metric = 'not_a_metric'
    
    err_msg = re.escape(
                (f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error', "
                 f"'mean_absolute_percentage_error' and 'mean_squared_log_error'. Got {metric}.")
              )
    with pytest.raises(ValueError, match = err_msg):
        _get_metric(metric)


def test_get_metric_import_and_calculate_mean_squared_error_correctly():
    """
    Test get_metric import and calculate mean_squared_error correctly.
    """
    metric_str = 'mean_squared_error'
    metric = _get_metric(metric_str)
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    expected = mean_squared_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected


def test_get_metric_import_and_calculate_mean_absolute_error_correctly():
    """
    Test get_metric import and calculate mean_absolute_error correctly.
    """
    metric_str = 'mean_absolute_error'
    metric = _get_metric(metric_str)
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    expected = mean_absolute_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected


def test_get_metric_import_and_calculate_mean_absolute_percentage_error_correctly():
    """
    Test get_metric import and calculate mean_absolute_percentage_error correctly.
    """
    metric_str = 'mean_absolute_percentage_error'
    metric = _get_metric(metric_str)
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    expected = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected


def test_get_metric_import_and_calculate_mean_squared_log_error_correctly():
    """
    Test get_metric import and calculate mean_squared_log_error correctly.
    """
    metric_str = 'mean_squared_log_error'
    metric = _get_metric(metric_str)
    y_true = [3, 5, 2.5, 7]
    y_pred = [2.5, 5, 4, 8]
    expected = mean_squared_log_error(y_true=y_true, y_pred=y_pred)

    assert metric(y_true=y_true, y_pred=y_pred) == expected