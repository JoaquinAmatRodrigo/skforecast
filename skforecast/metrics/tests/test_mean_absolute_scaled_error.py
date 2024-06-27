# Unit test mean_absolute_scaled_error
# ==============================================================================
from skforecast.metrics import mean_absolute_scaled_error
import pytest
import numpy as np
import pandas as pd

def test_mean_absolute_scaled_error_output():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    expected_mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_train)))
    assert np.isclose(mean_absolute_scaled_error(y_true, y_pred, y_train), expected_mase)

def test_mean_absolute_scaled_error_input_types():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    with pytest.raises(TypeError):
        mean_absolute_scaled_error([1, 2, 3], y_pred, y_train)
    with pytest.raises(TypeError):
        mean_absolute_scaled_error(y_true, [1, 2, 3], y_train)
    with pytest.raises(TypeError):
        mean_absolute_scaled_error(y_true, y_pred, [1, 2, 3])

def test_mean_absolute_scaled_error_input_length():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    with pytest.raises(ValueError):
        mean_absolute_scaled_error(y_true, y_pred, y_train)
    with pytest.raises(ValueError):
        mean_absolute_scaled_error(y_true, y_pred, y_train)

def test_mean_absolute_scaled_error_empty_input():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    with pytest.raises(ValueError):
        mean_absolute_scaled_error(np.array([]), y_pred, y_train)
    with pytest.raises(ValueError):
        mean_absolute_scaled_error(y_true, np.array([]), y_train)
    
def test_mean_absolute_scaled_error_pandas_series_input():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    y_train_series = pd.Series(y_train)
    expected_mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_train)))
    assert np.isclose(mean_absolute_scaled_error(y_true_series, y_pred_series, y_train_series), expected_mase)
    
