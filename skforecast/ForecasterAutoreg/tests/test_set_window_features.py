# Unit test set_window_features ForecasterAutoreg
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from skforecast.preprocessing import RollingFeatures
from skforecast.ForecasterAutoreg import ForecasterAutoreg


@pytest.mark.parametrize("wf", 
                         [RollingFeatures(stats='mean', window_sizes=6),
                          [RollingFeatures(stats='mean', window_sizes=6)]],
                         ids = lambda wf: f'window_features: {type(wf)}')
def test_set_window_features_with_different_inputs(wf):
    """
    Test how attributes change with window_features argument.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=5)
    forecaster.set_window_features(window_features=wf)

    assert forecaster.max_lag == 5
    assert forecaster.max_size_window_features == 6
    assert forecaster.window_features_names == ['roll_mean_6']
    assert forecaster.window_features_class_names == ['RollingFeatures']
    assert forecaster.window_size == 6


def test_set_window_features_when_differentiation_is_not_None():
    """
    Test how `window_size` is also updated when the forecaster includes 
    differentiation.
    """
    forecaster = ForecasterAutoreg(
                     regressor       = LinearRegression(),
                     lags            = 3,
                     window_features = RollingFeatures(stats='median', window_sizes=2),
                     differentiation = 1
                 )
    
    rolling = RollingFeatures(stats='mean', window_sizes=6)
    forecaster.set_window_features(window_features=rolling)

    assert forecaster.max_lag == 3
    assert forecaster.max_size_window_features == 6
    assert forecaster.window_features_names == ['roll_mean_6']
    assert forecaster.window_features_class_names == ['RollingFeatures']
    assert forecaster.window_size == 6 + 1


def test_set_window_features_when_lags():
    """
    Test how `window_size` is also updated when the forecaster includes
    lags.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=10)
    forecaster = ForecasterAutoreg(
                     regressor       = LinearRegression(),
                     lags            = 9,
                     window_features = rolling
                 )
    
    rolling = RollingFeatures(stats='median', window_sizes=5)
    forecaster.set_window_features(window_features=rolling)

    assert forecaster.max_lag == 9
    assert forecaster.max_size_window_features == 5
    assert forecaster.window_features_names == ['roll_median_5']
    assert forecaster.window_features_class_names == ['RollingFeatures']
    assert forecaster.window_size == 9


def test_set_window_features_to_None():
    """
    Test how attributes change when window_features is set to None.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=6)
    forecaster = ForecasterAutoreg(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     window_features = rolling
                 )
    
    forecaster.set_window_features(window_features=None)

    assert forecaster.max_lag == 5
    assert forecaster.max_size_window_features is None
    assert forecaster.window_features_names is None
    assert forecaster.window_features_class_names is None
    assert forecaster.window_size == 5
