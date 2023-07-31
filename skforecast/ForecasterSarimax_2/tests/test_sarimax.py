# Unit test fit Sarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax_2 import Sarimax

# Fixtures
from .fixtures_ForecasterSarimax import y
from .fixtures_ForecasterSarimax import y_datetime


def test_Sarimax_sarimax_params_are_stored_during_initialization():
    """
    Check if `_sarimax_params` are stored correctly during initialization
    """
    sarimax = Sarimax()
    results = sarimax._sarimax_params

    expected_params = {
        'order': (1, 0, 0),
        'seasonal_order': (0, 0, 0, 0),
        'trend': None,
        'measurement_error': False,
        'time_varying_regression': False,
        'mle_regression': True,
        'simple_differencing': False,
        'enforce_stationarity': True,
        'enforce_invertibility': True,
        'hamilton_representation': False,
        'concentrate_scale': False,
        'trend_offset': 1,
        'use_exact_diffuse': False,
        'dates': None,
        'freq': None,
        'missing': 'none',
        'validate_specification': True,
        'method': 'lbfgs',
        'maxiter': 50,
        'start_params': None,
        'disp': False,
        'sm_init_kwargs': {},
        'sm_fit_kwargs': {},
        'sm_predict_kwargs': {}
    }

    assert isinstance(results, dict)
    assert results == expected_params


def test_Sarimax__repr__(capfd):
    """
    Check if `_sarimax_params` are stored correctly during initialization
    """
    sarimax = Sarimax(order=(1, 2, 3), seasonal_order=(4, 5, 6, 12))
    expected_out = sarimax.__repr__

    out, _ = capfd.readouterr()

    assert out == expected_out 

