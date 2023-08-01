# Unit test fit Sarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterSarimax_2 import Sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

# Fixtures
from .fixtures_ForecasterSarimax import y
from .fixtures_ForecasterSarimax import y_datetime
from .fixtures_ForecasterSarimax import exog
from .fixtures_ForecasterSarimax import exog_datetime
from .fixtures_ForecasterSarimax import exog_predict
from .fixtures_ForecasterSarimax import exog_predict_datetime


def test_decorator_check_fitted():
    """
    Test Sarimax fit with pandas `y` and `exog`.
    """

    sarimax = Sarimax(order=(1, 1, 1))

    err_msg = re.escape(
                ("Sarimax instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using this method.")
              )
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.predict(steps=5)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.append(y=y)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.apply(y=y)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.extend(y=y)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.params()
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.summary()


def test_Sarimax_sarimax_params_are_stored_during_initialization():
    """
    Test if `sarimax._sarimax_params` are stored correctly during initialization.
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
    Check information printed by a Sarimax object.
    """
    sarimax = Sarimax(order=(1, 2, 3), seasonal_order=(4, 5, 6, 12))
    expected_out = "Sarimax(1,2,3)(4,5,6)[12]\n"

    print(sarimax)
    out, _ = capfd.readouterr()

    assert out == expected_out


@pytest.mark.parametrize("kwargs, _init_kwargs", 
                         [({'order': (1,1,1), 
                            'sm_init_kwargs': {'order': (2,2,2)}}, 
                           {'order': (1,1,1)}), 
                          ({'order': (2,2,2), 
                            'sm_init_kwargs': {'test': 'fake_param'}}, 
                           {'order': (2,2,2), 
                            'test': 'fake_param'})], 
                         ids = lambda values : f'kwargs, _init_kwargs: {values}')
def test_Sarimax_consolidate_kwargs_init_kwargs(kwargs, _init_kwargs):
    """
    Test if `sarimax._init_kwargs` are correctly consolidate.
    """
    sarimax = Sarimax(**kwargs)
    results = (sarimax._init_kwargs, sarimax.sm_init_kwargs)

    # First expected _init_kwargs, second sm_init_kwargs
    expected = ({
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
        'validate_specification': True
    },
    kwargs['sm_init_kwargs']
    )
    expected[0].update(_init_kwargs)

    assert all(isinstance(x, dict) for x in results)
    assert results[0] == expected[0]
    assert results[1] == expected[1]


@pytest.mark.parametrize("kwargs, _fit_kwargs", 
                         [({'maxiter': 100, 
                            'sm_fit_kwargs': {'maxiter': 200}}, 
                           {'maxiter': 100}), 
                          ({'maxiter': 200, 
                            'sm_fit_kwargs': {'test': 'fake_param'}}, 
                           {'maxiter': 200, 
                            'test': 'fake_param'})], 
                         ids = lambda values : f'kwargs, _fit_kwargs: {values}')
def test_Sarimax_consolidate_kwargs_fit_kwargs(kwargs, _fit_kwargs):
    """
    Test if `sarimax._fit_kwargs` are correctly consolidate.
    """
    sarimax = Sarimax(**kwargs)
    results = (sarimax._fit_kwargs, sarimax.sm_fit_kwargs)

    # First expected _fit_kwargs, second sm_fit_kwargs
    expected = ({
        'method': 'lbfgs',
        'maxiter': 50,
        'start_params': None,
        'disp': False
    },
    kwargs['sm_fit_kwargs']
    )
    expected[0].update(_fit_kwargs)

    assert all(isinstance(x, dict) for x in results)
    assert results[0] == expected[0]
    assert results[1] == expected[1]


@pytest.mark.parametrize("kwargs, _predict_kwargs", 
                         [({'sm_predict_kwargs': {'test': 'fake_param'}}, 
                           {'test': 'fake_param'})], 
                         ids = lambda values : f'kwargs, _predict_kwargs: {values}')
def test_Sarimax_consolidate_kwargs_predict_kwargs(kwargs, _predict_kwargs):
    """
    Test if `sarimax._predict_kwargs` are correctly consolidate.
    """
    sarimax = Sarimax(**kwargs)
    results = (sarimax._predict_kwargs, sarimax.sm_predict_kwargs)

    # First expected _predict_kwargs, second sm_predict_kwargs
    expected = (
    {},
    kwargs['sm_predict_kwargs']
    )
    expected[0].update(_predict_kwargs)

    assert all(isinstance(x, dict) for x in results)
    assert results[0] == expected[0]
    assert results[1] == expected[1]


def test_Sarimax_create_sarimax():
    """
    Test statsmodels SARIMAX is correctly create with _create_sarimax.
    """
    sarimax = Sarimax(order=(1, 1, 1))
    sarimax._create_sarimax(endog=y)
    sarimax_stats = sarimax.sarimax

    assert isinstance(sarimax_stats, SARIMAX)
    assert sarimax_stats.order == (1, 1, 1)


def test_Sarimax_fit_with_numpy():
    """
    Test Sarimax fit with numpy `y` and `exog`.
    """
    y_numpy = y.to_numpy()
    exog_numpy = exog.to_numpy()

    sarimax = Sarimax(order=(1, 1, 1))

    assert sarimax.output_type is None
    assert sarimax.sarimax_res is None
    assert sarimax.fitted == False
    assert sarimax.training_index is None

    sarimax.fit(y=y_numpy, exog=exog_numpy)

    assert sarimax.output_type == 'numpy'
    assert isinstance(sarimax.sarimax_res, SARIMAXResultsWrapper)
    assert sarimax.fitted == True
    assert sarimax.training_index == None


@pytest.mark.parametrize("y, exog", 
                         [(y, exog),
                          (y_datetime, exog_datetime),
                          (y.to_frame(), exog.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame())], 
                         ids = lambda values : f'y, exog: {type(values)}')
def test_Sarimax_fit_with_pandas(y, exog):
    """
    Test Sarimax fit with pandas `y` and `exog`.
    """

    sarimax = Sarimax(order=(1, 1, 1))

    assert sarimax.output_type is None
    assert sarimax.sarimax_res is None
    assert sarimax.fitted == False
    assert sarimax.training_index is None

    sarimax.fit(y=y, exog=exog)

    assert sarimax.output_type == 'pandas'
    assert isinstance(sarimax.sarimax_res, SARIMAXResultsWrapper)
    assert sarimax.fitted == True
    pd.testing.assert_index_equal(sarimax.training_index, y.index)


def test_Sarimax_predict_with_numpy():
    """
    Test Sarimax fit with numpy `y` and `exog`.
    """
    y_numpy = y.to_numpy()
    exog_numpy = exog.to_numpy()
    exog_predict_numpy = exog_predict.to_numpy()

    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, trend=None, method='nm', 
                      sm_fit_kwargs={'ftol': 3e-16})
    sarimax.fit(y=y_numpy, exog=exog_numpy)

    predictions = sarimax.predict(steps=10, exog=exog_predict_numpy)
    expected = np.array([0.56646657, 0.60395361, 0.63537604, 0.66633791, 0.70051882,
                         0.73498455, 0.78119222, 0.8035615 , 0.75481498, 0.69921447])

    np.testing.assert_almost_equal(predictions, expected)


@pytest.mark.parametrize("y, exog", 
                         [(y, exog),
                          (y_datetime, exog_datetime),
                          (y.to_frame(), exog.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame())], 
                         ids = lambda values : f'y, exog: {type(values)}')
def test_Sarimax_predict_with_pandas(y, exog):
    """
    Test Sarimax fit with pandas `y` and `exog`.
    """
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, trend=None, method='nm', 
                      sm_fit_kwargs={'ftol': 3e-16})
    sarimax.fit(y=y, exog=exog)

    predictions = sarimax.predict(steps=10, exog=exog_predict)

    if isinstance(y.index, pd.RangeIndex):
        idx = pd.RangeIndex(start=50, stop=60, step=1)
    else:
        idx = pd.date_range(start='2050', periods=10, freq='A')

    expected = pd.DataFrame(
        data    = np.array([0.56646657, 0.60395361, 0.63537604, 0.66633791, 0.70051882,
                            0.73498455, 0.78119222, 0.8035615 , 0.75481498, 0.69921447]),
        index   = idx,
        columns = ['pred']
    )

    pd.testing.assert_frame_equal(predictions, expected)


