# Unit test set_out_sample_residuals ForecasterAutoreg
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_set_out_sample_residuals_exception_when_residuals_is_not_pd_Series():
    '''
    Test exception is raised when residuals argument is not pd.Series.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster.set_out_sample_residuals(residuals=[1, 2, 3])


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    '''
    Test len residuals stored when its length is greater than 1000.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(2000)))
    assert len(forecaster.out_sample_residuals) == 1000


def test_same_out_sample_residuals_stored_when_residuals_length_is_greater_than_1000():
    '''
    Test same out sample residuals are stored when its length is greater than 1000
    executing function two times.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(2000)))
    out_sample_residuals_1 = forecaster.out_sample_residuals
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(2000)))
    out_sample_residuals_2 = forecaster.out_sample_residuals
    assert (out_sample_residuals_1 == out_sample_residuals_2).all()


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append():
    '''
    Test residuals stored when new residuals length is less than 1000 and append is False.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(20)))
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(10)), append=False)
    expected = pd.Series(np.arange(10))
    results = forecaster.out_sample_residuals
    assert (results == expected).all()


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append():
    '''
    Test residuals stored when new residuals length is less than 1000 and append is True.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(10)))
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(10)), append=True)
    expected = pd.Series(np.hstack([np.arange(10), np.arange(10)]))
    results = forecaster.out_sample_residuals
    assert (results == expected).all()


def test_set_out_sample_residuals_when_residuals_length_is_more_than_1000_and_append():
    '''
    Test residuals stored when new residuals length is more than 1000 and append is True.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(10)))
    forecaster.set_out_sample_residuals(residuals=pd.Series(np.arange(1000)), append=True)
    expected = pd.Series(np.hstack([np.arange(10), np.arange(1200)])[:1000])
    results = forecaster.out_sample_residuals
    assert (results == expected).all()