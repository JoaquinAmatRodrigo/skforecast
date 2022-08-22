# Unit test set_out_sample_residuals ForecasterAutoreg
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


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


def test_set_out_sample_residuals_when_transform_is_True():
    '''
    Test residuals stored when using a StandardScaler() as transformer.
    '''

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, transformer_y=StandardScaler())
    y = pd.Series(
            np.array([
                12.5, 10.3,  9.9, 10.4,  9.9,  8.5, 10.6, 11.4, 10. ,  9.5, 10.1,
                11.5, 11.4, 11.3, 10.5,  9.6, 10.4, 11.7,  8.7, 10.6])
        )
    forecaster.fit(y=y)
    new_residuals = pd.Series(np.random.normal(size=100), name='residuals')
    new_residuals_transformed = forecaster.transformer_y.transform(new_residuals.to_frame())
    new_residuals_transformed = pd.Series(new_residuals_transformed.flatten(), name='residuals')
    forecaster.set_out_sample_residuals(residuals=new_residuals, transform=True)

    pd.testing.assert_series_equal(new_residuals_transformed, forecaster.out_sample_residuals)