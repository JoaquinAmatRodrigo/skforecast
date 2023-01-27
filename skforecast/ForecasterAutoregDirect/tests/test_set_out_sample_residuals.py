# Unit test set_params ForecasterAutoregDirect
# ==============================================================================
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import re
import pytest


def test_set_out_sample_residuals_exception_when_residuals_is_not_dict():
    """
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(fit_intercept=True), lags=3, steps=2)
    residuals = [1, 2, 3]
    err_msg = re.escape(
                f"`residuals` argument must be a dict of `pd.Series`. Got {type(residuals)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_exception_when_residuals_is_dict_of_non_pandas_series():
    """
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(fit_intercept=True), lags=3, steps=2)
    residuals = {1:[1,2,3,4]}
    err_msg = re.escape(
                f"`residuals` argument must be a dict of `pd.Series`. Got {type(residuals)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    residuals = {1:pd.Series(np.arange(10)), 2:pd.Series(np.arange(10))}
    forecaster = ForecasterAutoregDirect(LinearRegression(fit_intercept=True), lags=3, steps=2)
    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=residuals, append=False)
    expected = {1:pd.Series(np.arange(10)), 2:pd.Series(np.arange(10))}
    results = forecaster.out_sample_residuals

    assert expected.keys() == results.keys()
    assert all(all(expected[k] ==results[k]) for k in expected.keys())


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    residuals = {1:pd.Series(np.arange(10)), 2:pd.Series(np.arange(10))}
    forecaster = ForecasterAutoregDirect(LinearRegression(fit_intercept=True), lags=3, steps=2)
    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=residuals, append=True)
    expected = {1:pd.concat([pd.Series(np.arange(10)), pd.Series(np.arange(10))], ignore_index=True),
                2:pd.concat([pd.Series(np.arange(10)), pd.Series(np.arange(10))], ignore_index=True)}
    results = forecaster.out_sample_residuals

    assert expected.keys() == results.keys()
    assert all(all(expected[k] == results[k]) for k in expected.keys())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    """
    Test len residuals stored when its length is greater than 1000.
    """
    residuals = {1:pd.Series(np.arange(2000)), 2:pd.Series(np.arange(2000))}
    forecaster = ForecasterAutoregDirect(LinearRegression(fit_intercept=True), lags=3, steps=2)
    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals

    assert list(results.keys()) == [1, 2]
    assert all(len(value)==1000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_keys_do_not_match():
    """
    """
    residuals = {3:pd.Series(np.arange(10)), 4:pd.Series(np.arange(10))}
    forecaster = ForecasterAutoregDirect(LinearRegression(fit_intercept=True), lags=3, steps=2)
    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals

    assert results == {1: None, 2: None}


def test_set_out_sample_residuals_when_residuals_keys_partially_match():
    """
    """
    residuals = {1:pd.Series(np.arange(10)), 4:pd.Series(np.arange(10))}
    forecaster = ForecasterAutoregDirect(LinearRegression(fit_intercept=True), lags=3, steps=2)
    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals

    assert list(results.keys()) == [1, 2]
    assert all(results[1] == pd.Series(np.arange(10)))
    assert results[2] is None