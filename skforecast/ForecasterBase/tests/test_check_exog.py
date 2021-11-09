import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_check_exog_exception_when_exog_is_int():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(exog=10)
        
        
def test_check_exog_exception_when_exog_is_list():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(exog=[1, 2, 3])
        
        
def test_check_exog_exception_when_exog_is_numpy_array_with_more_than_2_dimensions():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(exog=np.arange(30).reshape(-1, 10, 3))
        
        
def test_check_exog_exception_when_ref_type_is_pandas_series_and_exog_is_numpy_array_with_more_than_1_column():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog     = np.arange(30).reshape(-1, 2),
            ref_type = pd.Series
        )
        
        
def test_check_exog_exception_when_ref_type_is_pandas_series_and_exog_is_numpy_array_with_more_than_2_dimensions():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(exog=np.arange(30).reshape(-1, 10, 3))
        
        
def test_check_exog_exception_when_exog_has_diferent_number_of_columns_than_ref_shape():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = np.arange(30).reshape(-1, 3),
            ref_type  = np.ndarray,
            ref_shape = (1, 2)
        )

def test_check_exog_exception_when_exog_is_1d_numpy_array_and_ref_shape_has_more_than_1_column():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = np.arange(30),
            ref_type  = np.ndarray,
            ref_shape = (1, 5)
        )
        
        
def test_check_exog_exception_when_exog_is_pandas_series_and_ref_shape_has_more_than_1_column():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = pd.Series(np.arange(30)),
            ref_type  = np.ndarray,
            ref_shape = (1, 5)
        )

def test_check_exog_exception_when_exog_is_pandas_dataframe_with_3_columns_and_ref_shape_has_2_columns():
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = pd.DataFrame(np.arange(30).reshape(-1, 3)),
            ref_type  = pd.DataFrame,
            ref_shape = (10, 2)
        )