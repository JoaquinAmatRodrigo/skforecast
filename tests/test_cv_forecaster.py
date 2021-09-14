import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
from skforecast.model_selection import cv_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


# Test cv_forecaster for ForecasterAutoreg
#-------------------------------------------------------------------------------
def test_cv_forecaster_output_when_y_is_nparange_20_initial_train_size_10_steps_5_allow_incomplete_fold_True():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])

    results = cv_forecaster(
                forecaster=forecaster,
                y=np.arange(20),
                initial_train_size=10,
                steps=5,
                metric='mean_absolute_error',
                exog=None,
                allow_incomplete_fold=True,
                verbose=True
            )[0]
    expected = np.array([0,0])
    assert results == approx(expected)
       

def test_cv_forecaster_output_when_y_is_nparange_20_initial_train_size_10_steps_4_allow_incomplete_fold_True():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])

    results = cv_forecaster(
                forecaster=forecaster,
                y=np.arange(20),
                initial_train_size=10,
                steps=4,
                metric='mean_absolute_error',
                exog=None,
                allow_incomplete_fold=True,
                verbose=True
            )[0]
    expected = np.array([0, 0, 0])
    assert results == approx(expected)


def test_cv_forecaster_output_when_y_is_nparange_20_initial_train_size_10_steps_5_allow_incomplete_fold_False():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])

    results = cv_forecaster(
                forecaster=forecaster,
                y=np.arange(20),
                initial_train_size=10,
                steps=4,
                metric='mean_absolute_error',
                exog=None,
                allow_incomplete_fold=False,
                verbose=True
            )[0]
    expected = np.array([0, 0])
    assert results == approx(expected)


def test_cv_forecaster_output_when_y_and_initial_train_size_have_same_lenght():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])

    results = cv_forecaster(
                forecaster=forecaster,
                y=np.arange(20),
                initial_train_size=20,
                steps=5,
                metric='mean_absolute_error',
                exog=None,
                allow_incomplete_fold=True,
                verbose=True
            )
    expected = (np.array([]), np.array([]))
    assert results == approx(expected)

def test_cv_forecaster_output_when_initial_train_size_is_higer_than_y_lenght():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])

    with pytest.raises(Exception):
        cv_forecaster(
            forecaster=forecaster,
            y=np.arange(20),
            initial_train_size=50,
            steps=5,
            metric='mean_absolute_error',
            exog=None,
            allow_incomplete_fold=True,
            verbose=True
        )


def test_cv_forecaster_exception_when_forcaster_lags_is_higer_than_y_lenght():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=40)

    with pytest.raises(Exception):
        results = cv_forecaster(
                        forecaster=forecaster,
                        y=np.arange(20),
                        initial_train_size=10,
                        steps=5,
                        metric='mean_absolute_error',
                        exog=None,
                        allow_incomplete_fold=True,
                        verbose=True
                    )
