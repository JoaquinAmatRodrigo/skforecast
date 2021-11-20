import numpy as np
import pandas as pd
from pytest import approx
from skforecast import __version__
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression


def create_predictors(y):
    '''
    Create first 5 lags of a time series.
    '''
    X_train = pd.DataFrame({'y':y.copy()})
    for i in range(0, 5):
        X_train[f'lag_{i+1}'] = X_train['y'].shift(i)
    X_train = X_train.drop(columns='y').tail(1).to_numpy()  
    
    return X_train  



def test_fit_last_window_stored():
    '''
    Test that values of last window are stored after fitting.
    '''   
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=pd.Series(np.arange(50)))
    expected = pd.Series(np.array([45, 46, 47, 48, 49]), index=[45, 46, 47, 48, 49])
    assert (forecaster.last_window == expected).all()
    
def test_in_sample_residuals_stored_when_fit_forecaster():
    '''
    Test that values of in_sample_residuals are stored after fitting.
    '''   
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )

    forecaster.fit(y=pd.Series(np.arange(7)))
    expected = np.array([0, 0])
    results = forecaster.in_sample_residuals  
    assert results.values == approx(expected)