from pytest import approx
import numpy as np
import pandas as pd
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
    

def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    '''
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    1 step is predicted using in-sample residuals.
    '''
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20., 20.]])
    results = forecaster._estimate_boot_interval(steps=1, in_sample_residuals=True, n_boot=2)  
    assert results == approx(expected)
