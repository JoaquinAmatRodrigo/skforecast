import pytest
import pandas as pd
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


def test_init_exception_when_window_size_argument_is_string():
   '''
   '''
   with pytest.raises(Exception):
        forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = '5'
                    )

def test_init_exception_when_fun_predictors_argument_is_string():
   '''
   '''
   with pytest.raises(Exception):
        forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = 'create_predictors',
                        window_size    = 5
                    )