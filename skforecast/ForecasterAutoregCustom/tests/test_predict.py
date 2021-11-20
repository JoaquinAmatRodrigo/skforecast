from pytest import approx
import numpy as np
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


def test_predict_output_when_regressor_is_LinearRegression():
    '''
    Test predict output when using LinearRegression as regressor.
    '''
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=pd.Series(np.arange(50)))
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                data = np.array([50., 51., 52., 53., 54.]),
                index = [3, 4, 5, 6, 7]
               )
    assert (predictions.values == approx(expected.values))
    assert (predictions.index == expected.index).all()

