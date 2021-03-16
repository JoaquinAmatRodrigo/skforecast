# test_ForecasterCustom.py 
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast import __version__
from skforecast.ForecasterCustom import ForecasterCustom
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


def test_init_lags():
    '''
    Check creation of self.lags attribute when initialize.
    '''    
    forecaster = ForecasterCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                 )
    
    assert forecaster.window_size == 5
    
    
def test_init_lags_exceptions():
    '''
    Check exceptions when initialize lags.
    '''    
    with pytest.raises(Exception):
        forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = [5]
                     )
    with pytest.raises(Exception):
        forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = '5'
                     )
        
def test_fit_exceptions():
    '''
    Check exceptions during fit.
    '''
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                     )
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=pd.Series(np.arange(10)))
        
        
def test_fit_last_window():
    '''
    Check last window stored during fit.
    '''
    
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50))
    
    assert (forecaster.last_window == np.arange(50)[-5:]).all()
    
    
def test_predict_exceptions():
    '''
    Check exceptions when predict.
    '''
    
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50))
        forecaster.predict(steps=10, exog=np.arange(10))
        
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(50))
        forecaster.predict(steps=10)
        
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(50))
        forecaster.predict(steps=10, exog=np.arange(5)) 
        
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(50))
        forecaster.predict(steps=10, exog=np.arange(5))
        
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50))
        forecaster.predict(steps=10, last_window=[1, 2, 3])
        
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50))
        forecaster.predict(steps=10, last_window=pd.Series([1, 2]))
        
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50))
        forecaster.predict(steps=10, last_window=np.array([1, 2]))
        
def test_predict_output():
    '''
    Check prediction output. Use LinearRegression() since the output is deterministic.
    '''
    
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    
    forecaster.fit(y=np.arange(50))
    predictions = forecaster.predict(steps=5)
    
    assert predictions == approx(np.array([50., 51., 52., 53., 54.]))
    
    
def test_check_y():
    '''
    Check _check_y() raises errors.
    '''
    
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    
    with pytest.raises(Exception):
        forecaster._check_y(y=10)
        
    with pytest.raises(Exception):
        forecaster._check_y(y=[1, 2, 3])
        
    with pytest.raises(Exception):
        forecaster._check_y(y=np.arange(10).reshape(-1, 1))
        
        
def test_check_exog():
    '''
    Check _check_exog() raises errors.
    '''
    
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    
    with pytest.raises(Exception):
        forecaster._check_exog(exog=10)
        
    with pytest.raises(Exception):
        forecaster._check_exog(exog=[1, 2, 3])
        
    with pytest.raises(Exception):
        forecaster._check_exog(exog=np.arange(30).reshape(-1, 10, 3))
        
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog     = np.arange(30).reshape(-1, 2),
            ref_type = pd.Series
        )
        
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = np.arange(30),
            ref_type  = np.ndarray,
            ref_shape = (1, 5)
        )
        
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = np.arange(30).reshape(-1, 3),
            ref_type  = np.ndarray,
            ref_shape = (1, 2)
        )
        
        
def test_preproces_y():
    '''
    Check _check_exog() raises errors.
    '''
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    
    assert (forecaster._preproces_y(y=np.arange(3)) == np.arange(3)).all()
    assert (forecaster._preproces_y(y=pd.Series([0, 1, 2])) == np.arange(3)).all()
    
    
def test_preproces_exog():
    '''
    Check _check_exog() raises errors.
    '''
    
    forecaster = ForecasterCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    
    assert (forecaster._preproces_exog(exog=np.arange(3)) == np.arange(3).reshape(-1, 1)).all()
    assert (forecaster._preproces_exog(exog=pd.Series([0, 1, 2])) == np.arange(3).reshape(-1, 1)).all()
