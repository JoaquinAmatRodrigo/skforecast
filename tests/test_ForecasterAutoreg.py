# test_ForecasterAutoreg.py 
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast import __version__
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_init_lags():
    '''
    Check creation of self.lags attribute when initialize.
    '''    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=10)
    assert (forecaster.lags == np.arange(10) + 1).all()
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])
    assert (forecaster.lags == np.array([1, 2, 3])).all()
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=range(1, 4))
    assert (forecaster.lags == np.array(range(1, 4))).all()
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=np.arange(1, 10))
    assert (forecaster.lags == np.arange(1, 10)).all()

    
    
def test_init_lags_exceptions():
    '''
    Check exceptions when initialize lags.
    '''    
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=-10)
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=range(0, 4))
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=np.arange(0, 4))
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=[0, 1, 2])
        
    
def test_create_lags():
    '''
    Check matrix of lags is created properly
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    results = forecaster.create_lags(y=np.arange(10))
    correct = (np.array([[2., 1., 0.],
                        [3., 2., 1.],
                        [4., 3., 2.],
                        [5., 4., 3.],
                        [6., 5., 4.],
                        [7., 6., 5.],
                        [8., 7., 6.]]),
             np.array([3., 4., 5., 6., 7., 8., 9.]))

    assert (results[0] == correct[0]).all()
    assert (results[1] == correct[1]).all()
    


def test_create_lags_exceptions():
    '''
    Check exceptions when creating lags.
    '''
    with pytest.raises(Exception):
        forecaster = ForecasterAutoreg(LinearRegression(), lags=10)
        forecaster.create_lags(y=np.arange(5))
        

def test_fit_exceptions():
    '''
    Check exceptions during fit.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=5)
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=pd.Series(np.arange(10)))
        
        
def test_fit_last_window():
    '''
    Check last window stored during fit.
    '''
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    
    assert (forecaster.last_window == np.array([47, 48, 49])).all()
    
    
def test_predict_exceptions():
    '''
    Check exceptions when predict.
    '''
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    
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
        forecaster.predict(steps=10, last_window=[1,2,3])
        
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
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    predictions = forecaster.predict(steps=5)
    
    assert predictions == approx(np.array([50., 51., 52., 53., 54.]))
    
    
def test_check_y():
    '''
    Check _check_y() raises errors.
    '''
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    
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
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    
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
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    
    assert (forecaster._preproces_y(y=np.arange(3)) == np.arange(3)).all()
    assert (forecaster._preproces_y(y=pd.Series([0, 1, 2])) == np.arange(3)).all()
    
    
def test_preproces_exog():
    '''
    Check _check_exog() raises errors.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    
    assert (forecaster._preproces_exog(exog=np.arange(3)) == np.arange(3).reshape(-1, 1)).all()
    assert (forecaster._preproces_exog(exog=pd.Series([0, 1, 2])) == np.arange(3).reshape(-1, 1)).all()
