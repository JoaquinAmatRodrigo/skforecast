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
    assert (forecaster.lags == np.arange(10)).all()
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])
    assert (forecaster.lags == np.array([1, 2, 3]) - 1).all()
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=range(1, 4))
    assert (forecaster.lags == np.array(range(1, 4)) - 1).all()
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=np.arange(1, 10))
    assert (forecaster.lags == np.arange(1, 10) - 1).all()
    

def test_create_lags():
    '''
    Check matrix of lags is created properly
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    results = forecaster.create_lags(y=np.arange(10))
    correct = (np.array([[0., 1., 2.],
                        [1., 2., 3.],
                        [2., 3., 4.],
                        [3., 4., 5.],
                        [4., 5., 6.],
                        [5., 6., 7.],
                        [6., 7., 8.]]),
                    np.array([3., 4., 5., 6., 7., 8., 9.]))
    
    assert (results[0] == correct[0]).all()
    assert (results[1] == correct[1]).all()

    
def test_fit_last_window():
    '''
    Check last window stored during fit.
    '''
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    
    assert (forecaster.last_window == np.array([47, 48, 49])).all()
    

def test_predict_output():
    '''
    Check prediction output. Use LinearRegression() since the output is deterministic.
    '''
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    predictions = forecaster.predict(steps=5)
    
    assert predictions == approx(np.array([50., 51., 52., 53., 54.]))
