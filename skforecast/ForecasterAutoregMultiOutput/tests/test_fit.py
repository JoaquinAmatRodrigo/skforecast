# Unit test fit
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from sklearn.linear_model import LinearRegression

        
def test_fit_last_window_stored():
    '''
    Test that values of last window are stored after fitting.
    '''    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=pd.Series(np.arange(50)))
    expected = pd.Series(np.array([46, 47, 48, 49]), index=[46, 47, 48, 49])
    results = forecaster.last_window
    pd.testing.assert_series_equal(expected, results)
