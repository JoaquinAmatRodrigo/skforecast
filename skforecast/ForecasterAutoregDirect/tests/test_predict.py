# Unit test predict ForecasterAutoregDirect
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression

        
def test_predict_output_when_regressor_is_LinearRegression():
    '''
    Test predict output when using LinearRegression as regressor.
    '''
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(50)))
    results = forecaster.predict()
    expected = pd.Series(
                data = np.array([50., 51., 52.]),
                index = pd.RangeIndex(start=50, stop=53, step=1),
                name = 'pred'
               )
    pd.testing.assert_series_equal(results, expected)
    