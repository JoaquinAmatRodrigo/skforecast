# Unit test predict
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from sklearn.linear_model import LinearRegression

        
def test_predict_output_when_regressor_is_LinearRegression():
    '''
    Test predict output when using LinearRegression as regressor.
    '''
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(50)))
    results = forecaster.predict()
    expected = pd.Series(
                data = np.array([50., 51., 52.]),
                index = [4, 5, 6],
                name = 'pred'
               )
    pd.testing.assert_series_equal(results, expected)
    