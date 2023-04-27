# Unit test get_feature_importances ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax import ForecasterSarimax
from pmdarima.arima import ARIMA
from sklearn.exceptions import NotFittedError


def test_exception_is_raised_when_forecaster_is_not_fitted():
    """
    Test exception is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
              )
    with pytest.raises(NotFittedError, match=err_msg):         
        forecaster.get_feature_importances()


def test_output_get_feature_importances_ForecasterSarimax():
    """
    Test output of get_feature_importances ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=pd.Series(np.arange(10)))
    expected = pd.DataFrame({
                    'feature': ['intercept', 'ar.L1', 'ma.L1', 'sigma2'],
                    'importance': np.array([0.49998574676910396, 0.5000130662306124, 
                                            7.479723906909597e-11, 2.658043128694438e-12])
                })
    results = forecaster.get_feature_importances()

    pd.testing.assert_frame_equal(expected, results)