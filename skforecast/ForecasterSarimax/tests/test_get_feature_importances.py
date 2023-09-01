# Unit test get_feature_importances ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from sklearn.exceptions import NotFittedError

# Fixtures
from .fixtures_ForecasterSarimax import y
from .fixtures_ForecasterSarimax import exog


@pytest.mark.parametrize("regressor", 
                         [ARIMA(order=(1, 0, 0)), 
                          Sarimax(order=(1, 0, 0))], 
                         ids = lambda reg : f'regressor: {type(reg)}')
def test_NotFittedError_is_raised_when_forecaster_is_not_fitted(regressor):
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterSarimax(regressor=regressor)

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
              )
    with pytest.raises(NotFittedError, match=err_msg):         
        forecaster.get_feature_importances()


def test_output_get_feature_importances_ForecasterSarimax_pmdarima():
    """
    Test output of get_feature_importances ForecasterSarimax pmdarmia.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1, 1, 1)))
    forecaster.fit(y=pd.Series(np.arange(10)))
    expected = pd.DataFrame({
                   'feature': ['intercept', 'ar.L1', 'ma.L1', 'sigma2'],
                   'importance': np.array([0.49998574676910396, 0.5000130662306124, 
                                           7.479723906909597e-11, 2.658043128694438e-12])
               })
    results = forecaster.get_feature_importances()

    pd.testing.assert_frame_equal(expected, results)


def test_output_get_feature_importances_ForecasterSarimax_skforecast():
    """
    Test output of get_feature_importances ForecasterSarimax skforecast.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order= (1, 1, 1), maxiter=1000, method='cg', disp=False)
                 )
    forecaster.fit(y=y, exog=exog)
    expected = pd.DataFrame({
                   'feature': ['exog', 'ar.L1', 'ma.L1', 'sigma2'],
                   'importance': np.array([0.9690539855149568, 0.4666537980992382, 
                                           -0.5263430267037418, 0.7862622654382363])
               })
    results = forecaster.get_feature_importances()

    pd.testing.assert_frame_equal(expected, results)