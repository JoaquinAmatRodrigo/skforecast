# Unit test set_fit_kwargs ForecasterSarimax
# ==============================================================================
import re
import pytest
from pmdarima.arima import ARIMA
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.exceptions import IgnoredArgumentWarning


def test_set_fit_kwargs_pmdarima():
    """
    Test set_fit_kwargs method using pmdarima.
    """
    forecaster = ForecasterSarimax(
                     regressor  = ARIMA(order=(1,1,1)),
                     fit_kwargs = {'fit_args': {'optim_score': 'approx'}}
                 )
    
    new_fit_kwargs = {'fit_args': {'optim_score': 'harvey'}}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'fit_args': {'optim_score': 'harvey'}}

    assert results == expected


def test_set_fit_kwargs_skforecast():
    """
    Test set_fit_kwargs method using skforecast.
    """
    forecaster = ForecasterSarimax(
                     regressor  = Sarimax(order=(1, 0, 1))
                 )
    new_fit_kwargs = {'warning': 1}
    
    warn_msg = re.escape(
                 ("When using the skforecast Sarimax model, the fit kwargs should "
                  "be passed using the model parameter `sm_fit_kwargs`.")
             )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        forecaster.set_fit_kwargs(new_fit_kwargs)
    
    assert forecaster.fit_kwargs == {}