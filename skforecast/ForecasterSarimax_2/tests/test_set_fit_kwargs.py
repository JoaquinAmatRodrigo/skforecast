# Unit test set_fit_kwargs ForecasterSarimax
# ==============================================================================
from skforecast.ForecasterSarimax import ForecasterSarimax
from pmdarima.arima import ARIMA


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
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