# Unit test set_params ForecasterSarimax
# ==============================================================================
from skforecast.ForecasterSarimax import ForecasterSarimax
from pmdarima.arima import ARIMA


def test_ForecasterSarimax_set_params():
    """
    Test set_params() method.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    new_params = {'order': (2,2,2), 'seasonal_order': (1,1,1,2)}
    forecaster.set_params(new_params)
    expected = {'maxiter'           : 50,
                'method'            : 'lbfgs',
                'order'             : (2, 2, 2),
                'out_of_sample_size': 0,
                'scoring'           : 'mse',
                'scoring_args'      : None,
                'seasonal_order'    : (1, 1, 1, 2),
                'start_params'      : None,
                'suppress_warnings' : False,
                'trend'             : None,
                'with_intercept'    : True}
    results = forecaster.regressor.get_params()

    assert results == expected