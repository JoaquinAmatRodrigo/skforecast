# Unit test backtesting_sarimax
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from pmdarima.arima import ARIMA

# Fixtures
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import y_datetime
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import exog_datetime


def test_backtesting_sarimax_ValueError_when_initial_train_size_is_None():
    """
    Test ValueError is raised in backtesting_sarimax when initial_train_size is None.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

    initial_train_size = None
    
    err_msg = re.escape('`initial_train_size` must be an int smaller than the length of `y`.')
    with pytest.raises(ValueError, match = err_msg):
        backtesting_sarimax(
            forecaster         = forecaster,
            y                  = y_datetime,
            steps              = 4,
            metric             = 'mean_absolute_error',
            initial_train_size = initial_train_size,
            refit              = False,
            fixed_train_size   = False,
            exog               = None,
            alpha              = None,
            interval           = None,
            verbose            = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         [(len(y_datetime)), (len(y_datetime) + 1)], 
                         ids = lambda value : f'len: {value}' )
def test_backtesting_sarimax_ValueError_when_initial_train_size_more_than_or_equal_to_len_y(initial_train_size):
    """
    Test ValueError is raised in backtesting_sarimax when initial_train_size >= len(y).
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
    
    err_msg = re.escape('`initial_train_size` must be an int smaller than the length of `y`.')
    with pytest.raises(ValueError, match = err_msg):
        backtesting_sarimax(
            forecaster         = forecaster,
            y                  = y_datetime,
            steps              = 4,
            metric             = 'mean_absolute_error',
            initial_train_size = initial_train_size,
            refit              = False,
            fixed_train_size   = False,
            exog               = None,
            alpha              = None,
            interval           = None,
            verbose            = False
        )


def test_backtesting_sarimax_ValueError_when_initial_train_size_less_than_forecaster_window_size():
    """
    Test ValueError is raised in backtesting_sarimax when initial_train_size < forecaster.window_size.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

    initial_train_size = forecaster.window_size - 1
    
    err_msg = re.escape(
            (f"`initial_train_size` must be greater than "
             f"forecaster's window_size ({forecaster.window_size}).")
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_sarimax(
            forecaster         = forecaster,
            y                  = y_datetime,
            steps              = 4,
            metric             = 'mean_absolute_error',
            initial_train_size = initial_train_size,
            refit              = False,
            fixed_train_size   = False,
            exog               = None,
            alpha              = None,
            interval           = None,
            verbose            = False
        )


def test_backtesting_sarimax_TypeError_when_refit_not_bool():
    """
    Test TypeError is raised in backtesting_sarimax when refit is not bool.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

    refit = 'not_bool'
    
    err_msg = re.escape( f'`refit` must be boolean: `True`, `False`.')
    with pytest.raises(TypeError, match = err_msg):
        backtesting_sarimax(
            forecaster         = forecaster,
            y                  = y_datetime,
            steps              = 4,
            metric             = 'mean_absolute_error',
            initial_train_size = 12,
            refit              = refit,
            fixed_train_size   = False,
            exog               = None,
            alpha              = None,
            interval           = None,
            verbose            = False
        )


def test_output_backtesting_sarimax_no_refit_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 3,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = False,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = True
                                   )
    
    expected_metric = 0.10564120819988289
    expected_values = np.array(
                          [0.63793971, 0.55322609, 0.71445518, 0.81448569, 
                           0.73818098, 0.71739124, 0.33699337, 0.34024219, 
                           0.39178913, 0.66499417, 0.54139334, 0.53432513]
                      )

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_no_refit_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 5,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = False,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.10061680900744897
    expected_values = np.array(
                          [0.63793971, 0.55322609, 0.71445518, 0.73039455, 0.69111006,
                           0.49744496, 0.37841947, 0.3619615 , 0.43815806, 0.44457422,
                           0.70144038, 0.61758017]
                      )

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_yes_refit_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 3,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.11195734192693496
    expected_values = np.array(
                        [0.63823688, 0.5535291 , 0.71457415, 0.8237381 , 0.7671365 ,
                        0.7315812 , 0.46865075, 0.47620553, 0.47437724, 0.52047381,
                        0.51618542, 0.51596407]
                      )
    expected_values = pd.DataFrame(
                          data    = expected_values,
                          columns = ['pred'],
                          index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     
    print(metric)
    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_yes_refit_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 5,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.09205718721313501
    expected_values = np.array(
                        [0.63823688, 0.5535291, 0.71457415, 0.73068107, 0.69155181,
                         0.47290713, 0.49963264, 0.49367393, 0.49133867, 0.49348299,
                         0.72450554, 0.63766057]
                      )
    expected_values = pd.DataFrame(
                          data    = expected_values,
                          columns = ['pred'],
                          index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.1)

def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 3,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = True,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = True
                                   )
    
    expected_metric = 0.08703290568510223
    expected_values = np.array(
                          [0.63793971, 0.55322609, 0.71445518, 0.56816834, 0.60577745,
                           0.61543978, 0.2837735 , 0.4109889 , 0.34055892, 0.54244717,
                           0.35633378, 0.55692658]
                      )

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 5,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = True,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.14972067408401932
    expected_values = np.array(
                          [0.63793971, 0.55322609, 0.71445518, 0.73039455, 0.69111006,
                           0.31724739, 0.1889182 , 0.29580158, 0.18153829, 0.2240739 ,
                           0.69979474, 0.58754509]
                      )

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0005)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.006)


def test_output_backtesting_sarimax_no_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 3,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = False,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.08924856486084219
    expected_values = np.array(
                         [0.65196283, 0.57702538, 0.66580046, 0.714828, 0.71623124,
                         0.67056245, 0.34725379, 0.32326499, 0.4678328, 0.65149367,
                         0.55076885, 0.52887971]
                      )

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_yes_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 3,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.10409885910606446
    expected_values = np.array(
                         [0.65196283, 0.57702538, 0.66580046, 0.73326053, 0.73999565,
                         0.68009211, 0.31761307, 0.264506, 0.42963014, 0.52169281,
                         0.56901593, 0.54864362]
                     )

    expected_values = pd.DataFrame(
                         data    = expected_values,
                         columns = ['pred'],
                         index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0005)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0025)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 5,
                                        metric             = 'mean_squared_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = True,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.14113801542106508
    expected_values = np.array(
                          [0.65196283, 0.57702538, 0.66580046, 0.61586071, 0.69004112,
                           0.272719, 0.18853606, 0.25185503, 0.20299981, 0.21249287,
                           0.75480558, 0.63902921]
                      )

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0005)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0015)


def my_metric(y_true, y_pred): # pragma: no cover
    """
    Callable metric
    """
    metric = ((y_true - y_pred)/len(y_true)).mean()
    
    return metric


def test_output_backtesting_sarimax_no_refit_yes_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), callable metric. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 3,
                                        metric             = my_metric,
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = False,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = -0.0016005986216357211
    expected_values = np.array(
                          [0.65196283, 0.57702538, 0.66580046, 0.714828, 0.71623124,
                           0.67056245, 0.34725379, 0.32326499, 0.4678328, 0.65149367,
                           0.55076885, 0.52887971]
                      )

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_no_refit_no_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), list of metrics. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 3,
                                        metric             = [my_metric, 'mean_absolute_error'],
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = False,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = [-0.0024305433379636946, 0.25015256206055664]
    expected_values = np.array(
                          [0.63793971, 0.55322609, 0.71445518, 0.81448569, 0.73818098,
                           0.71739124, 0.33699337, 0.34024219, 0.39178913, 0.66499417,
                           0.54139334, 0.53432513]
                      )

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_yes_refit_no_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), callable metric. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        steps              = 3,
                                        metric             = my_metric,
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = 0.0039252414599997626
    expected_values = np.array(
                          [0.63823688, 0.5535291 , 0.71457415, 0.8237381 , 0.7671365 ,
                           0.7315812 , 0.46865075, 0.47620553, 0.47437724, 0.52047381,
                           0.51618542, 0.51596407]
                      )

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                    )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    list of metrics. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 3,
                                        metric             = [my_metric, 'mean_absolute_error'],
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = True,
                                        refit              = True,
                                        alpha              = None,
                                        interval           = None,
                                        verbose            = False
                                   )
    
    expected_metric = [-0.0003554763935739996, 0.2729350828631346]
    expected_values = np.array(
                          [0.65196283, 0.57702538, 0.66580046, 0.76232501, 0.78530427,
                           0.74674924, 0.32893953, 0.33545942, 0.49755038, 0.48101042,
                           0.37252349, 0.48195615]
                       )

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.01)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.001)
    

@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_output_backtesting_sarimax_no_refit_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_absolute_error',
    interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 3,
                                        metric             = ['mean_absolute_error'],
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = False,
                                        alpha              = alpha,
                                        interval           = interval,
                                        verbose            = False
                                   )
    
    expected_metric = [0.22662967568552414]
    expected_values = np.array([[ 0.65196283,  0.18723883,  1.11668683],
                                [ 0.57702538,  0.05511674,  1.09893402],
                                [ 0.66580046,  0.1217932 ,  1.20980771],
                                [ 0.714828  ,  0.25060163,  1.17905437],
                                [ 0.71623124,  0.19529936,  1.23716311],
                                [ 0.67056245,  0.12789764,  1.21322726],
                                [ 0.34725379, -0.11654659,  0.81105416],
                                [ 0.32326499, -0.19683011,  0.84336009],
                                [ 0.4678328 , -0.07368132,  1.00934692],
                                [ 0.65149367,  0.18806208,  1.11492527],
                                [ 0.55076885,  0.03139862,  1.07013908],
                                [ 0.52887971, -0.01163713,  1.06939656]])

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred', 'lower_bound', 'upper_bound'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.000025)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_output_backtesting_sarimax_yes_refit_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_absolute_error', interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 3,
                                        metric             = 'mean_absolute_error',
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = False,
                                        refit              = True,
                                        alpha              = alpha,
                                        interval           = interval,
                                        verbose            = False
                                   )
    
    expected_metric = 0.2423875933808799
    expected_values = np.array([[ 0.65196283,  0.18723883,  1.11668683],
                                [ 0.57702538,  0.05511674,  1.09893402],
                                [ 0.66580046,  0.1217932 ,  1.20980771],
                                [ 0.73326053,  0.27283432,  1.19368675],
                                [ 0.73999565,  0.2194705 ,  1.26052079],
                                [ 0.68009211,  0.14236641,  1.2178178 ],
                                [ 0.31761307, -0.15822946,  0.79345559],
                                [ 0.264506  , -0.27346558,  0.80247757],
                                [ 0.42963014, -0.14094026,  1.00020054],
                                [ 0.52169281,  0.07120324,  0.97218239],
                                [ 0.56901593,  0.10030656,  1.0377253 ],
                                [ 0.54864362,  0.07951009,  1.01777715]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.00025)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0075)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_absolute_error', interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(2,2,2)))

    metric, backtest_predictions = backtesting_sarimax(
                                        forecaster         = forecaster,
                                        y                  = y_datetime,
                                        exog               = exog_datetime,
                                        steps              = 3,
                                        metric             = ['mean_absolute_error'],
                                        initial_train_size = len(y_datetime)-12,
                                        fixed_train_size   = True,
                                        refit              = True,
                                        alpha              = alpha,
                                        interval           = interval,
                                        verbose            = False
                                   )
    
    expected_metric = [0.2729350828631346]
    expected_values = np.array([[ 0.65196283,  0.18723883,  1.11668683],
                                [ 0.57702538,  0.05511674,  1.09893402],
                                [ 0.66580046,  0.1217932 ,  1.20980771],
                                [ 0.76232501,  0.30338409,  1.22126592],
                                [ 0.78530427,  0.27799337,  1.29261517],
                                [ 0.74674924,  0.22580718,  1.2676913 ],
                                [ 0.32893953, -0.14761105,  0.8054901 ],
                                [ 0.33545942, -0.23328796,  0.9042068 ],
                                [ 0.49755038, -0.12170221,  1.11680296],
                                [ 0.48101042, -0.0407233 ,  1.00274415],
                                [ 0.37252349, -0.27208667,  1.01713366],
                                [ 0.48195615, -0.2650018 ,  1.22891409]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0002)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0015)