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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
    
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
                                        verbose            = False
                                   )
    
    expected_metric = 0.10564449116792594
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.8145688855,
                                0.7383540136, 0.7175581892, 0.3371878999, 0.3404664486,
                                0.3920279171, 0.6650885959, 0.541608158, 0.5345377188])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_no_refit_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.10059257357428625
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.7305617801,
                                0.6913258666, 0.4976197743, 0.3786960441, 0.36221788,
                                0.438474656, 0.4450083288, 0.7015845146, 0.6177119599])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.11191023692999214
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.8236634405,
                                0.7671966066, 0.7318098972, 0.4753534544, 0.4878701598,
                                0.4884851128, 0.5211869394, 0.5170146169, 0.516800887 ])
    expected_values = pd.DataFrame(
                          data    = expected_values,
                          columns = ['pred'],
                          index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     
    print(metric)
    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.09044168640090938
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.7305617801,
                                0.6913258666, 0.4709966524, 0.5031823976, 0.5018454029,
                                0.5010863298, 0.5036654154, 0.6937025582, 0.5734803045])
    expected_values = pd.DataFrame(
                          data    = expected_values,
                          columns = ['pred'],
                          index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
                                        verbose            = False
                                   )
    
    expected_metric = 0.06989905151853813
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.534674915 ,
                                0.6146465644, 0.5849857098, 0.4912121605, 0.5191775648,
                                0.5308123973, 0.5424695163, 0.3564268548, 0.5570288022])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.08922580416788294
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.7305617801,
                                0.6913258666, 0.4775083012, 0.515039201 , 0.5153595286,
                                0.5153285596, 0.5190251665, 0.6955141571, 0.5829237462])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)
    

def test_output_backtesting_sarimax_no_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.08925532261358403
    expected_values = np.array([0.6519047238, 0.5772943726, 0.6657697849, 0.7147063689,
                                0.7166754268, 0.6709517945, 0.3477544267, 0.3237557088,
                                0.4683742738, 0.6513165305, 0.5512283372, 0.5293714288])

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.1040532152699317
    expected_values = np.array([0.6519047238, 0.5772943726, 0.6657697849, 0.7337365893,
                                0.7402027263, 0.6800253246, 0.3175607719, 0.2646546682,
                                0.4297918732, 0.522778418, 0.5725599904, 0.5547692757])

    expected_values = pd.DataFrame(
                         data    = expected_values,
                         columns = ['pred'],
                         index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.0783413602892565
    expected_values = np.array([0.6519047238, 0.5772943726, 0.6657697849, 0.6159358116,
                                0.6905630512, 0.4165855067, 0.4912115236, 0.4881126381,
                                0.5738226384, 0.5427397898, 0.7548294773, 0.6390792395])
    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = -0.001622813940319886
    expected_values = np.array([0.6519047238, 0.5772943726, 0.6657697849, 0.7147063689,
                                0.7166754268, 0.6709517945, 0.3477544267, 0.3237557088,
                                0.4683742738, 0.6513165305, 0.5512283372, 0.5293714288])
    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_no_refit_no_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), list of metrics. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = [-0.0024431777939957766, 0.2501673978304879]
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.8145688855,
                                0.7383540136, 0.7175581892, 0.3371878999, 0.3404664486,
                                0.3920279171, 0.6650885959, 0.541608158, 0.5345377188])
    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_no_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), callable metric. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = -0.004165283959601966
    expected_values = np.array([0.6380200425, 0.5532980001, 0.714519703 , 0.8236634405,
                                0.7671966066, 0.7318098972, 0.4753534544, 0.4878701598,
                                0.4884851128, 0.5211869394, 0.5170146169, 0.516800887 ])

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    list of metrics. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = [-0.001810459272016009, 0.25796710568368647]
    expected_values = np.array([0.6519047238, 0.5772943726, 0.6657697849, 0.7620030163,
                              0.7858998429, 0.7475340329, 0.3289957937, 0.3357500345,
                              0.4982306108, 0.4963807234, 0.5321684767, 0.5141926925])

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)
    

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = [0.22667983000050848]
    expected_values = np.array([[ 0.6519047238,  0.1867398521,  1.1170695955],
                                [ 0.5772943726,  0.0550837622,  1.099504983 ],
                                [ 0.6657697849,  0.121451421,  1.2100881488],
                                [ 0.7147063689,  0.2500398108,  1.179372927 ],
                                [ 0.7166754268,  0.1954420873,  1.2379087663],
                                [ 0.6709517945,  0.127976669,  1.21392692  ],
                                [ 0.3477544267, -0.1164855382,  0.8119943916],
                                [ 0.3237557088, -0.1966403909,  0.8441518084],
                                [ 0.4683742738, -0.0734494424,  1.0101979901],
                                [ 0.6513165305,  0.1874458832,  1.1151871779],
                                [ 0.5512283372,  0.031557534,  1.0708991405],
                                [ 0.5293714288, -0.0114543467,  1.0701972043]])

    expected_values = pd.DataFrame(
                        data    = expected_values,
                        columns = ['pred', 'lower_bound', 'upper_bound'],
                        index   = pd.date_range(start='2038', periods=12, freq='A')
                      )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_values, backtest_predictions, atol=0.0001)


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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = 0.24216132863296913
    expected_values = np.array([[ 0.6519047238,  0.1867398521,  1.1170695955],
                                [ 0.5772943726,  0.0550837622,  1.099504983 ],
                                [ 0.6657697849,  0.121451421,  1.2100881488],
                                [ 0.7337365893,  0.2737189574,  1.1937542212],
                                [ 0.7402027263,  0.2200736651,  1.2603317874],
                                [ 0.6800253246,  0.1428285965,  1.2172220527],
                                [ 0.3175607719, -0.158283587,  0.7934051309],
                                [ 0.2646546682, -0.2733190927,  0.8026284291],
                                [ 0.4297918732, -0.1407959766,  1.0003797231],
                                [ 0.522778418,  0.0723826109,  0.9731742251],
                                [ 0.5725599904,  0.1043004567,  1.040819524 ],
                                [ 0.5547692757,  0.085779059,  1.0237594924]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(2,2,2)))

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
    
    expected_metric = [0.25796710568368647]
    expected_values = np.array([[ 0.6519047238,  0.1867398521,  1.1170695955],
                                [ 0.5772943726,  0.0550837622,  1.099504983 ],
                                [ 0.6657697849,  0.121451421 ,  1.2100881488],
                                [ 0.7620030163,  0.3034036355,  1.2206023971],
                                [ 0.7858998429,  0.2794773829,  1.292322303 ],
                                [ 0.7475340329,  0.22787058  ,  1.2671974858],
                                [ 0.3289957937, -0.1473156992,  0.8053072867],
                                [ 0.3357500345, -0.2326164776,  0.9041165466],
                                [ 0.4982306108, -0.120464111 ,  1.1169253327],
                                [ 0.4963807234,  0.0420330534,  0.9507283935],
                                [ 0.5321684767,  0.05767149  ,  1.0066654633],
                                [ 0.5141926925,  0.0380659504,  0.9903194346]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)