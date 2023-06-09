# Unit test backtesting_sarimax
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from pmdarima.arima import ARIMA

# Fixtures
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import y_datetime
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import exog_datetime


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    err_msg = re.escape(
            ("`forecaster` must be of type `ForecasterSarimax`, for all other "
             "types of forecasters use the functions available in the other "
             "`model_selection` modules.")
        )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_sarimax(
            forecaster            = forecaster,
            y                     = y_datetime,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y_datetime[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            alpha                 = None,
            interval              = None,
            verbose               = False,
            show_progress         = False
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
                                        verbose            = True
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


@pytest.mark.parametrize("n_jobs", [-1, 1],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_sarimax_yes_refit_no_exog_no_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, 
    no exog, yes refit, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'. (Mocked done with skforecast 0.7.0.)
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
                                        n_jobs             = n_jobs,
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


@pytest.mark.parametrize("n_jobs", [-1, 1],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_sarimax_yes_refit_no_exog_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, 
    no exog, yes refit, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. (Mocked done with skforecast 0.7.0.)
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
                                        n_jobs             = n_jobs,
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,0,0)))

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
    
    expected_metric = 0.06443497388158415
    expected_values = np.array([0.5808796661, 0.5170345946, 0.4546504329, 0.4518464581,
                                0.5349945619, 0.4592103268, 0.4659649981, 0.448978798 ,
                                0.5612031446, 0.500060416 , 0.5448058138, 0.5246443326])

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,0,0)))

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
    
    expected_metric = 0.2018507318439613
    expected_values = np.array([[0.5808796661, 0.164441322 , 0.9973180102],
                                [0.5170345946, 0.0940599402, 0.9400092491],
                                [0.4546504329, 0.0314706312, 0.8778302345],
                                [0.4518464581, 0.0327714211, 0.870921495 ],
                                [0.5349945619, 0.1059645448, 0.9640245791],
                                [0.4592103268, 0.0297075509, 0.8887131028],
                                [0.4659649981, 0.0481543449, 0.8837756513],
                                [0.448978798 , 0.0223625105, 0.8755950854],
                                [0.5612031446, 0.1342158091, 0.98819048  ],
                                [0.500060416 , 0.0726297766, 0.9274910554],
                                [0.5448058138, 0.1120978921, 0.9775137355],
                                [0.5246443326, 0.0918061134, 0.9574825517]])

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,0,0)))

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
    
    expected_metric = [0.20744848519166362]
    expected_values = np.array([[0.5808796661, 0.164441322 , 0.9973180102],
                                [0.5170345946, 0.0940599402, 0.9400092491],
                                [0.4546504329, 0.0314706312, 0.8778302345],
                                [0.4818642089, 0.0605997277, 0.9031286902],
                                [0.5347689152, 0.1022767621, 0.9672610683],
                                [0.4795745631, 0.0464842333, 0.9126648928],
                                [0.4687425354, 0.0429977408, 0.89448733  ],
                                [0.4599063136, 0.0169252478, 0.9028873794],
                                [0.5514034666, 0.1070284241, 0.995778509 ],
                                [0.4824086925, 0.0556563027, 0.9091610824],
                                [0.5174687336, 0.0852978136, 0.9496396537],
                                [0.5003253491, 0.0680168459, 0.9326338523]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric, abs=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)