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
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))

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
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    
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
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))

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
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))

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
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(2,2,2)))

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

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_no_refit_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(2,2,2)))

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

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_yes_refit_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(2,2,2)))

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
    
    expected_metric = 0.1245724897025159
    expected_values = np.array(
                          [0.63793971, 0.55322609, 0.71445518, 0.82489505, 0.77191505,
                           0.73750377, 0.30807646, 0.27854996, 0.28835934, 0.51981128,
                           0.51690468, 0.51726221]
                      )

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_yes_refit_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(2,2,2)))

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
    
    expected_metric = 0.09556639597015694
    expected_values = np.array(
                          [0.63793971, 0.55322609, 0.71445518, 0.73039455, 0.69111006,
                           0.4814143 , 0.5014026 , 0.48131545, 0.47225091, 0.4734432 ,
                           0.78337242, 0.65285038]
                      )

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred'],
                                        index   = pd.date_range(start='2038', periods=12, freq='A')
                                    )                                                     

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(2,2,2)))

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

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(2,2,2)))

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

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)







# Hechos todos sin exog, no refit, refit y refit + fixed, con y sin remainder













def test_output_backtesting_sarimax_no_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    steps = 3
    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_sarimax(
                                        y          = y,
                                        exog       = exog,        
                                        steps      = steps,
                                        metric     = 'mean_squared_error',
                                        initial_train_size = len(y_train),
                                        fixed_train_size   = False,
                                        refit      = False,
                                        order      = (1, 0, 0),
                                        verbose    = False
                                   )
    
    expected_metric = 0.08042580124608663
    expected_backtest_predictions = pd.DataFrame({
    'predicted_mean':np.array([0.77506303, 0.68675642, 0.57725733, 0.51868684, 0.54281652, 0.43401386,
                               0.3892287 , 0.33261601, 0.36472075, 0.40337179, 0.38182328, 0.32970631]),
    'lower y':np.array([ 0.24507798, -0.02871066, -0.26052729, -0.0112982 , -0.17265056, -0.40377076, 
                        -0.14075634, -0.38285107, -0.47306387, -0.12661325, -0.3336438 , -0.50807831]),
    'upper y':np.array([1.30504807, 1.4022235 , 1.41504195, 1.04867188, 1.2582836 , 1.27179848, 
                        0.91921374, 1.04808309, 1.20250537, 0.93335683, 1.09729036, 1.16749093])                                                                
                                                                         }, index=np.arange(38, 50))

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_yes_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    steps = 3
    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_sarimax(
                                        y          = y,
                                        exog       = exog,        
                                        steps      = steps,
                                        metric     = 'mean_squared_error',
                                        initial_train_size = len(y_train),
                                        fixed_train_size   = False,
                                        refit      = True,
                                        order      = (1, 0, 0),
                                        verbose    = False
                                   )
    
    expected_metric = 0.08442615943508115
    expected_backtest_predictions = pd.DataFrame({
    'predicted_mean':np.array([0.77506303, 0.68675642, 0.57725733, 0.52073138, 0.53115375, 0.42776686,
                               0.38307311, 0.32059731, 0.35042998, 0.38540627, 0.36253005, 0.30099599]),
    'lower y':np.array([ 0.24507798, -0.02871066, -0.26052729, -0.00679734, -0.17885993, -0.4013203 , 
                        -0.14460782, -0.38731126, -0.47372304, -0.16511761, -0.37462024, -0.55570653]),
    'upper y':np.array([1.30504807, 1.4022235 , 1.41504195, 1.0482601 , 1.24116743, 1.25685401,
                        0.91075403, 1.02850588, 1.174583  , 0.93593015, 1.09968035, 1.15769851])                                                                
                                                                         }, index=np.arange(38, 50))

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, rtol=1e-4)


def my_metric(y_true, y_pred):
    """
    Callable metric
    """
    metric = ((y_true - y_pred)/len(y_true)).mean()
    
    return metric


def test_output_backtesting_sarimax_no_refit_no_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    no refit, 12 observations to backtest, steps=4 (no remainder), callable metric
    """
    steps = 3
    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_sarimax(
                                        y          = y,
                                        steps      = steps,
                                        metric     = my_metric,
                                        initial_train_size = len(y_train),
                                        fixed_train_size   = False,
                                        refit      = False,
                                        order      = (1, 0, 0),
                                        verbose    = False
                                   )
    
    expected_metric = 0.006764050531706984
    expected_backtest_predictions = pd.DataFrame({
    'predicted_mean':np.array([0.78762096, 0.69437463, 0.61216773, 0.55008326, 0.48495899, 0.42754477, 
                               0.365715  , 0.32241806, 0.28424703, 0.42584791, 0.37543184, 0.33098452]),
    'lower y':np.array([ 0.24826801, -0.02465313, -0.22013845,  0.01073031, -0.23406878, -0.40476141, 
                        -0.17363794, -0.39660971, -0.54805914, -0.11350503, -0.34359592, -0.50132165]),
    'upper y':np.array([1.3269739 , 1.4134024 , 1.44447391, 1.0894362 , 1.20398675, 1.25985094, 
                        0.90506794, 1.04144582, 1.11655321, 0.96520085, 1.09445961, 1.1632907 ])                                                                
                                                                         }, index=np.arange(38, 50))

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_yes_refit_no_exog_fixed_train_size_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed train size
    """
    steps = 3
    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_sarimax(
                                        y          = y,
                                        steps      = steps,
                                        metric     = 'mean_squared_error',
                                        initial_train_size = len(y_train),
                                        fixed_train_size   = True,
                                        refit      = True,
                                        order      = (1, 0, 0),
                                        verbose    = False
                                   )
    
    expected_metric = 0.08732478802417498
    expected_backtest_predictions = pd.DataFrame({
    'predicted_mean':np.array([0.78762096, 0.69437463, 0.61216773, 0.54666203, 0.47894536, 0.41961696, 
                               0.36824258, 0.32689014, 0.29018145, 0.40309841, 0.33639089, 0.2807226 ]),
    'lower y':np.array([ 0.24826801, -0.02465313, -0.22013845,  0.0170064 , -0.22523743, -0.39350466,
                        -0.15419253, -0.37169314, -0.52068564, -0.15280075, -0.38764766, -0.54031653]),
    'upper y':np.array([1.3269739 , 1.4134024 , 1.44447391, 1.07631765, 1.18312815, 1.23273858,
                        0.89067769, 1.02547342, 1.10104854, 0.95899756, 1.06042944, 1.10176172])                                                                
                                                                         }, index=np.arange(38, 50))

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_yes_refit_yes_exog_fixed_train_size_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed train size
    """
    steps = 3
    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_sarimax(
                                        y          = y,
                                        exog       = exog,
                                        steps      = steps,
                                        metric     = 'mean_squared_error',
                                        initial_train_size = len(y_train),
                                        fixed_train_size   = True,
                                        refit      = True,
                                        order      = (1, 0, 0),
                                        verbose    = False
                                   )
    
    expected_metric = 0.08480706997117245
    expected_backtest_predictions = pd.DataFrame({
    'predicted_mean':np.array([0.77506303, 0.68675642, 0.57725733, 0.53279056, 0.50702637, 0.42388899,
                               0.39033963, 0.3390865 , 0.36431203, 0.38600385, 0.34999042, 0.28746299]),
    'lower y':np.array([ 0.24507798, -0.02871066, -0.26052729,  0.00541859, -0.1990523 , -0.39662692, 
                        -0.12436057, -0.35775318, -0.45385542, -0.16140765, -0.37591687, -0.5487951 ]),
    'upper y':np.array([1.30504807, 1.4022235 , 1.41504195, 1.06016252, 1.21310503, 1.24440491,
                        0.90503984, 1.03592617, 1.18247947, 0.93341534, 1.07589771, 1.12372107])                                                                
                                                                         }, index=np.arange(38, 50))

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)