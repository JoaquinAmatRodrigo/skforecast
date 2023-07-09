# Unit test _backtesting_forecaster Refit
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from pytest import approx
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection.model_selection import _backtesting_forecaster

# Fixtures
from .fixtures_model_selection import y
from .fixtures_model_selection import exog
from .fixtures_model_selection import out_sample_residuals


# ******************************************************************************
# * Test _backtesting_forecaster No Interval                                   *
# ******************************************************************************

@pytest.mark.parametrize("n_jobs", [-1, 1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterAutoreg_with_mocked(n_jobs):
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoreg.
    """
    expected_metric = 0.06598802629306816
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 
                         0.38969292, 0.52778339, 0.49152015, 0.4841678 , 
                         0.4076433 , 0.50904672, 0.50249462, 0.49232817])}, 
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        n_jobs              = n_jobs,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    
    lags = y[-1:-4:-1]
    
    return lags 


@pytest.mark.parametrize("refit", [True, 1],
                         ids=lambda n: f'refit: {n}')
def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterAutoregCustom_with_mocked(refit):
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoregCustom.
    """
    expected_metric = 0.06598802629306816
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339,
                     0.49152015, 0.4841678 , 0.4076433 , 0.50904672, 0.50249462, 0.49232817])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(), 
                     fun_predictors = create_predictors,
                     window_size    = 3
                 )

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = refit,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterAutoregDirect_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoregDirect.
    """
    expected_metric = 0.07076203468824617
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.5468482 , 0.44670961, 0.57651222, 0.52511275, 0.3686309 , 0.56234835, 
                     0.44276032, 0.52260065, 0.37665741, 0.5382938 , 0.48755548, 0.44534071])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 4
                 )

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = 0.06916732087926723
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.5096801 ,
                     0.49519677, 0.47997916, 0.49177914, 0.495797  , 0.57738724, 0.44370472])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 5,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metric = 0.05663345135204598
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.42295275, 0.46286083,
                     0.43618422, 0.43552906, 0.48687517, 0.55455072, 0.55577332, 0.53943402])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = 0.061723961096013524
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.43595809,
                     0.4349167 , 0.42381237, 0.55165332, 0.53442833, 0.65361802, 0.51297419])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 5,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
    
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Test _backtesting_forecaster Interval                                      *
# ******************************************************************************

def test_output_backtesting_forecaster_interval_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.06598802629306816
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339, 
                     0.49152015, 0.4841678, 0.4076433, 0.50904672, 0.50249462, 0.49232817]),
    'lower_bound':np.array([0.19882822, 0.08272406, 0.18106389, 0.18777395, 0.03520425, 0.12926384,
                            0.0495347 , 0.04527341, 0.0113795 , 0.13676538, 0.12478441, 0.06814153]),
    'upper_bound':np.array([0.95368172, 0.81704742, 0.93685716, 0.9407976 , 0.78486946, 0.93084605,
                            0.84533191, 0.90255909, 0.80099612, 0.88747244, 0.88292664, 0.88718366])                                                                 
                                                                         }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = [5, 95],
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   ) 
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.06916732087926723
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.5096801 , 
                     0.49519677, 0.47997916, 0.49177914, 0.495797  , 0.57738724, 0.44370472]),
    'lower_bound':np.array([0.19882822, 0.08272406, 0.18106389, 0.18777395, 0.1238825 , 0.06681772,
                            0.09795868, 0.08383945, 0.10160946, 0.08917676, 0.23321023, 0.08685352]),
    'upper_bound':np.array([0.95368172, 0.81704742, 0.93685716, 0.9407976 , 0.85396419, 0.86172991,
                            0.88313129, 0.82354636, 0.93875053, 0.86176335, 0.96037185, 0.84205069])                                                                 
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 5,
                                        metric              = 'mean_squared_error',
                                        interval            = [5, 95],
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
    
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.05663345135204598
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.42295275, 0.46286083,
                     0.43618422, 0.43552906, 0.48687517, 0.55455072, 0.55577332, 0.53943402]),
    'lower_bound':np.array([0.24619375, 0.10545295, 0.13120713, 0.08044217, 0.07440334, 0.11331854,
                            0.01436362, 0.02747413, 0.14867238, 0.19834047, 0.19884259, 0.16964474]),
    'upper_bound':np.array([0.95777604, 0.88685543, 0.90755063, 0.87811336, 0.8225198 , 0.81894689,
                            0.81179723, 0.84420112, 0.89407425, 0.93903702, 0.91748574, 0.93705358])                                                                 
                                                                         }, index=pd.RangeIndex(start=38, stop=50, step=1))
    
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = [5, 95],
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
    
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.061723961096013524
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.43595809,
                     0.4349167 , 0.42381237, 0.55165332, 0.53442833, 0.65361802, 0.51297419]),
    'lower_bound':np.array([0.24619375, 0.10545295, 0.13120713, 0.08044217, 0.13725077, 0.08041239,
                            0.05015513, 0.07677812, 0.17434611, 0.16051962, 0.29167326, 0.15775686]),
    'upper_bound':np.array([0.95777604, 0.88685543, 0.90755063, 0.87811336, 0.86891022, 0.74808834,
                            0.80296989, 0.77919033, 0.97680126, 0.8877086 , 1.07608747, 0.90555785])                                                                 
                                                                         }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 5,
                                        metric              = 'mean_squared_error',
                                        interval            = [5, 95],
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Out sample residuals                                                       *
# ******************************************************************************

def test_output_backtesting_forecaster_interval_out_sample_residuals_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = False'
    """
    expected_metric = 0.06598802629306816
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339, 
                         0.49152015, 0.4841678 , 0.4076433 , 0.50904672, 0.50249462, 0.49232817]),
        'lower_bound':np.array([0.63654358, 0.62989756, 0.67156167, 0.70363265, 0.46905871, 0.73507276, 
                                0.64554988, 0.64183541, 0.48700909, 0.68845988, 0.63865297, 0.60684242]),
        'upper_bound':np.array([1.54070487, 1.5131313 , 1.56749058, 1.62564968, 1.37322   , 1.61930035, 
                                1.54870568, 1.55335041, 1.39117037, 1.56935123, 1.51973211, 1.50300901])                                                                
                                                                            }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=out_sample_residuals, append=False)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = [5, 95],
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = False,
                                        verbose             = False
                                   )

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Callable metric                                                            *
# ******************************************************************************

def my_metric(y_true, y_pred): # pragma: no cover
    """
    Callable metric
    """
    metric = ((y_true - y_pred)/len(y_true)).mean()
    
    return metric

def test_callable_metric_backtesting_forecaster_no_exog_no_remainder_with_mocked():
    """
    Test callable metric in _backtesting_forecaster with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metric = 0.005283745900436151
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339,
                     0.49152015, 0.4841678 , 0.4076433 , 0.50904672, 0.50249462, 0.49232817])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = my_metric,
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_list_metrics_backtesting_forecaster_no_exog_no_remainder_with_mocked():
    """
    Test list of metrics in _backtesting_forecaster with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metrics = [0.06598802629306816, 0.06598802629306816]
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339,
                     0.49152015, 0.4841678 , 0.4076433 , 0.50904672, 0.50249462, 0.49232817])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metrics, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = False,
                                        steps               = 4,
                                        metric              = ['mean_squared_error', mean_squared_error],
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )

    assert expected_metrics == approx(metrics)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * fixed_train_size = True                                                    *
# ******************************************************************************

def test_output_backtesting_forecaster_fixed_train_size_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = 0.06720844584333846
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.34597367, 0.50223873,
                     0.47833829, 0.46082257, 0.37810191, 0.49508366, 0.48808014, 0.47323313])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = True,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
    
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_fixed_train_size_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = 0.07217085374372428
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.4909399 , 
                     0.47942107, 0.46025344, 0.46649132, 0.47061725, 0.57603136, 0.41480551])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = True,
                                        steps               = 5,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_fixed_train_size_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = 0.05758244401484334
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.37689967, 0.44267729, 
                     0.42642836, 0.41604275, 0.45047245, 0.53784704, 0.53726274, 0.51516772])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = True,
                                        steps               = 4,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_fixed_train_size_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = 0.06425019123005545
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.41975558, 
                     0.4256614 , 0.41176005, 0.52357817, 0.509974  , 0.65354628, 0.48210726])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        refit               = True,
                                        initial_train_size  = len(y_train),
                                        fixed_train_size    = True,
                                        steps               = 5,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Gap                                                                        *
# ******************************************************************************


def test_output_backtesting_forecaster_interval_yes_exog_yes_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.0839045861490063
    expected_predictions = pd.DataFrame(
        data = np.array([[ 0.65022999,  0.27183154,  1.06503914],
                         [ 0.52364637,  0.19453326,  0.923044  ],
                         [ 0.46809256,  0.12440678,  0.75395058],
                         [ 0.48759732,  0.12107577,  0.83145059],
                         [ 0.47172445,  0.14533367,  0.74697552],
                         [ 0.54075007,  0.15924753,  1.01588603],
                         [ 0.50283999,  0.19115121,  0.85076388],
                         [ 0.49737535,  0.22956778,  0.7939908 ],
                         [ 0.49185456,  0.22055651,  0.85300844],
                         [ 0.48906044,  0.17652033,  0.80604025],
                         [ 0.27751064, -0.07159784,  0.70858646],
                         [ 0.25859617, -0.03545793,  0.57770947],
                         [ 0.32853669,  0.03467979,  0.81698814],
                         [ 0.43751884,  0.13949436,  0.82693778],
                         [ 0.33016371,  0.03065767,  0.64473596],
                         [ 0.58126262,  0.22355106,  1.01410171],
                         [ 0.52955296,  0.21810115,  0.93120556]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.RangeIndex(start=33, stop=50, step=1)
    )

    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 8
                 )

    n_backtest = 20
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster            = forecaster,
                                        y                     = y,
                                        exog                  = exog,
                                        refit                 = True,
                                        initial_train_size    = len(y_train),
                                        fixed_train_size      = False,
                                        gap                   = 3,
                                        allow_incomplete_fold = True,
                                        steps                 = 5,
                                        metric                = 'mean_squared_error',
                                        interval              = [5, 95],
                                        n_boot                = 500,
                                        random_state          = 123,
                                        in_sample_residuals   = True,
                                        verbose               = False
                                   )

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_not_allow_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'in_sample_residuals = True', allow_incomplete_fold = False
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = 0.09133694038363274
    expected_predictions = pd.DataFrame(
        data = np.array([[ 0.65022999,  0.27183154,  1.06503914],
                         [ 0.52364637,  0.19453326,  0.923044  ],
                         [ 0.46809256,  0.12440678,  0.75395058],
                         [ 0.48759732,  0.12107577,  0.83145059],
                         [ 0.47172445,  0.14533367,  0.74697552],
                         [ 0.50952084,  0.17367515,  0.96199755],
                         [ 0.52131987,  0.19133495,  0.77700048],
                         [ 0.50289964,  0.261816  ,  0.89080844],
                         [ 0.54941988,  0.2254731 ,  0.81990615],
                         [ 0.51121195,  0.23007235,  0.76212766],
                         [ 0.25123452, -0.02303138,  0.68451499],
                         [ 0.2791409 ,  0.05980179,  0.67369153],
                         [ 0.28390161,  0.0752468 ,  0.58609101],
                         [ 0.38317935,  0.0929118 ,  0.70099806],
                         [ 0.42183128,  0.10993796,  0.69111769]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.date_range(start='2022-02-03', periods=15, freq='D')
    )

    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 8
                 )

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster            = forecaster,
                                        y                     = y_with_index,
                                        exog                  = exog_with_index,
                                        refit                 = True,
                                        initial_train_size    = len(y_with_index) - 20,
                                        fixed_train_size      = True,
                                        gap                   = 3,
                                        allow_incomplete_fold = False,
                                        steps                 = 5,
                                        metric                = 'mean_squared_error',
                                        interval              = [5, 95],
                                        n_boot                = 500,
                                        random_state          = 123,
                                        in_sample_residuals   = True,
                                        verbose               = False
                                   )
    backtest_predictions = backtest_predictions.asfreq('D')

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Refit int                                                                  *
# ******************************************************************************
        

def test_output_backtesting_forecaster_refit_int_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of backtesting_forecaster refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=0, metric='mean_squared_error',
    'in_sample_residuals = True'. Refit int.
    """
    expected_metric = 0.06099110404144631
    expected_predictions = pd.DataFrame(
        data = np.array([[0.55616986, 0.15288789, 0.89198752],
                         [0.48751797, 0.05863911, 0.83169303],
                         [0.57764391, 0.17436194, 0.91346157],
                         [0.51298667, 0.08410781, 0.85716173],
                         [0.47430051, 0.0796644 , 0.82587748],
                         [0.49192271, 0.15006787, 0.84017409],
                         [0.52213783, 0.12750172, 0.8737148 ],
                         [0.54492575, 0.20307092, 0.89317713],
                         [0.52501537, 0.13641764, 0.86685356],
                         [0.4680474 , 0.08515461, 0.81080044],
                         [0.51059498, 0.12199725, 0.85243317],
                         [0.53067132, 0.14777853, 0.87342436],
                         [0.4430938 , 0.0509291 , 0.69854202],
                         [0.49911716, 0.1231365 , 0.8711497 ],
                         [0.44546347, 0.05329877, 0.70091169],
                         [0.46530749, 0.08932683, 0.83734003],
                         [0.46901878, 0.08173407, 0.82098555],
                         [0.55371362, 0.14618224, 0.98199137],
                         [0.60759064, 0.22030593, 0.9595574 ],
                         [0.50415336, 0.09662198, 0.93243111]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.RangeIndex(start=30, stop=50, step=1)
    )

    forecaster = ForecasterAutoregDirect(
                     regressor = Ridge(random_state=123), 
                     lags      = 3,
                     steps     = 8
                 )

    n_backtest = 20
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                       forecaster            = forecaster,
                                       y                     = y,
                                       exog                  = exog,
                                       refit                 = 2,
                                       initial_train_size    = len(y_train),
                                       fixed_train_size      = False,
                                       gap                   = 0,
                                       allow_incomplete_fold = True,
                                       steps                 = 2,
                                       metric                = 'mean_squared_error',
                                       interval              = [5, 95],
                                       n_boot                = 500,
                                       random_state          = 123,
                                       in_sample_residuals   = True,
                                       verbose               = False,
                                       n_jobs                = 1
                                   )

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_int_interval_yes_exog_not_allow_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'in_sample_residuals = True', allow_incomplete_fold = False. Refit int.
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = 0.060991643719298785
    expected_predictions = pd.DataFrame(
        data = np.array([[0.51878642, 0.17324039, 0.87771447],
                         [0.49269791, 0.15223278, 0.86847335],
                         [0.49380441, 0.14003517, 0.81226679],
                         [0.51467463, 0.18225936, 0.91506854],
                         [0.52690045, 0.18135443, 0.8858285 ],
                         [0.51731996, 0.17685482, 0.8930954 ],
                         [0.51290311, 0.15913388, 0.83136549],
                         [0.50334306, 0.17092779, 0.90373697],
                         [0.50171526, 0.15616923, 0.86064331],
                         [0.50946908, 0.16900395, 0.88524452],
                         [0.50200357, 0.14823433, 0.82046595],
                         [0.50436041, 0.17194514, 0.90475432],
                         [0.4534189 , 0.09320851, 0.83561058],
                         [0.52621695, 0.17305963, 0.91133042],
                         [0.50477802, 0.10690002, 0.88750077],
                         [0.52235258, 0.1700762 , 0.91128311]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.date_range(start='2022-02-03', periods=16, freq='D')
    )

    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=3)

    metric, backtest_predictions = _backtesting_forecaster(
                                       forecaster            = forecaster,
                                       y                     = y_with_index,
                                       exog                  = exog_with_index,
                                       refit                 = 3,
                                       initial_train_size    = len(y_with_index) - 20,
                                       fixed_train_size      = True,
                                       gap                   = 3,
                                       allow_incomplete_fold = False,
                                       steps                 = 4,
                                       metric                = 'mean_squared_error',
                                       interval              = [5, 95],
                                       n_boot                = 500,
                                       random_state          = 123,
                                       in_sample_residuals   = True,
                                       verbose               = False,
                                       n_jobs                = 1
                                   )
    backtest_predictions = backtest_predictions.asfreq('D')

    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)