# Unit test _backtesting_forecaster No refit
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.0646438286283131]})
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 
                         0.39585199, 0.55935949, 0.45263533, 0.4578669 , 
                         0.36988237, 0.57912951, 0.48686057, 0.45709952])}, 
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                       forecaster         = forecaster,
                                       y                  = y,
                                       exog               = None,
                                       refit              = False,
                                       initial_train_size = len(y_train),
                                       steps              = 4,
                                       metric             = 'mean_squared_error',
                                       n_jobs             = n_jobs,
                                       verbose            = False
                                   )
                                   
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def create_predictors(y):  # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]
    return lags 


@pytest.mark.parametrize("refit", [False, 0],
                         ids=lambda n: f'refit: {n}')
def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterAutoregCustom_with_mocked(refit):
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoregCustom.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06464382862831312]})
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                     0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(), 
                     fun_predictors = create_predictors,
                     window_size    = 3
                 )
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        refit              = refit,
                                        initial_train_size = len(y_train),
                                        steps              = 4,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )
                                   
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterAutoregDirect_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoregDirect.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06999891546733726]})
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.5468482 , 0.44670961, 0.57651222, 0.52511275, 0.39745044, 0.53365885, 
                     0.39390143, 0.47002119, 0.38080838, 0.56987255, 0.42302249, 0.45580163])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 4
                 )
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        refit              = False,
                                        initial_train_size = len(y_train),
                                        steps              = 4,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )
                                   
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_no_initial_train_size_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    no initial_train_size, steps=1, ForecasterAutoreg.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.05194702533929101]})
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.50394528, 0.53092847, 0.51638165, 0.47382814, 0.61996956,
                         0.47685471, 0.52565717, 0.50842469, 0.4925563 , 0.55717972,
                         0.45707228, 0.45990822, 0.53762873, 0.5230163 , 0.41661072,
                         0.51097738, 0.52448483, 0.48179537, 0.5307759 , 0.55580453,
                         0.51780297, 0.53189442, 0.55356883, 0.46142853, 0.52517734,
                         0.46241276, 0.49292214, 0.53169102, 0.40448875, 0.55686226,
                         0.46860633, 0.5098154 , 0.49041677, 0.48435035, 0.51152271,
                         0.56870534, 0.53226143, 0.49091506, 0.56878395, 0.42767269,
                         0.53335856, 0.48167273, 0.5658333 , 0.41464667, 0.56733702,
                         0.5724869 , 0.45299923])
        }, 
        index=pd.RangeIndex(start=3, stop=50, step=1)
    )
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)
    initial_train_size = None
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        refit              = False,
                                        initial_train_size = initial_train_size,
                                        steps              = 1,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )                
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07085869503962372]})
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.54252612, 
                     0.46139434, 0.44730047, 0.50031862, 0.49831103, 0.55613172, 0.42970914])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        refit              = False,
                                        initial_train_size = len(y_train),
                                        steps              = 5,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )         
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.05585411566592716]})
    expected_predictions = pd.DataFrame({
    'pred': np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.41316188, 0.51416101, 
                      0.42705363, 0.4384041 , 0.42611891, 0.59547291, 0.5170294 , 0.4982889 ])
    }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = exog,
                                        refit              = False,
                                        initial_train_size = len(y_train),
                                        steps              = 4,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06313056651237414]})
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.49157332, 
                     0.43327346, 0.42420804, 0.53389427, 0.51693094, 0.60207937, 0.48227974])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = exog,
                                        refit              = False,
                                        initial_train_size = len(y_train),
                                        steps              = 5,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_yes_remainder_skip_folds_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.044538146622722964]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.50035119,
                    0.60207937,
                    0.48227974,
                ]
            )
        },
        index=pd.Index([38, 39, 40, 41, 42, 48, 49], dtype="int64"),
    )
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        exog=exog,
        refit=False,
        initial_train_size=len(y_train),
        steps=5,
        skip_folds=2,
        metric="mean_squared_error",
        verbose=True,
    )
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Test _backtesting_forecaster Interval                                      *
# ******************************************************************************

def test_output_backtesting_forecaster_interval_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06464382862831312]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.55717779, 0.19882822, 0.95368172],
            [0.43355138, 0.05699193, 0.8646772 ],
            [0.54969767, 0.20857548, 0.93889056],
            [0.52945466, 0.17670586, 0.87242694],
            [0.39585199, 0.03750242, 0.79235592],
            [0.55935949, 0.18280004, 0.99048531],
            [0.45263533, 0.11151314, 0.84182822],
            [0.4578669 , 0.1051181 , 0.80083918],
            [0.36988237, 0.01153279, 0.76638629],
            [0.57912951, 0.20257006, 1.01025533],
            [0.48686057, 0.14573839, 0.87605347],
            [0.45709952, 0.10435072, 0.8000718 ]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
    forecaster = ForecasterAutoreg(
        regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15}
    )
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = None,
                                        refit                   = False,
                                        initial_train_size      = len(y_train),
                                        steps                   = 4,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False,
                                        show_progress           = False
                                   )
                            
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07085869503962372]})
    expected_predictions = pd.DataFrame(
        data = np.array([
                [0.55717779, 0.19882822, 0.95368172],
                [0.43355138, 0.05699193, 0.8646772 ],
                [0.54969767, 0.20857548, 0.93889056],
                [0.52945466, 0.17670586, 0.87242694],
                [0.48308861, 0.12598277, 0.8744728 ],
                [0.54252612, 0.18417655, 0.93903005],
                [0.46139434, 0.08483489, 0.89252016],
                [0.44730047, 0.10617829, 0.83649337],
                [0.50031862, 0.14756982, 0.8432909 ],
                [0.49831103, 0.14120519, 0.88969522],
                [0.55613172, 0.19778215, 0.95263565],
                [0.42970914, 0.05314969, 0.86083496]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
    forecaster = ForecasterAutoreg(
        regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15}
    )
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = None,
                                        refit                   = False,
                                        initial_train_size      = len(y_train),
                                        steps                   = 5,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )
    
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.05585411566592716]})
    expected_predictions = pd.DataFrame(
    data = np.array([
            [0.59059622, 0.24619375, 0.95777604],
            [0.47257504, 0.11801273, 0.84870551],
            [0.53024098, 0.15322187, 0.91489125],
            [0.46163343, 0.0970681 , 0.84865478],
            [0.41316188, 0.06875941, 0.7803417 ],
            [0.51416101, 0.1595987 , 0.89029149],
            [0.42705363, 0.05003453, 0.81170391],
            [0.4384041 , 0.07383877, 0.82542545],
            [0.42611891, 0.08171644, 0.79329874],
            [0.59547291, 0.2409106 , 0.97160338],
            [0.5170294 , 0.1400103 , 0.90167968],
            [0.4982889 , 0.13372357, 0.88531025]]
        ),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = exog,
                                        refit                   = False,
                                        initial_train_size      = len(y_train),
                                        steps                   = 4,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )
    
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06313056651237414]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.59059622, 0.24619375, 0.95777604],
            [0.47257504, 0.11801273, 0.84870551],
            [0.53024098, 0.15322187, 0.91489125],
            [0.46163343, 0.0970681 , 0.84865478],
            [0.50035119, 0.14541886, 0.8608513 ],
            [0.49157332, 0.14717085, 0.85875315],
            [0.43327346, 0.07871115, 0.80940394],
            [0.42420804, 0.04718894, 0.80885832],
            [0.53389427, 0.16932894, 0.92091562],
            [0.51693094, 0.16199861, 0.87743105],
            [0.60207937, 0.25767689, 0.96925919],
            [0.48227974, 0.12771743, 0.85841022]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(
        regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15}
    )
    n_backtest = 12
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = exog,
                                        refit                   = False,
                                        initial_train_size      = len(y_train),
                                        steps                   = 5,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False,
                                        show_progress           = False
                                   )
    
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Out sample residuals                                                       *
# ******************************************************************************

def test_output_backtesting_forecaster_interval_out_sample_residuals_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'use_in_sample_residuals = False'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.0646438286283131]})
    expected_predictions = pd.DataFrame({
        'pred': np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                         0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952]),
        'lower_bound': np.array([0.63654358, 0.62989756, 0.67156167, 0.70363265, 0.47521778, 0.75570567, 
                                0.57449933, 0.63204489, 0.44924816, 0.7754757 , 0.60872458, 0.63127751]),
        'upper_bound': np.array([1.54070487, 1.5131313 , 1.56749058, 1.62564968, 1.37937907, 1.63893941, 
                                1.47042824, 1.55406192, 1.35340944, 1.65870943, 1.50465348, 1.55329453])
                                                                            }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=out_sample_residuals, append=False)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = None,
                                        refit                   = False,
                                        initial_train_size      = len(y_train),
                                        steps                   = 4,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = False,
                                        verbose                 = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Callable metric                                                           *
# ******************************************************************************

def my_metric(y_true, y_pred):  # pragma: no cover
    """
    Callable metric
    """
    metric = ((y_true - y_pred) / len(y_true)).mean()
    
    return metric


def test_callable_metric_backtesting_forecaster_no_exog_no_remainder_with_mocked():
    """
    Test callable metric in _backtesting_forecaster with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"my_metric": [0.005603130564222017]})
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                     0.45263533, 0.4578669, 0.36988237, 0.57912951, 0.48686057, 0.45709952])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        refit              = False,
                                        initial_train_size = len(y_train),
                                        steps              = 4,
                                        metric             = my_metric,
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_list_metrics_backtesting_forecaster_no_exog_no_remainder_with_mocked():
    """
    Test list of metrics in _backtesting_forecaster with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metrics = pd.DataFrame({
        "mean_absolute_error": [0.20796216],
        "mean_squared_error": [0.0646438286283131]
    })
    expected_predictions = pd.DataFrame({
    'pred': np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                     0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metrics, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        refit              = False,
                                        initial_train_size = len(y_train),
                                        steps              = 4,
                                        metric             = ['mean_absolute_error', mean_squared_error],
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metrics, metrics)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Gap                                                                        *
# ******************************************************************************


def test_output_backtesting_forecaster_interval_yes_exog_yes_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07965887934114284]})
    expected_predictions = pd.DataFrame(
        {
            "pred": {
                33: 0.6502299892039047,
                34: 0.5236463733888286,
                35: 0.4680925560056311,
                36: 0.48759731640448023,
                37: 0.4717244484701478,
                38: 0.5758584470242223,
                39: 0.5359293308956417,
                40: 0.48995122056218404,
                41: 0.4527131747771053,
                42: 0.49519412870799573,
                43: 0.2716782430953111,
                44: 0.2781382241553876,
                45: 0.31569423787508544,
                46: 0.39832240179281636,
                47: 0.3482268061581242,
                48: 0.6209868551179476,
                49: 0.4719075668681217,
            },
            "lower_bound": {
                33: 0.2718315398459972,
                34: 0.19453326150156153,
                35: 0.12440677892063873,
                36: 0.12107577460096824,
                37: 0.14533367434351524,
                38: 0.19745999766631478,
                39: 0.2068162190083746,
                40: 0.14626544347719167,
                41: 0.08619163297359334,
                42: 0.1688033545813632,
                43: -0.10672020626259643,
                44: -0.05097488773187947,
                45: -0.027991539209906935,
                46: 0.031800859989304375,
                47: 0.021836032031491648,
                48: 0.2425884057600401,
                49: 0.14279445498085463,
            },
            "upper_bound": {
                33: 1.065039136744312,
                34: 0.9230439995748042,
                35: 0.7539505750052855,
                36: 0.8314505919311639,
                37: 0.7469755216091047,
                38: 0.9906675945646294,
                39: 0.9353269570816173,
                40: 0.7758092395618383,
                41: 0.796566450303789,
                42: 0.7704452018469528,
                43: 0.6864873906357182,
                44: 0.6775358503413632,
                45: 0.6015522568747398,
                46: 0.7421756773195001,
                47: 0.6234778792970812,
                48: 1.0357960026583548,
                49: 0.8713051930540974,
            },
        }
    )
    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 8
                 )
    n_backtest = 20
    y_train = y[:-n_backtest]
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = exog,
                                        refit                   = False,
                                        initial_train_size      = len(y_train),
                                        gap                     = 3,
                                        allow_incomplete_fold   = True,
                                        steps                   = 5,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_not_allow_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'use_in_sample_residuals = True', allow_incomplete_fold = False
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = pd.DataFrame({"mean_squared_error": [0.08826806818628832]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.65022999,  0.27183154,  1.06503914],
                         [0.52364637,  0.19453326,  0.923044  ],
                         [0.46809256,  0.12440678,  0.75395058],
                         [0.48759732,  0.12107577,  0.83145059],
                         [0.47172445,  0.14533367,  0.74697552],
                         [0.57585845,  0.19746   ,  0.99066759],
                         [0.53592933,  0.20681622,  0.93532696],
                         [0.48995122,  0.14626544,  0.77580924],
                         [0.45271317,  0.08619163,  0.79656645],
                         [0.49519413,  0.16880335,  0.7704452 ],
                         [0.27167824, -0.10672021,  0.68648739],
                         [0.27813822, -0.05097489,  0.67753585],
                         [0.31569424, -0.02799154,  0.60155226],
                         [0.3983224 ,  0.03180086,  0.74217568],
                         [0.34822681,  0.02183603,  0.62347788]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.date_range(start='2022-02-03', periods=15, freq='D')
    )

    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 8
                 )

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y_with_index,
                                        exog                    = exog_with_index,
                                        refit                   = False,
                                        initial_train_size      = len(y_with_index) - 20,
                                        gap                     = 3,
                                        allow_incomplete_fold   = False,
                                        steps                   = 5,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )
    backtest_predictions = backtest_predictions.asfreq('D')

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)
