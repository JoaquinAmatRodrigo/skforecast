# Unit test _backtesting_forecaster_no_refit
# ==============================================================================
import numpy as np
import pandas as pd
from pytest import approx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection.model_selection import _backtesting_forecaster_no_refit

# Fixtures _backtesting_forecaster_no_refit Series (skforecast==0.5.1)
# np.random.seed(123)
# y = np.random.rand(50)
# exog = np.random.rand(50)
# out_sample_residuals = np.random.rand(50)

y = pd.Series(
    np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
              0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
              0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
              0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
              0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
              0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
              0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
              0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
              0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
              0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]))

exog = pd.Series(
    np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
              0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
              0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
              0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
              0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
              0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
              0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
              0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
              0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
              0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]), 
    name='exog')

out_sample_residuals = np.array([
                        0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061,
                        0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781,
                        0.3167879 , 0.35426468, 0.17108183, 0.82911263, 0.33867085,
                        0.55237008, 0.57855147, 0.52153306, 0.00268806, 0.98834542,
                        0.90534158, 0.20763586, 0.29248941, 0.52001015, 0.90191137,
                        0.98363088, 0.25754206, 0.56435904, 0.80696868, 0.39437005,
                        0.73107304, 0.16106901, 0.60069857, 0.86586446, 0.98352161,
                        0.07936579, 0.42834727, 0.20454286, 0.45063649, 0.54776357,
                        0.09332671, 0.29686078, 0.92758424, 0.56900373, 0.457412  ,
                        0.75352599, 0.74186215, 0.04857903, 0.7086974 , 0.83924335
                      ])


# ******************************************************************************
# * Test _backtesting_forecaster_no_refit No Interval                             *
# ******************************************************************************

def test_output_backtesting_forecaster_no_refit_no_exog_no_remainder_ForecasterAutoreg_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoreg.
    """
    expected_metric = 0.0646438286283131
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949,
                     0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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


def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    
    lags = y[-1:-4:-1]
    
    return lags 


def test_output_backtesting_forecaster_no_refit_no_exog_no_remainder_ForecasterAutoregCustom_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoregCustom.
    """
    expected_metric = 0.06464382862831312
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

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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


def test_output_backtesting_forecaster_no_refit_no_exog_no_remainder_ForecasterAutoregDirect_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterAutoregDirect.
    """
    expected_metric = 0.06999891546733726
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

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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


def test_output_backtesting_forecaster_no_refit_no_exog_no_initial_train_size_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    no initial_train_size, steps=1, ForecasterAutoreg.
    """
    expected_metric = 0.05194702533929101
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
                         0.5724869 , 0.45299923])}, index=pd.RangeIndex(start=3, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    initial_train_size = None

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = initial_train_size,
                                        steps               = 1,
                                        metric              = 'mean_squared_error',
                                        interval            = None,
                                        n_boot              = 500,
                                        random_state        = 123,
                                        in_sample_residuals = True,
                                        verbose             = False
                                   )
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_refit_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = 0.07085869503962372
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.54252612, 
                     0.46139434, 0.44730047, 0.50031862, 0.49831103, 0.55613172, 0.42970914])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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


def test_output_backtesting_forecaster_no_refit_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metric = 0.05585411566592716
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.41316188, 0.51416101, 
                     0.42705363, 0.4384041 , 0.42611891, 0.59547291, 0.5170294 , 0.4982889 ])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        initial_train_size  = len(y_train),
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


def test_output_backtesting_forecaster_no_refit_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = 0.06313056651237414
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.49157332, 
                     0.43327346, 0.42420804, 0.53389427, 0.51693094, 0.60207937, 0.48227974])
                                                                 }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        initial_train_size  = len(y_train),
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
# * Test _backtesting_forecaster_no_refit Interval                                *
# ******************************************************************************

def test_output_backtesting_forecaster_no_refit_interval_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.0646438286283131
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                     0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952]),
    'lower_bound':np.array([0.19882822, 0.08272406, 0.18106389, 0.18777395, 0.03750242, 0.20853217, 
                            0.08400155, 0.11618618, 0.01153279, 0.22830219, 0.11822679, 0.1154188 ]),
    'upper_bound':np.array([0.95368172, 0.81704742, 0.93685716, 0.9407976 , 0.79235592, 0.94285553, 
                            0.83979482, 0.86920984, 0.76638629, 0.96262555, 0.87402007, 0.86844245])                                                                 
                                                                         }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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


def test_output_backtesting_forecaster_no_refit_interval_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.07085869503962372
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.54252612, 
                     0.46139434, 0.44730047, 0.50031862, 0.49831103, 0.55613172, 0.42970914]),
    'lower_bound':np.array([0.19882822, 0.08272406, 0.18106389, 0.18777395, 0.1238825 , 0.18417655, 
                            0.11056702, 0.07866669, 0.1586379 , 0.13910492, 0.19778215, 0.07888182]),
    'upper_bound':np.array([0.95368172, 0.81704742, 0.93685716, 0.9407976 , 0.85396419, 0.93903005, 
                            0.84489038, 0.83445996, 0.91166155, 0.86918661, 0.95263565, 0.81320518])                                                                 
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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


def test_output_backtesting_forecaster_no_refit_interval_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.05585411566592716
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.41316188, 0.51416101, 
                     0.42705363, 0.4384041 , 0.42611891, 0.59547291,0.5170294 , 0.4982889 ]),
    'lower_bound':np.array([0.24619375, 0.10545295, 0.13120713, 0.08044217, 0.06875941, 0.14703893, 
                            0.02801978, 0.05721284, 0.08171644, 0.22835082, 0.11799555, 0.11709763]),
    'upper_bound':np.array([0.95777604, 0.88685543, 0.90755063, 0.87811336, 0.7803417 , 0.9284414 , 
                            0.80436328, 0.85488403, 0.79329874, 1.0097533 , 0.89433905, 0.91476883])                                                                 
                                                                         }, index=pd.RangeIndex(start=38, stop=50, step=1))
    
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        initial_train_size  = len(y_train),
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


def test_output_backtesting_forecaster_no_refit_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    """
    expected_metric = 0.06313056651237414
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.49157332, 
                     0.43327346, 0.42420804, 0.53389427, 0.51693094, 0.60207937, 0.48227974]),
    'lower_bound':np.array([0.24619375, 0.10545295, 0.13120713, 0.08044217, 0.13725077, 0.14717085, 
                            0.06615137, 0.02517419, 0.15270301, 0.15383052, 0.25767689, 0.11515766]),
    'upper_bound':np.array([0.95777604, 0.88685543, 0.90755063, 0.87811336, 0.86891022, 0.85875315, 
                            0.84755385, 0.80151769, 0.9503742 , 0.88548997, 0.96925919, 0.89656013])                                                                 
                                                                         }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
                                        initial_train_size  = len(y_train),
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

def test_output_backtesting_forecaster_no_refit_interval_out_sample_residuals_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = False'
    """
    expected_metric = 0.0646438286283131
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                         0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952]),
        'lower_bound':np.array([0.63654358, 0.62989756, 0.67156167, 0.70363265, 0.47521778, 0.75570567, 
                                0.57449933, 0.63204489, 0.44924816, 0.7754757 , 0.60872458, 0.63127751]),
        'upper_bound':np.array([1.54070487, 1.5131313 , 1.56749058, 1.62564968, 1.37937907, 1.63893941, 
                                1.47042824, 1.55406192, 1.35340944, 1.65870943, 1.50465348, 1.55329453])                                                                
                                                                            }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=out_sample_residuals, append=False)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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
# * Callable metric                                                           *
# ******************************************************************************

def my_metric(y_true, y_pred): # pragma: no cover
    """
    Callable metric
    """
    metric = ((y_true - y_pred)/len(y_true)).mean()
    
    return metric

def test_callable_metric_backtesting_forecaster_no_refit_no_exog_no_remainder_with_mocked():
    """
    Test callable metric in _backtesting_forecaster_no_refit with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metric = 0.005603130564222017
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                     0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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


def test_list_metrics_backtesting_forecaster_no_refit_no_exog_no_remainder_with_mocked():
    """
    Test list of metrics in _backtesting_forecaster_no_refit with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metrics = [0.0646438286283131, 0.0646438286283131]
    expected_predictions = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.39585199, 0.55935949, 
                     0.45263533, 0.4578669 , 0.36988237, 0.57912951, 0.48686057, 0.45709952])
                                                                }, index=pd.RangeIndex(start=38, stop=50, step=1))

    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metrics, backtest_predictions = _backtesting_forecaster_no_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
                                        initial_train_size  = len(y_train),
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