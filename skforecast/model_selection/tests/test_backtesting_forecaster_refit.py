# Unit test _backtesting_forecaster_refit
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection.model_selection import _backtesting_forecaster_refit

# Fixtures _backtesting_forecaster_refit Series (skforecast==0.4.2)
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

out_sample_residuals = pd.Series(
    np.array([0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061,
              0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781,
              0.3167879 , 0.35426468, 0.17108183, 0.82911263, 0.33867085,
              0.55237008, 0.57855147, 0.52153306, 0.00268806, 0.98834542,
              0.90534158, 0.20763586, 0.29248941, 0.52001015, 0.90191137,
              0.98363088, 0.25754206, 0.56435904, 0.80696868, 0.39437005,
              0.73107304, 0.16106901, 0.60069857, 0.86586446, 0.98352161,
              0.07936579, 0.42834727, 0.20454286, 0.45063649, 0.54776357,
              0.09332671, 0.29686078, 0.92758424, 0.56900373, 0.457412  ,
              0.75352599, 0.74186215, 0.04857903, 0.7086974 , 0.83924335]),
    name='out_sample_residuals')


# ******************************************************************************
# * Test _backtesting_forecaster_refit No Interval                             *
# ******************************************************************************

# Fixtures _backtesting_forecaster_refit No exog No remainder (skforecast==0.4.2)
metric_mocked_no_exog_no_remainder = 0.06598802629306816
backtest_predictions_mocked_no_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339,
                     0.49152015, 0.4841678 , 0.4076433 , 0.50904672, 0.50249462, 0.49232817])
                                                                }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit No exog Yes remainder (skforecast==0.4.2)
metric_mocked_no_exog_yes_remainder = 0.06916732087926723
backtest_predictions_mocked_no_exog_yes_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.5096801 ,
                    0.49519677, 0.47997916, 0.49177914, 0.495797  , 0.57738724, 0.44370472])
                                                                 }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit Yes exog No remainder (skforecast==0.4.2)
metric_mocked_yes_exog_no_remainder = 0.05663345135204598
backtest_predictions_mocked_yes_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.42295275, 0.46286083,
                     0.43618422, 0.43552906, 0.48687517, 0.55455072, 0.55577332, 0.53943402])
                                                                 }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit Yes exog Yes remainder (skforecast==0.4.2)
metric_mocked_yes_exog_yes_remainder = 0.061723961096013524
backtest_predictions_mocked_yes_exog_yes_remainder = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.43595809,
                     0.4349167 , 0.42381237, 0.55165332, 0.53442833, 0.65361802, 0.51297419])
                                                                 }, index=np.arange(38, 50))


def test_output_backtesting_forecaster_refit_no_exog_no_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_no_exog_no_remainder
    expected_backtest_predictions = backtest_predictions_mocked_no_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_no_exog_yes_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_no_exog_yes_remainder
    expected_backtest_predictions = backtest_predictions_mocked_no_exog_yes_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_yes_exog_no_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_yes_exog_no_remainder
    expected_backtest_predictions = backtest_predictions_mocked_yes_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_yes_exog_yes_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_yes_exog_yes_remainder
    expected_backtest_predictions = backtest_predictions_mocked_yes_exog_yes_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


# ******************************************************************************
# * Test _backtesting_forecaster_refit Interval                                *
# ******************************************************************************

# Fixtures _backtesting_forecaster_refit Interval No exog No remainder (skforecast==0.4.2)
metric_mocked_interval_no_exog_no_remainder = 0.06598802629306816
backtest_predictions_mocked_interval_no_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339, 
                     0.49152015, 0.4841678, 0.4076433, 0.50904672, 0.50249462, 0.49232817]),
    'lower_bound':np.array([0.19882822, 0.08272406, 0.18106389, 0.18777395, 0.03520425, 0.12926384,
                            0.0495347 , 0.04527341, 0.0113795 , 0.13676538, 0.12478441, 0.06814153]),
    'upper_bound':np.array([0.95368172, 0.81704742, 0.93685716, 0.9407976 , 0.78486946, 0.93084605,
                            0.84533191, 0.90255909, 0.80099612, 0.88747244, 0.88292664, 0.88718366])                                                                 
                                                                         }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit Interval No exog Yes remainder (skforecast==0.4.2)
metric_mocked_interval_no_exog_yes_remainder = 0.06916732087926723
backtest_predictions_mocked_interval_no_exog_yes_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.5096801 , 
                     0.49519677, 0.47997916, 0.49177914, 0.495797  , 0.57738724, 0.44370472]),
    'lower_bound':np.array([0.19882822, 0.08272406, 0.18106389, 0.18777395, 0.1238825 , 0.06681772,
                            0.09795868, 0.08383945, 0.10160946, 0.08917676, 0.23321023, 0.08685352]),
    'upper_bound':np.array([0.95368172, 0.81704742, 0.93685716, 0.9407976 , 0.85396419, 0.86172991,
                            0.88313129, 0.82354636, 0.93875053, 0.86176335, 0.96037185, 0.84205069])                                                                 
                                                                }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit Interval Yes exog No remainder (skforecast==0.4.2)
metric_mocked_interval_yes_exog_no_remainder = 0.05663345135204598
backtest_predictions_mocked_interval_yes_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.42295275, 0.46286083,
                     0.43618422, 0.43552906, 0.48687517, 0.55455072, 0.55577332, 0.53943402]),
    'lower_bound':np.array([0.24619375, 0.10545295, 0.13120713, 0.08044217, 0.07440334, 0.11331854,
                            0.01436362, 0.02747413, 0.14867238, 0.19834047, 0.19884259, 0.16964474]),
    'upper_bound':np.array([0.95777604, 0.88685543, 0.90755063, 0.87811336, 0.8225198 , 0.81894689,
                            0.81179723, 0.84420112, 0.89407425, 0.93903702, 0.91748574, 0.93705358])                                                                 
                                                                         }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit Interval Yes exog Yes remainder (skforecast==0.4.2)
metric_mocked_interval_yes_exog_yes_remainder = 0.061723961096013524
backtest_predictions_mocked_interval_yes_exog_yes_remainder = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.43595809,
                     0.4349167 , 0.42381237, 0.55165332, 0.53442833, 0.65361802, 0.51297419]),
    'lower_bound':np.array([0.24619375, 0.10545295, 0.13120713, 0.08044217, 0.13725077, 0.08041239,
                            0.05015513, 0.07677812, 0.17434611, 0.16051962, 0.29167326, 0.15775686]),
    'upper_bound':np.array([0.95777604, 0.88685543, 0.90755063, 0.87811336, 0.86891022, 0.74808834,
                            0.80296989, 0.77919033, 0.97680126, 0.8877086 , 1.07608747, 0.90555785])                                                                 
                                                                         }, index=np.arange(38, 50))


def test_output_backtesting_forecaster_refit_interval_no_exog_no_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_interval_no_exog_no_remainder
    expected_backtest_predictions = backtest_predictions_mocked_interval_no_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_interval_no_exog_yes_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_interval_no_exog_yes_remainder
    expected_backtest_predictions = backtest_predictions_mocked_interval_no_exog_yes_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_interval_yes_exog_no_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_interval_yes_exog_no_remainder
    expected_backtest_predictions = backtest_predictions_mocked_interval_yes_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_interval_yes_exog_yes_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'in_sample_residuals = True'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_interval_yes_exog_yes_remainder
    expected_backtest_predictions = backtest_predictions_mocked_interval_yes_exog_yes_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


# ******************************************************************************
# * Out sample residuals                                                       *
# ******************************************************************************

# Fixtures _backtesting_forecaster_refit out_sample_residuals No exog No remainder (skforecast==0.4.2)
metric_mocked_interval_out_sample_residuals_no_exog_no_remainder = 0.06598802629306816
backtest_predictions_mocked_interval_out_sample_residuals_no_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339, 
                     0.49152015, 0.4841678 , 0.4076433 , 0.50904672, 0.50249462, 0.49232817]),
    'lower_bound':np.array([0.63654358, 0.62989756, 0.67156167, 0.70363265, 0.46905871, 0.73507276, 
                            0.64554988, 0.64183541, 0.48700909, 0.68845988, 0.63865297, 0.60684242]),
    'upper_bound':np.array([1.54070487, 1.5131313 , 1.56749058, 1.62564968, 1.37322   , 1.61930035, 
                            1.54870568, 1.55335041, 1.39117037, 1.56935123, 1.51973211, 1.50300901])                                                                
                                                                         }, index=np.arange(38, 50))

def test_output_backtesting_forecaster_refit_interval_out_sample_residuals_no_exog_no_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'in_sample_residuals = False'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)
    forecaster.set_out_sample_residuals(residuals=out_sample_residuals, append=False)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = metric_mocked_interval_out_sample_residuals_no_exog_no_remainder
    expected_backtest_predictions = backtest_predictions_mocked_interval_out_sample_residuals_no_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


# ******************************************************************************
# * Calleable metric                                                           *
# ******************************************************************************

# Fixtures _backtesting_forecaster_refit my_metric No exog No remainder (skforecast==0.4.2)
my_metric_mocked_no_exog_no_remainder = 0.005283745900436151
my_metric_backtest_predictions_mocked_no_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.38969292, 0.52778339,
                     0.49152015, 0.4841678 , 0.4076433 , 0.50904672, 0.50249462, 0.49232817])
                                                                }, index=np.arange(38, 50))

def my_metric(y_true, y_pred):
    '''
    Calleable metric
    '''
    metric = ((y_true - y_pred)/len(y_true)).mean()
    
    return metric

def test_calleable_metric_backtesting_forecaster_refit_no_exog_no_remainder_with_mocked():
    '''
    Test calleable metric in _backtesting_forecaster_refit with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
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
    expected_metric = my_metric_mocked_no_exog_no_remainder
    expected_backtest_predictions = my_metric_backtest_predictions_mocked_no_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


# ******************************************************************************
# * fixed_train_size = True                                                    *
# ******************************************************************************

# Fixtures _backtesting_forecaster_refit fixed_train_size No exog No remainder (skforecast==0.4.2)
metric_mocked_fixed_train_no_exog_no_remainder = 0.06720844584333846
backtest_predictions_mocked_fixed_train_no_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.34597367, 0.50223873,
                     0.47833829, 0.46082257, 0.37810191, 0.49508366, 0.48808014, 0.47323313])
                                                                }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit fixed_train_size No exog Yes remainder (skforecast==0.4.2)
metric_mocked_fixed_train_no_exog_yes_remainder = 0.07217085374372428
backtest_predictions_mocked_fixed_train_no_exog_yes_remainder = pd.DataFrame({
    'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 0.48308861, 0.4909399 , 
                     0.47942107, 0.46025344, 0.46649132, 0.47061725, 0.57603136, 0.41480551])
                                                                }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit fixed_train_size Yes exog No remainder (skforecast==0.4.2)
metric_mocked_fixed_train_yes_exog_no_remainder = 0.05758244401484334
backtest_predictions_mocked_fixed_train_yes_exog_no_remainder = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.37689967, 0.44267729, 
                     0.42642836, 0.41604275, 0.45047245, 0.53784704, 0.53726274, 0.51516772])
                                                                }, index=np.arange(38, 50))

# Fixtures _backtesting_forecaster_refit fixed_train_size Yes exog Yes remainder (skforecast==0.4.2)
metric_mocked_fixed_train_yes_exog_yes_remainder = 0.06425019123005545
backtest_predictions_mocked_fixed_train_yes_exog_yes_remainder = pd.DataFrame({
    'pred':np.array([0.59059622, 0.47257504, 0.53024098, 0.46163343, 0.50035119, 0.41975558, 
                     0.4256614 , 0.41176005, 0.52357817, 0.509974  , 0.65354628, 0.48210726])
                                                                }, index=np.arange(38, 50))

def test_output_backtesting_forecaster_refit_fixed_train_size_no_exog_no_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed_train_size=True
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
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
    expected_metric = metric_mocked_fixed_train_no_exog_no_remainder
    expected_backtest_predictions = backtest_predictions_mocked_fixed_train_no_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_fixed_train_size_no_exog_yes_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    fixed_train_size=True
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = None,
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
    expected_metric = metric_mocked_fixed_train_no_exog_yes_remainder
    expected_backtest_predictions = backtest_predictions_mocked_fixed_train_no_exog_yes_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_fixed_train_size_yes_exog_no_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed_train_size=True
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
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
    expected_metric = metric_mocked_fixed_train_yes_exog_no_remainder
    expected_backtest_predictions = backtest_predictions_mocked_fixed_train_yes_exog_no_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_fixed_train_size_yes_exog_yes_remainder_with_mocked():
    '''
    Test output of _backtesting_forecaster_refit with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    fixed_train_size=True
    '''
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = _backtesting_forecaster_refit(
                                        forecaster          = forecaster,
                                        y                   = y,
                                        exog                = exog,
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
    expected_metric = metric_mocked_fixed_train_yes_exog_yes_remainder
    expected_backtest_predictions = backtest_predictions_mocked_fixed_train_yes_exog_yes_remainder
    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)