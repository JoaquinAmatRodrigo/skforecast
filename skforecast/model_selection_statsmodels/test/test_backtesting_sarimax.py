# Unit test backtesting_sarimax
# ==============================================================================
import numpy as np
import pandas as pd
from pytest import approx
from skforecast.model_selection_statsmodels import backtesting_sarimax

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

def test_output_backtesting_sarimax_no_refit_no_exog_with_mocked():
    '''
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    no refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    '''
    steps = 3
    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_sarimax(
                                        y          = y,
                                        steps      = steps,
                                        metric     = 'mean_squared_error',
                                        initial_train_size = len(y_train),
                                        fixed_train_size   = False,
                                        refit      = False,
                                        order      = (1, 0, 0),
                                        verbose    = False
                                   )
    
    expected_metric = 0.0822951336284151
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


def test_output_backtesting_sarimax_yes_refit_no_exog_with_mocked():
    '''
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    '''
    steps = 3
    n_backtest = 12
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_sarimax(
                                        y          = y,
                                        steps      = steps,
                                        metric     = 'mean_squared_error',
                                        initial_train_size = len(y_train),
                                        fixed_train_size   = False,
                                        refit      = True,
                                        order      = (1, 0, 0),
                                        verbose    = False
                                   )
    
    expected_metric = 0.08690566590779003
    expected_backtest_predictions = pd.DataFrame({
    'predicted_mean':np.array([0.78762096, 0.69437463, 0.61216773, 0.54672076, 0.47904829, 0.41975224, 
                               0.35752885, 0.30814562, 0.26558338, 0.41059292, 0.34901571, 0.29667332]),
    'lower y':np.array([ 0.24826801, -0.02465313, -0.22013845,  0.01241147, -0.23135474, -0.40058642,
                        -0.17946813, -0.40077799, -0.54786123, -0.1528057 , -0.39042186, -0.54741457]),
    'upper y':np.array([1.3269739 , 1.4134024 , 1.44447391, 1.08103006, 1.18945132, 1.2400909 , 
                        0.89452583, 1.01706922, 1.07902799, 0.97399153, 1.08845327, 1.14076121])                                                                
                                                                         }, index=np.arange(38, 50))
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions)


def test_output_backtesting_sarimax_no_refit_yes_exog_with_mocked():
    '''
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    '''
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
    '''
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    '''
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
    '''
    Calleable metric
    '''
    metric = ((y_true - y_pred)/len(y_true)).mean()
    
    return metric


def test_output_backtesting_sarimax_no_refit_no_exog_calleable_metric_with_mocked():
    '''
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    no refit, 12 observations to backtest, steps=4 (no remainder), calleable metric
    '''
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
    '''
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed train size
    '''
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
    '''
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed train size
    '''
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