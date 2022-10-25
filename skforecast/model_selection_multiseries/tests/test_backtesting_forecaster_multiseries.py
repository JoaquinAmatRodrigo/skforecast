# Unit test backtesting_forecaster_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pytest import approx
import sys
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

# Fixtures
# series_1 = np.random.rand(50)
# series_2 = np.random.rand(50)
series = pd.DataFrame({'l1': pd.Series(np.array(
                                 [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                                  0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                                  0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                                  0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                                  0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                                  0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                                  0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                                  0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                                  0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                                  0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]
                                       )
                             ), 
                       'l2': pd.Series(np.array(
                                 [0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                                  0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                                  0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                                  0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                                  0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                                  0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                                  0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                                  0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                                  0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                                  0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]
                                       )
                             )
                       })


@pytest.mark.parametrize("initial_train_size", 
                         [(len(series)), (len(series) + 1)], 
                         ids = lambda value : f'len: {value}' )
def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_more_than_or_equal_to_len_series(initial_train_size):
    """
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size >= len(series).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 3
                 )
    
    err_msg = re.escape('If used, `initial_train_size` must be smaller than length of `series`.')
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            levels              = 'l1',
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_less_than_forecaster_window_size():
    """
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size < forecaster.window_size.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 3
                 )

    initial_train_size = forecaster.window_size - 1
    
    err_msg = re.escape(
            f"`initial_train_size` must be greater than "
            f"forecaster's window_size ({forecaster.window_size})."
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            levels              = ['l1'],
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_None_and_forecaster_not_fitted():
    """
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size is None and
    forecaster is not fitted.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )

    initial_train_size = None
    
    err_msg = re.escape('`forecaster` must be already trained if no `initial_train_size` is provided.')
    with pytest.raises(NotFittedError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            levels              = 'l1',
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_multiseries_exception_when_refit_not_bool():
    """
    Test Exception is raised in backtesting_forecaster_multiseries when refit is not bool.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )

    refit = 'not_bool'
    
    err_msg = re.escape( f'`refit` must be boolean: `True`, `False`.')
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            levels              = 'l1',
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            refit               = refit,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_None_and_refit_True():
    """
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size is None
    and refit is True.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    forecaster.fitted = True

    initial_train_size = None
    refit = True
    
    err_msg = re.escape(f'`refit` is only allowed when `initial_train_size` is not `None`.')
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            levels              = 'l1',
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = refit,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_multiseries_exception_when_forecaster_not_ForecasterAutoregMultiSeries():
    """
    Test Exception is raised in backtesting_forecaster_multiseries when forecaster is not of type
    ForecasterAutoregMultiSeries.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
            ('`forecaster` must be of type `ForecasterAutoregMultiSeries`, for all other '
             'types of forecasters use the functions in `model_selection` module.')
        )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            levels              = 'l1',
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


@pytest.mark.parametrize("levels, refit", 
                         [(1    , True), 
                          (1    , False)],
                         ids=lambda d: f'levels: {d}')
def test_backtesting_forecaster_multiseries_exception_when_levels_not_list_str_None(levels, refit):
    """
    Test Exception is raised in backtesting_forecaster_multiseries when 
    `levels` is not a `list`, `str` or `None`.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
                (f'`levels` must be a `list` of column names, a `str` '
                 f'of a column name or `None`.')
              )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            levels              = levels,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            refit               = refit,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_not_refit_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries without refit
    with mocked (mocked done in Skforecast v0.5.0).
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster          = forecaster,
                                               series              = series,
                                               steps               = steps,
                                               levels              = 'l1',
                                               metric              = 'mean_absolute_error',
                                               initial_train_size  = len(series) - n_validation,
                                               refit               = False,
                                               fixed_train_size    = False,
                                               exog                = None,
                                               interval            = None,
                                               n_boot              = 500,
                                               random_state        = 123,
                                               in_sample_residuals = True,
                                               verbose             = True
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'],
                                    'mean_absolute_error': [0.20754847190853098]})
    expected_predictions = pd.DataFrame({
                               'l1':np.array([0.4978839 , 0.46288427, 0.48433446, 
                                              0.48767779, 0.477799  , 0.48523814, 
                                              0.49341916, 0.48967772, 0.48517846, 
                                              0.49868447, 0.4859614 , 0.48480032])},
                               index=np.arange(38, 50)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_fixed_train_size_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries with refit and
    fixed_train_size with mocked (mocked done in Skforecast v0.5.0).
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster          = forecaster,
                                               series              = series,
                                               steps               = steps,
                                               levels              = ['l1'],
                                               metric              = 'mean_absolute_error',
                                               initial_train_size  = len(series) - n_validation,
                                               refit               = True,
                                               fixed_train_size    = True, 
                                               exog                = None,
                                               interval            = None,
                                               n_boot              = 500,
                                               random_state        = 123,
                                               in_sample_residuals = True,
                                               verbose             = True
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'],
                                    'mean_absolute_error': [0.21651617115803679]})
    expected_predictions = pd.DataFrame({
                               'l1':np.array([0.4978839 , 0.46288427, 0.48433446, 
                                              0.50853803, 0.50006415, 0.50105623,
                                              0.46764379, 0.46845675, 0.46768947, 
                                              0.48298309, 0.47778385, 0.47776533])},
                               index=np.arange(38, 50)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries with refit
    with mocked (mocked done in Skforecast v0.5.0).
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster          = forecaster,
                                               series              = series,
                                               steps               = steps,
                                               levels              = 'l1',
                                               metric              = 'mean_absolute_error',
                                               initial_train_size  = len(series) - n_validation,
                                               refit               = True,
                                               fixed_train_size    = False,
                                               exog                = None,
                                               interval            = None,
                                               n_boot              = 500,
                                               random_state        = 123,
                                               in_sample_residuals = True,
                                               verbose             = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.2124129141233719]})
    expected_predictions = pd.DataFrame({
                               'l1':np.array([0.4978838984103099, 0.46288426670127997, 0.48433446479429937, 
                                               0.510664891759972, 0.49734477162307983, 0.5009680695304023,
                                               0.48647770856843825, 0.4884651517014008, 0.48643766346259326, 
                                               0.4973047492523979, 0.4899104838474172, 0.4891085370228432])},
                               index=np.arange(38, 50)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_with_mocked_metrics():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    with mocked and list of metrics (mocked done in Skforecast v0.5.0). Only valid for
    python > 3.7.
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster          = forecaster,
                                               series              = series,
                                               steps               = steps,
                                               levels              = 'l1',
                                               metric              = ['mean_absolute_error', mean_absolute_error],
                                               initial_train_size  = len(series) - n_validation,
                                               refit               = True,
                                               fixed_train_size    = False,
                                               exog                = None,
                                               interval            = None,
                                               n_boot              = 500,
                                               random_state        = 123,
                                               in_sample_residuals = True,
                                               verbose             = False
                                           )
    
    expected_metric = pd.DataFrame(data    = [['l1', 0.2124129141233719, 0.2124129141233719]],
                                   columns = ['levels', 'mean_absolute_error', 'mean_absolute_error'])
    expected_predictions = pd.DataFrame({
                               'l1':np.array([0.4978838984103099, 0.46288426670127997, 0.48433446479429937, 
                                               0.510664891759972, 0.49734477162307983, 0.5009680695304023,
                                               0.48647770856843825, 0.4884651517014008, 0.48643766346259326, 
                                               0.4973047492523979, 0.4899104838474172, 0.4891085370228432])},
                               index=np.arange(38, 50)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_no_refit_levels_metrics_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries with no refit
    and multiple levels and metrics with mocked (mocked done in Skforecast v0.5.0).
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster          = forecaster,
                                               series              = series,
                                               steps               = steps,
                                               levels              = ['l1', 'l2'],
                                               metric              = ['mean_absolute_error', mean_absolute_error],
                                               initial_train_size  = len(series) - n_validation,
                                               refit               = False,
                                               fixed_train_size    = False,
                                               exog                = None,
                                               interval            = None,
                                               n_boot              = 500,
                                               random_state        = 123,
                                               in_sample_residuals = True,
                                               verbose             = False
                                           )
    
    expected_metric = pd.DataFrame(data    = [['l1', 0.20754847190853098, 0.20754847190853098],
                                              ['l2', 0.21399842651788026, 0.21399842651788026]],
                                   columns = ['levels', 'mean_absolute_error', 'mean_absolute_error'])
    expected_predictions = pd.DataFrame({
                               'l1':np.array([0.4978839 , 0.46288427, 0.48433446, 0.48767779, 0.477799  ,
                                              0.48523814, 0.49341916, 0.48967772, 0.48517846, 0.49868447, 
                                              0.4859614 , 0.48480032]),
                               'l2':np.array([0.50266337, 0.53045945, 0.50527774, 0.5150115 , 0.49356802,
                                              0.50376722, 0.51866166, 0.49041268, 0.50349225, 0.48625798,
                                              0.52405344, 0.50605782])},
                               index=np.arange(38, 50)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_not_refit_exog_interval_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries without refit
    with mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster          = forecaster,
                                               series              = series,
                                               steps               = steps,
                                               levels              = ['l1'],
                                               metric              = 'mean_absolute_error',
                                               initial_train_size  = len(series) - n_validation,
                                               refit               = False,
                                               fixed_train_size    = False,
                                               exog                = series['l1'].rename('exog_1'),
                                               interval            = [5, 95],
                                               n_boot              = 500,
                                               random_state        = 123,
                                               in_sample_residuals = True,
                                               verbose             = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.14238176570382063]})
    expected_predictions = pd.DataFrame(
                               data = np.array([[0.64371728, 0.36845896, 0.91248693],
                                                [0.47208179, 0.19871058, 0.74002421],
                                                [0.52132498, 0.24592578, 0.78440458],
                                                [0.3685079 , 0.09324957, 0.63727755],
                                                [0.42192697, 0.14855575, 0.68986939],
                                                [0.46785602, 0.19245683, 0.73093562],
                                                [0.61543694, 0.34017861, 0.88420659],
                                                [0.41627752, 0.14290631, 0.68421995],
                                                [0.4765156 , 0.20111641, 0.7395952 ],
                                                [0.65858347, 0.38332514, 0.92735312],
                                                [0.49986428, 0.22649307, 0.7678067 ],
                                                [0.51750994, 0.24211075, 0.78058954]]),
                               columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
                               index = np.arange(38, 50)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_fixed_train_size_exog_interval_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries with refit and fixed_train_size
    with mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster          = forecaster,
                                               series              = series,
                                               steps               = steps,
                                               levels              = 'l1',
                                               metric              = 'mean_absolute_error',
                                               initial_train_size  = len(series) - n_validation,
                                               refit               = True,
                                               fixed_train_size    = True,
                                               exog                = series['l1'].rename('exog_1'),
                                               interval            = [5, 95],
                                               n_boot              = 500,
                                               random_state        = 123,
                                               in_sample_residuals = True,
                                               verbose             = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.1509587543248219]})
    expected_predictions = pd.DataFrame(
                               data = np.array([[0.64371728, 0.36845896, 0.91248693],
                                                [0.47208179, 0.19871058, 0.74002421],
                                                [0.52132498, 0.24592578, 0.78440458],
                                                [0.38179014, 0.15353798, 0.65675721],
                                                [0.43343713, 0.19888319, 0.70802106],
                                                [0.4695322 , 0.18716947, 0.74043601],
                                                [0.57891069, 0.31913315, 0.84636925],
                                                [0.41212578, 0.15090397, 0.69394422],
                                                [0.46851038, 0.20736343, 0.76349422],
                                                [0.63190066, 0.35803673, 0.90303047],
                                                [0.49132695, 0.21857874, 0.78112082],
                                                [0.51665452, 0.26139311, 0.80879351]]),
                               columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
                               index = np.arange(38, 50)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)