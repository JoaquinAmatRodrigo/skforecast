# Unit test backtesting_forecaster_multiseries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from pytest import approx
import sys
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

# Fixtures
series = pd.DataFrame({'1': pd.Series(np.arange(20)), 
                       '2': pd.Series(np.arange(20))
                      })
# series_1 = np.random.rand(50)
# series_2 = np.random.rand(50)
series = pd.DataFrame({'1': pd.Series(np.array(
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
                       '2': pd.Series(np.array(
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
                      }
         )


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_more_than_len_y():
    """
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size > len(series).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    initial_train_size = len(series) + 1
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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

@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0).
    """

    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12

    metric, backtest_predictions = backtesting_forecaster_multiseries(
                    forecaster          = forecaster,
                    series              = series,
                    steps               = steps,
                    level               = '1',
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
    
    expected_metric = 0.2124129141233719
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.4978838984103099, 0.46288426670127997, 0.48433446479429937, 
                         0.510664891759972, 0.49734477162307983, 0.5009680695304023,
                         0.48647770856843825, 0.4884651517014008, 0.48643766346259326, 
                         0.4973047492523979, 0.4899104838474172, 0.4891085370228432])},
        index=np.arange(38, 50))
                                   
    assert expected_metric == approx(metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
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

    metrics, backtest_predictions = backtesting_forecaster_multiseries(
                    forecaster          = forecaster,
                    series              = series,
                    steps               = steps,
                    level               = '1',
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
    
    expected_metrics = [0.2124129141233719, 0.2124129141233719]
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.4978838984103099, 0.46288426670127997, 0.48433446479429937, 
                         0.510664891759972, 0.49734477162307983, 0.5009680695304023,
                         0.48647770856843825, 0.4884651517014008, 0.48643766346259326, 
                         0.4973047492523979, 0.4899104838474172, 0.4891085370228432])},
        index=np.arange(38, 50))
                                   
    assert expected_metrics == metrics
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)