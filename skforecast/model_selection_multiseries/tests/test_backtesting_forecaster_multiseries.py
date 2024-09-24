# Unit test backtesting_forecaster_multiseries
# ==============================================================================
import re
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

# Fixtures
from .fixtures_model_selection_multiseries import series
THIS_DIR = Path(__file__).parent
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}
series_with_nans = series.copy()
series_with_nans.iloc[:10, series_with_nans.columns.get_loc('l2')] = np.nan

def create_predictors(y):  # pragma: no cover
    """
    Create first 2 lags of a time series.
    """
    lags = y[-1:-3:-1]

    return lags

def create_predictors_14(y):  # pragma: no cover
    """
    Create first 14 lags of a time series.
    """
    lags = y[-1:-15:-1]

    return lags

def create_predictors_1_7_14(y):  # pragma: no cover
    """
    Create lags 1 and 5 of a time series.
    """
    lag_1 = y[-1]
    lag_7 = y[-7]
    lag_14 = y[-14]
    lags = np.hstack([lag_1, lag_7, lag_14])

    return lags


def test_backtesting_forecaster_multiseries_TypeError_when_forecaster_not_a_forecaster_multiseries():
    """
    Test TypeError is raised in backtesting_forecaster_multiseries when 
    forecaster is not of type 'ForecasterAutoregMultiSeries', 
    'ForecasterAutoregMultiSeriesCustom' or 'ForecasterAutoregMultiVariate'.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )

    err_msg = re.escape(
        ("`forecaster` must be of type ['ForecasterAutoregMultiSeries', "
         "'ForecasterAutoregMultiSeriesCustom', 'ForecasterAutoregMultiVariate', "
         "'ForecasterRnn'], for all other types of forecasters use the functions "
         "available in the `model_selection` module. "
         f"Got {type(forecaster).__name__}")
    )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster_multiseries(
            forecaster            = forecaster,
            series                = series,
            steps                 = 4,
            levels                = 'l1',
            metric                = 'mean_absolute_error',
            add_aggregated_metric = False,
            initial_train_size    = 12,
            refit                 = False,
            fixed_train_size      = False,
            exog                  = None,
            verbose               = False
        )


# ForecasterAutoregMultiSeries and ForecasterAutoregMultiSeriesCustom
# ======================================================================================================================
@pytest.mark.parametrize("forecaster, n_jobs", 
                         [(ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), -1),
                          (ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), 1),
                          (ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), 'auto'),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), -1),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), 1),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), 'auto')], 
                         ids=lambda fc: f'forecaster, n_jobs: {fc}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_not_refit_with_mocked(forecaster, n_jobs):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom without refit with mocked 
    (mocked done in Skforecast v0.5.0).
    """
    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = 'l1',
                                               metric                = 'mean_absolute_error',
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = False,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = True,
                                               n_jobs                = n_jobs
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'],
                                    'mean_absolute_error': [0.20754847190853098]})
    expected_predictions = pd.DataFrame({
                               'l1': np.array([0.4978839 , 0.46288427, 0.48433446, 
                                               0.48767779, 0.477799  , 0.48523814, 
                                               0.49341916, 0.48967772, 0.48517846, 
                                               0.49868447, 0.4859614 , 0.48480032])},
                               index=pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_not_refit_not_initial_train_size_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom without refit and initial_train_size 
    is None with mocked, forecaster must be fitted, (mocked done in Skforecast v0.5.0).
    """

    forecaster.fit(series=series)

    steps = 1
    initial_train_size = None

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = 'l1',
                                               metric                = mean_absolute_error,
                                               add_aggregated_metric = False,
                                               initial_train_size    = initial_train_size,
                                               refit                 = False,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'],
                                    'mean_absolute_error': [0.18616882305307128]})
    expected_predictions = pd.DataFrame({
        'l1': np.array([0.48459053, 0.49259742, 0.51314434, 0.51387387, 0.49192289,
                        0.53266761, 0.49986433, 0.496257  , 0.49677997, 0.49641078,
                        0.52024409, 0.49255581, 0.47860725, 0.50888892, 0.51923275,
                        0.4773962 , 0.49249923, 0.51342903, 0.50350073, 0.50946515,
                        0.51912045, 0.50583902, 0.50272475, 0.51237963, 0.48600893,
                        0.49942566, 0.49056705, 0.49810661, 0.51591527, 0.47512221,
                        0.51005943, 0.5003548 , 0.50409177, 0.49838669, 0.49366925,
                        0.50348344, 0.52748975, 0.51740335, 0.49023212, 0.50969436,
                        0.47668736, 0.50262471, 0.50267211, 0.52623492, 0.47776998,
                        0.50850968, 0.53127329, 0.49010354])},
        index=pd.RangeIndex(start=2, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster, n_jobs", 
                         [(ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), -1),
                          (ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), 1),
                          (ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), 'auto'),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), -1),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), 1),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), 'auto')], 
                         ids=lambda fc: f'forecaster, n_jobs: {fc}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_fixed_train_size_with_mocked(forecaster, n_jobs):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit, fixed_train_size and 
    custom metric with mocked (mocked done in Skforecast v0.5.0).
    """

    steps = 3
    n_validation = 12

    def custom_metric(y_true, y_pred):  # pragma: no cover
        """
        """
        metric = mean_absolute_error(y_true, y_pred)
        
        return metric

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = ['l1'],
                                               metric                = custom_metric,
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = True,
                                               fixed_train_size      = True, 
                                               exog                  = None,
                                               verbose               = True,
                                               n_jobs                = n_jobs
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'],
                                    'custom_metric': [0.21651617115803679]})
    expected_predictions = pd.DataFrame({
                               'l1':np.array([0.4978839 , 0.46288427, 0.48433446, 
                                              0.50853803, 0.50006415, 0.50105623,
                                              0.46764379, 0.46845675, 0.46768947, 
                                              0.48298309, 0.47778385, 0.47776533])},
                               index=pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster, n_jobs", 
                         [(ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), -1),
                          (ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), 1),
                          (ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                        lags=2, transformer_series=None,
                                                        encoding='onehot'), 'auto'),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), -1),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), 1),
                          (ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                              fun_predictors=create_predictors, 
                                                              window_size=2, transformer_series=None,
                                                              encoding='onehot'), 'auto')], 
                         ids=lambda fc: f'forecaster, n_jobs: {fc}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_with_mocked(forecaster, n_jobs):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit with mocked 
    (mocked done in Skforecast v0.5.0).
    """

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = 'l1',
                                               metric                = 'mean_absolute_error',
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = True,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False,
                                               n_jobs                = n_jobs
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.2124129141233719]})
    expected_predictions = pd.DataFrame({
        'l1': np.array([0.4978838984103099, 0.46288426670127997, 0.48433446479429937, 
                        0.510664891759972, 0.49734477162307983, 0.5009680695304023,
                        0.48647770856843825, 0.4884651517014008, 0.48643766346259326, 
                        0.4973047492523979, 0.4899104838474172, 0.4891085370228432])},
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_list_metrics_with_mocked_metrics(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit and list of metrics with 
    mocked and list of metrics (mocked done in Skforecast v0.5.0).
    """

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = 'l1',
                                               metric                = ['mean_absolute_error', mean_absolute_error],
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = True,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False
                                           )
    
    expected_metric = pd.DataFrame(
        data    = [['l1', 0.2124129141233719, 0.2124129141233719]],
        columns = ['levels', 'mean_absolute_error', 'mean_absolute_error']
    )
    expected_predictions = pd.DataFrame({
        'l1': np.array([0.4978838984103099, 0.46288426670127997, 0.48433446479429937, 
                        0.510664891759972, 0.49734477162307983, 0.5009680695304023,
                        0.48647770856843825, 0.4884651517014008, 0.48643766346259326, 
                        0.4973047492523979, 0.4899104838474172, 0.4891085370228432])},
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_no_refit_levels_metrics_remainder_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with no refit, remainder, multiple 
    levels and metrics with mocked (mocked done in Skforecast v0.5.0).
    """

    steps = 5
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = None,
                                               metric                = ['mean_absolute_error', mean_absolute_error],
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = False,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False
                                           )
    
    expected_metric = pd.DataFrame(
        data    = [['l1', 0.21143995953996186, 0.21143995953996186],
                   ['l2', 0.2194174144550234, 0.2194174144550234]],
        columns = ['levels', 'mean_absolute_error', 'mean_absolute_error']
    )
    expected_predictions = pd.DataFrame({
        'l1': np.array([0.4978839 , 0.46288427, 0.48433446, 0.48677605, 0.48562473,
                        0.50259242, 0.49536197, 0.48478881, 0.48496106, 0.48555902,
                        0.49673897, 0.4576795 ]),
        'l2': np.array([0.50266337, 0.53045945, 0.50527774, 0.50315834, 0.50452649,
                        0.47372756, 0.51226827, 0.50650107, 0.50420766, 0.50448097,
                        0.52211914, 0.51092531])},
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_levels_metrics_remainder_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit, remainder, multiple levels 
    and metrics with mocked (mocked done in Skforecast v0.5.0).
    """

    steps = 5
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = None,
                                               metric                = ['mean_absolute_error', mean_absolute_error],
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = True,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False
                                           )
    
    expected_metric = pd.DataFrame(
        data    = [['l1', 0.20809130188099298, 0.20809130188099298],
                   ['l2', 0.22082212805693338, 0.22082212805693338]],
        columns = ['levels', 'mean_absolute_error', 'mean_absolute_error']
    )
    expected_predictions = pd.DataFrame({
        'l1': np.array([0.4978839 , 0.46288427, 0.48433446, 0.48677605, 0.48562473,
                        0.49724331, 0.4990606 , 0.4886555 , 0.48776085, 0.48830266,
                        0.52381728, 0.47432451]),
        'l2': np.array([0.50266337, 0.53045945, 0.50527774, 0.50315834, 0.50452649,
                        0.46847508, 0.5144631 , 0.51135241, 0.50842259, 0.50838289,
                        0.52555989, 0.51801796])},
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_not_refit_exog_interval_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom without refit with mocked using exog 
    and intervals (mocked done in Skforecast v0.5.0).
    """

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = ['l1'],
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               refit                   = False,
                                               fixed_train_size        = False,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 500,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
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
                               index = pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_fixed_train_size_exog_interval_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit and fixed_train_size with 
    mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               refit                   = True,
                                               fixed_train_size        = True,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 500,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
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
                               index = pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_no_refit_exog_interval_gap_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with no refit and gap with 
    mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """
    n_validation = 20
    steps = 5
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = True,
                                               refit                   = False,
                                               fixed_train_size        = False,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.12454132173965098]})
    expected_predictions = pd.DataFrame(
                               data = np.array([[0.49730036, 0.21114279, 0.77715089],
                                                [0.47420843, 0.264856  , 0.7065774 ],
                                                [0.43537767, 0.2249109 , 0.66681921],
                                                [0.47436444, 0.22133275, 0.70531502],
                                                [0.6336613 , 0.3459817 , 0.86401859],
                                                [0.6507444 , 0.36458683, 0.93059493],
                                                [0.49896782, 0.28961539, 0.73133679],
                                                [0.54030017, 0.3298334 , 0.7717417 ],
                                                [0.36847564, 0.11544395, 0.59942623],
                                                [0.43668441, 0.14900482, 0.6670417 ],
                                                [0.47126095, 0.18510337, 0.75111147],
                                                [0.62443149, 0.41507906, 0.85680046],
                                                [0.41464206, 0.20417529, 0.64608359],
                                                [0.49248163, 0.23944994, 0.72343221],
                                                [0.66520692, 0.37752732, 0.89556421],
                                                [0.50609184, 0.21993426, 0.78594236],
                                                [0.53642897, 0.32707654, 0.76879794]]),
                               columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
                               index = pd.RangeIndex(start=33, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_exog_interval_gap_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit, gap, allow_incomplete_fold 
    False with mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """
    n_validation = 20
    steps = 5
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = True,
                                               fixed_train_size        = False,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.136188772278576]})
    expected_predictions = pd.DataFrame(
                               data = np.array([[0.49730036, 0.21114279, 0.77715089],
                                                [0.47420843, 0.264856  , 0.7065774 ],
                                                [0.43537767, 0.2249109 , 0.66681921],
                                                [0.47436444, 0.22133275, 0.70531502],
                                                [0.6336613 , 0.3459817 , 0.86401859],
                                                [0.64530446, 0.40972155, 0.88380085],
                                                [0.48649557, 0.24491857, 0.72423562],
                                                [0.5353094 , 0.32581737, 0.77006447],
                                                [0.34635064, 0.10940796, 0.55257192],
                                                [0.42546209, 0.21943098, 0.66773177],
                                                [0.47082006, 0.22579712, 0.7436865 ],
                                                [0.61511518, 0.35650569, 0.90150613],
                                                [0.42930559, 0.20426756, 0.64016313],
                                                [0.4836202 , 0.22363998, 0.74885113],
                                                [0.6539521 , 0.42099386, 0.91924259]]),
                               columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
                               index = pd.RangeIndex(start=33, stop=48, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_exog_fixed_train_size_interval_gap_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit, fixed_train_size, gap, 
    with mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    n_validation = 20
    steps = 5
    gap = 5

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series_datetime,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series_datetime) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = True,
                                               fixed_train_size        = True,
                                               exog                    = series_datetime['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.13924838761692487]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.43537767, 0.2249109 , 0.66681921],
                         [0.47436444, 0.22133275, 0.70531502],
                         [0.6336613 , 0.3459817 , 0.86401859],
                         [0.6507473 , 0.43958576, 0.88320161],
                         [0.49896782, 0.28839553, 0.65221992],
                         [0.52865114, 0.30307925, 0.72246706],
                         [0.33211557, 0.11784593, 0.51606139],
                         [0.4205642 , 0.16345217, 0.60841454],
                         [0.44468828, 0.22213009, 0.63950501],
                         [0.61301873, 0.37677291, 0.81173965],
                         [0.41671844, 0.13568381, 0.67943385],
                         [0.47385345, 0.19025177, 0.76849597],
                         [0.62360146, 0.39888165, 0.85075261],
                         [0.49407875, 0.24011478, 0.78855477],
                         [0.51234652, 0.26828928, 0.81339595]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.date_range(start='2022-02-05', periods=15, freq='D')
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_no_refit_different_lengths_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with no refit and gap with 
    mocked using exog and intervals and series with different lengths 
    (mocked done in Skforecast v0.5.0).
    """
    n_validation = 20
    steps = 5
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series_with_nans,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series_with_nans) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = True,
                                               refit                   = False,
                                               fixed_train_size        = False,
                                               exog                    = series_with_nans['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.11243765852459384]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.49746847, 0.29550116, 0.74555927],
                        [0.46715961, 0.2507391 , 0.64732398],
                        [0.42170236, 0.16610213, 0.64208341],
                        [0.47242371, 0.29346814, 0.68859381],
                        [0.66450589, 0.42917887, 0.87949769],
                        [0.67311291, 0.47114561, 0.92120371],
                        [0.48788629, 0.27146579, 0.66805066],
                        [0.55141437, 0.29581413, 0.77179541],
                        [0.33369285, 0.15473728, 0.54986294],
                        [0.43287852, 0.19755151, 0.64787033],
                        [0.46643379, 0.26446648, 0.71452459],
                        [0.6535986 , 0.43717809, 0.83376297],
                        [0.38328273, 0.12768249, 0.60366377],
                        [0.49931045, 0.32035488, 0.71548055],
                        [0.70119564, 0.46586862, 0.91618744],
                        [0.49290276, 0.29093545, 0.74099356],
                        [0.54653561, 0.3301151 , 0.72669998]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.RangeIndex(start=33, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_different_lengths_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit, gap, allow_incomplete_fold 
    False with mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """
    n_validation = 20
    steps = 5
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series_with_nans,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series_with_nans) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = True,
                                               fixed_train_size        = False,
                                               exog                    = series_with_nans['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.12472922864372042]})
    expected_predictions = pd.DataFrame(
                               data = np.array([[0.49746847, 0.24620019, 0.74208308],
                                                [0.46715961, 0.27280913, 0.69897019],
                                                [0.42170236, 0.21384761, 0.6338432 ],
                                                [0.47242371, 0.23550111, 0.68264162],
                                                [0.66450589, 0.4128072 , 0.87549193],
                                                [0.66852682, 0.46973936, 0.89016477],
                                                [0.47796299, 0.2686537 , 0.69760042],
                                                [0.5477168 , 0.35928395, 0.75677995],
                                                [0.31418572, 0.10972171, 0.51376307],
                                                [0.42390085, 0.22962754, 0.65199941],
                                                [0.46927962, 0.25401323, 0.72621914],
                                                [0.63294789, 0.39385723, 0.89726095],
                                                [0.41355599, 0.20713645, 0.62092038],
                                                [0.48437311, 0.24174172, 0.73198965],
                                                [0.67731074, 0.46831444, 0.9265058 ]]),
                               columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
                               index = pd.RangeIndex(start=33, stop=48, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_fixed_train_size_different_lengths_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit, fixed_train_size, gap, 
    with mocked using exog and intervals (mocked done in Skforecast v0.5.0).
    """
    series_with_nans_datetime = series_with_nans.copy()
    series_with_nans_datetime.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    n_validation = 20
    steps = 5
    gap = 5

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series_with_nans_datetime,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series_with_nans_datetime) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = True,
                                               fixed_train_size        = True,
                                               exog                    = series_with_nans_datetime['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.1355099897175138]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.42170236, 0.21384761, 0.6338432 ],
                         [0.47242371, 0.23550111, 0.68264162],
                         [0.66450589, 0.4128072 , 0.87549193],
                         [0.67311586, 0.46534536, 0.89662809],
                         [0.487886  , 0.27487735, 0.65050566],
                         [0.52759202, 0.30254031, 0.7316064 ],
                         [0.33021382, 0.11679102, 0.52056867],
                         [0.4226274 , 0.16874895, 0.61548288],
                         [0.44647715, 0.22033296, 0.65301466],
                         [0.61435678, 0.37793909, 0.81285962],
                         [0.41671844, 0.13568381, 0.67943385],
                         [0.47385345, 0.19025177, 0.76849597],
                         [0.62360146, 0.39888165, 0.85075261],
                         [0.49407875, 0.24011478, 0.78855477],
                         [0.51234652, 0.26828928, 0.81339595]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.date_range(start='2022-02-05', periods=15, freq='D')
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_int_interval_yes_exog_yes_remainder_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit int, interval, gap, 
    with mocked using exog and intervals (mocked done in Skforecast v0.9.0).
    """
    refit = 2
    n_validation = 20
    steps = 2

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = None,
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               gap                     = 0,
                                               allow_incomplete_fold   = False,
                                               refit                   = refit,
                                               fixed_train_size        = True,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame(
        {'levels': ['l1', 'l2'], 
         'mean_absolute_error': [0.13341641696298234, 0.2532943228025243]}
    )
    expected_predictions = pd.DataFrame(
        data = np.array([[ 3.62121311e-01,  1.50820320e-01,  5.93493644e-01,
                             3.65807102e-01, -1.66846896e-01,  7.41872674e-01],
                         [ 4.75603912e-01,  2.64338068e-01,  7.55477618e-01,
                             4.77660870e-01, -2.09016888e-02,  8.50640767e-01],
                         [ 4.78856340e-01,  2.67555349e-01,  7.10228673e-01,
                             4.79828611e-01, -5.28253867e-02,  8.55894183e-01],
                         [ 4.97626059e-01,  2.86360215e-01,  7.77499764e-01,
                             4.98181879e-01, -3.80679351e-04,  8.71161776e-01],
                         [ 4.57502853e-01,  2.34854548e-01,  6.90295515e-01,
                             4.93394245e-01, -8.59476649e-02,  8.26877562e-01],
                         [ 4.19762089e-01,  1.89123646e-01,  5.98698797e-01,
                             4.37345761e-01, -2.47590943e-02,  8.09946951e-01],
                         [ 4.64704145e-01,  2.42055840e-01,  6.97496807e-01,
                             4.95977483e-01, -8.33644270e-02,  8.29460800e-01],
                         [ 6.27792451e-01,  3.97154009e-01,  8.06729160e-01,
                             6.80737238e-01,  2.18632382e-01,  1.05333843e+00],
                         [ 5.87334375e-01,  3.26921031e-01,  8.69376004e-01,
                             6.65280778e-01,  1.23472110e-01,  1.05838353e+00],
                         [ 4.53164888e-01,  1.98770357e-01,  7.31112253e-01,
                             5.17357914e-01, -3.46981422e-02,  9.17114685e-01],
                         [ 4.91347698e-01,  2.30934354e-01,  7.73389327e-01,
                             5.56066550e-01,  1.42578818e-02,  9.49169307e-01],
                         [ 3.58133021e-01,  1.03738489e-01,  6.36080385e-01,
                             3.94861192e-01, -1.57194865e-01,  7.94617962e-01],
                         [ 4.23074894e-01,  1.46402575e-01,  6.85243368e-01,
                             4.56570627e-01, -8.44076809e-02,  9.13597456e-01],
                         [ 4.77639455e-01,  2.00288697e-01,  7.45269910e-01,
                             4.52788025e-01, -6.08964382e-02,  9.16193940e-01],
                         [ 5.90866263e-01,  3.14193944e-01,  8.53034737e-01,
                             6.06068550e-01,  6.50902424e-02,  1.06309538e+00],
                         [ 4.29943139e-01,  1.52592381e-01,  6.97573594e-01,
                             4.22379461e-01, -9.13050017e-02,  8.85785376e-01],
                         [ 4.71297777e-01,  1.92467142e-01,  7.32960612e-01,
                             5.16307783e-01, -5.04225194e-02,  9.64821047e-01],
                         [ 6.45316619e-01,  4.47497018e-01,  9.25177996e-01,
                             6.51701762e-01,  1.69673821e-01,  1.01740457e+00],
                         [ 5.42727946e-01,  2.63897311e-01,  8.04390781e-01,
                             5.10069712e-01, -5.66605897e-02,  9.58582976e-01],
                         [ 5.30915933e-01,  3.33096333e-01,  8.10777311e-01,
                             5.43494604e-01,  6.14666630e-02,  9.09197415e-01]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound', 'l2', 'l2_lower_bound', 'l2_upper_bound'],
        index = pd.RangeIndex(start=30, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_refit_int_interval_yes_exog_not_allow_remainder_gap_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with refit int, interval, gap, 
    with mocked using exog and intervals (mocked done in Skforecast v0.9.0).
    """
    series_with_index = series.copy()
    series_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = series['l1'].rename('exog_1').copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    refit = 3
    n_validation = 20
    steps = 4
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series_with_index,
                                               steps                   = steps,
                                               levels                  = ['l2'],
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series_with_index) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = refit,
                                               fixed_train_size        = False,
                                               exog                    = exog_with_index,
                                               interval                = [5, 95],
                                               n_boot                  = 100,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame(
        {'levels': ['l2'], 
         'mean_absolute_error': [0.2791832310123065]}
    )
    expected_predictions = pd.DataFrame(
        data = np.array([[ 0.49984026,  0.00509982,  0.82464616],
                         [ 0.4767447 , -0.0798953 ,  0.80487146],
                         [ 0.43791384, -0.12045877,  0.81344287],
                         [ 0.47690063, -0.01831127,  0.85480226],
                         [ 0.63619086,  0.14145042,  0.96099675],
                         [ 0.65328344,  0.09664345,  0.98141021],
                         [ 0.50150406, -0.05686855,  0.8770331 ],
                         [ 0.54283634,  0.04762444,  0.92073797],
                         [ 0.37097633, -0.1237641 ,  0.69578223],
                         [ 0.43922059, -0.11741941,  0.76734736],
                         [ 0.47379722, -0.08457539,  0.84932626],
                         [ 0.62696782,  0.13175592,  1.00486945],
                         [ 0.45263688, -0.09848725,  0.84471127],
                         [ 0.5005012 , -0.02743982,  0.95029956],
                         [ 0.66342359,  0.10579924,  1.20771224],
                         [ 0.53547991,  0.02357046,  0.89862351]]),
        columns = ['l2', 'l2_lower_bound', 'l2_upper_bound'],
        index = pd.date_range(start='2022-02-03', periods=16, freq='D')
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
    [ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    ), 
    ForecasterAutoregMultiSeriesCustom(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        fun_predictors=create_predictors_14, 
        window_size=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )], 
    ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_series_and_exog_dict_with_mocked(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom when series and exog are
    dictionaries (mocked done in Skforecast v0.12.0).
    """
    
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series_dict,
        exog                  = exog_dict,
        steps                 = 24,
        metric                = 'mean_absolute_error',
        add_aggregated_metric = False,
        initial_train_size    = len(series_dict_train['id_1000']),
        fixed_train_size      = True,
        gap                   = 0,
        allow_incomplete_fold = True,
        refit                 = False,
        n_jobs                = 'auto',
        verbose               = False,
        show_progress         = False,
        suppress_warnings     = True
    )

    expected_metrics = pd.DataFrame(
        data={
        'levels': ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004'],
        'mean_absolute_error': [286.6227398656757, 1364.7345740769094,
                                np.nan, 237.4894217124842, 1267.85941538558]
        },
        columns=['levels', 'mean_absolute_error']
    )
    expected_predictions = pd.DataFrame(
        data=np.array([
        [1438.14154717, 2090.79352613, 2166.9832933 , 7285.52781428],
        [1438.14154717, 2089.11038884, 2074.55994929, 7488.18398744],
        [1438.14154717, 2089.11038884, 2035.99448247, 7488.18398744],
        [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
        [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
        [1403.93625654, 2076.10228838, 2035.99448247, 7250.69119259],
        [1403.93625654, 2076.10228838,        np.nan, 7085.32315355],
        [1403.93625654, 2000.42985714,        np.nan, 7285.52781428],
        [1403.93625654, 2013.4379576 ,        np.nan, 7285.52781428],
        [1403.93625654, 2013.4379576 ,        np.nan, 7285.52781428]]),
        index=pd.date_range('2016-08-01', periods=10, freq='D'),
        columns=['id_1000', 'id_1001', 'id_1003', 'id_1004']
    )

    pd.testing.assert_frame_equal(metrics, expected_metrics)
    pd.testing.assert_frame_equal(predictions.head(10), expected_predictions)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), 
                                                       lags=2, transformer_series=None,
                                                       encoding='onehot'), 
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(random_state=123), 
                                                             fun_predictors=create_predictors, 
                                                             window_size=2, transformer_series=None,
                                                             encoding='onehot')], 
                         ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_aggregate_metrics_true(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom with no refit, remainder, multiple 
    levels and add_aggregated_metric.
    """
    steps = 5
    n_validation = 12

    metrics, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = None,
                                               metric                = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                                               add_aggregated_metric = True,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = False,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False
                                           )

    expected_metrics = pd.DataFrame(
        {
            "levels": ["l1", "l2", "average", "weighted_average", "pooling"],
            "mean_absolute_error": [
                0.21143995953996186,
                0.2194174144550234,
                0.21542868699749262,
                0.21542868699749262,
                0.21542868699749262,
            ],
            "mean_absolute_scaled_error": [
                0.9265126347211138,
                0.7326447294149787,
                0.8295786820680462,
                0.8295786820680462,
                0.8164857848143883,
            ],
        }
    )
    expected_predictions = pd.DataFrame(
                            {
                                "l1": [
                                    0.4978839,
                                    0.46288427,
                                    0.48433446,
                                    0.48677605,
                                    0.48562473,
                                    0.50259242,
                                    0.49536197,
                                    0.48478881,
                                    0.48496106,
                                    0.48555902,
                                    0.49673897,
                                    0.4576795,
                                ],
                                "l2": [
                                    0.50266337,
                                    0.53045945,
                                    0.50527774,
                                    0.50315834,
                                    0.50452649,
                                    0.47372756,
                                    0.51226827,
                                    0.50650107,
                                    0.50420766,
                                    0.50448097,
                                    0.52211914,
                                    0.51092531,
                                ],
                            },
                            index=pd.RangeIndex(start=38, stop=50, step=1),
                            )

    pd.testing.assert_frame_equal(expected_metrics, metrics)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("forecaster", 
    [ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    ), 
    ForecasterAutoregMultiSeriesCustom(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        fun_predictors=create_predictors_14, 
        window_size=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )], 
    ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_series_and_exog_dict_with_mocked_multiple_aggregated_metrics(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom when series and exog are
    dictionaries and multiple aggregated metrics are calculated.
    (mocked done in Skforecast v0.12.0).
    """
    
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series_dict,
        exog                  = exog_dict,
        steps                 = 24,
        metric                = ['mean_absolute_error', 'mean_squared_error'],
        add_aggregated_metric = True,
        initial_train_size    = len(series_dict_train['id_1000']),
        fixed_train_size      = True,
        gap                   = 0,
        allow_incomplete_fold = True,
        refit                 = False,
        n_jobs                = 'auto',
        verbose               = False,
        show_progress         = False,
        suppress_warnings     = True
    )

    expected_metrics = pd.DataFrame({
                        'levels': {0: 'id_1000',
                        1: 'id_1001',
                        2: 'id_1002',
                        3: 'id_1003',
                        4: 'id_1004',
                        5: 'average',
                        6: 'weighted_average',
                        7: 'pooling'},
                        'mean_absolute_error': {0: 286.6227398656757,
                        1: 1364.7345740769094,
                        2: np.nan,
                        3: 237.4894217124842,
                        4: 1267.85941538558,
                        5: 789.1765377601623,
                        6: 745.7085483145497,
                        7: 745.7085483145497},
                        'mean_squared_error': {0: 105816.86051259708,
                        1: 2175934.9583102698,
                        2: np.nan,
                        3: 95856.72602398091,
                        4: 2269796.338792736,
                        5: 1161851.2209098958,
                        6: 1024317.153152019,
                        7: 1024317.1531520189}
                })
    
    expected_predictions = pd.DataFrame(
    data=np.array([[1438.14154717, 2090.79352613, 2166.9832933, 7285.52781428],
                   [1438.14154717, 2089.11038884, 2074.55994929, 7488.18398744],
                   [1438.14154717, 2089.11038884, 2035.99448247, 7488.18398744],
                   [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
                   [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
                   [1403.93625654, 2076.10228838, 2035.99448247, 7250.69119259],
                   [1403.93625654, 2076.10228838, np.nan, 7085.32315355],
                   [1403.93625654, 2000.42985714, np.nan, 7285.52781428],
                   [1403.93625654, 2013.4379576, np.nan, 7285.52781428],
                   [1403.93625654, 2013.4379576, np.nan, 7285.52781428]]),
    columns=['id_1000', 'id_1001', 'id_1003', 'id_1004'],
    index=pd.date_range('2016-08-01', periods=10, freq='D')
)

    pd.testing.assert_frame_equal(metrics, expected_metrics)
    pd.testing.assert_frame_equal(predictions.head(10), expected_predictions)


@pytest.mark.parametrize("forecaster", 
    [ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    ), 
    ForecasterAutoregMultiSeriesCustom(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        fun_predictors=create_predictors_14, 
        window_size=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )], 
    ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_series_and_exog_dict_with_mocked_skip_folds_2(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom when series and exog are
    dictionaries (mocked done in Skforecast v0.12.0).
    """
    
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series_dict,
        exog                  = exog_dict,
        steps                 = 5,
        metric                = 'mean_absolute_error',
        initial_train_size    = len(series_dict_train['id_1000']),
        fixed_train_size      = True,
        gap                   = 0,
        skip_folds            = 2,
        allow_incomplete_fold = True,
        refit                 = False,
        n_jobs                = 'auto',
        verbose               = True,
        show_progress         = True,
        suppress_warnings     = True
    )

    expected_metrics = pd.DataFrame(
        data={
        'levels': ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004',
                   'average', 'weighted_average', 'pooling'],
        'mean_absolute_error': [258.53959034, 1396.68222457, np.nan, 253.72012285,
                                1367.69574404, 819.1594204522052, 753.0251104677614,
                                753.0251104677616]
        },
        columns=['levels', 'mean_absolute_error']
    )
    expected_predictions = pd.DataFrame(
        data=np.array([
            [1438.14154717, 2090.79352613, 2166.9832933 , 7285.52781428],
            [1438.14154717, 2089.11038884, 2074.55994929, 7488.18398744],
            [1438.14154717, 2089.11038884, 2035.99448247, 7488.18398744],
            [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
            [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
            [1432.18965392, 2013.4379576 ,        np.nan, 7488.18398744],
            [1403.93625654, 2013.4379576 ,        np.nan, 7488.18398744],
            [1403.93625654, 2136.7144822 ,        np.nan, 7250.69119259],
            [1403.93625654, 2136.7144822 ,        np.nan, 7085.32315355],
            [1438.14154717, 2013.4379576 ,        np.nan, 7285.52781428]]),
        index=pd.DatetimeIndex([
                '2016-08-01', '2016-08-02', '2016-08-03', '2016-08-04',
                '2016-08-05', '2016-08-11', '2016-08-12', '2016-08-13',
                '2016-08-14', '2016-08-15'],
                dtype='datetime64[ns]', freq=None),
        columns=['id_1000', 'id_1001', 'id_1003', 'id_1004']
    )

    pd.testing.assert_frame_equal(metrics, expected_metrics)
    pd.testing.assert_frame_equal(predictions.head(10), expected_predictions)


@pytest.mark.parametrize("forecaster", 
    [ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=30, random_state=123, verbose=-1, max_depth=4
        ),
        lags=[1, 7, 14],
        encoding=None,
        differentiation=1,
        dropna_from_series=False,
        transformer_series=None,
        transformer_exog=StandardScaler(),
    ), 
    ForecasterAutoregMultiSeriesCustom(
        regressor=LGBMRegressor(
            n_estimators=30, random_state=123, verbose=-1, max_depth=4
        ),
        fun_predictors=create_predictors_1_7_14, 
        window_size=14,
        encoding=None,
        differentiation=1,
        dropna_from_series=False,
        transformer_series=None,
        transformer_exog=StandardScaler(),
    )], 
    ids=lambda forecaster: f'forecaster: {type(forecaster).__name__}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiSeries_series_and_exog_dict_encoding_None_differentiation(forecaster):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom when series and exog are
    dictionaries, encoding=None, and differentiation=1 (mocked done in Skforecast v0.13.0).
    """
    
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series_dict,
        exog                  = exog_dict,
        steps                 = 24,
        metric                = ['mean_absolute_error', 'mean_absolute_scaled_error'],
        initial_train_size    = len(series_dict_train['id_1000']),
        fixed_train_size      = True,
        gap                   = 0,
        allow_incomplete_fold = True,
        refit                 = False,
        n_jobs                = 'auto',
        verbose               = True,
        show_progress         = True,
        suppress_warnings     = True
    )

    expected_metrics = pd.DataFrame(
        data={
            'levels': [
                'id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004',
                'average', 'weighted_average', 'pooling'
            ],
            'mean_absolute_error': [
                234.51032919, 1145.83513569, np.nan, 1342.06986733,
                1025.76779699, 937.0457823, 818.85514869, 818.85514869
            ],
            'mean_absolute_scaled_error': [
                1.06505383, 3.25617103, np.nan, 5.62127465,
                0.85811137, 2.70015272, 2.6047229, 1.9453509
            ]
        }
    )
    expected_predictions = pd.DataFrame(
        data=np.array([
            [1351.44874045, 3267.13419659, 3843.06135374, 7220.15909652],
            [1411.73786344, 3537.21728977, 3823.777024  , 7541.54119992],
            [1367.56233886, 3595.37617537, 3861.96220571, 7599.93110962],
            [1349.98221742, 3730.36025071, 3900.14738742, 7997.86353537],
            [1330.33321825, 3752.7395927 , 3850.39111741, 7923.44398373],
            [1045.2013386 , 2652.15007539, 4227.38385608, 6651.39787084],
            [ 801.6084919 , 2059.57604816,        np.nan, 5625.61804572],
            [1473.95692547, 3159.99921219,        np.nan, 6740.53028097],
            [1596.51499989, 3449.40467366,        np.nan, 7172.21235605],
            [1572.31317919, 3507.56355926,        np.nan, 7506.35124681]]),
        index=pd.date_range('2016-08-01', periods=10, freq='D'),
        columns=['id_1000', 'id_1001', 'id_1003', 'id_1004']
    )

    pd.testing.assert_frame_equal(metrics, expected_metrics)
    pd.testing.assert_frame_equal(predictions.head(10), expected_predictions)


# ForecasterAutoregMultiVariate
# ======================================================================================================================
@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_not_refit_with_mocked(n_jobs):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate without refit
    with mocked (mocked done in Skforecast v0.6.0).
    """

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = 'l1',
                                               metric                = 'mean_absolute_error',
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = False,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               n_jobs                = n_jobs,
                                               verbose               = True
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'],
                                    'mean_absolute_error': [0.2056686186667702]})
    expected_predictions = pd.DataFrame({
                               'l1':np.array([0.55397908, 0.48026456, 0.52368724,
                                              0.48490132, 0.46928502, 0.52511441, 
                                              0.46529858, 0.45430583, 0.51706306, 
                                              0.50561424, 0.47109786, 0.45568319])},
                               index=pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_not_refit_not_initial_train_size_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    without refit and initial_train_size is None with mocked, forecaster must be fitted,
    (mocked done in Skforecast v0.6.0).
    """

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series)

    steps = 1
    initial_train_size = None

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = ['l1'],
                                               metric                = mean_absolute_error,
                                               add_aggregated_metric = False,
                                               initial_train_size    = initial_train_size,
                                               refit                 = False,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'],
                                    'mean_absolute_error': [0.17959810844511925]})
    expected_predictions = pd.DataFrame({
        'l1': np.array([0.42102839, 0.47138953, 0.52252844, 0.54594276, 0.51253167,
                        0.57557448, 0.45455673, 0.42240387, 0.48555804, 0.46974027,
                        0.52264755, 0.45674877, 0.43440543, 0.47187135, 0.59653789,
                        0.41804686, 0.53408781, 0.58853203, 0.49880785, 0.57834799,
                        0.47345798, 0.46890693, 0.45765737, 0.59034503, 0.46198262,
                        0.49384858, 0.54212837, 0.56867955, 0.5095804 , 0.47751184,
                        0.50402253, 0.48993588, 0.52583999, 0.4306855 , 0.42782129,
                        0.52841356, 0.62570147, 0.55585762, 0.48719966, 0.48508799,
                        0.37122115, 0.53115279, 0.47119561, 0.52734455, 0.41557646,
                        0.57546277, 0.57700474, 0.50898628])},
        index=pd.RangeIndex(start=2, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_fixed_train_size_with_mocked(n_jobs):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate with refit,
    fixed_train_size and custom metric with mocked (mocked done in Skforecast v0.6.0).
    """

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l2',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12

    def custom_metric(y_true, y_pred):  # pragma: no cover
        """
        """
        metric = mean_absolute_error(y_true, y_pred)
        
        return metric

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = None,
                                               metric                = custom_metric,
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = True,
                                               fixed_train_size      = True, 
                                               exog                  = None,
                                               n_jobs                = n_jobs,
                                               verbose               = True
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l2'],
                                    'custom_metric': [0.2326510995879597]})
    expected_predictions = pd.DataFrame({
                               'l2': np.array([0.58478895, 0.56729494, 0.54469663,
                                               0.50326485, 0.53339207, 0.50892268, 
                                               0.46841857, 0.48498214, 0.52778775,
                                               0.51476103, 0.48480385, 0.53470992])},
                               index=pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_with_mocked(n_jobs):
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with refit with mocked (mocked done in Skforecast v0.6.0).
    """

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = 'l1',
                                               metric                = 'mean_absolute_error',
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = True,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               n_jobs                = n_jobs,
                                               verbose               = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.20733067815663564]})
    expected_predictions = pd.DataFrame({
                               'l1': np.array([0.55397908, 0.48026456, 0.52368724, 
                                               0.49816586, 0.48470807, 0.54162611,
                                               0.45270749, 0.47194035, 0.53386908,
                                               0.55296942, 0.53498642, 0.44772825])},
                               index=pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_list_metrics_with_mocked_metrics():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with refit and list of metrics with mocked and list of metrics 
    (mocked done in Skforecast v0.6.0).
    """

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster            = forecaster,
                                               series                = series,
                                               steps                 = steps,
                                               levels                = 'l1',
                                               metric                = ['mean_absolute_error', mean_absolute_error],
                                               add_aggregated_metric = False,
                                               initial_train_size    = len(series) - n_validation,
                                               refit                 = True,
                                               fixed_train_size      = False,
                                               exog                  = None,
                                               verbose               = False
                                           )
    
    expected_metric = pd.DataFrame(
        data    = [['l1', 0.20733067815663564, 0.20733067815663564]],
        columns = ['levels', 'mean_absolute_error', 'mean_absolute_error']
    )
    expected_predictions = pd.DataFrame({
                               'l1': np.array([0.55397908, 0.48026456, 0.52368724, 
                                               0.49816586, 0.48470807, 0.54162611, 
                                               0.45270749, 0.47194035, 0.53386908, 
                                               0.55296942, 0.53498642, 0.44772825])},
                               index=pd.RangeIndex(start=38, stop=50, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_not_refit_exog_interval_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    without refit with mocked using exog and intervals 
    (mocked done in Skforecast v0.7.0).
    """

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = {'l1': 3, 'l2': None},
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = ['l1'],
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               refit                   = False,
                                               fixed_train_size        = False,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 500,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.0849883385916299]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.76991301, 0.6530355 , 0.87499155],
                         [0.4856994 , 0.33808808, 0.63146337],
                         [0.59391504, 0.44464521, 0.7193584 ],
                         [0.2924861 , 0.17560859, 0.39756464],
                         [0.37961054, 0.23199922, 0.5253745 ],
                         [0.42813627, 0.27886644, 0.55357963],
                         [0.69375682, 0.57687931, 0.79883536],
                         [0.3375219 , 0.18991058, 0.48328587],
                         [0.49705665, 0.34778682, 0.62250001],
                         [0.80370716, 0.68682964, 0.9087857 ],
                         [0.49265658, 0.34504526, 0.63842054],
                         [0.5482819 , 0.39901207, 0.67372527]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.RangeIndex(start=38, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_fixed_train_size_exog_interval_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with refit and fixed_train_size with mocked using exog and intervals 
    (mocked done in Skforecast v0.7.0).
    """

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = {'l1': None, 'l2': [1, 3]},
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               refit                   = True,
                                               fixed_train_size        = True,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 500,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.08191170312799796]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.76727126, 0.6427192 , 0.87291353],
                         [0.48402505, 0.32602185, 0.62277114],
                         [0.55855356, 0.41628182, 0.70625225],
                         [0.23916462, 0.10047205, 0.36932685],
                         [0.38545642, 0.26845512, 0.54060936],
                         [0.45551238, 0.31284488, 0.61746892],
                         [0.72213794, 0.58748583, 0.86084704],
                         [0.33389321, 0.2059126 , 0.4880578 ],
                         [0.4767024 , 0.34677964, 0.62049076],
                         [0.81570257, 0.68317749, 0.95105791],
                         [0.5040059 , 0.36942232, 0.65356959],
                         [0.54173456, 0.41187539, 0.69844758]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.RangeIndex(start=38, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_no_refit_exog_interval_gap_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with no refit and gap with mocked using exog and intervals 
    (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = {'l1': 2, 'l2': [1, 3]},
                     steps              = 8,
                     transformer_series = None
                 )
    
    n_validation = 20
    steps = 5
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = True,
                                               refit                   = False,
                                               fixed_train_size        = False,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.11791887332493929]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.55880533, 0.38128442, 0.73357531],
                         [0.46285725, 0.29691631, 0.61599977],
                         [0.35358667, 0.15258482, 0.48264849],
                         [0.44404948, 0.27047171, 0.61037957],
                         [0.64659616, 0.46670518, 0.81968729],
                         [0.70306475, 0.52554383, 0.87783473],
                         [0.48677757, 0.32083663, 0.63992009],
                         [0.49848981, 0.29748796, 0.62755162],
                         [0.31544893, 0.14187117, 0.48177903],
                         [0.4450306 , 0.26513962, 0.61812173],
                         [0.50164877, 0.32412785, 0.67641875],
                         [0.62883248, 0.46289154, 0.781975  ],
                         [0.33387601, 0.13287416, 0.46293782],
                         [0.45961408, 0.28603631, 0.62594417],
                         [0.63726975, 0.45737877, 0.81036088],
                         [0.54013414, 0.36261322, 0.71490412],
                         [0.52550978, 0.35956884, 0.6786523 ]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.RangeIndex(start=33, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_exog_interval_gap_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with refit, gap, allow_incomplete_fold False with mocked using exog and 
    intervals (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 8,
                     transformer_series = None
                 )
    
    n_validation = 20
    steps = 5
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = True,
                                               fixed_train_size        = False,
                                               exog                    = series['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.09906557188440204]})
    expected_predictions = pd.DataFrame(
                               data = np.array([[0.5137416 , 0.28565005, 0.64960534],
                                                [0.51578881, 0.36235612, 0.71123006],
                                                [0.38569703, 0.20965762, 0.51379331],
                                                [0.41784889, 0.23241384, 0.58140101],
                                                [0.69352545, 0.51075098, 0.88011232],
                                                [0.73709708, 0.60256116, 0.87889663],
                                                [0.49379408, 0.34235281, 0.70108827],
                                                [0.54129634, 0.38342955, 0.66423569],
                                                [0.2889167 , 0.13545924, 0.45541277],
                                                [0.41762991, 0.26652975, 0.58517659],
                                                [0.4281944 , 0.26680135, 0.56842703],
                                                [0.6800295 , 0.53425961, 0.86349246],
                                                [0.3198172 , 0.18545173, 0.44111693],
                                                [0.45269985, 0.29948117, 0.57043644],
                                                [0.76214215, 0.63320969, 0.90810797]]),
                               columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
                               index = pd.RangeIndex(start=33, stop=48, step=1)
                           )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_fixed_train_size_exog_interval_gap_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with refit, fixed_train_size, gap, with mocked using exog and intervals 
    (mocked done in Skforecast v0.5.0).
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 10,
                     transformer_series = None
                 )

    n_validation = 20
    steps = 5
    gap = 5

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series_datetime,
                                               steps                   = steps,
                                               levels                  = 'l1',
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series_datetime) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = True,
                                               fixed_train_size        = True,
                                               exog                    = series_datetime['l1'].rename('exog_1'),
                                               interval                = [5, 95],
                                               n_boot                  = 150,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame({'levels': ['l1'], 
                                    'mean_absolute_error': [0.12073089072357064]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.40010139, 0.21491844, 0.52790634],
                         [0.42717215, 0.23067944, 0.57988887],
                         [0.68150611, 0.48700348, 0.81952219],
                         [0.66167269, 0.46929506, 0.87762411],
                         [0.48946946, 0.36803878, 0.69919005],
                         [0.52955424, 0.36121274, 0.7342489 ],
                         [0.30621348, 0.11465701, 0.46702924],
                         [0.44140106, 0.27497225, 0.62523765],
                         [0.41294246, 0.26739719, 0.58273245],
                         [0.66798745, 0.51402456, 0.82567436],
                         [0.35206832, 0.19485472, 0.6001923 ],
                         [0.41828203, 0.28154037, 0.58081107],
                         [0.6605119 , 0.44992937, 0.79751069],
                         [0.49225437, 0.31367508, 0.67624609],
                         [0.52528842, 0.33700505, 0.74482225]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.date_range(start='2022-02-05', periods=15, freq='D')
    )
    backtest_predictions = backtest_predictions.asfreq('D')
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_int_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with refit int, interval, gap, with mocked using exog and intervals 
    (mocked done in Skforecast v0.9.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 2,
                     transformer_series = None
                 )
    
    refit = 2
    n_jobs = 2
    n_validation = 20
    steps = 2

    warn_msg = re.escape(
        ("If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
         "is set to 1 to avoid unexpected results during parallelization.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                                   forecaster              = forecaster,
                                                   series                  = series,
                                                   steps                   = steps,
                                                   levels                  = 'l1',
                                                   metric                  = 'mean_absolute_error',
                                                   add_aggregated_metric   = False,
                                                   initial_train_size      = len(series) - n_validation,
                                                   gap                     = 0,
                                                   allow_incomplete_fold   = False,
                                                   refit                   = refit,
                                                   fixed_train_size        = True,
                                                   exog                    = series['l1'].rename('exog_1'),
                                                   interval                = [5, 95],
                                                   n_boot                  = 100,
                                                   random_state            = 123,
                                                   use_in_sample_residuals = True,
                                                   verbose                 = False,
                                                   n_jobs                  = n_jobs
                                               )
    
    expected_metric = pd.DataFrame(
        {'levels': ['l1'], 
         'mean_absolute_error': [0.09237017311770493]}
    )
    expected_predictions = pd.DataFrame(
        data = np.array([[0.2983264 , 0.14814112, 0.52611729],
                         [0.45367569, 0.3061465 , 0.56042116],
                         [0.4802857 , 0.33010042, 0.70807659],
                         [0.48384823, 0.33631904, 0.59059371],
                         [0.44991269, 0.27452568, 0.6567239 ],
                         [0.38062305, 0.20946396, 0.50828437],
                         [0.42817363, 0.25278662, 0.63498485],
                         [0.68209515, 0.51093605, 0.80975646],
                         [0.72786565, 0.58843297, 0.8652939 ],
                         [0.45147711, 0.32772617, 0.57845425],
                         [0.52904599, 0.38961331, 0.66647424],
                         [0.28657426, 0.16282332, 0.4135514 ],
                         [0.36296117, 0.24577117, 0.52321437],
                         [0.46902166, 0.37455162, 0.62846802],
                         [0.68519422, 0.56800422, 0.84544742],
                         [0.35197602, 0.25750597, 0.51142238],
                         [0.47338638, 0.29477692, 0.61652158],
                         [0.74059646, 0.6075235 , 0.88275308],
                         [0.55112717, 0.37251772, 0.69426237],
                         [0.55778417, 0.4247112 , 0.69994079]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.RangeIndex(start=30, stop=50, step=1)
    )
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_multiseries_ForecasterAutoregMultiVariate_refit_int_interval_yes_exog_not_allow_remainder_gap_with_mocked():
    """
    Test output of backtesting_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with refit int, interval, gap, with mocked using exog and intervals 
    (mocked done in Skforecast v0.9.0).
    """
    series_with_index = series.copy()
    series_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = series['l1'].rename('exog_1').copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 7,
                     transformer_series = None
                 )

    refit = 3
    n_validation = 30
    steps = 4
    gap = 3

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                               forecaster              = forecaster,
                                               series                  = series_with_index,
                                               steps                   = steps,
                                               levels                  = ['l1'],
                                               metric                  = 'mean_absolute_error',
                                               add_aggregated_metric   = False,
                                               initial_train_size      = len(series_with_index) - n_validation,
                                               gap                     = gap,
                                               allow_incomplete_fold   = False,
                                               refit                   = refit,
                                               fixed_train_size        = False,
                                               exog                    = exog_with_index,
                                               interval                = [5, 95],
                                               n_boot                  = 100,
                                               random_state            = 123,
                                               use_in_sample_residuals = True,
                                               verbose                 = False
                                           )
    
    expected_metric = pd.DataFrame(
        {'levels': ['l1'], 
         'mean_absolute_error': [0.10066454067329249]}
    )
    expected_predictions = pd.DataFrame(
        data = np.array([[0.53203113, 0.27318916, 0.69931226],
                         [0.62129188, 0.46220068, 0.85870486],
                         [0.40145922, 0.17189084, 0.59797159],
                         [0.38821275, 0.17098498, 0.61456487],
                         [0.37636995, 0.11752798, 0.54365108],
                         [0.36599243, 0.20690123, 0.60340541],
                         [0.47654862, 0.24698023, 0.67306099],
                         [0.33725755, 0.12002978, 0.56360967],
                         [0.46122163, 0.20237966, 0.62850276],
                         [0.47541934, 0.31632814, 0.71283232],
                         [0.50601999, 0.27645161, 0.70253236],
                         [0.39293452, 0.17570675, 0.61928664],
                         [0.43793405, 0.27695885, 0.5990429 ],
                         [0.52600886, 0.38167775, 0.72224797],
                         [0.70882239, 0.55319058, 0.83377003],
                         [0.70063755, 0.51962777, 0.85415566],
                         [0.48631217, 0.32533698, 0.64742102],
                         [0.52248417, 0.37815306, 0.71872328],
                         [0.30010035, 0.14446853, 0.42504798],
                         [0.43342987, 0.25242008, 0.58694797],
                         [0.42479569, 0.2638205 , 0.58590454],
                         [0.68566159, 0.54133048, 0.8819007 ],
                         [0.34632377, 0.19069195, 0.4712714 ],
                         [0.44695116, 0.26594138, 0.60046927]]),
        columns = ['l1', 'l1_lower_bound', 'l1_upper_bound'],
        index = pd.date_range(start='2022-01-24', periods=24, freq='D')
    )
    backtest_predictions = backtest_predictions.asfreq('D')
                                   
    pd.testing.assert_frame_equal(expected_metric, metrics_levels)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)
