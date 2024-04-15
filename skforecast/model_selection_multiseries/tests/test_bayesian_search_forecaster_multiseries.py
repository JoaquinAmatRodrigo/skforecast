# Unit test bayesian_search_forecaster_multiseries
# ==============================================================================
import re
import pytest
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import bayesian_search_forecaster_multiseries
from skforecast.model_selection_multiseries import bayesian_search_forecaster_multivariate

# Fixtures
from .fixtures_model_selection_multiseries import series

series_dict = joblib.load('./fixture_sample_multi_series.joblib')
exog_dict = joblib.load('./fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}

def create_predictors(y): # pragma: no cover
    """
    Create first 4 lags of a time series.
    """

    lags = y[-1:-5:-1]

    return lags

def create_predictors_14(y): # pragma: no cover
    """
    Create first 14 lags of a time series.
    """
    lags = y[-1:-15:-1]

    return lags


def test_ValueError_bayesian_search_forecaster_multiseries_when_return_best_and_len_series_exog_different():
    """
    Test ValueError is raised in bayesian_search_forecaster_multiseries when 
    return_best and length of `series` and `exog` do not match.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    exog = series[:30].copy()

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `series`. "
         f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series,
            exog               = exog,
            search_space       = search_space,
            steps              = 3,
            metric             = 'mean_absolute_error',
            refit              = True,
            initial_train_size = len(series[:-12]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
            engine             = 'optuna'
        )


def test_bayesian_search_forecaster_multiseries_ValueError_when_engine_not_optuna():
    """
    Test ValueError in bayesian_search_forecaster_multiseries is raised when engine 
    is not 'optuna'.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    engine = 'not_optuna'
    
    err_msg = re.escape(f"`engine` only allows 'optuna', got {engine}.")
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series,
            search_space       = search_space,
            steps              = 3,
            metric             = 'mean_absolute_error',
            refit              = True,
            initial_train_size = len(series[:-12]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
            engine             = engine
        )


def test_results_output_bayesian_search_forecaster_multiseries_optuna_engine_ForecasterAutoregMultiSeries():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterAutoregMultiSeries with mocked using 
    optuna engine (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    steps = 3
    n_validation = 12

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags' : trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  steps              = steps,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  refit              = False,
                  initial_train_size = len(series) - n_validation,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.23598059857016607}, 0.20880273566554797,
            0.23598059857016607],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.398196343012209}, 0.2088108334929226,
            0.398196343012209],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.4441865222328282}, 0.2088131177203144,
            0.4441865222328282],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.53623586010342}, 0.2088176743151537,
            0.53623586010342],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.7252189487445193}, 0.2088269659234972,
            0.7252189487445193],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.5558016213920624},
            0.20913532870732662, 0.5558016213920624],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.6995044937418831},
            0.20913908163271674, 0.6995044937418831],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.7406154516747153},
            0.20914015254956903, 0.7406154516747153],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.8509374761370117},
            0.20914302038975385, 0.8509374761370117],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.9809565564007693},
            0.20914638910039665, 0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([9, 3, 4, 6, 8, 1, 0, 5, 7, 2], dtype='int64')
    ).astype({
        'mean_absolute_error': float, 
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_forecaster_multiseries_optuna_engine_ForecasterAutoregMultiSeriesCustom():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterAutoregMultiSeriesCustom with mocked using 
    optuna engine (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4
                 )
    steps = 3
    n_validation = 12

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        
        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  steps              = steps,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.2345829390285611}, 0.21476183342122757,
            0.2345829390285611],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.29327794160087567}, 0.21476722307839208,
            0.29327794160087567],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.398196343012209}, 0.21477679099211167,
            0.398196343012209],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.42887539552321635}, 0.2147795727959785,
            0.42887539552321635],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.48612258246951734}, 0.21478474449018078,
            0.48612258246951734],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.5558016213920624}, 0.2147910057902114,
            0.5558016213920624],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.6879814411990146}, 0.21480278321679924,
            0.6879814411990146],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.6995044937418831}, 0.2148038037681593,
            0.6995044937418831],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.7222742800877074}, 0.2148058175049471,
            0.7222742800877074],
        [list(['l1', 'l2']), 'custom function: create_predictors',
            {'alpha': 0.9809565564007693}, 0.2148284279387544,
            0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([2, 1, 9, 5, 8, 3, 7, 0, 4, 6], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


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
def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_with_mocked(forecaster):
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterAutoregMultiSeries 
    and ForecasterAutoregMultiSeriesCustom when series and exog are
    dictionaries (mocked done in Skforecast v0.12.0).
    """

    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
        
        results_search, best_trial = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict,
            exog               = exog_dict,
            search_space       = search_space,
            metric             = 'mean_absolute_error',
            initial_train_size = len(series_dict_train['id_1000']),
            steps              = 24,
            refit              = False,
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True
        )
    
    expected = pd.DataFrame(
        np.array([[list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
        np.array([ 1,  7, 14]), {'n_estimators': 15, 'max_depth': 3},
        569.1242067387045, 15, 3],
       [list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
        np.array([ 1,  7, 14]), {'n_estimators': 11, 'max_depth': 3},
        609.0967098805875, 11, 3],
       [list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
        np.array([5]), {'n_estimators': 15, 'max_depth': 3},
        709.9766964468846, 15, 3]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'n_estimators', 'max_depth'],
        index=pd.Index([0, 2, 1], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'n_estimators': int,
        'max_depth': int
    })

    pd.testing.assert_frame_equal(expected, results_search)


def test_results_output_bayesian_search_forecaster_multivariate_optuna_engine_ForecasterAutoregMultiVariate():
    """
    Test output of bayesian_search_forecaster_multivariate in 
    ForecasterAutoregMultiVariate with mocked using 
    optuna engine (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags' : trial.suggest_categorical('lags', [2, {'l1': 4, 'l2': [2, 3]}])
        }

        return search_space

    results = bayesian_search_forecaster_multivariate(
                  forecaster         = forecaster,
                  series             = series,
                  steps              = steps,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  refit              = False,
                  initial_train_size = len(series) - n_validation,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.23598059857016607}, 0.19308110319514993,
            0.23598059857016607],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.398196343012209}, 0.1931744420708601,
            0.398196343012209],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.4441865222328282}, 0.1932004954044704,
            0.4441865222328282],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.53623586010342}, 0.19325210858832276,
            0.53623586010342],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.7252189487445193}, 0.19335589494249983,
            0.7252189487445193],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.5558016213920624},
            0.20131081099888368, 0.5558016213920624],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.6995044937418831},
            0.2013710017368262, 0.6995044937418831],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.7406154516747153},
            0.2013880862681147, 0.7406154516747153],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.8509374761370117},
            0.20143363961627603, 0.8509374761370117],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.9809565564007693},
            0.20148678375852938, 0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([9, 3, 4, 6, 8, 1, 0, 5, 7, 2], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)