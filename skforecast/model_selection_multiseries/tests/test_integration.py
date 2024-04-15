# Integration test model_selection_multiseries
# ==============================================================================
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import (
    backtesting_forecaster_multiseries,
    bayesian_search_forecaster_multiseries,
    grid_search_forecaster_multiseries
)

# Fixtures
series_dict = joblib.load('./fixture_sample_multi_series.joblib')
exog_dict = joblib.load('./fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}


def test_integration_backtesting_multiseries():
    """
    Integration test for backtesting_forecaster_multiseries function.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster=forecaster,
        series=series_dict,
        exog=exog_dict,
        levels=None,
        steps=24,
        metric= "mean_absolute_error",
        initial_train_size=len(series_dict_train["id_1000"]),
        fixed_train_size=True,
        gap=0,
        allow_incomplete_fold=True,
        refit=False,
        n_jobs="auto",
        verbose=False,
        show_progress=False,
        suppress_warnings=True,
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


def test_integration_bayesian_search_multiseries():
    """
    Integration test for bayesian_search_multiseries function.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

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
            forecaster=forecaster,
            series=series_dict,
            exog=exog_dict,
            search_space=search_space,
            metric="mean_absolute_error",
            initial_train_size=len(series_dict_train["id_1000"]),
            steps=10,
            refit=False,
            n_trials=3,
            return_best=False,
            show_progress=False,
            verbose=False,
            suppress_warnings=True,
        )
    
    expected = pd.Series(
        data = [725.19927565, 743.64911807, 764.96414019],
        index = [0, 2, 1],
        name = 'mean_absolute_error'
    )

    pd.testing.assert_series_equal(expected, results_search.mean_absolute_error)