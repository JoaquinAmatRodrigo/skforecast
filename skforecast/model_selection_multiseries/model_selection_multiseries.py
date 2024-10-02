################################################################################
#                  skforecast.model_selection_multiseries                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable, Generator
import os
import re
from copy import deepcopy
import logging
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import optuna
from optuna.samplers import TPESampler

from ..exceptions import warn_skforecast_categories
from ..exceptions import LongTrainingWarning
from ..exceptions import IgnoredArgumentWarning
from ..metrics import add_y_train_argument, _get_metric
from ..model_selection.model_selection import _create_backtesting_folds
from ..utils import check_backtesting_input
from ..utils import select_n_jobs_backtesting
from ..utils import initialize_lags
from ..utils import initialize_lags_grid
from ..utils import set_skforecast_warnings

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def _initialize_levels_model_selection_multiseries(
    forecaster: object, 
    series: Union[pd.DataFrame, dict],
    levels: Optional[Union[str, list]] = None
) -> list:
    """
    Initialize levels for model_selection_multiseries functions.

    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate, ForecasterRnn
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    Returns
    -------
    levels : list
        List of levels to be used in model_selection_multiseries functions.
    
    """

    multi_series_forecasters_with_levels = [
        'ForecasterAutoregMultiSeries', 
        'ForecasterAutoregMultiSeriesCustom', 
        'ForecasterRnn'
    ]

    if type(forecaster).__name__ in multi_series_forecasters_with_levels  \
        and not isinstance(levels, (str, list, type(None))):
        raise TypeError(
            (f"`levels` must be a `list` of column names, a `str` of a column "
             f"name or `None` when using a forecaster of type "
             f"{multi_series_forecasters_with_levels}. If the forecaster is of "
             f"type `ForecasterAutoregMultiVariate`, this argument is ignored.")
        )

    if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate':
        if levels and levels != forecaster.level and levels != [forecaster.level]:
            warnings.warn(
                (f"`levels` argument have no use when the forecaster is of type "
                 f"`ForecasterAutoregMultiVariate`. The level of this forecaster "
                 f"is '{forecaster.level}', to predict another level, change "
                 f"the `level` argument when initializing the forecaster. \n"),
                 IgnoredArgumentWarning
            )
        levels = [forecaster.level]
    else:
        if levels is None:
            # Forecaster could be untrained, so self.series_col_names cannot be used.
            if isinstance(series, pd.DataFrame):
                levels = list(series.columns)
            else:
                levels = list(series.keys())
        elif isinstance(levels, str):
            levels = [levels]

    return levels


def _extract_data_folds_multiseries(
    series: Union[pd.Series, pd.DataFrame, dict],
    folds: list,
    span_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    window_size: int,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    dropna_last_window: bool = False,
    externally_fitted: bool = False
) -> Generator[
        Tuple[
            Union[pd.Series, pd.DataFrame, dict],
            pd.DataFrame,
            list,
            Optional[Union[pd.Series, pd.DataFrame, dict]],
            Optional[Union[pd.Series, pd.DataFrame, dict]],
            list
        ],
        None,
        None
    ]:
    """
    Select the data from series and exog that corresponds to each fold created using the
    skforecast.model_selection._create_backtesting_folds function.

    Parameters
    ----------
    series : pandas Series, pandas DataFrame, dict
        Time series.
    folds : list
        Folds created using the skforecast.model_selection._create_backtesting_folds
        function.
    span_index : pandas DatetimeIndex, pandas RangeIndex
        Full index from the minimum to the maximum index among all series.
    window_size : int
        Size of the window needed to create the predictors.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    dropna_last_window : bool, default `False`
        If `True`, drop the columns of the last window that have NaN values.
    externally_fitted : bool, default `False`
        Flag indicating whether the forecaster is already trained. Only used when 
        `initial_train_size` is None and `refit` is False.

    Yield
    -----
    series_train : pandas Series, pandas DataFrame, dict
        Time series corresponding to the training set of the fold.
    series_last_window: pandas DataFrame
        Time series corresponding to the last window of the fold.
    levels_last_window: list
        Levels of the time series present in the last window of the fold.
    exog_train: pandas Series, pandas DataFrame, dict, None
        Exogenous variable corresponding to the training set of the fold.
    exog_test: pandas Series, pandas DataFrame, dict, None
        Exogenous variable corresponding to the test set of the fold.
    fold: list
        Fold created using the skforecast.model_selection._create_backtesting_folds

    """

    for fold in folds:
        train_iloc_start       = fold[0][0]
        train_iloc_end         = fold[0][1]
        last_window_iloc_start = fold[1][0]
        last_window_iloc_end   = fold[1][1]
        test_iloc_start        = fold[2][0]
        test_iloc_end          = fold[2][1]

        if isinstance(series, dict) or isinstance(exog, dict):
            # Substract 1 to the iloc indexes to get the loc indexes
            train_loc_start       = span_index[train_iloc_start]
            train_loc_end         = span_index[train_iloc_end - 1]
            last_window_loc_start = span_index[last_window_iloc_start]
            last_window_loc_end   = span_index[last_window_iloc_end - 1]
            test_loc_start        = span_index[test_iloc_start]
            test_loc_end          = span_index[test_iloc_end - 1]

        if isinstance(series, pd.DataFrame):
            series_train = series.iloc[train_iloc_start:train_iloc_end, ]

            series_to_drop = []
            for col in series_train.columns:
                if series_train[col].isna().all():
                    series_to_drop.append(col)
                else:
                    first_valid_index = series_train[col].first_valid_index()
                    last_valid_index = series_train[col].last_valid_index()
                    if (
                        len(series_train[col].loc[first_valid_index:last_valid_index])
                        < window_size
                    ):
                        series_to_drop.append(col)

            series_last_window = series.iloc[
                last_window_iloc_start:last_window_iloc_end,
            ]
            
            series_train = series_train.drop(columns=series_to_drop)
            if not externally_fitted:
                series_last_window = series_last_window.drop(columns=series_to_drop)
        else:
            series_train = {}
            for k in series.keys():
                v = series[k].loc[train_loc_start:train_loc_end]
                if not v.isna().all():
                    first_valid_index = v.first_valid_index()
                    last_valid_index  = v.last_valid_index()
                    if first_valid_index is not None and last_valid_index is not None:
                        v = v.loc[first_valid_index : last_valid_index]
                        if len(v) >= window_size:
                            series_train[k] = v

            series_last_window = {}
            for k, v in series.items():
                v = series[k].loc[last_window_loc_start:last_window_loc_end]
                if ((externally_fitted or k in series_train) and len(v) >= window_size):
                    series_last_window[k] = v

            series_last_window = pd.DataFrame(series_last_window)

        if dropna_last_window:
            series_last_window = series_last_window.dropna(axis=1, how="any")
            # TODO: add the option to drop the series without minimum non NaN values.
            # Similar to how pandas does in the rolling window function.
        
        levels_last_window = list(series_last_window.columns)

        if exog is not None:
            if isinstance(exog, (pd.Series, pd.DataFrame)):
                exog_train = exog.iloc[train_iloc_start:train_iloc_end, ]
                exog_test = exog.iloc[test_iloc_start:test_iloc_end, ]
            else:
                exog_train = {
                    k: v.loc[train_loc_start:train_loc_end] 
                    for k, v in exog.items()
                }
                exog_train = {k: v for k, v in exog_train.items() if len(v) > 0}

                exog_test = {
                    k: v.loc[test_loc_start:test_loc_end]
                    for k, v in exog.items()
                    if externally_fitted or k in exog_train
                }

                exog_test = {k: v for k, v in exog_test.items() if len(v) > 0}
        else:
            exog_train = None
            exog_test = None

        yield series_train, series_last_window, levels_last_window, exog_train, exog_test, fold


def _calculate_metrics_multiseries(
    series: Union[pd.DataFrame, dict],
    predictions: pd.DataFrame,
    folds: Union[list, tqdm],
    span_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    window_size: int,
    metrics: list,
    levels: list,
    add_aggregated_metric: bool = True
) -> pd.DataFrame:
    """   
    Calculate metrics for each level and also for all levels aggregated using
    average, weighted average or pooling.

    - 'average': the average (arithmetic mean) of all levels.
    - 'weighted_average': the average of the metrics weighted by the number of
    predicted values of each level.
    - 'pooling': the values of all levels are pooled and then the metric is
    calculated.

    Parameters
    ----------
    series : pandas DataFrame, dict
        Series data used for backtesting.
    predictions : pandas DataFrame
        Predictions generated during the backtesting process.
    folds : list, tqdm
        Folds created during the backtesting process.
    span_index : pandas DatetimeIndex, pandas RangeIndex
        Full index from the minimum to the maximum index among all series.
    window_size : int
        Size of the window used by the forecaster to create the predictors.
        This is used remove the first `window_size` (`window_size_diff` if 
        differentiation is included) values from y_train since they are not part
        of the training matrix.
    metrics : list
        List of metrics to calculate.
    levels : list
        Levels to calculate the metrics.
    add_aggregated_metric : bool, default `True`
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s).
    
    """

    if not isinstance(series, (pd.DataFrame, dict)):
        raise TypeError(
            ("`series` must be a pandas DataFrame or a dictionary of pandas "
             "DataFrames.")
        )
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("`predictions` must be a pandas DataFrame.")
    if not isinstance(folds, (list, tqdm)):
        raise TypeError("`folds` must be a list or a tqdm object.")
    if not isinstance(span_index, (pd.DatetimeIndex, pd.RangeIndex)):
        raise TypeError("`span_index` must be a pandas DatetimeIndex or pandas RangeIndex.")
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("`window_size` must be an integer.")
    if not isinstance(metrics, list):
        raise TypeError("`metrics` must be a list.")
    if not isinstance(levels, list):
        raise TypeError("`levels` must be a list.")
    if not isinstance(add_aggregated_metric, bool):
        raise TypeError("`add_aggregated_metric` must be a boolean.")
    
    metric_names = [(m if isinstance(m, str) else m.__name__) for m in metrics]

    y_true_pred_levels = []
    y_train_levels = []
    for level in levels:
        y_true_pred_level = None
        y_train = None
        if level in predictions.columns:
            # TODO: avoid merges inside the loop, instead merge outside and then filter
            y_true_pred_level = pd.merge(
                series[level],
                predictions[level],
                left_index  = True,
                right_index = True,
                how         = "inner",
            ).dropna(axis=0, how="any")
            y_true_pred_level.columns = ['y_true', 'y_pred']

            train_indexes = []
            for i, fold in enumerate(folds):
                fit_fold = fold[-1]
                if i == 0 or fit_fold:
                    train_iloc_start = fold[0][0]
                    train_iloc_end = fold[0][1]
                    train_indexes.append(np.arange(train_iloc_start, train_iloc_end))
            train_indexes = np.unique(np.concatenate(train_indexes))
            train_indexes = span_index[train_indexes]
            y_train = series[level].loc[series[level].index.intersection(train_indexes)]

        y_true_pred_levels.append(y_true_pred_level)
        y_train_levels.append(y_train)
            
    metrics_levels = []
    for i, level in enumerate(levels):
        if y_true_pred_levels[i] is not None and not y_true_pred_levels[i].empty:
            metrics_level = [
                m(
                    y_true = y_true_pred_levels[i].iloc[:, 0],
                    y_pred = y_true_pred_levels[i].iloc[:, 1],
                    y_train = y_train_levels[i].iloc[window_size:]  # Exclude observations used to create predictors
                )
                for m in metrics
            ]
            metrics_levels.append(metrics_level)
        else:
            metrics_levels.append([None for _ in metrics])

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = [m if isinstance(m, str) else m.__name__
                                    for m in metrics]
                     )
    metrics_levels.insert(0, 'levels', levels)

    if len(levels) < 2:
        add_aggregated_metric = False
    
    if add_aggregated_metric:

        # aggragation: average
        average = metrics_levels.drop(columns='levels').mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'

        # aggregation: weighted_average
        weighted_averages = {}
        n_predictions_levels = (
            predictions
            .notna()
            .sum()
            .to_frame(name='n_predictions')
            .reset_index(names='levels')
        )
        metrics_levels_no_missing = (
            metrics_levels.merge(n_predictions_levels, on='levels', how='inner')
        )
        for col in metric_names:
            weighted_averages[col] = np.average(
                metrics_levels_no_missing[col],
                weights=metrics_levels_no_missing['n_predictions']
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'

        # aggregation: pooling
        y_true_pred_levels, y_train_levels = zip(
            *[
                (a, b.iloc[window_size:])  # Exclude observations used to create predictors
                for a, b in zip(y_true_pred_levels, y_train_levels)
                if a is not None
            ]
        )
        y_train_levels = list(y_train_levels)
        y_true_pred_levels = pd.concat(y_true_pred_levels)
        y_train_levels_concat = pd.concat(y_train_levels)

        pooled = []
        for m, m_name in zip(metrics, metric_names):
            if m_name in ['mean_absolute_scaled_error', 'root_mean_squared_scaled_error']:
                pooled.append(
                    m(
                        y_true = y_true_pred_levels.loc[:, 'y_true'],
                        y_pred = y_true_pred_levels.loc[:, 'y_pred'],
                        y_train = y_train_levels
                    )
                )
            else:
                pooled.append(
                    m(
                        y_true = y_true_pred_levels.loc[:, 'y_true'],
                        y_pred = y_true_pred_levels.loc[:, 'y_pred'],
                        y_train = y_train_levels_concat
                    )
                )
        pooled = pd.DataFrame([pooled], columns=metric_names)
        pooled['levels'] = 'pooling'

        metrics_levels = pd.concat(
            [metrics_levels, average, weighted_average, pooled],
            axis=0,
            ignore_index=True
        )

    return metrics_levels


def _predict_and_calculate_metrics_multiseries_one_step_ahead(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, dict],
    X_train_encoding: pd.Series,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, dict],
    X_test_encoding: pd.Series,
    levels: list,
    metrics: list,
    add_aggregated_metric: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """   
    One-step-ahead predictions and metrics for each level and also for all levels
    aggregated using average, weighted average or pooling.
    Input matrices (X_train, y_train, X_test, y_test, X_train_encoding, y_test_encoding)
    are should have been generated using the forecaster._train_test_split_one_step_ahead().

    - 'average': the average (arithmetic mean) of all levels.
    - 'weighted_average': the average of the metrics weighted by the number of
    predicted values of each level.
    - 'pooling': the values of all levels are pooled and then the metric is
    calculated.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    series : pandas DataFrame, dict
        Series data used to train and test the forecaster.
    X_train : pandas DataFrame
        Training matrix.
    y_train : pandas Series, dict
        Target values of the training set.
    X_train_encoding : pandas Series
            Series identifiers for each row of `X_train`.
    X_test : pandas DataFrame
        Test matrix.
    y_test : pandas Series, dict
        Target values of the test set.
    X_test_encoding : pandas Series
        Series identifiers for each row of `X_test`.
    metrics : list
        List of metrics to calculate.
    levels : list
        Levels to calculate the metrics.
    add_aggregated_metric : bool, default `True`
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas Dataframe
        Value of predictions for each level.
    
    """

    if not isinstance(series, (pd.DataFrame, dict)):
        raise TypeError(
            ("`series` must be a pandas DataFrame or a dictionary of pandas "
             "DataFrames.")
        )
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"`X_train` must be a pandas DataFrame. Got: {type(X_train)}")
    if not isinstance(y_train, (pd.Series, dict)):
        raise TypeError(
            (f"`y_train` must be a pandas Series or a dictionary of pandas Series. "
                f"Got: {type(y_train)}")
        )        
    if not isinstance(X_train_encoding, pd.Series):
        raise TypeError(
            (f"`X_train_encoding` must be a pandas Series. Got: {type(X_train_encoding)}")
        )
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"`X_test` must be a pandas DataFrame. Got: {type(X_test)}")
    if not isinstance(y_test, (pd.Series, dict)):
        raise TypeError(
            (f"`y_test` must be a pandas Series or a dictionary of pandas Series. "
             f"Got: {type(y_test)}")
        )
    if not isinstance(X_test_encoding, pd.Series):
        raise TypeError(
            (f"`y_test_encoding` must be a pandas Series. Got: {type(X_test_encoding)}")
        )
    if not isinstance(metrics, list):
        raise TypeError("`metrics` must be a list.")
    if not isinstance(levels, list):
        raise TypeError("`levels` must be a list.")
    if not isinstance(add_aggregated_metric, bool):
        raise TypeError("`add_aggregated_metric` must be a boolean.")
    
    metrics = [
        _get_metric(metric=m)
        if isinstance(m, str)
        else add_y_train_argument(m) 
        for m in metrics
    ]
    metric_names = [(m if isinstance(m, str) else m.__name__) for m in metrics]

    if isinstance(series[levels[0]].index, pd.DatetimeIndex):
        freq = series[levels[0]].index.freq
    else:
        freq = series[levels[0]].index.step

    if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate':
        step = 1
        X_train, y_train = forecaster.filter_train_X_y_for_step(
                               step    = step,
                               X_train = X_train,
                               y_train = y_train
                           )
        X_test, y_test = forecaster.filter_train_X_y_for_step(
                             step    = step,  
                             X_train = X_test,
                             y_train = y_test
                         )                 
        forecaster.regressors_[step].fit(X_train, y_train)
        pred = forecaster.regressors_[step].predict(X_test)
    else:
        forecaster.regressor.fit(X_train, y_train)
        pred = forecaster.regressor.predict(X_test)

    predictions_per_level = pd.DataFrame(
        {
            'y_true': y_test,
            'y_pred': pred,
            '_level_skforecast': X_test_encoding,
        },
        index=y_test.index,
    ).groupby('_level_skforecast')
    predictions_per_level = {key: group for key, group in predictions_per_level}

    y_train_per_level = pd.DataFrame(
        {"y_train": y_train, "_level_skforecast": X_train_encoding},
        index=y_train.index,
    ).groupby("_level_skforecast")
    # Interleaved Nan values were excluded fom y_train. They are reestored
    y_train_per_level = {key: group.asfreq(freq) for key, group in y_train_per_level}

    if forecaster.differentiation is not None:
        for level in predictions_per_level:
            differentiator = deepcopy(forecaster.differentiator_[level])
            differentiator.initial_values = [
                series[level].iloc[
                    forecaster.window_size_diff - forecaster.differentiation
                ]
            ]
            predictions_per_level[level]["y_pred"] = differentiator.inverse_transform_next_window(
                predictions_per_level[level]["y_pred"].to_numpy()
            )
            predictions_per_level[level]["y_true"] = differentiator.inverse_transform_next_window(
                predictions_per_level[level]["y_true"].to_numpy()
            )
            y_train_per_level[level]["y_true"] = differentiator.inverse_transform(
                y_train_per_level[level]["y_train"].to_numpy()
            )[forecaster.differentiation:]

    if forecaster.transformer_series is not None:
        for level in predictions_per_level:
            transformer = forecaster.transformer_series_[level]
            predictions_per_level[level]["y_pred"] = transformer.inverse_transform(
                predictions_per_level[level][["y_pred"]]
            )
            predictions_per_level[level]["y_true"] = transformer.inverse_transform(
                predictions_per_level[level][["y_true"]]
            )
            y_train_per_level[level]["y_train"] = transformer.inverse_transform(
                y_train_per_level[level][["y_train"]]
            )

    metrics_levels = []
    for level in levels:
        if level in predictions_per_level:
            metrics_level = [
                m(
                    y_true = predictions_per_level[level].loc[:, 'y_true'],
                    y_pred = predictions_per_level[level].loc[:, 'y_pred'],
                    y_train = y_train_per_level[level].loc[:, 'y_train']
                )
                for m in metrics
            ]
            metrics_levels.append(metrics_level)
        else:
            metrics_levels.append([None for _ in metrics])

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = [m if isinstance(m, str) else m.__name__
                                    for m in metrics]
                     )
    metrics_levels.insert(0, 'levels', levels)

    if len(levels) < 2:
        add_aggregated_metric = False

    if add_aggregated_metric:

        # aggragation: average
        average = metrics_levels.drop(columns='levels').mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'

        # aggregation: weighted_average
        weighted_averages = {}
        n_predictions_levels = {
            k: v['y_pred'].notna().sum()
            for k, v in predictions_per_level.items()
        }
        n_predictions_levels = pd.DataFrame(
            n_predictions_levels.items(),
            columns=['levels', 'n_predictions']
        )
        metrics_levels_no_missing = (
            metrics_levels.merge(n_predictions_levels, on='levels', how='inner')
        )
        for col in metric_names:
            weighted_averages[col] = np.average(
                metrics_levels_no_missing[col],
                weights=metrics_levels_no_missing['n_predictions']
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'

        # aggregation: pooling
        list_y_train_by_level = [
            v['y_train'].to_numpy()
            for k, v in y_train_per_level.items()
            if k in predictions_per_level
        ]
        predictions_pooled = pd.concat(predictions_per_level.values())
        y_train_pooled = pd.concat(
            [v for k, v in y_train_per_level.items() if k in predictions_per_level]
        )
        pooled = []
        for m, m_name in zip(metrics, metric_names):
            if m_name in ['mean_absolute_scaled_error', 'root_mean_squared_scaled_error']:
                pooled.append(
                    m(
                        y_true = predictions_pooled['y_true'],
                        y_pred = predictions_pooled['y_pred'],
                        y_train = list_y_train_by_level
                    )
                )
            else:
                pooled.append(
                    m(
                        y_true = predictions_pooled['y_true'],
                        y_pred = predictions_pooled['y_pred'],
                        y_train = y_train_pooled['y_train']
                    )
                )
        pooled = pd.DataFrame([pooled], columns=metric_names)
        pooled['levels'] = 'pooling'

        metrics_levels = pd.concat(
            [metrics_levels, average, weighted_average, pooled],
            axis=0,
            ignore_index=True
        )

    predictions = (
        pd.concat(predictions_per_level.values())
        .loc[:, ["y_pred", "_level_skforecast"]]
        .pivot(columns="_level_skforecast", values="y_pred")
        .rename_axis(columns=None, index=None)
    )
    predictions = predictions.asfreq(X_test.index.freq)

    return metrics_levels, predictions


def _backtesting_forecaster_multiseries(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int] = None,
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    levels: Optional[Union[str, list]] = None,
    add_aggregated_metric: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    refit: Union[bool, int] = False,
    interval: Optional[list] = None,
    n_boot: int = 500,
    random_state: int = 123,
    use_in_sample_residuals: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting for multi-series and multivariate forecasters.

    - If `refit` is `False`, the model will be trained only once using the 
    `initial_train_size` first observations. 
    - If `refit` is `True`, the model is trained on each iteration, increasing
    the training set. 
    - If `refit` is an `integer`, the model will be trained every that number 
    of iterations.
    - If `forecaster` is already trained and `initial_train_size` is `None`,
    no initial train will be done and all data will be used to evaluate the model.
    However, the first `len(forecaster.last_window)` observations are needed
    to create the initial predictors, so no predictions are calculated for them.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error'}
        - If `Callable`: Function with arguments y_true, y_pred that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is 
        already trained, no initial train is done and all data is used to evaluate the 
        model. However, the first `len(forecaster.last_window)` observations are needed 
        to create the initial predictors, so no predictions are calculated for them. 
        This useful to backtest the model on the same data used to train it.
        `None` is only allowed when `refit` is `False` and `forecaster` is already
        trained.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    skip_folds : int, list, default `None`
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
    add_aggregated_metric : bool, default `False`
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an 
        integer, the Forecaster will be trained every that number of iterations.
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated.
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    use_in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction
        error to create prediction intervals. If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.
    backtest_predictions : pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`. 
        If there is more than one level, this structure will be repeated for each of them.

        - column pred: predictions.
        - column lower_bound: lower bound of the interval.
        - column upper_bound: upper bound of the interval.
    
    """

    set_skforecast_warnings(suppress_warnings, action='ignore')

    forecaster = deepcopy(forecaster)

    if n_jobs == 'auto':
        n_jobs = select_n_jobs_backtesting(
                     forecaster = forecaster,
                     refit      = refit
                 )
    elif not isinstance(refit, bool) and refit != 1 and n_jobs != 1:
        warnings.warn(
            ("If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
             "is set to 1 to avoid unexpected results during parallelization."),
             IgnoredArgumentWarning
        )
        n_jobs = 1
    else:
        n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster,
                 series     = series,
                 levels     = levels
             )

    if not isinstance(metric, list):
        metrics = [
            _get_metric(metric=metric)
            if isinstance(metric, str)
            else add_y_train_argument(metric)
        ]
    else:
        metrics = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]

    store_in_sample_residuals = False if interval is None else True

    if isinstance(series, dict):
        min_index = min([v.index[0] for v in series.values()])
        max_index = max([v.index[-1] for v in series.values()])
        # All series must have the same frequency
        frequency = series[list(series.keys())[0]].index.freqstr
        span_index = pd.date_range(start=min_index, end=max_index, freq=frequency)
    else:
        span_index = series.index

    if initial_train_size is not None:
        # First model training, this is done to allow parallelization when `refit`
        # is `False`. The initial Forecaster fit is outside the auxiliary function.
        window_size = forecaster.window_size_diff
        fold_initial_train = [
            [0, initial_train_size],
            [initial_train_size - window_size, initial_train_size],
            [0, 0],  # dummy values
            [0, 0],  # dummy values
            True
        ]
        data_fold = _extract_data_folds_multiseries(
                        series             = series,
                        folds              = [fold_initial_train],
                        span_index         = span_index,
                        window_size        = window_size,
                        exog               = exog,
                        dropna_last_window = forecaster.dropna_from_series,
                        externally_fitted  = False
                    )
        series_train, _, last_window_levels, exog_train, _, _ = next(data_fold)

        forecaster.fit(
            series                    = series_train,
            exog                      = exog_train,
            store_last_window         = last_window_levels,
            store_in_sample_residuals = store_in_sample_residuals,
            suppress_warnings         = suppress_warnings
        )
        externally_fitted = False
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        window_size = forecaster.window_size_diff
        initial_train_size = window_size
        externally_fitted = True

    # TODO: remove when all forecaster include differentiation
    if type(forecaster).__name__ in ['ForecasterAutoregMultiSeries', 
                                     'ForecasterAutoregMultiSeriesCustom']:
        differentiation = forecaster.differentiation
    else:
        differentiation = None

    folds = _create_backtesting_folds(
                data                  = span_index,
                window_size           = window_size,
                initial_train_size    = initial_train_size,
                test_size             = steps,
                externally_fitted     = externally_fitted,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                skip_folds            = skip_folds,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = False,
                differentiation       = differentiation,
                verbose               = verbose
            )

    if refit:
        n_of_fits = int(len(folds) / refit)
        if type(forecaster).__name__ != 'ForecasterAutoregMultiVariate' and n_of_fits > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits} times. This can take substantial "
                 f"amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning,
            )
        elif type(forecaster).__name__ == 'ForecasterAutoregMultiVariate' and n_of_fits * forecaster.steps > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits * forecaster.steps} times "
                 f"({n_of_fits} folds * {forecaster.steps} regressors). This can take "
                 f"substantial amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )

    if show_progress:
        folds = tqdm(folds)

    data_folds = _extract_data_folds_multiseries(
                     series             = series,
                     folds              = folds,
                     span_index         = span_index,
                     window_size        = window_size,
                     exog               = exog,
                     dropna_last_window = forecaster.dropna_from_series,
                     externally_fitted  = externally_fitted
                 )

    def _fit_predict_forecaster(data_fold, forecaster, interval, levels):
        """
        Fit the forecaster and predict `steps` ahead. This is an auxiliary 
        function used to parallelize the backtesting_forecaster_multiseries
        function.
        """

        (
            series_train,
            last_window_series,
            last_window_levels,
            exog_train,
            next_window_exog,
            fold
        ) = data_fold

        if fold[4] is True:
            forecaster.fit(
                series                    = series_train, 
                exog                      = exog_train,
                store_last_window         = last_window_levels,
                store_in_sample_residuals = store_in_sample_residuals,
                suppress_warnings         = suppress_warnings
            )

        test_iloc_start = fold[2][0]
        test_iloc_end   = fold[2][1]
        steps = len(range(test_iloc_start, test_iloc_end))
        if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate' and gap > 0:
            # Select only the steps that need to be predicted if gap > 0
            test_iloc_start = fold[3][0]
            test_iloc_end   = fold[3][1]
            steps = list(np.arange(len(range(test_iloc_start, test_iloc_end))) + gap + 1)

        levels_predict = [level for level in levels 
                          if level in last_window_levels]
        if interval is None:

            pred = forecaster.predict(
                       steps             = steps, 
                       levels            = levels_predict, 
                       last_window       = last_window_series,
                       exog              = next_window_exog,
                       suppress_warnings = suppress_warnings
                   )
        else:
            pred = forecaster.predict_interval(
                       steps                   = steps,
                       levels                  = levels_predict, 
                       last_window             = last_window_series,
                       exog                    = next_window_exog,
                       interval                = interval,
                       n_boot                  = n_boot,
                       random_state            = random_state,
                       use_in_sample_residuals = use_in_sample_residuals,
                       suppress_warnings       = suppress_warnings
                   )

        if type(forecaster).__name__ != 'ForecasterAutoregMultiVariate' and gap > 0:
            pred = pred.iloc[gap:, ]

        return pred

    backtest_predictions = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_forecaster)(
            data_fold  = data_fold,
            forecaster = forecaster,
            interval   = interval,
            levels     = levels,
        )
        for data_fold in data_folds
    )

    backtest_predictions = pd.concat(backtest_predictions, axis=0)

    levels_in_backtest_predictions = backtest_predictions.columns
    if interval is not None:
        levels_in_backtest_predictions = [
            level 
            for level in levels_in_backtest_predictions
            if not re.search(r'_lower_bound|_upper_bound', level)
        ]
    for level in levels_in_backtest_predictions:
        valid_index = series[level][series[level].notna()].index
        no_valid_index = backtest_predictions.index.difference(valid_index, sort=False)
        cols = [level]
        if interval:
            cols = cols + [f'{level}_lower_bound', f'{level}_upper_bound']
        backtest_predictions.loc[no_valid_index, cols] = np.nan

    metrics_levels = _calculate_metrics_multiseries(
        series                = series,
        predictions           = backtest_predictions,
        folds                 = folds,
        span_index            = span_index,
        window_size           = forecaster.window_size_diff,
        metrics               = metrics,
        levels                = levels,
        add_aggregated_metric = add_aggregated_metric
    )
       
    set_skforecast_warnings(suppress_warnings, action='default')

    return metrics_levels, backtest_predictions


def backtesting_forecaster_multiseries(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int],
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    levels: Optional[Union[str, list]] = None,
    add_aggregated_metric: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    refit: Union[bool, int] = False,
    interval: Optional[list] = None,
    n_boot: int = 500,
    random_state: int = 123,
    use_in_sample_residuals: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting for multi-series and multivariate forecasters.

    - If `refit` is `False`, the model will be trained only once using the 
    `initial_train_size` first observations. 
    - If `refit` is `True`, the model is trained on each iteration, increasing
    the training set. 
    - If `refit` is an `integer`, the model will be trained every that number 
    of iterations.
    - If `forecaster` is already trained and `initial_train_size` is `None`,
    no initial train will be done and all data will be used to evaluate the model.
    However, the first `len(forecaster.last_window)` observations are needed
    to create the initial predictors, so no predictions are calculated for them.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.

    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate, ForecasterRnn
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is 
        already trained, no initial train is done and all data is used to evaluate the 
        model. However, the first `len(forecaster.last_window)` observations are needed 
        to create the initial predictors, so no predictions are calculated for them. 
        This useful to backtest the model on the same data used to train it.
        `None` is only allowed when `refit` is `False` and `forecaster` is already
        trained.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    skip_folds : int, list, default `None`
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
    add_aggregated_metric : bool, default `True`
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an 
        integer, the Forecaster will be trained every that number of iterations.
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated.
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    use_in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals. If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
        If there is more than one level, this structure will be repeated for each of them.

        - column pred: predictions.
        - column lower_bound: lower bound of the interval.
        - column upper_bound: upper bound of the interval.
    
    """

    multi_series_forecasters = [
        'ForecasterAutoregMultiSeries', 
        'ForecasterAutoregMultiSeriesCustom', 
        'ForecasterAutoregMultiVariate',
        'ForecasterRnn'
    ]

    forecaster_name = type(forecaster).__name__

    if forecaster_name not in multi_series_forecasters:
        raise TypeError(
            (f"`forecaster` must be of type {multi_series_forecasters}, "
             f"for all other types of forecasters use the functions available in "
             f"the `model_selection` module. Got {forecaster_name}")
        )
    
    check_backtesting_input(
        forecaster              = forecaster,
        steps                   = steps,
        metric                  = metric,
        add_aggregated_metric   = add_aggregated_metric,
        series                  = series,
        exog                    = exog,
        initial_train_size      = initial_train_size,
        fixed_train_size        = fixed_train_size,
        gap                     = gap,
        skip_folds              = skip_folds,
        allow_incomplete_fold   = allow_incomplete_fold,
        refit                   = refit,
        interval                = interval,
        n_boot                  = n_boot,
        random_state            = random_state,
        use_in_sample_residuals = use_in_sample_residuals,
        n_jobs                  = n_jobs,
        verbose                 = verbose,
        show_progress           = show_progress,
        suppress_warnings       = suppress_warnings
    )

    metrics_levels, backtest_predictions = _backtesting_forecaster_multiseries(
        forecaster              = forecaster,
        series                  = series,
        steps                   = steps,
        levels                  = levels,
        metric                  = metric,
        add_aggregated_metric   = add_aggregated_metric,
        initial_train_size      = initial_train_size,
        fixed_train_size        = fixed_train_size,
        gap                     = gap,
        skip_folds              = skip_folds,
        allow_incomplete_fold   = allow_incomplete_fold,
        exog                    = exog,
        refit                   = refit,
        interval                = interval,
        n_boot                  = n_boot,
        random_state            = random_state,
        use_in_sample_residuals = use_in_sample_residuals,
        n_jobs                  = n_jobs,
        verbose                 = verbose,
        show_progress           = show_progress,
        suppress_warnings       = suppress_warnings
    )

    return metrics_levels, backtest_predictions


def grid_search_forecaster_multiseries(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    param_grid: dict,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    steps: Optional[int] = None,
    method: str = 'backtesting',
    aggregate_metric: Union[str, list] = ['weighted_average', 'average', 'pooling'],
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    levels: Optional[Union[str, list]] = None,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    refit: Union[bool, int] = False,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    steps : int, default `None`
        Number of steps to predict.
    method : str, default `'backtesting'`
        Method used to evaluate the model.

        - 'backtesting': the model is evaluated using backtesting process in which
        the model predicts `steps` ahead in each iteration.
        - 'one_step_ahead': the model is evaluated using only one step ahead predictions.
        This is faster than backtesting but the results may not reflect the actual
        performance of the model when predicting multiple steps ahead. Arguments `steps`,
        `fixed_train_size`, `gap`, `skip_folds`, `allow_incomplete_fold` and `refit` are 
        ignored when `method` is 'one_step_ahead'.
        **New in version 0.14.0**
    aggregate_metric : str, list, default `['weighted_average', 'average', 'pooling']`
        Aggregation method/s used to combine the metric/s of all levels (series)
        when multiple levels are predicted. If list, the first aggregation method
        is used to select the best parameters.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    skip_folds : int, list, default `None`
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an 
        integer, the Forecaster will be trained every that number of iterations.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the hyperparameter 
        search. See skforecast.exceptions.warn_skforecast_categories for more
        information.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column levels: levels configuration for each iteration.
        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration. The resulting 
        metric will be the average of the optimization of all levels.
        - additional n columns with param = value.
    
    """

    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters_multiseries(
                forecaster            = forecaster,
                series                = series,
                param_grid            = param_grid,
                steps                 = steps,
                method                = method,
                metric                = metric,
                aggregate_metric      = aggregate_metric,
                initial_train_size    = initial_train_size,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                skip_folds            = skip_folds,
                allow_incomplete_fold = allow_incomplete_fold,
                levels                = levels,
                exog                  = exog,
                lags_grid             = lags_grid,
                refit                 = refit,
                n_jobs                = n_jobs,
                return_best           = return_best,
                verbose               = verbose,
                show_progress         = show_progress,
                suppress_warnings     = suppress_warnings,
                output_file           = output_file
            )

    return results


def random_search_forecaster_multiseries(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    param_distributions: dict,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    steps: Optional[int] = None,
    method: str = 'backtesting',
    aggregate_metric: Union[str, list] = ['weighted_average', 'average', 'pooling'],
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    levels: Optional[Union[str, list]] = None,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    refit: Union[bool, int] = False,
    n_iter: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions or 
        lists of parameters to try.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    steps : int, default `None`
        Number of steps to predict.
    method : str, default `'backtesting'`
        Method used to evaluate the model.

        - 'backtesting': the model is evaluated using backtesting process in which
        the model predicts `steps` ahead in each iteration.
        - 'one_step_ahead': the model is evaluated using only one step ahead predictions.
        This is faster than backtesting but the results may not reflect the actual
        performance of the model when predicting multiple steps ahead. Arguments `steps`,
        `fixed_train_size`, `gap`, `skip_folds`, `allow_incomplete_fold` and `refit` are 
        ignored when `method` is 'one_step_ahead'.
        **New in version 0.14.0**
    aggregate_metric : str, list, default `['weighted_average', 'average', 'pooling']`
        Aggregation method/s used to combine the metric/s of all levels (series)
        when multiple levels are predicted. If list, the first aggregation method
        is used to select the best parameters.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    skip_folds : int, list, default `None`
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an 
        integer, the Forecaster will be trained every that number of iterations.
    n_iter : int, default `10`
        Number of parameter settings that are sampled per lags configuration. 
        n_iter trades off runtime vs quality of the solution.
    random_state : int, default `123`
        Sets a seed to the random sampling for reproducible output.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the hyperparameter 
        search. See skforecast.exceptions.warn_skforecast_categories for more
        information.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column levels: levels configuration for each iteration.
        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration. The resulting 
        metric will be the average of the optimization of all levels.
        - additional n columns with param = value.
    
    """

    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )
    
    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, 
                                       random_state=random_state))

    results = _evaluate_grid_hyperparameters_multiseries(
                forecaster            = forecaster,
                series                = series,
                param_grid            = param_grid,
                steps                 = steps,
                method                = method,
                metric                = metric,
                aggregate_metric      = aggregate_metric,
                initial_train_size    = initial_train_size,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                skip_folds            = skip_folds,
                allow_incomplete_fold = allow_incomplete_fold,
                levels                = levels,
                exog                  = exog,
                lags_grid             = lags_grid,
                refit                 = refit,
                return_best           = return_best,
                n_jobs                = n_jobs,
                verbose               = verbose,
                show_progress         = show_progress,
                suppress_warnings     = suppress_warnings,
                output_file            = output_file
            )

    return results


def _evaluate_grid_hyperparameters_multiseries(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    param_grid: dict,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    steps: Optional[int] = None,
    method: str = 'backtesting',
    aggregate_metric: Union[str, list] = ['weighted_average', 'average', 'pooling'],
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    levels: Optional[Union[str, list]] = None,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    refit: Union[bool, int] = False,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    steps : int, default `None`
        Number of steps to predict.
    method : str, default `'backtesting'`
        Method used to evaluate the model.

        - 'backtesting': the model is evaluated using backtesting process in which
        the model predicts `steps` ahead in each iteration.
        - 'one_step_ahead': the model is evaluated using only one step ahead predictions.
        This is faster than backtesting but the results may not reflect the actual
        performance of the model when predicting multiple steps ahead. Arguments `steps`,
        `fixed_train_size`, `gap`, `skip_folds`, `allow_incomplete_fold` and `refit` are 
        ignored when `method` is 'one_step_ahead'.
        **New in version 0.14.0**
    aggregate_metric : str, list, default `['weighted_average', 'average', 'pooling']`
        Aggregation method/s used to combine the metric/s of all levels (series)
        when multiple levels are predicted. If list, the first aggregation method
        is used to select the best parameters.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    skip_folds : int, list, default `None`
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an 
        integer, the Forecaster will be trained every that number of iterations.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting. 
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the hyperparameter 
        search. See skforecast.exceptions.warn_skforecast_categories for more
        information.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column levels: levels configuration for each iteration.
        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - n columns with metrics: metric/s value/s estimated for each iteration.
        There is one column for each metric and aggregation method. The name of
        the column flollows the pattern `metric__aggregation`.
        - additional n columns with param = value.
    
    """

    set_skforecast_warnings(suppress_warnings, action='ignore')

    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )
    
    if method == 'one_step_ahead':
        warnings.warn(
            ("One-step-ahead predictions are used for faster model comparison, but they "
             "may not fully represent multi-step prediction performance. It is recommended "
             "to backtest the final model for a more accurate multi-step performance "
             "estimate.")
        )

    if return_best and exog is not None and (len(exog) != len(series)):
        raise ValueError(
            (f"`exog` must have same number of samples as `series`. "
             f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
        )
    
    if isinstance(aggregate_metric, str):
        aggregate_metric = [aggregate_metric]
    allowed_aggregate_metrics = ['average', 'weighted_average', 'pooling']
    if not set(aggregate_metric).issubset(allowed_aggregate_metrics):
        raise ValueError(
            (f"Allowed `aggregate_metric` are: {allowed_aggregate_metrics}. "
             f"Got: {aggregate_metric}.")
        )
    
    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster,
                 series     = series,
                 levels     = levels
             )

    add_aggregated_metric = True if len(levels) > 1 else False

    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
   
    if not isinstance(metric, list):
        metric = [metric]
    metric = [
        _get_metric(metric=m)
        if isinstance(m, str)
        else add_y_train_argument(m) 
        for m in metric
    ]
    metric_names = [(m if isinstance(m, str) else m.__name__) for m in metric]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
    
    if add_aggregated_metric:
        metric_names = [
            f"{metric_name}__{aggregation}"
            for metric_name in metric_names
            for aggregation in aggregate_metric
        ]

    if verbose:
        print(
            f"{len(param_grid) * len(lags_grid)} models compared for {len(levels)} "
            f"level(s). Number of iterations: {len(param_grid) * len(lags_grid)}."
        )

    if show_progress:
        lags_grid_tqdm = tqdm(lags_grid.items(), desc='lags grid', position=0)  # ncols=90
        param_grid = tqdm(param_grid, desc='params grid', position=1, leave=False)
    else:
        lags_grid_tqdm = lags_grid.items()
    
    if output_file is not None and os.path.isfile(output_file):
        os.remove(output_file)

    lags_list = []
    lags_label_list = []
    params_list = []
    metrics_list = []
    for lags_k, lags_v in lags_grid_tqdm:

        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(lags_v)
            lags_v = forecaster.lags.copy()
            if lags_label == 'values':
                lags_k = lags_v

        if method == 'one_step_ahead':

            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_train_encoding,
                X_test_encoding
            ) = forecaster._train_test_split_one_step_ahead(
                series=series, exog=exog, initial_train_size=initial_train_size
            )
        
        for params in param_grid:

            forecaster.set_params(params)
        
            if method == 'backtesting':

                metrics, _ = backtesting_forecaster_multiseries(
                    forecaster            = forecaster,
                    series                = series,
                    exog                  = exog,
                    steps                 = steps,
                    levels                = levels,
                    metric                = metric,
                    add_aggregated_metric = add_aggregated_metric,
                    initial_train_size    = initial_train_size,
                    fixed_train_size      = fixed_train_size,
                    gap                   = gap,
                    skip_folds            = skip_folds,
                    allow_incomplete_fold = allow_incomplete_fold,
                    refit                 = refit,
                    interval              = None,
                    verbose               = verbose,
                    n_jobs                = n_jobs,
                    show_progress         = False,
                    suppress_warnings     = suppress_warnings
                )

            else:

                metrics, _ = _predict_and_calculate_metrics_multiseries_one_step_ahead(
                    forecaster            = forecaster,
                    series                = series,
                    X_train               = X_train,
                    y_train               = y_train,
                    X_train_encoding      = X_train_encoding,
                    X_test                = X_test,
                    y_test                = y_test,
                    X_test_encoding       = X_test_encoding,
                    levels                = levels,
                    metrics               = metric,
                    add_aggregated_metric = add_aggregated_metric
                )

            if add_aggregated_metric:
                metrics = metrics.loc[metrics['levels'].isin(aggregate_metric), :]
            else:
                metrics = metrics.loc[metrics['levels'] == levels[0], :]
            metrics = pd.DataFrame(
                          data    = [metrics.iloc[:, 1:].transpose().stack().to_numpy()],
                          columns = metric_names
                      )

            for warn_category in warn_skforecast_categories:
                warnings.filterwarnings('ignore', category=warn_category)

            lags_list.append(lags_v)
            lags_label_list.append(lags_k)
            params_list.append(params)
            metrics_list.append(metrics)

            if output_file is not None:
                header = ['levels', 'lags', 'lags_label', 'params', 
                          *metric_names, *params.keys()]
                row = [
                    levels,
                    lags_v,
                    lags_k,
                    params,
                    *metrics.loc[0, :].to_list(),
                    *params.values()
                ]
                if not os.path.isfile(output_file):
                    with open(output_file, 'w', newline='') as f:
                        f.write('\t'.join(header) + '\n')
                        f.write('\t'.join([str(r) for r in row]) + '\n')
                else:
                    with open(output_file, 'a', newline='') as f:
                        f.write('\t'.join([str(r) for r in row]) + '\n')

    results = pd.concat(metrics_list, axis=0)
    results.insert(0, 'levels', [levels] * len(results))
    results.insert(1, 'lags', lags_list)
    results.insert(2, 'lags_label', lags_label_list)
    results.insert(3, 'params', params_list)
    results = (
        results
        .sort_values(by=metric_names[0], ascending=True)
        .reset_index(drop=True)
    )
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results.loc[0, 'lags']
        best_params = results.loc[0, 'params']
        best_metric = results.loc[0, metric_names[0]]
        
        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
            f"  Levels: {levels}\n"
        )

    set_skforecast_warnings(suppress_warnings, action='default')
    
    return results


def bayesian_search_forecaster_multiseries(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    search_space: Callable,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    steps: Optional[int] = None,
    method: str = 'backtesting',
    aggregate_metric: Union[str, list] = ['weighted_average', 'average', 'pooling'],
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    levels: Optional[Union[str, list]] = None,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    refit: Union[bool, int] = False,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: Optional[str] = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using multi-series backtesting 
    and optuna library.
    **New in version 0.12.0**
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    search_space : Callable
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    steps : int, default `None`
        Number of steps to predict.
    method : str, default `'backtesting'`
        Method used to evaluate the model.

        - 'backtesting': the model is evaluated using backtesting process in which
        the model predicts `steps` ahead in each iteration.
        - 'one_step_ahead': the model is evaluated using only one step ahead predictions.
        This is faster than backtesting but the results may not reflect the actual
        performance of the model when predicting multiple steps ahead. Arguments `steps`,
        `fixed_train_size`, `gap`, `skip_folds`, `allow_incomplete_fold` and `refit` are 
        ignored when `method` is 'one_step_ahead'.
        **New in version 0.14.0**
    aggregate_metric : str, list, default `['weighted_average', 'average', 'pooling']`
        Aggregation method/s used to combine the metric/s of all levels (series)
        when multiple levels are predicted. If list, the first aggregation method
        is used to select the best parameters.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    skip_folds : int, list, default `None`
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an 
        integer, the Forecaster will be trained every that number of iterations.
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the hyperparameter
        search. See skforecast.exceptions.warn_skforecast_categories for more
        information.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**
    kwargs_create_study : dict, default `{}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study().
        If default, the direction is set to 'minimize' and a TPESampler(seed=123) 
        sampler is used during optimization.
    kwargs_study_optimize : dict, default `{}`
        Other keyword arguments (key, value mappings) to pass to study.optimize().

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column levels: levels configuration for each iteration.
        - column lags: lags configuration for each iteration.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration. The resulting 
        metric will be the average of the optimization of all levels.
        - additional n columns with param = value.
    best_trial : optuna object
        The best optimization result returned as a FrozenTrial optuna object.
    
    """

    if return_best and exog is not None and (len(exog) != len(series)):
        raise ValueError(
            (f"`exog` must have same number of samples as `series`. "
             f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
        )

    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )
    
    results, best_trial = _bayesian_search_optuna_multiseries(
                            forecaster            = forecaster,
                            series                = series,
                            exog                  = exog,
                            levels                = levels, 
                            search_space          = search_space,
                            steps                 = steps,
                            metric                = metric,
                            method                = method,
                            aggregate_metric      = aggregate_metric,
                            refit                 = refit,
                            initial_train_size    = initial_train_size,
                            fixed_train_size      = fixed_train_size,
                            gap                   = gap,
                            skip_folds            = skip_folds,
                            allow_incomplete_fold = allow_incomplete_fold,
                            n_trials              = n_trials,
                            random_state          = random_state,
                            return_best           = return_best,
                            n_jobs                = n_jobs,
                            verbose               = verbose,
                            show_progress         = show_progress,
                            suppress_warnings     = suppress_warnings,
                            output_file           = output_file,
                            kwargs_create_study   = kwargs_create_study,
                            kwargs_study_optimize = kwargs_study_optimize
                        )
        
    return results, best_trial


def _bayesian_search_optuna_multiseries(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    search_space: Callable,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    steps: Optional[int] = None,
    method: str = 'backtesting',
    aggregate_metric: Union[str, list] = ['weighted_average', 'average', 'pooling'],
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    levels: Optional[Union[str, list]] = None,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    refit: Union[bool, int] = False,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: Optional[str] = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using multi-series backtesting 
    and optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    search_space : Callable
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    steps : int, default `None`
        Number of steps to predict.
    method : str, default `'backtesting'`
        Method used to evaluate the model.

        - 'backtesting': the model is evaluated using backtesting process in which
        the model predicts `steps` ahead in each iteration.
        - 'one_step_ahead': the model is evaluated using only one step ahead predictions.
        This is faster than backtesting but the results may not reflect the actual
        performance of the model when predicting multiple steps ahead. Arguments `steps`,
        `fixed_train_size`, `gap`, `skip_folds`, `allow_incomplete_fold` and `refit` are
        ignored when `method` is 'one_step_ahead'.
        **New in version 0.14.0**
    aggregate_metric : str, list, default `['weighted_average', 'average', 'pooling']`
        Aggregation method/s used to combine the metric/s of all levels (series)
        when multiple levels are predicted. If list, the first aggregation method
        is used to select the best parameters.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    skip_folds : int, list, default `None`
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an 
        integer, the Forecaster will be trained every that number of iterations.
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the hyperparameter
        search. See skforecast.exceptions.warn_skforecast_categories for more
        information.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**
    kwargs_create_study : dict, default `{}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study().
        If default, the direction is set to 'minimize' and a TPESampler(seed=123) 
        sampler is used during optimization.
    kwargs_study_optimize : dict, default `{}`
        Other keyword arguments (key, value mappings) to pass to study.optimize().

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column levels: levels configuration for each iteration.
        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - n columns with metrics: metric/s value/s estimated for each iteration.
        There is one column for each metric and aggregation method. The name of
        the column flollows the pattern `metric__aggregation`.
        - additional n columns with param = value.
    best_trial : optuna object
        The best optimization result returned as an optuna FrozenTrial object.

    """
    
    set_skforecast_warnings(suppress_warnings, action='ignore')

    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )
    
    if method == 'one_step_ahead':
        warnings.warn(
            ("One-step-ahead predictions are used for faster model comparison, but they "
             "may not fully represent multi-step prediction performance. It is recommended "
             "to backtest the final model for a more accurate multi-step performance "
             "estimate.")
        )
    
    if isinstance(aggregate_metric, str):
        aggregate_metric = [aggregate_metric]
    allowed_aggregate_metrics = ['average', 'weighted_average', 'pooling']
    if not set(aggregate_metric).issubset(allowed_aggregate_metrics):
        raise ValueError(
            (f"Allowed `aggregate_metric` are: {allowed_aggregate_metrics}. "
             f"Got: {aggregate_metric}.")
        )
    
    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster,
                 series     = series,
                 levels     = levels
             )
    add_aggregated_metric = True if len(levels) > 1 else False

    if not isinstance(metric, list):
        metric = [metric]
    metric = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]
    metric_names = [(m if isinstance(m, str) else m.__name__) for m in metric]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
    
    if add_aggregated_metric:
        metric_names = [
            f"{metric_name}__{aggregation}"
            for metric_name in metric_names
            for aggregation in aggregate_metric
        ]

    # Objective function using backtesting_forecaster_multiseries
    if method == 'backtesting':
        
        def _objective(
            trial,
            search_space          = search_space,
            forecaster            = forecaster,
            series                = series,
            exog                  = exog,
            steps                 = steps,
            levels                = levels,
            metric                = metric,
            add_aggregated_metric = add_aggregated_metric,
            aggregate_metric      = aggregate_metric,
            metric_names          = metric_names,
            initial_train_size    = initial_train_size,
            fixed_train_size      = fixed_train_size,
            gap                   = gap,
            skip_folds            = skip_folds,
            allow_incomplete_fold = allow_incomplete_fold,
            refit                 = refit,
            n_jobs                = n_jobs,
            verbose               = verbose,
            suppress_warnings     = suppress_warnings
        ) -> float:
            
            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != 'lags'}
            forecaster.set_params(sample_params)
            if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
                if "lags" in sample:
                    forecaster.set_lags(sample['lags'])
            
            metrics, _ = backtesting_forecaster_multiseries(
                             forecaster            = forecaster,
                             series                = series,
                             exog                  = exog,
                             steps                 = steps,
                             levels                = levels,
                             metric                = metric,
                             add_aggregated_metric = add_aggregated_metric,
                             initial_train_size    = initial_train_size,
                             fixed_train_size      = fixed_train_size,
                             gap                   = gap,
                             skip_folds            = skip_folds,
                             allow_incomplete_fold = allow_incomplete_fold,
                             refit                 = refit,
                             n_jobs                = n_jobs,
                             verbose               = verbose,
                             show_progress         = False,
                             suppress_warnings     = suppress_warnings
                         )

            if add_aggregated_metric:
                metrics = metrics.loc[metrics['levels'].isin(aggregate_metric), :]
            else:
                metrics = metrics.loc[metrics['levels'] == levels[0], :]
            metrics = pd.DataFrame(
                        data    = [metrics.iloc[:, 1:].transpose().stack().to_numpy()],
                        columns = metric_names
                    )
            
            # Store metrics in the variable `metrics_list` defined outside _objective.
            nonlocal metrics_list
            metrics_list.append(metrics)

            return metrics.loc[0, metric_names[0]]
    
    else:

        def _objective(
            trial,
            search_space          = search_space,
            forecaster            = forecaster,
            series                = series,
            exog                  = exog,
            levels                = levels,
            metric                = metric,
            add_aggregated_metric = add_aggregated_metric,
            aggregate_metric      = aggregate_metric,
            metric_names          = metric_names,
            initial_train_size    = initial_train_size
        ) -> float:
            
            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != 'lags'}
            forecaster.set_params(sample_params)
            if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
                if "lags" in sample:
                    forecaster.set_lags(sample['lags'])
            
            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_train_encoding,
                X_test_encoding
            ) = forecaster._train_test_split_one_step_ahead(
                series=series, exog=exog, initial_train_size=initial_train_size,
            )

            metrics, _ = _predict_and_calculate_metrics_multiseries_one_step_ahead(
                forecaster            = forecaster,
                series                = series,
                X_train               = X_train,
                y_train               = y_train,
                X_train_encoding      = X_train_encoding,
                X_test                = X_test,
                y_test                = y_test,
                X_test_encoding       = X_test_encoding,
                levels                = levels,
                metrics               = metric,
                add_aggregated_metric = add_aggregated_metric
            )

            if add_aggregated_metric:
                metrics = metrics.loc[metrics['levels'].isin(aggregate_metric), :]
            else:
                metrics = metrics.loc[metrics['levels'] == levels[0], :]
            metrics = pd.DataFrame(
                          data    = [metrics.iloc[:, 1:].transpose().stack().to_numpy()],
                          columns = metric_names
                      )
            
            # Store metrics in the variable `metrics_list` defined outside _objective.
            nonlocal metrics_list
            metrics_list.append(metrics)

            return metrics.loc[0, metric_names[0]]

    if show_progress:
        kwargs_study_optimize['show_progress_bar'] = True
    
    if output_file is not None:
        # Redirect optuna logging to file
        optuna.logging.disable_default_handler()
        logger = logging.getLogger('optuna')
        logger.setLevel(logging.INFO)
        for handler in logger.handlers.copy():
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
        handler = logging.FileHandler(output_file, mode="w")
        logger.addHandler(handler)
    else:
        logging.getLogger("optuna").setLevel(logging.WARNING)
        optuna.logging.disable_default_handler()

    # `metrics_list` will be modified inside _objective function. 
    # It is a trick to extract multiple values from _objective since
    # only the optimized value can be returned.
    metrics_list = []

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Choices for a categorical distribution should be*"
    )

    study = optuna.create_study(**kwargs_create_study)

    if 'sampler' not in kwargs_create_study.keys():
        study.sampler = TPESampler(seed=random_state)

    study.optimize(_objective, n_trials=n_trials, **kwargs_study_optimize)
    best_trial = study.best_trial

    if output_file is not None:
        handler.close()
       
    if search_space(best_trial).keys() != best_trial.params.keys():
        raise ValueError(
            (f"Some of the key values do not match the search_space key names.\n"
             f"  Search Space keys  : {list(search_space(best_trial).keys())}\n"
             f"  Trial objects keys : {list(best_trial.params.keys())}")
        )
    warnings.filterwarnings('default')
    
    lags_list = []
    params_list = []
    for trial in study.get_trials():
        regressor_params = {k: v for k, v in trial.params.items() if k != 'lags'}
        lags = trial.params.get(
                   'lags',
                   forecaster.lags if hasattr(forecaster, 'lags') else None
               )
        params_list.append(regressor_params)
        lags_list.append(lags)
    
    if type(forecaster).__name__ not in ['ForecasterAutoregMultiSeriesCustom',
                                         'ForecasterAutoregMultiVariate']:
        lags_list = [
            initialize_lags(forecaster_name=type(forecaster).__name__, lags = lag)[0]
            for lag in lags_list
        ]
    elif type(forecaster).__name__ == 'ForecasterAutoregMultiSeriesCustom':
        lags_list = [
            f"custom function: {forecaster.fun_predictors.__name__}"
            for _ in lags_list
        ]
    else:
        lags_list_initialized = []
        for lags in lags_list:
            if isinstance(lags, dict):
                for key in lags:
                    if lags[key] is None:
                        lags[key] = None
                    else:
                        lags[key] = initialize_lags(
                                        forecaster_name = type(forecaster).__name__,
                                        lags            = lags[key]
                                    )[0]
            else:
                lags = initialize_lags(
                           forecaster_name = type(forecaster).__name__,
                           lags            = lags
                       )[0]
            lags_list_initialized.append(lags)
        
        lags_list = lags_list_initialized

    results = pd.concat(metrics_list, axis=0)
    results.insert(0, 'levels', [levels] * len(results))
    results.insert(1, 'lags', lags_list)
    results.insert(2, 'params', params_list)
    results = (
        results
        .sort_values(by=metric_names[0], ascending=True)
        .reset_index(drop=True)
    )
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results.loc[0, 'lags']
        best_params = results.loc[0, 'params']
        best_metric = results.loc[0, metric_names[0]]
        
        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
            f"  Levels: {levels}\n"
        )

    set_skforecast_warnings(suppress_warnings, action='default')
            
    return results, best_trial


def select_features_multiseries(
    forecaster: object,
    selector: object,
    series: Union[pd.DataFrame, dict],
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    select_only: Optional[str] = None,
    force_inclusion: Optional[Union[list, str]] = None,
    subsample: Union[int, float] = 0.5,
    random_state: int = 123,
    verbose: bool = True,
) -> Union[list, list]:
    """
    Feature selection using any of the sklearn.feature_selection module selectors 
    (such as `RFECV`, `SelectFromModel`, etc.). Two groups of features are
    evaluated: autoregressive features and exogenous features. By default, the 
    selection process is performed on both sets of features at the same time, 
    so that the most relevant autoregressive and exogenous features are selected. 
    However, using the `select_only` argument, the selection process can focus 
    only on the autoregressive or exogenous features without taking into account 
    the other features. Therefore, all other features will remain in the model. 
    It is also possible to force the inclusion of certain features in the final 
    list of selected features using the `force_inclusion` parameter.

    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiseriesCustom
        Forecaster model.
    selector : object
        A feature selector from sklearn.feature_selection.
    series : pandas DataFrame
        Target time series to which the feature selection will be applied.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    select_only : str, default `None`
        Decide what type of features to include in the selection process. 
        
        - If `'autoreg'`, only autoregressive features (lags or custom 
        predictors) are evaluated by the selector. All exogenous features are 
        included in the output (`selected_exog`).
        - If `'exog'`, only exogenous features are evaluated without the presence
        of autoregressive features. All autoregressive features are included 
        in the output (`selected_autoreg`).
        - If `None`, all features are evaluated by the selector.
    force_inclusion : list, str, default `None`
        Features to force include in the final list of selected features.
        
        - If `list`, list of feature names to force include.
        - If `str`, regular expression to identify features to force include. 
        For example, if `force_inclusion="^sun_"`, all features that begin 
        with "sun_" will be included in the final list of selected features.
    subsample : int, float, default `0.5`
        Proportion of records to use for feature selection.
    random_state : int, default `123`
        Sets a seed for the random subsample so that the subsampling process 
        is always deterministic.
    verbose : bool, default `True`
        Print information about feature selection process.

    Returns
    -------
    selected_autoreg : list
        List of selected autoregressive features.
    selected_exog : list
        List of selected exogenous features.

    """

    valid_forecasters = [
        'ForecasterAutoregMultiSeries',
        'ForecasterAutoregMultiSeriesCustom',
    ]

    if type(forecaster).__name__ not in valid_forecasters:
        raise TypeError(
            f"`forecaster` must be one of the following classes: {valid_forecasters}."
        )
    
    if select_only not in ['autoreg', 'exog', None]:
        raise ValueError(
            "`select_only` must be one of the following values: 'autoreg', 'exog', None."
        )

    if subsample <= 0 or subsample > 1:
        raise ValueError(
            "`subsample` must be a number greater than 0 and less than or equal to 1."
        )
    
    forecaster = deepcopy(forecaster)
    forecaster.fitted = False
    output = forecaster._create_train_X_y(series=series, exog=exog)
    X_train = output[0]
    y_train = output[1]
    series_col_names = output[3]

    if forecaster.encoding == 'onehot':
        encoding_cols = series_col_names
    else:
        encoding_cols = ['_level_skforecast']

    if hasattr(forecaster, 'lags'):
        autoreg_cols = [f"lag_{lag}" for lag in forecaster.lags]
    else:
        if forecaster.name_predictors is not None:
            autoreg_cols = forecaster.name_predictors
        else:
            autoreg_cols = [
                col
                for col in X_train.columns
                if re.match(r'^custom_predictor_\d+', col)
            ]
    exog_cols = [
        col
        for col in X_train.columns
        if col not in autoreg_cols and col not in encoding_cols
    ]

    forced_autoreg = []
    forced_exog = []
    if force_inclusion is not None:
        if isinstance(force_inclusion, list):
            forced_autoreg = [col for col in force_inclusion if col in autoreg_cols]
            forced_exog = [col for col in force_inclusion if col in exog_cols]
        elif isinstance(force_inclusion, str):
            forced_autoreg = [col for col in autoreg_cols if re.match(force_inclusion, col)]
            forced_exog = [col for col in exog_cols if re.match(force_inclusion, col)]

    if select_only == 'autoreg':
        X_train = X_train.drop(columns=exog_cols + encoding_cols)
    elif select_only == 'exog':
        X_train = X_train.drop(columns=autoreg_cols + encoding_cols)
    else:
        X_train = X_train.drop(columns=encoding_cols)

    if isinstance(subsample, float):
        subsample = int(len(X_train) * subsample)

    rng = np.random.default_rng(seed=random_state)
    sample = rng.choice(X_train.index, size=subsample, replace=False)
    X_train_sample = X_train.loc[sample, :]
    y_train_sample = y_train.loc[sample]
    selector.fit(X_train_sample, y_train_sample)
    selected_features = selector.get_feature_names_out()

    if select_only == 'exog':
        selected_autoreg = autoreg_cols
    else:
        selected_autoreg = [
            feature
            for feature in selected_features
            if feature in autoreg_cols
        ]

    if select_only == 'autoreg':
        selected_exog = exog_cols
    else:
        selected_exog = [
            feature
            for feature in selected_features
            if feature in exog_cols
        ]

    if force_inclusion is not None: 
        if select_only != 'autoreg':
            forced_exog_not_selected = set(forced_exog) - set(selected_features)
            selected_exog.extend(forced_exog_not_selected)
            selected_exog.sort(key=exog_cols.index)
        if select_only != 'exog':
            forced_autoreg_not_selected = set(forced_autoreg) - set(selected_features)
            selected_autoreg.extend(forced_autoreg_not_selected)
            selected_autoreg.sort(key=autoreg_cols.index)

    if len(selected_autoreg) == 0:
        warnings.warn(
            ("No autoregressive features have been selected. Since a Forecaster "
             "cannot be created without them, be sure to include at least one "
             "using the `force_inclusion` parameter.")
        )
    else:
        if hasattr(forecaster, 'lags'):
            selected_autoreg = [int(feature.replace('lag_', '')) 
                                for feature in selected_autoreg] 

    if verbose:
        print(f"Recursive feature elimination ({selector.__class__.__name__})")
        print("--------------------------------" + "-" * len(selector.__class__.__name__))
        print(f"Total number of records available: {X_train.shape[0]}")
        print(f"Total number of records used for feature selection: {X_train_sample.shape[0]}")
        print(f"Number of features available: {len(autoreg_cols) + len(exog_cols)}") 
        print(f"    Autoreg (n={len(autoreg_cols)})")
        print(f"    Exog    (n={len(exog_cols)})")
        print(f"Number of features selected: {len(selected_features)}")
        print(f"    Autoreg (n={len(selected_autoreg)}) : {selected_autoreg}")
        print(f"    Exog    (n={len(selected_exog)}) : {selected_exog}")

    return selected_autoreg, selected_exog