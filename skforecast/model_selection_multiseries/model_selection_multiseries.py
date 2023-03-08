################################################################################
#                  skforecast.model_selection_multiseries                      #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Any
import numpy as np
import pandas as pd
import warnings
import logging
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn.exceptions import NotFittedError

from ..model_selection.model_selection import _get_metric
from ..model_selection.model_selection import _backtesting_forecaster_verbose

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def _backtesting_forecaster_multiseries_refit(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster model with a re-fitting strategy. A copy of the  
    original forecaster is created so it is not modified during the process.
    
    In each iteration:
        - Fit forecaster with the training set.
        - A number of `steps` ahead are predicted.
        - The training set increases with `steps` observations.
        - The model is re-fitted using the new training set.

    In order to apply backtesting with re-fit, an initial training set must be
    available, otherwise it would not be possible to increase the training set 
    after each iteration. `initial_train_size` must be provided.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forecaster model.
        
    series : pandas DataFrame
        Training time series.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.
    
    initial_train_size : int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
        
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals. If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.

    backtest_predictions : pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`.
        If there is more than one level, this structure will be repeated for each of them.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    
    """

    forecaster = deepcopy(forecaster)

    if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate':
        levels = [forecaster.level]
    else:
        if levels is None:
            # Forecaster can be not fitted, so cannot use self.series_col_names
            levels = list(series.columns) 
        elif isinstance(levels, str):
            levels = [levels]

    if isinstance(metric, str):
        metrics = [_get_metric(metric=metric)]
    elif isinstance(metric, list):
        metrics = [_get_metric(metric=m) if isinstance(m, str) else m for m in metric]
    else:
        metrics = [metric]
    
    backtest_predictions = []
    
    folds = int(np.ceil((len(series.index) - initial_train_size) / steps))
    remainder = (len(series.index) - initial_train_size) % steps

    if type(forecaster).__name__ != 'ForecasterAutoregMultiVariate' and folds > 50:
        warnings.warn(
            (f"The forecaster will be fit {folds} times. This can take substantial amounts of time. "
             f"If not feasible, try with `refit = False`. \n"),
            RuntimeWarning
        )
    elif type(forecaster).__name__ == 'ForecasterAutoregMultiVariate' and folds*forecaster.steps > 50:
        warnings.warn(
            (f"The forecaster will be fit {folds*forecaster.steps} times ({folds} folds * {forecaster.steps} regressors). "
             f"This can take substantial amounts of time. If not feasible, try with `refit = False`. \n"),
             RuntimeWarning
        )
    
    if verbose:
        _backtesting_forecaster_verbose(
            index_values       = series.index,
            steps              = steps,
            initial_train_size = initial_train_size,
            folds              = folds,
            remainder          = remainder,
            refit              = True,
            fixed_train_size   = fixed_train_size
        )

    store_in_sample_residuals = False if interval is None else True

    for i in range(folds):
        # In each iteration the model is fitted before making predictions.
        # if fixed_train_size the train size doesn't increase but moves by `steps` in each iteration.
        # if false the train size increases by `steps` in each iteration.
        train_idx_start = i * steps if fixed_train_size else 0
        train_idx_end = initial_train_size + i * steps
        
        exog_train_values = exog.iloc[train_idx_start:train_idx_end, ] if exog is not None else None
        next_window_exog = exog.iloc[train_idx_end:train_idx_end + steps, ] if exog is not None else None

        forecaster.fit(
            series                    = series.iloc[train_idx_start:train_idx_end, ], 
            exog                      = exog_train_values,
            store_in_sample_residuals = store_in_sample_residuals
        )

        if i == folds - 1: # last fold
            # If remainder > 0, only the remaining steps need to be predicted
            steps = steps if remainder == 0 else remainder

        if interval is None:
            pred = forecaster.predict(
                       steps       = steps, 
                       levels      = levels, 
                       exog        = next_window_exog
                   )
        else:
            pred = forecaster.predict_interval(
                       steps               = steps,
                       levels              = levels, 
                       exog                = next_window_exog,
                       interval            = interval,
                       n_boot              = n_boot,
                       random_state        = random_state,
                       in_sample_residuals = in_sample_residuals
                   )

        backtest_predictions.append(pred)
    
    backtest_predictions = pd.concat(backtest_predictions)

    metrics_levels = []

    for level in levels:
        metrics_values = [m(
                            y_true = series[level].iloc[initial_train_size:initial_train_size + len(backtest_predictions)],
                            y_pred = backtest_predictions[level]
                          ) for m in metrics
                         ]
 
        metrics_levels.append(metrics_values)

    metrics_levels = pd.concat([pd.DataFrame({'levels': levels}), 
                                pd.DataFrame(data    = metrics_levels,
                                             columns = [m if isinstance(m, str) else m.__name__ for m in metrics])],
                               axis=1)

    return metrics_levels, backtest_predictions


def _backtesting_forecaster_multiseries_no_refit(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: Optional[int]=None,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster without iterative re-fitting. In each iteration,
    a number of `steps` are predicted. A copy of the original forecaster is
    created so it is not modified during the process.

    If `forecaster` is already trained and `initial_train_size` is `None`,
    no initial train is done and all data is used to evaluate the model.
    However, the first `len(forecaster.last_window)` observations are needed
    to create the initial predictors, so no predictions are calculated for them.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forecaster model.
        
    series : pandas DataFrame
        Training time series.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.
    
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.

    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
        If there is more than one level, this structure will be repeated for each of them.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    
    """

    forecaster = deepcopy(forecaster)

    if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate':
        levels = [forecaster.level]
    else:
        if levels is None:
            # Forecaster can be not fitted, so cannot use self.series_col_names
            levels = list(series.columns) 
        elif isinstance(levels, str):
            levels = [levels]

    if isinstance(metric, str):
        metrics = [_get_metric(metric=metric)]
    elif isinstance(metric, list):
        metrics = [_get_metric(metric=m) if isinstance(m, str) else m for m in metric]
    else:
        metrics = [metric]
    
    backtest_predictions = []
    
    if initial_train_size is not None:
        exog_train_values = exog.iloc[:initial_train_size, ] if exog is not None else None
        store_in_sample_residuals = False if interval is None else True
        forecaster.fit(
            series                    = series.iloc[:initial_train_size, ],
            exog                      = exog_train_values,
            store_in_sample_residuals = store_in_sample_residuals
        )
        window_size = forecaster.window_size
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        window_size = forecaster.window_size
        initial_train_size = window_size

    folds     = int(np.ceil((len(series.index) - initial_train_size) / steps))
    remainder = (len(series.index) - initial_train_size) % steps

    if verbose:
        _backtesting_forecaster_verbose(
            index_values       = series.index,
            steps              = steps,
            initial_train_size = initial_train_size,
            folds              = folds,
            remainder          = remainder,
            refit              = False
        )

    for i in range(folds):
        # Since the model is only fitted with the initial_train_size, last_window
        # and next_window_exog must be updated to include the data needed to make
        # predictions.
        last_window_end    = initial_train_size + i * steps
        last_window_start  = last_window_end - window_size 
        last_window_series = series.iloc[last_window_start:last_window_end, ]

        next_window_exog = exog.iloc[last_window_end:last_window_end + steps, ] if exog is not None else None

        if i == folds - 1: # last fold
            # If remainder > 0, only the remaining steps need to be predicted
            steps = steps if remainder == 0 else remainder

        if interval is None:
            pred = forecaster.predict(
                       steps       = steps,
                       levels      = levels, 
                       last_window = last_window_series,
                       exog        = next_window_exog
                   )
        else:
            pred = forecaster.predict_interval(
                       steps               = steps,
                       levels              = levels, 
                       last_window         = last_window_series,
                       exog                = next_window_exog,
                       interval            = interval,
                       n_boot              = n_boot,
                       random_state        = random_state,
                       in_sample_residuals = in_sample_residuals
                   )
            
        backtest_predictions.append(pred)

    backtest_predictions = pd.concat(backtest_predictions)

    metrics_levels = []

    for level in levels:
        metrics_values = [m(
                            y_true = series[level].iloc[initial_train_size:initial_train_size + len(backtest_predictions)],
                            y_pred = backtest_predictions[level]
                          ) for m in metrics
                         ]
 
        metrics_levels.append(metrics_values)

    metrics_levels = pd.concat([pd.DataFrame({'levels': levels}), 
                                pd.DataFrame(data    = metrics_levels,
                                             columns = [m if isinstance(m, str) else m.__name__ for m in metrics])],
                               axis=1)                 

    return metrics_levels, backtest_predictions


def backtesting_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: Optional[int],
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting for multi-series and multivariate forecasters.

    If `refit` is False, the model is trained only once using the `initial_train_size`
    first observations. If `refit` is True, the model is trained in each iteration
    increasing the training set. A copy of the original forecaster is created so 
    it is not modified during the process.

    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forecaster model.

    series : pandas DataFrame
        Training time series.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.
    
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already 
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

        `None` is only allowed when `refit` is `False`.
    
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
        **New in version 0.6.0**
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration.

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
                  
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.

    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
        If there is more than one level, this structure will be repeated for each of them.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    
    """

    if initial_train_size is not None and not isinstance(initial_train_size, (int, np.int64, np.int32)):
        raise TypeError(
            (f'If used, `initial_train_size` must be an integer greater than '
             f'the window_size of the forecaster. Got {type(initial_train_size)}.')
        )

    if initial_train_size is not None and initial_train_size >= len(series):
        raise ValueError(
            (f'If used, `initial_train_size` must be an integer '
             f'smaller than the length of `series` ({len(series)}).')
        )
        
    if initial_train_size is not None and initial_train_size < forecaster.window_size:
        raise ValueError(
            (f'If used, `initial_train_size` must be an integer greater than '
             f'the window_size of the forecaster ({forecaster.window_size}).')
        )

    if initial_train_size is None and not forecaster.fitted:
        raise NotFittedError(
            '`forecaster` must be already trained if no `initial_train_size` is provided.'
        )

    if not isinstance(refit, bool):
        raise TypeError(
            f'`refit` must be boolean: `True`, `False`.'
        )

    if initial_train_size is None and refit:
        raise ValueError(
            f'`refit` is only allowed when `initial_train_size` is not `None`.'
        )

    if type(forecaster).__name__ not in ['ForecasterAutoregMultiSeries', 
                                         'ForecasterAutoregMultiSeriesCustom', 
                                         'ForecasterAutoregMultiVariate']:
        raise TypeError(
            ("`forecaster` must be of type `ForecasterAutoregMultiSeries`, "
             "`ForecasterAutoregMultiSeriesCustom` or `ForecasterAutoregMultiVariate`, "
             "for all other types of forecasters use the functions available in "
             f"the `model_selection` module. Got {type(forecaster).__name__}")
        )

    if type(forecaster).__name__ in ['ForecasterAutoregMultiSeries', 
                                     'ForecasterAutoregMultiSeriesCustom'] \
        and levels is not None and not isinstance(levels, (str, list)):
        raise TypeError(
            (f"`levels` must be a `list` of column names, a `str` of a column name "
             f"or `None` when using a `ForecasterAutoregMultiSeries` or "
             f"`ForecasterAutoregMultiSeriesCustom`. If the forecaster is of type "
             f"`ForecasterAutoregMultiVariate`, this argument is ignored.")
        )

    if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate' \
        and levels and levels != forecaster.level and levels != [forecaster.level]:
        warnings.warn(
            (f"`levels` argument have no use when the forecaster is of type "
             f"`ForecasterAutoregMultiVariate`. The level of this forecaster is "
             f"{forecaster.level}, to predict another level, change the `level` "
             f"argument when initializing the forecaster.")
        )
    
    if refit:
        metrics_levels, backtest_predictions = _backtesting_forecaster_multiseries_refit(
            forecaster          = forecaster,
            series              = series,
            steps               = steps,
            levels              = levels,
            metric              = metric,
            initial_train_size  = initial_train_size,
            fixed_train_size    = fixed_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            random_state        = random_state,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )
    else:
        metrics_levels, backtest_predictions = _backtesting_forecaster_multiseries_no_refit(
            forecaster          = forecaster,
            series              = series,
            steps               = steps,
            levels              = levels,
            metric              = metric,
            initial_train_size  = initial_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            random_state        = random_state,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )

    return metrics_levels, backtest_predictions


def grid_search_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forcaster model.
        
    series : pandas DataFrame
        Training time series.
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoregMultiSeries` or `ForecasterAutoregMultiVariate`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results : pandas DataFrame
        Results for each combination of parameters.
            column levels = levels.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric(s) value(s) estimated for each combination of parameters. The resulting metric will be
                            the average of the optimization of all levels.
            additional n columns with param = value.
    
    """

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        param_grid         = param_grid,
        steps              = steps,
        metric             = metric,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        levels             = levels,
        exog               = exog,
        lags_grid          = lags_grid,
        refit              = refit,
        return_best        = return_best,
        verbose            = verbose
    )

    return results


def random_search_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_distributions: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forcaster model.
        
    series : pandas DataFrame
        Training time series.
        
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.

    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoregMultiSeries` or `ForecasterAutoregMultiVariate`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.

    n_iter : int, default `10`
        Number of parameter settings that are sampled per lags configuration. 
        n_iter trades off runtime vs quality of the solution.

    random_state : int, default `123`
        Sets a seed to the random sampling for reproducible output.

    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results : pandas DataFrame
        Results for each combination of parameters.
            column levels = levels.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric(s) value(s) estimated for each combination of parameters. The resulting metric will be
                            the average of the optimization of all levels.
            additional n columns with param = value.
    
    """

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        param_grid         = param_grid,
        steps              = steps,
        metric             = metric,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        levels             = levels,
        exog               = exog,
        lags_grid          = lags_grid,
        refit              = refit,
        return_best        = return_best,
        verbose            = verbose
    )

    return results


def _evaluate_grid_hyperparameters_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forcaster model.

    series : pandas DataFrame
        Training time series.
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoregMultiSeries` or `ForecasterAutoregMultiVariate`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results : pandas DataFrame
        Results for each combination of parameters.
            column levels = levels.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric(s) value(s) estimated for each combination of parameters.
            The resulting metric will be the average of the optimization of all levels.
            additional n columns with param = value.
    
    """

    if return_best and exog is not None and (len(exog) != len(series)):
        raise ValueError(
            f'`exog` must have same number of samples as `series`. '
            f'length `exog`: ({len(exog)}), length `series`: ({len(series)})'
        )

    if type(forecaster).__name__ in ['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiSeriesCustom']  \
        and levels is not None and not isinstance(levels, (str, list)):
        raise TypeError(
            f'`levels` must be a `list` of column names, a `str` of a column name or `None`.'
        )

    if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate':
        if levels and levels != forecaster.level and levels != [forecaster.level]:
            warnings.warn(
                (f"`levels` argument have no use when the forecaster is of type "
                 f"ForecasterAutoregMultiVariate. "
                 f"The level of this forecaster is {forecaster.level}, to predict another "
                 f"level, change the `level` argument when initializing the forecaster. \n")
            )
        levels = [forecaster.level]
    else:
        if levels is None:
            # Forecaster can be not fitted, so cannot use self.series_col_names
            levels = list(series.columns) 
        elif isinstance(levels, str):
            levels = [levels]

    if type(forecaster).__name__ == 'ForecasterAutoregMultiSeriesCustom':
        if lags_grid is not None:
            warnings.warn(
                '`lags_grid` ignored if forecaster is an instance of `ForecasterAutoregMultiSeriesCustom`.'
            )
        lags_grid = ['custom predictors']
    elif lags_grid is None:
        lags_grid = [forecaster.lags]
   
    lags_list = []
    params_list = []
    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            'When `metric` is a `list`, each metric name must be unique.'
        )

    print(
        f'{len(param_grid)*len(lags_grid)} models compared for {len(levels)} level(s). '
        f'Number of iterations: {len(param_grid)*len(lags_grid)}.'
    )

    for lags in tqdm(lags_grid, desc='loop lags_grid', position=0, ncols=90):

        if type(forecaster).__name__ in ['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiVariate']:
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        for params in tqdm(param_grid, desc='loop param_grid', position=1, leave=False, ncols=90):

            forecaster.set_params(params)
            metrics_levels = backtesting_forecaster_multiseries(
                                 forecaster         = forecaster,
                                 series             = series,
                                 steps              = steps,
                                 levels             = levels,
                                 metric             = metric,
                                 initial_train_size = initial_train_size,
                                 fixed_train_size   = fixed_train_size,
                                 exog               = exog,
                                 refit              = refit,
                                 interval           = None,
                                 verbose            = verbose
                             )[0]
            warnings.filterwarnings('ignore', category=RuntimeWarning, message= "The forecaster will be fit.*")
            lags_list.append(lags)
            params_list.append(params)
            for m in metric:
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(metrics_levels[m_name].mean())

    results = pd.DataFrame({
                  'levels': [levels]*len(lags_list),
                  'lags'  : lags_list,
                  'params': params_list,
                  **metric_dict
              })
    
    results = results.sort_values(by=list(metric_dict.keys())[0], ascending=True)
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results['lags'].iloc[0]
        best_params = results['params'].iloc[0]
        best_metric = results[list(metric_dict.keys())[0]].iloc[0]
        
        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)
        forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, and the whole data set: \n"
            f"  Lags: {best_lags}\n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
            f"  Levels: {results['levels'].iloc[0]}\n"
        )
            
    return results


# Alias MultiVariate
# ==============================================================================
def backtesting_forecaster_multivariate(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: Optional[int],
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function is an alias of backtesting_forecaster_multiseries.

    Backtesting for multi-series and multivariate forecasters.

    If `refit` is False, the model is trained only once using the `initial_train_size`
    first observations. If `refit` is True, the model is trained in each iteration
    increasing the training set. A copy of the original forecaster is created so 
    it is not modified during the process.

    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forecaster model.

    series : pandas DataFrame
        Training time series.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.
    
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already 
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

        `None` is only allowed when `refit` is `False`.
    
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
        **New in version 0.6.0**
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration.

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
                  
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.

    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
        If there is more than one level, this structure will be repeated for each of them.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    
    """

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
        forecaster          = forecaster,
        series              = series,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        levels              = levels,
        exog                = exog,
        refit               = refit,
        interval            = interval,
        n_boot              = n_boot,
        random_state        = random_state,
        in_sample_residuals = in_sample_residuals,
        verbose             = verbose
    )

    return metrics_levels, backtest_predictions


def grid_search_forecaster_multivariate(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    This function is an alias of grid_search_forecaster_multiseries.

    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forcaster model.
        
    series : pandas DataFrame
        Training time series.
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoregMultiSeries` or `ForecasterAutoregMultiVariate`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results : pandas DataFrame
        Results for each combination of parameters.
            column levels = levels.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric(s) value(s) estimated for each combination of parameters. The resulting metric will be
                            the average of the optimization of all levels.
            additional n columns with param = value.
    
    """

    results = grid_search_forecaster_multiseries(
        forecaster         = forecaster,
        series             = series,
        param_grid         = param_grid,
        steps              = steps,
        metric             = metric,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        levels             = levels,
        exog               = exog,
        lags_grid          = lags_grid,
        refit              = refit,
        return_best        = return_best,
        verbose            = verbose
    )

    return results


def random_search_forecaster_multivariate(
    forecaster,
    series: pd.DataFrame,
    param_distributions: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    This function is an alias of random_search_forecaster_multiseries.

    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries
        Forcaster model.
        
    series : pandas DataFrame
        Training time series.
        
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.

    steps : int
        Number of steps to predict.
        
    metric : str, callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoregMultiSeries` or `ForecasterAutoregMultiVariate`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.

    n_iter : int, default `10`
        Number of parameter settings that are sampled per lags configuration. 
        n_iter trades off runtime vs quality of the solution.

    random_state : int, default `123`
        Sets a seed to the random sampling for reproducible output.

    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results : pandas DataFrame
        Results for each combination of parameters.
            column levels = levels.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric(s) value(s) estimated for each combination of parameters. The resulting metric will be
                            the average of the optimization of all levels.
            additional n columns with param = value.
    
    """

    results = random_search_forecaster_multiseries(
        forecaster          = forecaster,
        series              = series,
        param_distributions = param_distributions,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        levels              = levels,
        exog                = exog,
        lags_grid           = lags_grid,
        refit               = refit,
        n_iter              = n_iter,
        random_state        = random_state,
        return_best         = return_best,
        verbose             = verbose
    ) 

    return results