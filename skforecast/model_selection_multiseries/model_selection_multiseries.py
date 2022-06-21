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
import optuna
from optuna.samplers import TPESampler, RandomSampler
optuna.logging.set_verbosity(optuna.logging.WARNING) # disable optuna logs
from skopt.utils import use_named_args
from skopt import gp_minimize

from ..ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

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
    level: str,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[float, pd.DataFrame]:
    '''
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

    level : str
        Time series to be predicted.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.
    
    initial_train_size : int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
        
    fixed_train_size : bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.
        
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
    metric_value : float
        Value of the metric.

    backtest_predictions : pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    '''

    forecaster = deepcopy(forecaster)
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
    backtest_predictions = []
    
    folds = int(np.ceil((len(series.index) - initial_train_size) / steps))
    remainder = (len(series.index) - initial_train_size) % steps
    
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
        
    if folds > 50:
        print(
            f"Forecaster will be fit {folds} times. This can take substantial amounts of time. "
            f"If not feasible, try with `refit = False`. \n"
        )

    for i in range(folds):
        # In each iteration (except the last one) the model is fitted before making predictions.
        if fixed_train_size:
            # The train size doesn't increases but moves by `steps` in each iteration.
            train_idx_start = i * steps
        else:
            # The train size increases by `steps` in each iteration.
            train_idx_start = 0
        train_idx_end = initial_train_size + i * steps
            
        if exog is not None:
            next_window_exog = exog.iloc[train_idx_end:train_idx_end + steps, ]

        if interval is None:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(
                        series = series.iloc[train_idx_start:train_idx_end, ],
                        store_in_sample_residuals = False
                    )
                    pred = forecaster.predict(steps=steps, level=level)
                else:
                    forecaster.fit(
                        series = series.iloc[train_idx_start:train_idx_end, ], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ],
                        store_in_sample_residuals = False
                    )
                    pred = forecaster.predict(steps=steps, level=level, exog=next_window_exog)
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end, ],
                            store_in_sample_residuals = False
                        )
                        pred = forecaster.predict(steps=steps, level=level)
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end, ], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ],
                            store_in_sample_residuals = False
                        )
                        pred = forecaster.predict(steps=steps, level=level, exog=next_window_exog)
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end],
                            store_in_sample_residuals = False
                        )
                        pred = forecaster.predict(steps=steps, level=level)
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ],
                            store_in_sample_residuals = False
                        )
                        pred = forecaster.predict(steps=steps, level=level, exog=next_window_exog)
        else:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(
                        series = series.iloc[train_idx_start:train_idx_end],
                        store_in_sample_residuals = True
                    )
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
                else:
                    forecaster.fit(
                        series = series.iloc[train_idx_start:train_idx_end], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ],
                        store_in_sample_residuals = True
                    )
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end],
                            store_in_sample_residuals = True
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ],
                            store_in_sample_residuals = True
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end],
                            store_in_sample_residuals = True
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ],
                            store_in_sample_residuals = True
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )

        backtest_predictions.append(pred)
    
    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    metric_value = metric(
                    y_true = series[level].iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return metric_value, backtest_predictions


def _backtesting_forecaster_multiseries_no_refit(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    level: str,
    metric: Union[str, callable],
    initial_train_size: Optional[int]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[float, pd.DataFrame]:
    '''
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

    level : str
        Time series to be predicted.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.
    
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.
        
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
    metric_value : float
        Value of the metric.

    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    '''

    forecaster = deepcopy(forecaster)
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
    backtest_predictions = []
    
    if initial_train_size is not None:
        if exog is None:
            forecaster.fit(
                series = series.iloc[:initial_train_size, ],
                store_in_sample_residuals = True
            )      
        else:
            forecaster.fit(
                series = series.iloc[:initial_train_size, ],
                exog = exog.iloc[:initial_train_size, ],
                store_in_sample_residuals = True
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
        last_window_end   = initial_train_size + i * steps
        last_window_start = last_window_end - window_size 
        last_window_y     = series[[level]].iloc[last_window_start:last_window_end, ]
        if exog is not None:
            next_window_exog = exog.iloc[last_window_end:last_window_end + steps, ]
    
        if interval is None:  

            if i < folds - 1: 
                if exog is None:
                    pred = forecaster.predict(
                                steps       = steps,
                                level       = level,
                                last_window = last_window_y
                           )
                else:
                    pred = forecaster.predict(
                                steps       = steps,
                                level       = level,
                                last_window = last_window_y,
                                exog        = next_window_exog
                           )            
            else:    
                if remainder == 0:
                    if exog is None:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    level       = level,
                                    last_window = last_window_y
                               )
                    else:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    level       = level,
                                    last_window = last_window_y,
                                    exog        = next_window_exog
                               )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    level       = level,
                                    last_window = last_window_y
                               )
                    else:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    level       = level,
                                    last_window = last_window_y,
                                    exog        = next_window_exog
                               )
            
            backtest_predictions.append(pred)

        else:
            if i < folds - 1:
                if exog is None:
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                last_window  = last_window_y,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
                else:
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                last_window  = last_window_y,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )            
            else:    
                if remainder == 0:
                    if exog is None:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    level        = level,
                                    last_window  = last_window_y,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                               )
                    else:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    level        = level,
                                    last_window  = last_window_y,
                                    exog         = next_window_exog,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                               )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    level        = level,
                                    last_window  = last_window_y,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                               )
                    else:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    level        = level,
                                    last_window  = last_window_y,
                                    exog         = next_window_exog,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                               )
            
            backtest_predictions.append(pred)

    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    metric_value = metric(
                    y_true = series[level].iloc[initial_train_size : initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return metric_value, backtest_predictions


def backtesting_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    level: str,
    metric: Union[str, callable],
    initial_train_size: Optional[int],
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[float, pd.DataFrame]:
    '''
    Backtesting of forecaster multi-series model.

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

    level : str
        Time series to be predicted.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.
    
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already 
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

        `None` is only allowed when `refit` is `False`.
    
    fixed_train_size : bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.
        
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
    metric_value : float
        Value of the metric.

    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    '''

    if initial_train_size is not None and initial_train_size > len(series):
        raise Exception(
            'If used, `initial_train_size` must be smaller than length of `series`.'
        )
        
    if initial_train_size is not None and initial_train_size < forecaster.window_size:
        raise Exception(
            f"`initial_train_size` must be greater than "
            f"forecaster's window_size ({forecaster.window_size})."
        )

    if initial_train_size is None and not forecaster.fitted:
        raise Exception(
            '`forecaster` must be already trained if no `initial_train_size` is provided.'
        )

    if not isinstance(refit, bool):
        raise Exception(
            f'`refit` must be boolean: True, False.'
        )

    if initial_train_size is None and refit:
        raise Exception(
            f'`refit` is only allowed when there is a initial_train_size.'
        )

    if not isinstance(forecaster, ForecasterAutoregMultiSeries):
        raise Exception(
            ('`forecaster` must be of type `ForecasterAutoregMultiSeries`, for all other '
             'types of forecasters use the functions in `model_selection` module.')
        )
    
    if refit:
        metric_value, backtest_predictions = _backtesting_forecaster_multiseries_refit(
            forecaster          = forecaster,
            series              = series,
            steps               = steps,
            level               = level,
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
        metric_value, backtest_predictions = _backtesting_forecaster_multiseries_no_refit(
            forecaster          = forecaster,
            series              = series,
            steps               = steps,
            level               = level,
            metric              = metric,
            initial_train_size  = initial_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            random_state        = random_state,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )

    return metric_value, backtest_predictions


def grid_search_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels_list: list=None,
    levels_weights: dict=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    '''
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
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    levels_list : list, default `None`
        levels on which the forecaster is optimized. If `None`, all levels are 
        taken into acount.The resulting metric will be a weighted average of the 
        optimization of all levels. See also `levels_weights`.

    levels_weights : dict, default `None`
        Weights associated with levels in the form `{level: weight}`. 
        If `None`, all levels have the same weight.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg`, `ForecasterAutoregDirect` or `ForecasterAutoregMultiOutput`.
        
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
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.
    '''

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster          = forecaster,
        series              = series,
        param_grid          = param_grid,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        levels_list         = levels_list,
        levels_weights      = levels_weights,
        exog                = exog,
        lags_grid           = lags_grid,
        refit               = refit,
        return_best         = return_best,
        verbose             = verbose
    )

    return results


def random_search_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_distributions: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels_list: list=None,
    levels_weights: dict=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    '''
    Random search over specified parameter values or distributions for a Forecaster object.
    Validation is done using multi-series backtesting.
    
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
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    levels_list : list, default `None`
        levels on which the forecaster is optimized. If `None`, all levels are 
        taken into acount.The resulting metric will be a weighted average of the 
        optimization of all levels. See also `levels_weights`.

    levels_weights : dict, default `None`
        Weights associated with levels in the form `{level: weight}`. 
        If `None`, all levels have the same weight.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg`, `ForecasterAutoregDirect` or `ForecasterAutoregMultiOutput`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.

    n_iter : int, default `10`
        Number of parameter settings that are sampled. 
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
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.
    '''

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster          = forecaster,
        series              = series,
        param_grid          = param_grid,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        levels_list         = levels_list,
        levels_weights      = levels_weights,
        exog                = exog,
        lags_grid           = lags_grid,
        refit               = refit,
        return_best         = return_best,
        verbose             = verbose
    )

    return results


def _evaluate_grid_hyperparameters_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    levels_list: list=None,
    levels_weights: dict=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    '''
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
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    levels_list : list, default `None`
        levels on which the forecaster is optimized. If `None`, all levels are 
        taken into acount.The resulting metric will be a weighted average of the 
        optimization of all levels. See also `levels_weights`.

    levels_weights : dict, default `None`
        Weights associated with levels in the form `{level: weight}`. 
        If `None`, all levels have the same weight.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg`, `ForecasterAutoregDirect` or `ForecasterAutoregMultiOutput`.
        
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
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.
    '''

    if levels_list is not None and not isinstance(levels_list, list):
        raise Exception(
            f'`levels_list` must be a list of column names or `None`.'
        )

    if levels_weights is not None and not isinstance(levels_weights, dict):
        raise Exception(
            f'`levels_weights` must be a dict or `None`.'
        )

    if levels_list is None:
        levels_list = list(series.columns)

    if levels_weights is None:
        levels_weights = {col: 1./len(levels_list) for col in levels_list}

    if levels_list != list(levels_weights.keys()):
        raise Exception(
            f'`levels_weights` keys must be the same as the column names of series, `levels_list`.'
        )

    if sum(levels_weights.values()) != 1.0:
        raise Exception(
            f'Weights in `levels_weights` must add up to 1.0.'
        )

    if lags_grid is None:
        lags_grid = [forecaster.lags]
   
    lags_list = []
    params_list = []
    metric_list = []

    print(
        f"Number of models compared: {len(param_grid)*len(lags_grid)*len(levels_list)}."
    )

    for lags in tqdm(lags_grid, desc='loop lags_grid', position=0, ncols=90):
        
        forecaster.set_lags(lags)
        lags = forecaster.lags.copy()
        
        for params in tqdm(param_grid, desc='loop param_grid', position=1, leave=False, ncols=90):

            forecaster.set_params(**params)

            metric_level_list = []

            for level in levels_list:

                metric_level = backtesting_forecaster_multiseries(
                                    forecaster         = forecaster,
                                    series             = series,
                                    steps              = steps,
                                    level              = level,
                                    metric             = metric,
                                    initial_train_size = initial_train_size,
                                    fixed_train_size   = fixed_train_size,
                                    exog               = exog,
                                    refit              = refit,
                                    interval           = None,
                                    verbose            = verbose
                               )[0]

                metric_level_list.append(metric_level*levels_weights[level])

            lags_list.append(lags)
            params_list.append(params)
            metric_list.append(sum(metric_level_list))
            
    results = pd.DataFrame({
                'levels': [levels_list]*len(lags_list),
                'lags'  : lags_list,
                'params': params_list,
                'metric': metric_list})
    
    results = results.sort_values(by='metric', ascending=True)
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results['lags'].iloc[0]
        best_params = results['params'].iloc[0]
        best_metric = results['metric'].iloc[0]
        
        forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
            f"  Levels: {results['levels'].iloc[0]} \n"
            f"  Levels weights: {levels_weights} \n"
        )
            
    return results