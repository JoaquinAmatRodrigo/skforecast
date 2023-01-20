################################################################################
#                        skforecast.model_selection_sarimax                            #
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


def _backtesting_sarimax_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    alpha: Optional[float]=None,
    interval: Optional[list]=None,
    verbose: bool=False
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of ForecasterSarimax model with a re-fitting strategy. A copy of the  
    original forecaster is created so it is not modified during the process.
    
    In each iteration:
        - Fit forecaster with the training set.
        - A number of `steps` ahead are predicted.
        - The training set increases with `steps` observations.
        - The model is re-fitted using the new training set.

    In order to apply backtesting with refit, an initial training set must be
    available, otherwise it would not be possible to increase the training set 
    after each iteration. `initial_train_size` must be provided.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
        
    y : pandas Series
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
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    alpha : float, default `0.05`
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
        
    interval : list, default `None`
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metrics_value : float, list
        Value(s) of the metric(s).

    backtest_predictions : pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    
    """

    forecaster = deepcopy(forecaster)

    if isinstance(metric, str):
        metrics = _get_metric(metric=metric)
    elif isinstance(metric, list):
        metrics = [_get_metric(metric=m) if isinstance(m, str) else m for m in metric]
    else:
        metrics = metric

    backtest_predictions = []
    
    folds = int(np.ceil((len(y) - initial_train_size) / steps))
    remainder = (len(y) - initial_train_size) % steps
    
    if folds > 50:
        warnings.warn(
            (f"The forecaster will be fit {folds} times. This can take substantial amounts of time. "
             f"If not feasible, try with `refit = False`. \n"),
            RuntimeWarning
        )
    
    if verbose:
        _backtesting_forecaster_verbose(
            index_values       = y.index,
            steps              = steps,
            initial_train_size = initial_train_size,
            folds              = folds,
            remainder          = remainder,
            refit              = True,
            fixed_train_size   = fixed_train_size
        )
    
    for i in range(folds):
        # In each iteration the model is fitted before making predictions.
        # if fixed_train_size the train size doesn't increase but moves by `steps` in each iteration.
        # if false the train size increases by `steps` in each iteration.
        train_idx_start = i * steps if fixed_train_size else 0
        train_idx_end = initial_train_size + i * steps

        exog_train_values = exog.iloc[train_idx_start:train_idx_end, ] if exog is not None else None
        next_window_exog = exog.iloc[train_idx_end:train_idx_end + steps, ] if exog is not None else None

        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end, ], exog=exog_train_values)

        if i == folds - 1: # last fold
            # If remainder > 0, only the remaining steps need to be predicted
            steps = steps if remainder == 0 else remainder

        if alpha is None and interval is None:
            pred = forecaster.predict(steps=steps, exog=next_window_exog)
        else:
            pred = forecaster.predict_interval(
                       steps    = steps,
                       alpha    = alpha,
                       interval = interval,
                       exog     = next_window_exog,
                   )
            
        backtest_predictions.append(pred)
    
    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    if isinstance(metric, list):
        metrics_values = [m(
                            y_true = y.iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                            y_pred = backtest_predictions['pred']
                          ) for m in metrics
                         ]
    else:
        metrics_values = metrics(
                            y_true = y.iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                            y_pred = backtest_predictions['pred']
                         )

    return metrics_values, backtest_predictions


def _backtesting_sarimax_no_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: Optional[int]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    alpha: Optional[float]=None,
    interval: Optional[list]=None,
    verbose: bool=False
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of ForecasterSarimax without iterative re-fitting. In each iteration,
    a number of `steps` are predicted. A copy of the original forecaster is
    created so it is not modified during the process.

    If `forecaster` is already trained and `initial_train_size` is `None`,
    no initial train is done and all data is used to evaluate the model.
    However, the first `len(forecaster.last_window)` observations are needed
    to create the initial predictors, so no predictions are calculated for them.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
        
    y : pandas Series
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
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    alpha : float, default `0.05`
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
        
    interval : list, default `None`
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metrics_value : float, list
        Value(s) of the metric(s).

    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    
    """

    forecaster = deepcopy(forecaster)

    if isinstance(metric, str):
        metrics = _get_metric(metric=metric)
    elif isinstance(metric, list):
        metrics = [_get_metric(metric=m) if isinstance(m, str) else m for m in metric]
    else:
        metrics = metric
    
    backtest_predictions = []

    if initial_train_size is not None:
        exog_train_values = exog.iloc[:initial_train_size, ] if exog is not None else None
        forecaster.fit(y=y.iloc[:initial_train_size], exog=exog_train_values)
        window_size = forecaster.window_size
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        window_size = forecaster.window_size
        initial_train_size = window_size
    
    folds     = int(np.ceil((len(y) - initial_train_size) / steps))
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        _backtesting_forecaster_verbose(
            index_values       = y.index,
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
        if i == 0 :
            last_window_y = None
            last_window_exog = None
        else: 
            last_window_start = initial_train_size + steps * (i-1)
            last_window_end   = initial_train_size + steps * i

            last_window_y     = y.iloc[last_window_start:last_window_end]
            last_window_exog  = exog.iloc[last_window_start:last_window_end, ] if exog is not None else None
        
        next_window_exog  = exog.iloc[last_window_end:last_window_end + steps, ] if exog is not None else None
    
        if i == folds - 1: # last fold
            # If remainder > 0, only the remaining steps need to be predicted
            steps = steps if remainder == 0 else remainder
        
        if alpha is None and interval is None:
            pred = forecaster.predict(
                       steps            = steps,
                       last_window      = last_window_y,
                       last_window_exog = last_window_exog,
                       exog             = next_window_exog
                   )
        else:
            pred = forecaster.predict_interval(
                       steps            = steps,
                       exog             = next_window_exog,
                       alpha            = alpha,
                       interval         = interval,
                       last_window      = last_window_y,
                       last_window_exog = last_window_exog
                   )
        
        backtest_predictions.append(pred)

    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    if isinstance(metric, list):
        metrics_values = [m(
                            y_true = y.iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                            y_pred = backtest_predictions['pred']
                          ) for m in metrics
                         ]
    else:
        metrics_values = metrics(
                             y_true = y.iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                             y_pred = backtest_predictions['pred']
                         )

    return metrics_values, backtest_predictions


def backtesting_sarimax(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: Optional[int]=None,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    alpha: Optional[float]=None,
    interval: Optional[list]=None,
    verbose: bool=False
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of ForecasterSarimax.

    If `refit` is False, the model is trained only once using the `initial_train_size`
    first observations. If `refit` is True, the model is trained in each iteration
    increasing the training set. A copy of the original forecaster is created so 
    it is not modified during the process.

    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
        
    y : pandas Series
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
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration.
            
    alpha : float, default `0.05`
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
        
    interval : list, default `None`
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
                  
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metrics_value : float, list
        Value(s) of the metric(s).

    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.
    
    """

    if initial_train_size is not None and initial_train_size >= len(y):
        raise ValueError(
            'If used, `initial_train_size` must be smaller than length of `y`.'
        )
        
    if initial_train_size is not None and initial_train_size < forecaster.window_size:
        raise ValueError(
            (f"`initial_train_size` must be greater than "
             f"forecaster's window_size ({forecaster.window_size}).")
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
    
    if refit:
        metrics_values, backtest_predictions = _backtesting_sarimax_refit(
            forecaster          = forecaster,
            y                   = y,
            steps               = steps,
            metric              = metric,
            initial_train_size  = initial_train_size,
            fixed_train_size    = fixed_train_size,
            exog                = exog,
            alpha               = alpha,
            interval            = interval,
            verbose             = verbose
        )
    else:
        metrics_values, backtest_predictions = _backtesting_sarimax_no_refit(
            forecaster          = forecaster,
            y                   = y,
            steps               = steps,
            metric              = metric,
            initial_train_size  = initial_train_size,
            exog                = exog,
            alpha               = alpha,
            interval            = interval,
            verbose             = verbose
        )

    return metrics_values, backtest_predictions


def grid_search_sarimax(
    forecaster,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a ForecasterSarimax object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
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

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
        
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
    
    """

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters_sarimax(
        forecaster          = forecaster,
        y                   = y,
        param_grid          = param_grid,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        exog                = exog,
        refit               = refit,
        return_best         = return_best,
        verbose             = verbose
    )

    return results


def random_search_sarimax(
    forecaster,
    y: pd.Series,
    param_distributions: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forcaster model.
        
    y : pandas Series
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

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
        
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
    
    """

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters_sarimax(
        forecaster          = forecaster,
        y                   = y,
        param_grid          = param_grid,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        exog                = exog,
        refit               = refit,
        return_best         = return_best,
        verbose             = verbose
    )

    return results


def _evaluate_grid_hyperparameters_sarimax(
    forecaster,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
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

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
        
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
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            f'`exog` must have same number of samples as `y`. '
            f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
        )

    params_list = []
    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            'When `metric` is a `list`, each metric name must be unique.'
        )

    print(f"Number of models compared: {len(param_grid)}.")
  
    for params in tqdm(param_grid, desc='loop param_grid', position=0, ncols=90):

        forecaster.set_params(**params)
        metrics_values = backtesting_sarimax(
                                forecaster         = forecaster,
                                y                  = y,
                                steps              = steps,
                                metric             = metric,
                                initial_train_size = initial_train_size,
                                fixed_train_size   = fixed_train_size,
                                exog               = exog,
                                refit              = refit,
                                alpha              = None,
                                interval           = None,
                                verbose            = verbose
                            )[0]
        warnings.filterwarnings('ignore', category=RuntimeWarning, message= "The forecaster will be fit.*")   
        params_list.append(params)
        for m, m_value in zip(metric, metrics_values):
            m_name = m if isinstance(m, str) else m.__name__
            metric_dict[m_name].append(m_value)

    results = pd.DataFrame({
                 'params': params_list,
                 **metric_dict
              })
    
    results = results.sort_values(by=list(metric_dict.keys())[0], ascending=True)
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_params = results['params'].iloc[0]
        best_metric = results[list(metric_dict.keys())[0]].iloc[0]
        
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog)
        
        print(
            f"`Forecaster` refitted using the best-found parameters, and the whole data set: \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results