################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8


import typing
from typing import Union, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
import tqdm
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg

from .model_selection import time_series_spliter

logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)

def backtesting_autoreg_statsmodels(
    y: Union[np.ndarray, pd.Series],
    lags: int, 
    initial_train_size: int,
    steps: int,
    metric: str,
    exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
    verbose: bool=False
) -> Tuple[np.array, np.array]:
    '''
    
    Backtesting (validation) of `AutoReg` model from statsmodels v0.12. The model is
    trained only once using the `initial_train_size` first observations. In each
    iteration, a number of `steps` predictions are evaluated. This evaluation is
    much faster than cross-validation since the model is trained only once.
        
    Parameters
    ----------
        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
        
    lags: int, list
        The number of lags to include in the model if an integer or the list of
        lag indices to include. For example, [1, 4] will only include lags 1 and
        4 while lags=4 will include lags 1, 2, 3, and 4.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    verbose : bool, default `False`
        Print number of folds used for backtesting.

    Returns 
    -------
        
    metric_value: 1D np.ndarray
        Value of the metric.
        
    backtest_predictions: 1D np.ndarray
        Value of predictions.
    '''
    

    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error']:
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error' and "
            f"'mean_absolute_percentage_error'. Got {metric}."
        )
    
    backtest_predictions = []
    
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error
    }
    
    metric = metrics[metric]
    
    if isinstance(y, pd.Series):
        y = y.to_numpy(copy=True)
        
    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog = exog.to_numpy(copy=True)
        
    if exog is None:
        model = AutoReg(endog=y[:initial_train_size], lags=lags).fit()
    else:
        model = AutoReg(
                    endog = y[:initial_train_size],
                    exog  = exog[:initial_train_size],
                    lags  = lags
                ).fit()
    
    
    folds     = (len(y) - initial_train_size) // steps + 1
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        print(f"Number of observations used for training: {initial_train_size}")
        print(f"Number of observations used for testing: {len(y) - initial_train_size}")
        print(f"    Number of folds: {folds - 1 * (remainder == 0)}")
        print(f"    Number of steps per fold: {steps}")
        if remainder != 0:
            print(f"    Last fold only includes {remainder} observations")
      
    for i in range(folds):
        last_window_end   = initial_train_size + i * steps
        last_window_start = (initial_train_size + i * steps) - steps 
        last_window       = y[last_window_start:last_window_end]
        
        if i == 0:
            if exog is None:
                pred = model.forecast(steps=steps)
                            
            else:
                pred = model.forecast(
                            steps       = steps,
                            exog        = exog[last_window_end:last_window_end + steps]
                        )
                
        elif i < folds - 1:
            # Update internal values stored by AutoReg
            model.model._y = np.vstack((
                                model.model._y,
                                last_window.reshape(-1,1)
                             ))
        
            if exog is None:
                pred = model.forecast(steps=steps)
                            
            else:
                pred = model.forecast(
                            steps       = steps,
                            exog        = exog[last_window_end:last_window_end + steps]
                        )
                
        elif remainder != 0:
            steps = remainder
            # Update internal values stored by AutoReg
            model.model._y = np.vstack((
                                model.model._y,
                                last_window.reshape(-1,1)
                             ))
            
            if exog is None:
                pred = model.forecast(steps=steps)
                
            else:
                pred = model.forecast(
                            steps       = steps,
                            exog        = exog[last_window_end:last_window_end + steps]
                        )
        else:
            continue
        
        backtest_predictions.append(pred)
    
    backtest_predictions = np.concatenate(backtest_predictions)
    metric_value = metric(
                        y_true = y[initial_train_size: initial_train_size + len(backtest_predictions)],
                        y_pred = backtest_predictions
                   )

    return np.array([metric_value]), backtest_predictions


def cv_autoreg_statsmodels(
    y: Union[np.ndarray, pd.Series],
    lags: int, 
    initial_train_size: int,
    steps: int,
    metric: str,
    exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
    allow_incomplete_fold: bool=True,
    verbose: bool=False
) -> Tuple[np.array, np.array]:
    '''
        
    Cross-validation of `AutoReg` model from statsmodels v0.12. The order of data
    is maintained and the training set increases in each iteration.
    
    Parameters
    ----------
        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
        
    lags: int, list
        The number of lags to include in the model if an integer or the list of
        lag indices to include. For example, [1, 4] will only include lags 1 and
        4 while lags=4 will include lags 1, 2, 3, and 4.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    verbose : bool, default `False`
        Print number of folds used for cross-validation.
        
    Returns 
    -------

    cv_metrics: 1D np.ndarray
        Value of the metric for each fold.
        
    cv_predictions: 1D np.ndarray
        Predictions.
    '''
    

    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error']:
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error' and "
            f"'mean_absolute_percentage_error'. Got {metric}."
        )
        
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error
    }
    
    metric = metrics[metric]
    
    if isinstance(y, pd.Series):
        y = y.to_numpy(copy=True)
        
    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog = exog.to_numpy(copy=True)
        
    cv_predictions = []
    cv_metrics = []
    
    splits = time_series_spliter(
                y                     = y,
                initial_train_size    = initial_train_size,
                steps                 = steps,
                allow_incomplete_fold = allow_incomplete_fold,
                verbose               = verbose
             )
    
    for train_index, test_index in splits:
        
        if exog is None:
            model = AutoReg(endog=y[train_index], lags=lags).fit()
            pred = model.forecast(steps=len(test_index))
            
        else:
            model = AutoReg(
                        endog = y[train_index],
                        exog  = exog[train_index],
                        lags  = lags
                    ).fit()
            pred = model.forecast(steps=len(test_index), exog=exog[test_index])
    
               
        metric_value = metric(
                            y_true = y[test_index],
                            y_pred = pred
                       )
        
        cv_metrics.append(metric_value)
        cv_predictions.append(pred)
                          
    return np.array(cv_metrics), np.concatenate(cv_predictions)


def backtesting_sarimax_statsmodels(
        y: Union[np.ndarray, pd.Series],
        initial_train_size: int,
        steps: int,
        metric: str,
        order: tuple=(1, 0, 0), 
        seasonal_order: tuple=(0, 0, 0, 0),
        trend: str=None,
        exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
        sarimax_kwargs: dict={},
        fit_kwargs: dict={'disp':0},
        verbose: bool=False
) -> Tuple[np.array, np.array]:
    '''
    
    Backtesting (validation) of `SARIMAX` model from statsmodels v0.12. The model
    is trained only once using the `initial_train_size` first observations. In each
    iteration, a number of `steps` predictions are evaluated. This evaluation is
    much faster than cross-validation since the model is trained only once.
    
    https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_forecasting.html
    
    Parameters
    ----------
        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
        
    order: tuple 
        The (p,d,q) order of the model for the number of AR parameters, differences,
        and MA parameters. d must be an integer indicating the integration order
        of the process, while p and q may either be an integers indicating the AR
        and MA orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is an
        AR(1) model: (1,0,0).
        
    seasonal_order: tuple
        The (P,D,Q,s) order of the seasonal component of the model for the AR parameters,
        differences, MA parameters, and periodicity. D must be an integer
        indicating the integration order of the process, while P and Q may either
        be an integers indicating the AR and MA orders (so that all lags up to
        those orders are included) or else iterables giving specific AR and / or
        MA lags to include. s is an integer giving the periodicity (number of
        periods in season), often it is 4 for quarterly data or 12 for monthly data.
        Default is no seasonal effect.
        
    trend: str {‘n’,’c’,’t’,’ct’}
        Parameter controlling the deterministic trend polynomial A(t). Can be
        specified as a string where ‘c’ indicates a constant (i.e. a degree zero
        component of the trend polynomial), ‘t’ indicates a linear trend with time,
        and ‘ct’ is both. Can also be specified as an iterable defining the non-zero
        polynomial exponents to include, in increasing order. For example, [1,1,0,1]
        denotes a+bt+ct3. Default is to not include a trend component.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
        
    sarimax_kwargs: dict, default `{}`
        Additional keyword arguments passed to SARIMAX constructor. See more in
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX
        
    fit_kwargs: dict, default `{'disp':0}`
        Additional keyword arguments passed to SARIMAX fit. See more in
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html#statsmodels.tsa.statespace.sarimax.SARIMAX.fit
        
    verbose : bool, default `False`
        Print number of folds used for backtesting.
        
    Returns 
    -------

    metric_value: np.ndarray shape (1,)
        Value of the metric.

    backtest_predictions: 1D np.ndarray
        Value of predictions.
    '''
    

    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error']:
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error' and "
            f"'mean_absolute_percentage_error'. Got {metric}."
        )
    
    backtest_predictions = []
    
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error
    }
    
    metric = metrics[metric]
    
    if isinstance(y, pd.Series):
        y = y.to_numpy(copy=True)
        
    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog = exog.to_numpy(copy=True)
        
    if exog is None:
        model = SARIMAX(
                    endog = y[:initial_train_size],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend,
                    **sarimax_kwargs
                ).fit(**fit_kwargs)
        
    else:
        model = SARIMAX(
                    endog = y[:initial_train_size],
                    exog  = exog[:initial_train_size],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend,
                    **sarimax_kwargs
                ).fit(**fit_kwargs)
    
    
    folds     = (len(y) - initial_train_size) // steps + 1
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        print(f"Number of observations used for training: {initial_train_size}")
        print(f"Number of observations used for testing: {len(y) - initial_train_size}")
        print(f"    Number of folds: {folds - 1 * (remainder == 0)}")
        print(f"    Number of steps per fold: {steps}")
        if remainder != 0:
            print(f"    Last fold only includes {remainder} observations")
      
    for i in range(folds):
        last_window_end     = initial_train_size + i * steps
        last_window_start   = (initial_train_size + i * steps) - steps 
        last_window_y       = y[last_window_start:last_window_end]
        if exog is not None:
            last_window_exog    = exog[last_window_start:last_window_end]
            next_window_exog    = exog[last_window_end:last_window_end + steps]
        
        if i == 0:
            if exog is None:
                pred = model.forecast(steps=steps)
                            
            else:
                pred = model.forecast(
                            steps       = steps,
                            exog        = next_window_exog
                       )
                
        elif i < folds - 1:
            if exog is None:
                model = model.extend(endog=last_window_y)
                pred = model.forecast(steps=steps)
                            
            else:
                model = model.extend(endog=last_window_y, exog=last_window_exog)
                pred = model.forecast(
                            steps = steps,
                            exog  = next_window_exog
                        )
                
        elif remainder != 0:
            steps = remainder
            
            if exog is None:
                model = model.extend(exog=last_window_y)
                pred = model.forecast(steps=steps)
                
            else:
                model = model.extend(endog=last_window_y, exog=last_window_exog)
                pred = model.forecast(
                            steps = steps,
                            exog  = next_window_exog
                       )
        else:
            continue
        
        backtest_predictions.append(pred)
    
    backtest_predictions = np.concatenate(backtest_predictions)
    metric_value = metric(
                        y_true = y[initial_train_size: initial_train_size + len(backtest_predictions)],
                        y_pred = backtest_predictions
                   )

    return np.array([metric_value]), backtest_predictions


def cv_sarimax_statsmodels(
        y: Union[np.ndarray, pd.Series],
        initial_train_size: int,
        steps: int,
        metric: str,
        order: tuple=(1, 0, 0), 
        seasonal_order: tuple=(0, 0, 0, 0),
        trend: str=None,
        exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
        allow_incomplete_fold: bool=True,
        sarimax_kwargs: dict={},
        fit_kwargs: dict={'disp':0},
        verbose: bool=False
) -> Tuple[np.array, np.array]:
    '''
        
    Cross-validation of `SARIMAX` model from statsmodels v0.12. The order of data
    is maintained and the training set increases in each iteration.
    
    Parameters
    ----------
        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
        
    order: tuple 
        The (p,d,q) order of the model for the number of AR parameters, differences,
        and MA parameters. d must be an integer indicating the integration order
        of the process, while p and q may either be an integers indicating the AR
        and MA orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is an
        AR(1) model: (1,0,0).
        
    seasonal_order: tuple
        The (P,D,Q,s) order of the seasonal component of the model for the AR parameters,
        differences, MA parameters, and periodicity. D must be an integer
        indicating the integration order of the process, while P and Q may either
        be an integers indicating the AR and MA orders (so that all lags up to
        those orders are included) or else iterables giving specific AR and / or
        MA lags to include. s is an integer giving the periodicity (number of
        periods in season), often it is 4 for quarterly data or 12 for monthly data.
        Default is no seasonal effect.
        
    trend: str {‘n’,’c’,’t’,’ct’}
        Parameter controlling the deterministic trend polynomial A(t). Can be
        specified as a string where ‘c’ indicates a constant (i.e. a degree zero
        component of the trend polynomial), ‘t’ indicates a linear trend with time,
        and ‘ct’ is both. Can also be specified as an iterable defining the non-zero
        polynomial exponents to include, in increasing order. For example, [1,1,0,1]
        denotes a+bt+ct3. Default is to not include a trend component.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
        
    sarimax_kwargs: dict, default {}
        Additional keyword arguments passed to SARIMAX initialization. See more in
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX
        
    fit_kwargs: dict, default `{'disp':0}`
        Additional keyword arguments passed to SARIMAX fit. See more in
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html#statsmodels.tsa.statespace.sarimax.SARIMAX.fit
        
    verbose : bool, default `False`
        Print number of folds used for cross-validation.
        
    Returns 
    -------

    cv_metrics: 1D np.ndarray
        Value of the metric for each partition.

    cv_predictions: 1D np.ndarray
        Predictions.
    '''
    

    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error']:
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error' and "
            f"'mean_absolute_percentage_error'. Got {metric}."
        )
        
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error
    }
    
    metric = metrics[metric]
    
    if isinstance(y, pd.Series):
        y = y.to_numpy(copy=True)
        
    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog = exog.to_numpy(copy=True)
        
    cv_predictions = []
    cv_metrics = []
    
    splits = time_series_spliter(
                y                     = y,
                initial_train_size    = initial_train_size,
                steps                 = steps,
                allow_incomplete_fold = allow_incomplete_fold,
                verbose               = verbose
             )
    
    for train_index, test_index in splits:
        
        if exog is None:
            model = SARIMAX(
                    endog = y[train_index],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend,
                    **sarimax_kwargs
                ).fit(**fit_kwargs)
            
            pred = model.forecast(steps=len(test_index))
            
        else:         
            model = SARIMAX(
                    endog = y[train_index],
                    exog  = exog[train_index],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend,
                    **sarimax_kwargs
                ).fit(**fit_kwargs)
            
            pred = model.forecast(steps=len(test_index), exog=exog[test_index])
    
               
        metric_value = metric(
                            y_true = y[test_index],
                            y_pred = pred
                       )
        
        cv_metrics.append(metric_value)
        cv_predictions.append(pred)
                          
    return np.array(cv_metrics), np.concatenate(cv_predictions)


def grid_search_sarimax_statsmodels(
        y: Union[np.ndarray, pd.Series],
        param_grid: dict,
        initial_train_size: int,
        steps: int,
        metric: str,
        exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
        method: str='cv',
        allow_incomplete_fold: bool=True,
        sarimax_kwargs: dict={},
        fit_kwargs: dict={'disp':0},
        verbose: bool=False
) -> pd.DataFrame:
    '''
    Exhaustive search over specified parameter values for a `SARIMAX` model from
    statsmodels v0.12. Validation is done using time series cross-validation or
    backtesting.
    
    Parameters
    ----------
        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values. Allowed parameters in the grid are: order,
        seasonal_order and trend.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
        
    method : {'cv', 'backtesting'}
        Method used to estimate the metric for each parameter combination.
        'cv' for time series crosvalidation and 'backtesting' for simple
        backtesting. 'backtesting' is much faster since the model is fitted only
        once.
        
    allow_incomplete_fold : bool, default `True`
        The last test set is allowed to be incomplete if it does not reach `steps`
        observations. Otherwise, the latest observations are discarded.
        
    return_best : bool
        Refit the `forecaster` using the best found parameters on the whole data.
        
    sarimax_kwargs: dict, default `{}`
        Additional keyword arguments passed to SARIMAX initialization. See more in
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX
        
    fit_kwargs: dict, default `{'disp':0}`
        Additional keyword arguments passed to SARIMAX fit. See more in
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html#statsmodels.tsa.statespace.sarimax.SARIMAX.fit
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    
    results: pandas.DataFrame
        Metric value estimated for each combination of parameters.

    '''

    
    if isinstance(y, pd.Series):
        y = y.to_numpy(copy=True)
        
    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog = exog.to_numpy(copy=True)
        
      
    params_list = []
    metric_list = []
    bic_list = []
    aic_list = []
    
    if 'order' not in param_grid:
        param_grid['order'] = [(1, 0, 0)]
    if 'seasonal_order' not in param_grid:
        param_grid['seasonal_order'] = [(0, 0, 0, 0)]
    if 'trend' not in param_grid:
        param_grid['trend'] = [None]

    keys_to_ignore = set(param_grid.keys()) - {'order', 'seasonal_order', 'trend'}
    if keys_to_ignore:
        print(
            f'Only arguments: order, seasonal_order and trend are allowed for grid serach.'
            f' Ignoring {keys_to_ignore}.'
        )
        for key in keys_to_ignore:
            del param_grid[key]
            
    param_grid =  list(ParameterGrid(param_grid))

    logging.info(
        f"Number of models compared: {len(param_grid)}"
    )
        
        
    for params in tqdm.tqdm(param_grid):

        if method == 'cv':
            metrics = cv_sarimax_statsmodels(
                            y              = y,
                            exog           = exog,
                            order          = params['order'],
                            seasonal_order = params['seasonal_order'],
                            trend          = params['trend'],
                            initial_train_size = initial_train_size,
                            steps          = steps,
                            metric         = metric,
                            sarimax_kwargs = sarimax_kwargs,
                            fit_kwargs     = fit_kwargs,
                            verbose        = verbose
                        )[0]
        else:
            metrics = backtesting_sarimax_statsmodels(
                            y              = y,
                            exog           = exog,
                            order          = params['order'],
                            seasonal_order = params['seasonal_order'],
                            trend          = params['trend'],
                            initial_train_size = initial_train_size,
                            steps          = steps,
                            metric         = metric,
                            sarimax_kwargs = sarimax_kwargs,
                            fit_kwargs     = fit_kwargs,
                            verbose        = verbose
                        )[0]

        params_list.append(params)
        metric_list.append(metrics.mean())
        
        model = SARIMAX(
                    endog = y,
                    exog  = exog,
                    order = params['order'],
                    seasonal_order = params['seasonal_order'],
                    trend = params['trend'],
                    **sarimax_kwargs
                ).fit(**fit_kwargs)
        
        bic_list.append(model.bic)
        aic_list.append(model.aic)
            
    results = pd.DataFrame({
                'params': params_list,
                'metric': metric_list,
                'bic'   : bic_list,
                'aic'   : aic_list
              })
    
    results = results.sort_values(by='metric', ascending=True)
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
            
    return results