################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by Joaquin Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8


from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
import warnings
import logging
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid

from ..ForecasterAutoreg import ForecasterAutoreg
from ..ForecasterAutoregCustom import ForecasterAutoregCustom
from ..ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def time_series_splitter(
    y: Union[np.ndarray, pd.Series],
    initial_train_size: int,
    steps: int,
    allow_incomplete_fold: bool=True,
    verbose: bool=True
) -> Union[np.ndarray, np.ndarray]:
    '''
    
    Split indices of a time series into multiple train-test pairs. The order of
    is maintained and the training set increases in each iteration.
    
    Parameters
    ----------        
    y : 1d numpy ndarray, pandas Series
        Training time series values. 
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    allow_incomplete_fold : bool, default `True`
        The last test set is allowed to be incomplete if it does not reach `steps`
        observations. Otherwise, the latest observations are discarded.
        
    verbose : bool, default `True`
        Print number of splits created.

    Yields
    ------
    train : 1d numpy ndarray
        Training indices.
        
    test : 1d numpy ndarray
        Test indices.
        
    '''
    
    if not isinstance(y, (np.ndarray, pd.Series)):

        raise Exception('`y` must be `1D np.ndarray` o `pd.Series`.')

    elif isinstance(y, np.ndarray) and y.ndim != 1:

        raise Exception(
            f"`y` must be `1D np.ndarray` o `pd.Series`, "
            f"got `np.ndarray` with {y.ndim} dimensions."
        )
        
    if initial_train_size > len(y):
        raise Exception(
            '`initial_train_size` must be smaller than length of `y`.'
            ' Try to reduce `initial_train_size` or `steps`.'
        )

    if isinstance(y, pd.Series):
        y = y.to_numpy().copy()
    
  
    folds = (len(y) - initial_train_size) // steps  + 1
    # +1 fold is needed to allow including the remainder in the last iteration.
    remainder = (len(y) - initial_train_size) % steps   
    
    if verbose:
        if folds == 1:
            print(f"Number of folds: {folds - 1}")
            print("Not enough observations in `y` to create even a complete fold."
                  " Try to reduce `initial_train_size` or `steps`."
            )

        elif remainder == 0:
            print(f"Number of folds: {folds - 1}")

        elif remainder != 0 and allow_incomplete_fold:
            print(f"Number of folds: {folds}")
            print(
                f"Since `allow_incomplete_fold=True`, "
                f"last fold only includes {remainder} observations instead of {steps}."
            )
            print(
                'Incomplete folds with few observations could overestimate or ',
                'underestimate validation metrics.'
            )
        elif remainder != 0 and not allow_incomplete_fold:
            print(f"Number of folds: {folds - 1}")
            print(
                f"Since `allow_incomplete_fold=False`, "
                f"last {remainder} observations are descarted."
            )

    if folds == 1:
        # There are no observations to create even a complete fold
        return []
    
    for i in range(folds):
          
        if i < folds - 1:
            train_end     = initial_train_size + i * steps    
            train_indices = range(train_end)
            test_indices  = range(train_end, train_end + steps)
            
        else:
            if remainder != 0 and allow_incomplete_fold:
                train_end     = initial_train_size + i * steps  
                train_indices = range(train_end)
                test_indices  = range(train_end, len(y))
            else:
                break
        
        yield train_indices, test_indices
        
        
def get_metric(metric:str) -> callable:
    '''
    Get the corresponding scikitlearn function to calculate the metric.
    
    Parameters
    ----------
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
    
    Returns 
    -------
    metric : callable
        scikitlearn function to calculate the desired metric.
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
    
    return metric
    

def cv_forecaster(
    forecaster,
    y: pd.Series,
    initial_train_size: int,
    steps: int,
    metric: str,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    allow_incomplete_fold: bool=True,
    verbose: bool=True
) -> Tuple[np.array, pd.DataFrame]:
    '''
    Cross-validation of forecaster. The order of data is maintained and the
    training set increases in each iteration.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    allow_incomplete_fold : bool, default `True`
        The last test partition is allowed to be incomplete if it does not reach `steps`
        observations. Otherwise, the latest observations are discarded.
            
    verbose : bool, default `True`
        Print number of folds used for cross validation.

    Returns 
    -------
    cv_metrics: 1d numpy ndarray
        Value of the metric for each fold.

    cv_predictions: pandas DataFrame
        Predictions.

    '''

    if initial_train_size > len(y):
        raise Exception(
            '`initial_train_size` must be smaller than length of `y`.'
        )
        
    if initial_train_size is not None and initial_train_size < forecaster.window_size:
        raise Exception(
            f"`initial_train_size` must be greater than "
            f"forecaster's window_size ({forecaster.window_size})."
        )
        
    forecaster = deepcopy(forecaster)
    metric = get_metric(metric=metric)
    
    splits = time_series_splitter(
                y                     = y,
                initial_train_size    = initial_train_size,
                steps                 = steps,
                allow_incomplete_fold = allow_incomplete_fold,
                verbose               = verbose
             )

    cv_predictions = []
    cv_metrics = []
    
    for train_index, test_index in splits:
        
        if exog is None:
            forecaster.fit(y=y.iloc[train_index])      
            pred = forecaster.predict(steps=len(test_index))
            
        else:
            forecaster.fit(y=y.iloc[train_index], exog=exog.iloc[train_index,])      
            pred = forecaster.predict(steps=len(test_index), exog=exog.iloc[test_index])
               
        metric_value = metric(
                            y_true = y.iloc[test_index],
                            y_pred = pred
                       )
        
        cv_predictions.append(pred)
        cv_metrics.append(metric_value)
            
    cv_predictions = pd.concat(cv_predictions)
    cv_predictions = pd.DataFrame(cv_predictions)
    cv_metrics = np.array(cv_metrics)
    
    return cv_metrics, cv_predictions


def _backtesting_forecaster_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: str,
    initial_train_size: int,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[np.array, pd.DataFrame]:
    '''
    Backtesting of forecaster with model re-fitting. In each iteration:
        - Fit forecaster with the training set.
        - A number of `steps` ahead are predicted.
        - The training set increases with `steps` observations.

    In order to apply backtesting with re-fit, an initial training set must be
    available, otherwise it would not be possible to increase the training set after each
    iteration. `initial_train_size` must be provided.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
        The object forecaster is not overwritten.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog :panda Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval: list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot: int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals. If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: numpy ndarray shape (1,)
        Value of the metric.

    backtest_predictions: pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.

    '''
    
    forecaster = deepcopy(forecaster)
    metric = get_metric(metric=metric)
    backtest_predictions = []
    
    folds = int(np.ceil((len(y) - initial_train_size) / steps))
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        print(f"Information of backtesting process")
        print(f"----------------------------------")
        print(f"Number of observations used for initial training: {initial_train_size}")
        print(f"Number of observations used for backtesting: {len(y) - initial_train_size}")
        print(f"    Number of folds: {folds}")
        print(f"    Number of steps per fold: {steps}")
        if remainder != 0:
            print(f"    Last fold only includes {remainder} observations.")
        print("")
        for i in range(folds):
            train_size = initial_train_size + i * steps
            print(f"Data partition in fold: {i}")
            if i < folds - 1:
                print(f"    Training:   {y.index[0]} -- {y.index[train_size - 1]}")
                print(f"    Validation: {y.index[train_size]} -- {y.index[train_size + steps - 1]}")
            else:
                print(f"    Training:   {y.index[0]} -- {y.index[train_size - 1]}")
                print(f"    Validation: {y.index[train_size]} -- {y.index[-1]}")
        print("")
        
    if folds > 50:
        print(
            f"Forecaster will be fit {folds} times. This can take substantial amounts of time. "
            f"If not feasible, try with `refit = False`. \n"
        )

    for i in range(folds):
        # In each iteration (except the last one) the model is fitted before
        # making predictions. The train size increases by `steps` in each iteration.
        train_size = initial_train_size + i * steps
        if exog is not None:
            next_window_exog = exog.iloc[train_size:train_size + steps, ]

        if interval is None:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(y=y.iloc[:train_size])
                    pred = forecaster.predict(steps=steps)
                else:
                    forecaster.fit(y=y.iloc[:train_size], exog=exog.iloc[:train_size, ])
                    pred = forecaster.predict(steps=steps,exog=next_window_exog)
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(y=y.iloc[:train_size])
                        pred = forecaster.predict(steps=steps)
                    else:
                        forecaster.fit(y=y.iloc[:train_size], exog=exog.iloc[:train_size, ])
                        pred = forecaster.predict(steps=steps, exog=next_window_exog)
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(y=y.iloc[:train_size])
                        pred = forecaster.predict(steps=steps)
                    else:
                        forecaster.fit(y=y.iloc[:train_size], exog=exog.iloc[:train_size, ])
                        pred = forecaster.predict(steps=steps, exog=next_window_exog)
        else:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(y=y.iloc[:train_size])
                    pred = forecaster.predict_interval(
                                steps       = steps,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                            )
                else:
                    forecaster.fit(y=y.iloc[:train_size], exog=exog.iloc[:train_size, ])
                    pred = forecaster.predict_interval(
                                steps       = steps,
                                exog        = next_window_exog,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                           )
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(y=y.iloc[:train_size])
                        pred = forecaster.predict_interval(
                                steps       = steps,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(y=y.iloc[:train_size], exog=exog.iloc[:train_size, ])
                        pred = forecaster.predict_interval(
                                steps       = steps,
                                exog        = next_window_exog,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                           )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(y=y.iloc[:train_size])
                        pred = forecaster.predict_interval(
                                steps       = steps,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(y=y.iloc[:train_size], exog=exog.iloc[:train_size, ])
                        pred = forecaster.predict_interval(
                                steps       = steps,
                                exog        = next_window_exog,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                           )

        backtest_predictions.append(pred)
    
    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
            backtest_predictions = pd.DataFrame(backtest_predictions)

    metric_value = metric(
                    y_true = y.iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return np.array([metric_value]), backtest_predictions


def _backtesting_forecaster_no_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: str,
    initial_train_size: Optional[int]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[np.array, pd.DataFrame]:
    '''
    Backtesting of forecaster without iterative re-fitting. In each iteration,
    a number of `steps` are predicted.

    If `forecaster` is already trained and `initial_train_size` is `None`,
    no initial train is done and all data is used to evaluate the model.
    However, the first `len(forecaster.last_window)` observations are needed
    to create the initial predictors, so no predictions are calculated for them.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int, default `None`
        Number of samples in the initial train split. The object forecaster 
        is not overwritten. If `None` and `forecaster` is already trained, 
        no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.
        
    steps : int, None
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog :panda Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval: list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot: int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: numpy ndarray shape (1,)
        Value of the metric.

    backtest_predictions: pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.

    '''
        
    forecaster = deepcopy(forecaster)
    metric = get_metric(metric=metric)
    backtest_predictions = []

    if initial_train_size is not None:
        if exog is None:
            forecaster.fit(y=y.iloc[:initial_train_size])      
        else:
            forecaster.fit(
                y = y.iloc[:initial_train_size],
                exog = exog.iloc[:initial_train_size, ]
            )
        window_size = forecaster.window_size
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        window_size = forecaster.window_size
        initial_train_size = window_size
    
    folds     = int(np.ceil((len(y) - initial_train_size) / steps))
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        print(f"Information of backtesting process")
        print(f"----------------------------------")
        print(f"Number of observations used for initial training or as initial window: {initial_train_size}")
        print(f"Number of observations used for backtesting: {len(y) - initial_train_size}")
        print(f"    Number of folds: {folds}")
        print(f"    Number of steps per fold: {steps}")
        if remainder != 0:
            print(f"    Last fold only includes {remainder} observations")
        print("")
        for i in range(folds):
            last_window_end = initial_train_size + i * steps
            print(f"Data partition in fold: {i}")
            if i < folds - 1:
                print(f"    Training:   {y.index[0]} -- {y.index[initial_train_size - 1]}")
                print(f"    Validation: {y.index[last_window_end]} -- {y.index[last_window_end + steps -1]}")
            else:
                print(f"    Training:   {y.index[0]} -- {y.index[initial_train_size - 1]}")
                print(f"    Validation: {y.index[last_window_end]} -- {y.index[-1]}")
        print("")

    for i in range(folds):
        # Since the model is only fitted with the initial_train_size, last_window
        # and next_window_exog must be updated to include the data needed to make
        # predictions.
        last_window_end   = initial_train_size + i * steps
        last_window_start = last_window_end - window_size 
        last_window_y     = y.iloc[last_window_start:last_window_end]
        if exog is not None:
            next_window_exog = exog.iloc[last_window_end:last_window_end + steps, ]
    
        if interval is None:  

            if i < folds - 1: 
                if exog is None:
                    pred = forecaster.predict(
                                steps       = steps,
                                last_window = last_window_y
                            )
                else:
                    pred = forecaster.predict(
                                steps       = steps,
                                last_window = last_window_y,
                                exog        = next_window_exog
                            )            
            else:    
                if remainder == 0:
                    if exog is None:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y
                                )
                    else:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    exog        = next_window_exog
                                )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y
                                )
                    else:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    exog        = next_window_exog
                                )
            
            backtest_predictions.append(pred)

        else:
            if i < folds - 1:
                if exog is None:
                    pred = forecaster.predict_interval(
                                steps       = steps,
                                last_window = last_window_y,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                            )
                else:
                    pred = forecaster.predict_interval(
                                steps       = steps,
                                last_window = last_window_y,
                                exog        = next_window_exog,
                                interval    = interval,
                                n_boot      = n_boot,
                                in_sample_residuals = in_sample_residuals
                            )            
            else:    
                if remainder == 0:
                    if exog is None:
                        pred = forecaster.predict_interval(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    interval    = interval,
                                    n_boot      = n_boot,
                                    in_sample_residuals = in_sample_residuals
                                )
                    else:
                        pred = forecaster.predict_interval(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    exog        = next_window_exog,
                                    interval    = interval,
                                    n_boot      = n_boot,
                                    in_sample_residuals = in_sample_residuals
                                )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        pred = forecaster.predict_interval(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    interval    = interval,
                                    n_boot      = n_boot,
                                    in_sample_residuals = in_sample_residuals
                                )
                    else:
                        pred = forecaster.predict_interval(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    exog        = next_window_exog,
                                    interval    = interval,
                                    n_boot      = n_boot,
                                    in_sample_residuals = in_sample_residuals
                                )
            
            backtest_predictions.append(pred)

    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
            backtest_predictions = pd.DataFrame(backtest_predictions)

    metric_value = metric(
                    y_true = y.iloc[initial_train_size : initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return np.array([metric_value]), backtest_predictions


def backtesting_forecaster(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: str,
    initial_train_size: Optional[int],
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    in_sample_residuals: bool=True,
    verbose: bool=False
) -> Tuple[np.array, pd.DataFrame]:
    '''
    Backtesting of forecaster model.

    If `refit` is False, the model is trained only once using the `initial_train_size`
    first observations. If `refit` is True, the model is trained in each iteration
    increasing the training set.

    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int, default `None`
        Number of samples in the initial train split. The object forecaster 
        is not overwritten. If `None` and `forecaster` is already trained, 
        no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

        `None` is only allowed when `refit` is False.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog :panda Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    refit: bool, default False
        Whether to re-fit the forecaster in each iteration.

    interval: list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot: int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
                  
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: numpy ndarray shape (1,)
        Value of the metric.

    backtest_predictions: pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.

    '''

    if initial_train_size is not None and initial_train_size > len(y):
        raise Exception(
            'If used, `initial_train_size` must be smaller than length of `y`.'
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

    if interval is not None and isinstance(forecaster, ForecasterAutoregMultiOutput):
        raise Exception(
            ('Interval prediction is only available when forecaster is of type '
            'ForecasterAutoreg or ForecasterAutoregCustom.')
        )
    
    if refit:
        metric_value, backtest_predictions = _backtesting_forecaster_refit(
            forecaster          = forecaster,
            y                   = y,
            steps               = steps,
            metric              = metric,
            initial_train_size  = initial_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )
    else:
        metric_value, backtest_predictions = _backtesting_forecaster_no_refit(
            forecaster          = forecaster,
            y                   = y,
            steps               = steps,
            metric              = metric,
            initial_train_size  = initial_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )

    return metric_value, backtest_predictions


def grid_search_forecaster(
    forecaster,
    y: pd.Series,
    param_grid: dict,
    initial_train_size: int,
    steps: int,
    metric: str,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    '''
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range. 
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg`.
        
    refit: bool, default False
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    return_best : bool
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results: pandas DataFrame
        Metric value estimated for each combination of parameters.

    '''

    if isinstance(forecaster, ForecasterAutoregCustom):
        if lags_grid is not None:
            warnings.warn(
                '`lags_grid` ignored if forecaster is an instance of `ForecasterAutoregCustom`.'
            )
        lags_grid = ['custom predictors']
        
    elif lags_grid is None:
        lags_grid = [forecaster.lags]
        
      
    lags_list = []
    params_list = []
    metric_list = []
    
    param_grid =  list(ParameterGrid(param_grid))

    print(
        f"Number of models compared: {len(param_grid)*len(lags_grid)}"
    )
    
    for lags in tqdm(lags_grid, desc='loop lags_grid', position=0, ncols=90):
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        for params in tqdm(param_grid, desc='loop param_grid', position=1, leave=False, ncols=90):

            forecaster.set_params(**params)
            metrics = backtesting_forecaster(
                            forecaster         = forecaster,
                            y                  = y,
                            exog               = exog,
                            initial_train_size = initial_train_size,
                            steps              = steps,
                            metric             = metric,
                            refit              = refit,
                            interval           = None,
                            verbose            = verbose
                            )[0]

            lags_list.append(lags)
            params_list.append(params)
            metric_list.append(metrics.mean())
            
    results = pd.DataFrame({
                'lags'  : lags_list,
                'params': params_list,
                'metric': metric_list})
    
    results = results.sort_values(by='metric', ascending=True)
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results['lags'].iloc[0]
        best_params = results['params'].iloc[0]
        best_metric = results['metric'].iloc[0]
        
        print(
            f"Refitting `forecaster` using the best found lags and parameters and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog)
            
    return results
