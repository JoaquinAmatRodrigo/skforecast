################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by Joaquin Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8


from typing import Union, Tuple, Optional, Any
import numpy as np
import pandas as pd
import warnings
import logging
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING) # disable optuna logs
from skopt.utils import use_named_args
from skopt import gp_minimize

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
        
        
def _get_metric(metric:str) -> callable:
    '''
    Get the corresponding scikitlearn function to calculate the metric.
    
    Parameters
    ----------
    metric : {'mean_squared_error', 'mean_absolute_error', 
              'mean_absolute_percentage_error', 'mean_squared_log_error'}
        Metric used to quantify the goodness of fit of the model.
    
    Returns 
    -------
    metric : callable
        scikitlearn function to calculate the desired metric.
    '''
    
    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error', 'mean_squared_log_error']:
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error', "
            f"'mean_absolute_percentage_error' and 'mean_squared_log_error'. Got {metric}."
        )
    
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error,
        'mean_squared_log_error': mean_squared_log_error
    }
    
    metric = metrics[metric]
    
    return metric
    

def cv_forecaster(
    forecaster,
    y: pd.Series,
    initial_train_size: int,
    steps: int,
    metric: Union[str, callable],
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
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        It callable:
            Function with arguments y_true, y_pred that returns a float.
        
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
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
    
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
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
        
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.
        
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

    random_state: int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals. If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: float
        Value of the metric.

    backtest_predictions: pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.

    '''

    forecaster = deepcopy(forecaster)
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
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
            if fixed_train_size:
                # The train size doesn't increase but moves by `steps` in each iteration.
                train_idx_start = i * steps
                train_idx_end = initial_train_size + i * steps
            else:
                # The train size increases by `steps` in each iteration.
                train_idx_start = 0
                train_idx_end = initial_train_size + i * steps
            print(f"Data partition in fold: {i}")
            if i < folds - 1:
                print(f"    Training:   {y.index[train_idx_start]} -- {y.index[train_idx_end - 1]}  (n={len(y.index[train_idx_start:train_idx_end])})")
                print(f"    Validation: {y.index[train_idx_end]} -- {y.index[train_idx_end + steps - 1]}  (n={len(y.index[train_idx_end:train_idx_end + steps])})")
            else:
                print(f"    Training:   {y.index[train_idx_start]} -- {y.index[train_idx_end - 1]}  (n={len(y.index[train_idx_start:train_idx_end])})")
                print(f"    Validation: {y.index[train_idx_end]} -- {y.index[-1]}  (n={len(y.index[train_idx_end:])})")
        print("")
        
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
            train_idx_end = initial_train_size + i * steps
        else:
            # The train size increases by `steps` in each iteration.
            train_idx_start = 0
            train_idx_end = initial_train_size + i * steps
            
        if exog is not None:
            next_window_exog = exog.iloc[train_idx_end:train_idx_end + steps, ]

        if interval is None:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                    pred = forecaster.predict(steps=steps)
                else:
                    forecaster.fit(
                        y = y.iloc[train_idx_start:train_idx_end], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ]
                    )
                    pred = forecaster.predict(steps=steps, exog=next_window_exog)
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict(steps=steps)
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict(steps=steps, exog=next_window_exog)
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict(steps=steps)
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict(steps=steps, exog=next_window_exog)
        else:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                else:
                    forecaster.fit(
                        y = y.iloc[train_idx_start:train_idx_end], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ]
                    )
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
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
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
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
                    y_true = y.iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return metric_value, backtest_predictions


def _backtesting_forecaster_no_refit(
    forecaster,
    y: pd.Series,
    steps: int,
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
    a number of `steps` are predicted. A copy of the  original forecaster is
    created so it is not modified during the process.

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
        Number of samples in the initial train split. If `None` and `forecaster` is already
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.
        
    steps : int, None
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.
        
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

    random_state: int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: float
        Value of the metric.

    backtest_predictions: pandas DataFrame
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
                print(f"    Training:   {y.index[0]} -- {y.index[initial_train_size - 1]}  (n={len(y.index[:initial_train_size])})")
                print(f"    Validation: {y.index[last_window_end]} -- {y.index[last_window_end + steps -1]}  (n={len(y.index[last_window_end:last_window_end + steps])})")
            else:
                print(f"    Training:   {y.index[0]} -- {y.index[initial_train_size - 1]}  (n={len(y.index[:initial_train_size])})")
                print(f"    Validation: {y.index[last_window_end]} -- {y.index[-1]}  (n={len(y.index[last_window_end:])})")
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
                                steps        = steps,
                                last_window  = last_window_y,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                else:
                    pred = forecaster.predict_interval(
                                steps        = steps,
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
                                    last_window  = last_window_y,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                                )
                    else:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
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
                                    last_window  = last_window_y,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                                )
                    else:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
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
                    y_true = y.iloc[initial_train_size : initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return metric_value, backtest_predictions


def backtesting_forecaster(
    forecaster,
    y: pd.Series,
    steps: int,
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
    Backtesting of forecaster model.

    If `refit` is False, the model is trained only once using the `initial_train_size`
    first observations. If `refit` is True, the model is trained in each iteration
    increasing the training set. A copy of the original forecaster is created so 
    it is not modified during the process.

    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already 
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

        `None` is only allowed when `refit` is `False`.
    
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.
        
    exog :panda Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    refit: bool, default `False`
        Whether to re-fit the forecaster in each iteration.

    interval: list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot: int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state: int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.
                  
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: float
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
            fixed_train_size    = fixed_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            random_state        = random_state,
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
            random_state        = random_state,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )

    return metric_value, backtest_predictions


def grid_search_forecaster(
    forecaster,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
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

    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size: int 
        Number of samples in the initial train split.
 
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregMultiOutput`.
        
    refit: bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results: pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    '''

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters(
        forecaster          = forecaster,
        y                   = y,
        param_grid          = param_grid,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        exog                = exog,
        lags_grid           = lags_grid,
        refit               = refit,
        return_best         = return_best,
        verbose             = verbose
    )

    return results


def random_search_forecaster(
    forecaster,
    y: pd.Series,
    param_distributions: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
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
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.

    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size: int 
        Number of samples in the initial train split.
 
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregMultiOutput`.
        
    refit: bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.

    n_iter: int, default `10`
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.

    random_state: int, default `123`
        Sets a seed to the random sampling for reproducible output.

    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results: pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    '''

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters(
        forecaster          = forecaster,
        y                   = y,
        param_grid          = param_grid,
        steps               = steps,
        metric              = metric,
        initial_train_size  = initial_train_size,
        fixed_train_size    = fixed_train_size,
        exog                = exog,
        lags_grid           = lags_grid,
        refit               = refit,
        return_best         = return_best,
        verbose             = verbose
    )

    return results


def _evaluate_grid_hyperparameters(
    forecaster,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    '''
    Evaluate parameter values for a Forecaster object using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size: int 
        Number of samples in the initial train split.
 
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregMultiOutput`.
        
    refit: bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results: pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

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

    print(
        f"Number of models compared: {len(param_grid)*len(lags_grid)}."
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
                            steps              = steps,
                            metric             = metric,
                            initial_train_size = initial_train_size,
                            fixed_train_size   = fixed_train_size,
                            refit              = refit,
                            interval           = None,
                            verbose            = verbose
                            )[0]

            lags_list.append(lags)
            params_list.append(params)
            metric_list.append(metrics)
            
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
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results


def bayesian_search_forecaster(
    forecaster,
    y: pd.Series,
    search_space: Union[callable, dict],
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True,
    engine: str='skopt',
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={},
    kwargs_gp_minimize: dict={},
) -> Tuple[pd.DataFrame, object]:
    '''
    Bayesian optimization for a Forecaster object using time series backtesting and 
    optuna or skopt library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    search_space : callable (optuna), dict (skopt)
        If optuna engine: callable
            Function with argument `trial` which returns a dictionary with parameters names 
            (`str`) as keys and Trial object from optuna (trial.suggest_float, 
            trial.suggest_int, trial.suggest_categorical) as values.

        If skopt engine: dict
            Dictionary with parameters names (`str`) as keys and Space object from skopt 
            (Real, Integer, Categorical) as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size: int 
        Number of samples in the initial train split.
 
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregMultiOutput`.
        
    refit: bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    n_trials: int, default `10`
        Number of parameter settings that are sampled in each lag configuration.

    random_state: int, default `123`
        Sets a seed to the sampling for reproducible output.

    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    engine : str, default `'skopt'`
        If 'optuna':
            Bayesian optimization runs through the optuna library 

        If 'skopt':
            Bayesian optimization runs through the skopt library

    kwargs_create_study : dict, default `{'direction':'minimize', 'sampler':TPESampler(seed=123)}`
        Only applies to engine='optuna'.
            Keyword arguments (key, value mappings) to pass to optuna.create_study.

    kwargs_study_optimize : dict, default `{}`
        Only applies to engine='optuna'.
            Other keyword arguments (key, value mappings) to pass to study.optimize().

    kwargs_gp_minimize : dict, default `{}`
        Only applies to engine='skopt'.
            Other keyword arguments (key, value mappings) to pass to skopt.gp_minimize().

    Returns 
    -------
    results: pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    results_opt_best: optuna object (optuna), scipy object (skopt)   
        If optuna engine:
            The best optimization result returned as a FrozenTrial optuna object.

        If skopt engine:
            The best optimization result returned as a OptimizeResult object.
    '''

    if engine not in ['optuna', 'skopt']:
        raise Exception(
                f'''`engine` only allows 'optuna' or 'skopt', got {engine}.'''
              )

    if engine == 'optuna':
        results, results_opt_best = _bayesian_search_optuna(
                                        forecaster            = forecaster,
                                        y                     = y,
                                        exog                  = exog,
                                        lags_grid             = lags_grid,
                                        search_space          = search_space,
                                        steps                 = steps,
                                        metric                = metric,
                                        refit                 = refit,
                                        initial_train_size    = initial_train_size,
                                        fixed_train_size      = fixed_train_size,
                                        n_trials              = n_trials,
                                        random_state          = random_state,
                                        return_best           = return_best,
                                        verbose               = verbose,
                                        kwargs_create_study   = kwargs_create_study,
                                        kwargs_study_optimize = kwargs_study_optimize
                                    )
    else:
        results, results_opt_best = _bayesian_search_skopt(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = exog,
                                        lags_grid          = lags_grid,
                                        search_space       = search_space,
                                        steps              = steps,
                                        metric             = metric,
                                        refit              = refit,
                                        initial_train_size = initial_train_size,
                                        fixed_train_size   = fixed_train_size,
                                        n_trials           = n_trials,
                                        random_state       = random_state,
                                        return_best        = return_best,
                                        verbose            = verbose,
                                        kwargs_gp_minimize = kwargs_gp_minimize
                                    )

    return results, results_opt_best


def _bayesian_search_optuna(
    forecaster,
    y: pd.Series,
    search_space: callable,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True,
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={}
) -> Tuple[pd.DataFrame, object]:
    '''
    Bayesian optimization for a Forecaster object using time series backtesting 
    and optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    search_space : callable
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        If callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size: int 
        Number of samples in the initial train split.
 
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregMultiOutput`.
        
    refit: bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    n_trials: int, default `10`
        Number of parameter settings that are sampled in each lag configuration.

    random_state: int, default `123`
        Sets a seed to the sampling for reproducible output.

    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    kwargs_create_study : dict, default `{'direction':'minimize', 'sampler':TPESampler(seed=123)}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study.

    kwargs_study_optimize : dict, default `{}`
        Other keyword arguments (key, value mappings) to pass to study.optimize().

    Returns 
    -------
    results: pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    results_opt_best: optuna object
        The best optimization result returned as a FrozenTrial optuna object.
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
    results_opt_best = None

    # Objective function using backtesting_forecaster
    def _objective(
        trial,
        forecaster         = forecaster,
        y                  = y,
        exog               = exog,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        steps              = steps,
        metric             = metric,
        refit              = refit,
        verbose            = verbose,
        search_space       = search_space,
    ) -> float:
        
        forecaster.set_params(**search_space(trial))
        
        metric, _ = backtesting_forecaster(
                        forecaster         = forecaster,
                        y                  = y,
                        exog               = exog,
                        steps              = steps,
                        metric             = metric,
                        initial_train_size = initial_train_size,
                        fixed_train_size   = fixed_train_size,
                        refit              = refit,
                        verbose            = verbose
                        )

        return abs(metric)

    print(
        f'''Number of models compared: {n_trials*len(lags_grid)}, {n_trials} bayesian search in each lag configuration.'''
    )

    for lags in tqdm(lags_grid, desc='loop lags_grid', position=0, ncols=90):
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        study = optuna.create_study(**kwargs_create_study)

        if 'sampler' not in kwargs_create_study.keys():
            study.sampler = TPESampler(seed=random_state)

        study.optimize(_objective, n_trials=n_trials, **kwargs_study_optimize)

        best_trial = study.best_trial

        if search_space(best_trial).keys() != best_trial.params.keys():
            raise Exception(
                f'''Some of the key values do not match the search_space key names.
                Dict keys     : {list(search_space(best_trial).keys())}
                Trial objects : {list(best_trial.params.keys())}.'''
                )

        for trial in study.get_trials():
            params_list.append(trial.params)
            lags_list.append(lags)
            metric_list.append(trial.value)
        
        if results_opt_best is None:
            results_opt_best = best_trial
        else:
            if best_trial.value < results_opt_best.value:
                results_opt_best = best_trial
        
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
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results, results_opt_best


def _bayesian_search_skopt(
    forecaster,
    y: pd.Series,
    search_space: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True,
    kwargs_gp_minimize: dict={}
) -> Tuple[pd.DataFrame, object]:
    '''
    Bayesian optimization for a Forecaster object using time series backtesting and skopt library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    search_space : dict
        Dictionary with parameters names (`str`) as keys and Space object from skopt 
        (Real, Integer, Categorical) as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        It callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size: int 
        Number of samples in the initial train split.
 
    fixed_train_size: bool, default `True`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregMultiOutput`.
        
    refit: bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    n_trials: int, default `10`
        Number of parameter settings that are sampled in each lag configuration.

    random_state: int, default `123`
        Sets a seed to the sampling for reproducible output.

    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    kwargs_gp_minimize : dict, default `{}`
        Other keyword arguments (key, value mappings) to pass to skopt.gp_minimize().

    Returns 
    -------
    results: pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    results_opt_best: scipy object
        The best optimization result returned as a OptimizeResult object.
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
    results_opt_best = None

    for key in search_space.keys():
        if key != search_space[key].name:
            raise Exception(
                f'''Some of the key values do not match the Space object name from skopt.
                    {key} != {search_space[key].name}.'''
            )

    search_space = list(search_space.values())

    # Objective function using backtesting_forecaster
    @use_named_args(search_space)
    def _objective(
        forecaster         = forecaster,
        y                  = y,
        exog               = exog,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        steps              = steps,
        metric             = metric,
        refit              = refit,
        verbose            = verbose,
        **params
    ) -> float:
        
        forecaster.set_params(**params)
        
        metric, _ = backtesting_forecaster(
                        forecaster         = forecaster,
                        y                  = y,
                        exog               = exog,
                        steps              = steps,
                        metric             = metric,
                        initial_train_size = initial_train_size,
                        fixed_train_size   = fixed_train_size,
                        refit              = refit,
                        verbose            = verbose
                    )

        return abs(metric)

    print(
        f'''Number of models compared: {n_trials*len(lags_grid)}, {n_trials} bayesian search in each lag configuration.'''
    )

    for lags in tqdm(lags_grid, desc='loop lags_grid', position=0, ncols=90):
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        results_opt = gp_minimize(
                        func         = _objective,
                        dimensions   = search_space,
                        n_calls      = n_trials,
                        random_state = random_state,
                        **kwargs_gp_minimize
                      )

        for i, x in enumerate(results_opt.x_iters):
            params = {}
            for j, x in enumerate(search_space):
                params[x.name] = results_opt.x_iters[i][j]
            
            params_list.append(params)
            lags_list.append(lags)
            metric_list.append(results_opt.func_vals[i])

        if results_opt_best is None:
            results_opt_best = results_opt
        else:
            if results_opt.fun < results_opt_best.fun:
                results_opt_best = results_opt
        
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
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )

    return results, results_opt_best