################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
import warnings
import logging
from copy import deepcopy
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn.exceptions import NotFittedError
import optuna
from optuna.samplers import TPESampler, RandomSampler
optuna.logging.set_verbosity(optuna.logging.WARNING) # disable optuna logs

from ..exceptions import LongTrainingWarning

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
    """
    Split indices of a time series into multiple train-test pairs. The order 
    is maintained and the training set increases in each iteration.
    
    Parameters
    ----------        
    y : 1d numpy ndarray, pandas Series
        Training time series values. 
    
    initial_train_size : int 
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
    
    """
    
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
                f"last {remainder} observations are discarded."
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


def _create_backtesting_folds(
    y: pd.Series,
    test_size: int,
    initial_train_size: Union[int, None],
    gap: int,
    refit: bool=False,
    fixed_train_size: bool=True,
    allow_incomplete_fold: bool=True,
    return_all_indexes: bool=False,
    verbose: bool=True
) -> list:
    """
    Provides train/test indices (position) to split time series data samples that
    are observed at fixed time intervals, in train/test sets. In each split, test
    indices must be higher than before.

    Three arrays are returned for each fold with the position of train, test
    including the gap, and test excluding the gap. The gap is the number of
    samples to exclude from the end of each train set before the test set. The
    test excluding the gap is the one that must be used to make evaluate the
    model. The test including the gap is provided for convenience.

    Returned indexes are not the indexes of the original time series, but the
    positional indexes of the samples in the time series. For example, if the   
    original time series is `y = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`, the
    returned indexes for the first fold if  `test_size = 4`, `gap = 1` and 
    `initial_train_size = 2` are: `[[0, 1], [2, 3, 4, 5], [3, 4, 5]]]`. This means
    that the first fold is using the samples with positional indexes 0 and 1 in
    the time series as training set, and the samples with positional indexes 2,
    3, 4 and 5 as test set, but only the samples with positional indexes 3, 4 and
    5 should be used to evaluate the model since `gap = 1`. The second fold would
    be `[[0, 1, 2, 3], [4, 5, 6, 7], [5, 6, 7]]`, and so on.
    
    Parameters
    ----------        
    y : pandas Series
        Time series values. 
    
    initial_train_size : int, None
        Size of the training set in the first fold. If `None` or 0, the initial
        fold does not include a training set.
        
    test_size : int
        Size of the test set in each fold.

    gap : int, default 0
        Number of samples to exclude from the end of each train set before the
        test set.
        
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to be incomplete if it does not reach `test_size`
        samples in the test set. Otherwise, the last fold is excluded.

    return_all_indexes : bool, default `False`
        If `True`, return all the indexes included in each fold. If `False`, return
        only the first and last index of each partition in each fold.
        
    verbose : bool, default `True`
        Print information if the folds created.

    Returns
    ------
    folds : list
        List containing the indices (position) of `y` for training, test including
        the gap, and test excluding the gap for each fold.
    
    """

    if not isinstance(y, pd.Series):
        raise ValueError("`y` must be a pandas Series.")
    
    if not isinstance(test_size, int):
        raise ValueError("`test_size` must be an integer.")
    if test_size < 1:
        raise ValueError("`test_size` must be greater than 0.")
    if not isinstance(gap, int):
        raise ValueError("`gap` must be an integer.")
    if gap < 1:
        raise ValueError("`gap` must be greater than 0.")
    if initial_train_size is not None and not isinstance(initial_train_size, int):
        raise ValueError("`initial_train_size` must be an integer or None.")
    if initial_train_size is not None and initial_train_size < 1:
        raise ValueError("`initial_train_size` must be greater than 0 or None.")

    if initial_train_size is None:
        initial_train_size = 0

    if initial_train_size + gap > len(y):
        raise ValueError(
            "The combination of initial_train_size and gap can not be larger"
            "than the length of y."
        )
    
    idx = range(len(y))
    folds = []
    i = 0
    last_fold_excluded = False

    while initial_train_size + (i * test_size) + gap <= len(y):

        if refit:
            # If fixed_train_size the train size doesn't increase but moves by 
            # `test_size` positions in each iteration. If False, the train size
            # increases by `test_size` in each iteration.
            train_idx_start = i * (test_size) if fixed_train_size else 0
            train_idx_end = initial_train_size + i * (test_size)
            test_idx_start = train_idx_end
        else:
            # The train size doesn't increase and doesn't move.
            train_idx_start = 0
            train_idx_end = initial_train_size
            test_idx_start = initial_train_size + i * (test_size)

        try:
            test_idx_end = test_idx_start + gap + test_size
        except:
            test_idx_end = len(y)

        partitions = [
            idx[train_idx_start : train_idx_end],
            idx[test_idx_start : test_idx_end],
            idx[test_idx_start + gap : test_idx_end]
        ]
        partitions =[partition if len(partition) > 0 else None for partition in partitions]
        folds.append(partitions)

        i += 1

    remainder = test_size - (test_size - len(folds[-1][1]))

    if not allow_incomplete_fold:
        if remainder != 0:
            folds = folds[:-1]
            last_fold_excluded = True
            remainder = 0

    if verbose:
        print(f"Information of backtesting process")
        print(f"----------------------------------")
        print(f"Number of observations used for initial training: {initial_train_size}")
        print(f"Number of observations used for backtesting: {len(y) - initial_train_size}")
        print(f"    Number of folds: {len(folds)}")
        print(f"    Number of steps per fold: {test_size}")
        print(f"    Number of steps to exclude from the end of each train set before test (gap): {gap}")
        if last_fold_excluded:
            print(f"    Last fold has been excluded because it was incomplete.")
        if remainder !=0:
            print(f"    Last fold only includes {remainder} observations.")
        print("")

        for i, fold in enumerate(folds):
            training_start = y.index[fold[0][0]] if fold[0] is not None else None
            training_end = y.index[fold[0][-1]] if fold[0] is not None else None
            training_length = len(fold[0]) if fold[0] is not None else 0
            validation_start = y.index[fold[2][0]]
            validation_end = y.index[fold[2][-1]]
            validation_length = len(fold[2])
            print(f"Fold: {i}")
            print(
                f"    Training:   {training_start} -- {training_end} (n={training_length})"
            )
            print(
                f"    Validation: {validation_start} -- {validation_end} (n={validation_length})"
            )

    if not return_all_indexes:
        folds = [
            [[fold[0][0], fold[0][-1]], [fold[1][0], fold[1][-1]], [fold[2][0], fold[2][-1]]] 
            for fold in folds
        ]

    return folds
        
        
def _get_metric(
    metric: str
) -> Callable:
    """
    Get the corresponding scikit-learn function to calculate the metric.
    
    Parameters
    ----------
    metric : str
        Metric used to quantify the goodness of fit of the model. Available metrics: 
        {'mean_squared_error', 'mean_absolute_error', 
         'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
    Returns 
    -------
    metric : Callable
        scikit-learn function to calculate the desired metric.
    
    """
    
    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error', 'mean_squared_log_error']:
        raise ValueError(
            (f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error', "
             f"'mean_absolute_percentage_error' and 'mean_squared_log_error'. Got {metric}.")
        )
    
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error,
        'mean_squared_log_error': mean_squared_log_error
    }
    
    metric = metrics[metric]
    
    return metric


def _backtesting_forecaster_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    gap: int = 0,
    fixed_train_size: bool=True,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False,
    show_progress: bool=True
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of forecaster model with a re-fitting strategy. A copy of the  
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
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
        
    y : pandas Series
        Training time series.
        
    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.
    
    initial_train_size : int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
        
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    gap : int, default 0
        Number of samples to exclude from the end of each train set before the
        test set.
        
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to be incomplete if it does not reach `steps`
        samples in the test set. Otherwise, the last fold is excluded.
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`. If `None`, no
        intervals are estimated. Only available for forecaster of type 
        ForecasterAutoreg and ForecasterAutoregCustom.
    
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

    show_progress: bool, default `True`
        Whether to show a progress bar. Defaults to True.

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

    folds = _create_backtesting_folds(
                y                     = y,
                test_size             = steps,
                initial_train_size    = initial_train_size,
                gap                   = gap,
                refit                 = True,
                fixed_train_size      = fixed_train_size,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = False,
                verbose               = verbose  
            )
        
    if type(forecaster).__name__ != 'ForecasterAutoregDirect' and len(folds) > 50:
        warnings.warn(
            (f"The forecaster will be fit {folds} times. This can take substantial "
             f"amounts of time. If not feasible, try with `refit = False`.\n"),
            LongTrainingWarning
        )
    elif type(forecaster).__name__ == 'ForecasterAutoregDirect' and len(folds)*forecaster.steps > 50:
        warnings.warn(
            (f"The forecaster will be fit {len(folds)*forecaster.steps} times "
             f"({len(folds)} folds * {forecaster.steps} regressors). This can take "
             f"substantial amounts of time. If not feasible, try with `refit = False`.\n"),
             LongTrainingWarning
        )

    backtest_predictions = []
    
    for fold in tqdm(folds) if show_progress else folds:
        # In each iteration the model is fitted before making predictions.
        # if fixed_train_size the train size doesn't increase but moves by `steps`
        # in each iteration. if false the train size increases by `steps` in each
        # iteration.
        train_idx_start = fold[0][0] if fixed_train_size else 0
        train_idx_end   = fold[0][1]
        test_idx_start  = fold[1][0]
        test_idx_end    = fold[1][1]
        pred_idx_start  = fold[2][0]
        pred_idx_end    = fold[2][1]

        y_train = y.iloc[train_idx_start:train_idx_end, ]
        exog_train = exog.iloc[train_idx_start:train_idx_end, ] if exog is not None else None
        next_window_exog = exog.iloc[test_idx_start:test_idx_end, ] if exog is not None else None

        forecaster.fit(y=y_train, exog=exog_train)

        steps = len(fold[1])

        if interval is None:
            pred = forecaster.predict(steps=steps, exog=next_window_exog)
        else:
            pred = forecaster.predict_interval(
                       steps               = steps,
                       exog                = next_window_exog,
                       interval            = interval,
                       n_boot              = n_boot,
                       random_state        = random_state,
                       in_sample_residuals = in_sample_residuals
                   )

        pred = pred.iloc[pred_idx_start:pred_idx_end, ]
        backtest_predictions.append(pred)
    
    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    if isinstance(metric, list):
        metrics_values = [
            m(
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


def _backtesting_forecaster_no_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    gap: int = 0,
    initial_train_size: Optional[int]=None,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False,
    show_progress: bool=True
) -> Tuple[Union[float, list], pd.DataFrame]:
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
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
        
    y : pandas Series
        Training time series.
        
    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.
    
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

    gap : int, default 0
        Number of samples to exclude from the end of each train set before the
        test set.
        
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to be incomplete if it does not reach `steps`
        samples in the test set. Otherwise, the last fold is excluded.
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`. If `None`, no
        intervals are estimated. Only available for forecaster of type 
        ForecasterAutoreg and ForecasterAutoregCustom.
            
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

    show_progress: bool, default `True`
        Whether to show a progress bar. Defaults to True.

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

    for i in tqdm(range(folds)) if show_progress else range(folds):
        # Since the model is only fitted with the initial_train_size, last_window
        # and next_window_exog must be updated to include the data needed to make
        # predictions.
        last_window_end   = initial_train_size + i * steps
        last_window_start = last_window_end - window_size 
        last_window_y     = y.iloc[last_window_start:last_window_end]
        
        next_window_exog = exog.iloc[last_window_end:last_window_end + steps, ] if exog is not None else None
    
        if i == folds - 1: # last fold
            # If remainder > 0, only the remaining steps need to be predicted
            steps = steps if remainder == 0 else remainder
        
        if interval is None:
            pred = forecaster.predict(
                       steps       = steps,
                       last_window = last_window_y,
                       exog        = next_window_exog
                   )
        else:
            pred = forecaster.predict_interval(
                       steps               = steps,
                       last_window         = last_window_y,
                       exog                = next_window_exog,
                       interval            = interval,
                       n_boot              = n_boot,
                       random_state        = random_state,
                       in_sample_residuals = in_sample_residuals
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


def backtesting_forecaster(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int]=None,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False,
    show_progress: bool=True
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of forecaster model.

    If `refit` is False, the model is trained only once using the `initial_train_size`
    first observations. If `refit` is True, the model is trained in each iteration
    increasing the training set. A copy of the original forecaster is created so 
    it is not modified during the process.

    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
        
    y : pandas Series
        Training time series.
    
    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.
    
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

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`. If `None`, no
        intervals are estimated. Only available for forecaster of type 
        ForecasterAutoreg and ForecasterAutoregCustom.
            
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

    show_progress: bool, default `True`
        Whether to show a progress bar. Defaults to True.

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

    if initial_train_size is not None and not isinstance(initial_train_size, (int, np.int64, np.int32)):
        raise TypeError(
            (f'If used, `initial_train_size` must be an integer greater than '
             f'the window_size of the forecaster. Got {type(initial_train_size)}.')
        )

    if initial_train_size is not None and initial_train_size >= len(y):
        raise ValueError(
            (f'If used, `initial_train_size` must be an integer '
             f'smaller than the length of `y` ({len(y)}).')
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

    if interval is not None and type(forecaster).__name__ == 'ForecasterAutoregDirect':
        raise TypeError(
            ('Interval prediction is only available when forecaster is of type '
             'ForecasterAutoreg or ForecasterAutoregCustom.')
        )
    
    if type(forecaster).__name__ not in ['ForecasterAutoreg', 'ForecasterAutoregCustom', 'ForecasterAutoregDirect']:
        raise TypeError(
            ('`forecaster` must be of type `ForecasterAutoreg`, `ForecasterAutoregCustom` '
             'or `ForecasterAutoregDirect`, for all other types of forecasters '
             'use the functions available in the `model_selection` module.')
        )
    
    if refit:
        metrics_values, backtest_predictions = _backtesting_forecaster_refit(
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
            verbose             = verbose,
            show_progress       = show_progress
        )
    else:
        metrics_values, backtest_predictions = _backtesting_forecaster_no_refit(
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
            verbose             = verbose,
            show_progress       = show_progress
        )

    return metrics_values, backtest_predictions


def grid_search_forecaster(
    forecaster,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
        
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
    metric: Union[str, Callable, list],
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
    """
    Random search over specified parameter values or distributions for a Forecaster object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forcaster model.
        
    y : pandas Series
        Training time series. 
        
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.

    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
        
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
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.
    
    """

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
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
        
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

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            f'`exog` must have same number of samples as `y`. '
            f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
        )

    if type(forecaster).__name__ == 'ForecasterAutoregCustom':
        if lags_grid is not None:
            warnings.warn(
                '`lags_grid` ignored if forecaster is an instance of `ForecasterAutoregCustom`.'
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

    print(f"Number of models compared: {len(param_grid)*len(lags_grid)}.")

    for lags in tqdm(lags_grid, desc='lags grid', position=0): #ncols=90
        
        if type(forecaster).__name__ in ['ForecasterAutoreg', 'ForecasterAutoregDirect']:
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        for params in tqdm(param_grid, desc='params grid', position=1, leave=False): #ncols=90

            forecaster.set_params(params)
            metrics_values = backtesting_forecaster(
                                 forecaster         = forecaster,
                                 y                  = y,
                                 steps              = steps,
                                 metric             = metric,
                                 initial_train_size = initial_train_size,
                                 fixed_train_size   = fixed_train_size,
                                 exog               = exog,
                                 refit              = refit,
                                 interval           = None,
                                 verbose            = verbose,
                                 show_progress      = False
                             )[0]
            warnings.filterwarnings('ignore', category=RuntimeWarning, message= "The forecaster will be fit.*")
            lags_list.append(lags)
            params_list.append(params)
            for m, m_value in zip(metric, metrics_values):
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(m_value)

    results = pd.DataFrame({
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
        
        if type(forecaster).__name__ in ['ForecasterAutoreg', 'ForecasterAutoregDirect']:
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)
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
    search_space: Union[Callable, dict],
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    verbose: bool=True,
    engine: str='optuna',
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={},
    kwargs_gp_minimize: Any='deprecated'
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using time series backtesting and 
    optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forcaster model.
        
    y : pandas Series
        Training time series. 
        
    search_space : Callable (optuna), dict (skopt)
        If optuna engine: Callable
            Function with argument `trial` which returns a dictionary with parameters names 
            (`str`) as keys and Trial object from optuna (trial.suggest_float, 
            trial.suggest_int, trial.suggest_categorical) as values.

        If skopt engine: dict
            Dictionary with parameters names (`str`) as keys and Space object from skopt 
            (Real, Integer, Categorical) as values.
            **Deprecated in version 0.7.0**

    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
        When using engine "skopt", the minimum value is 10.

    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output.

    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    engine : str, default `'optuna'`
        If 'optuna':
            Bayesian optimization runs through the optuna library.

        If 'skopt':
            Bayesian optimization runs through the skopt library.
            **Deprecated in version 0.7.0**

    kwargs_create_study : dict, default `{'direction':'minimize', 'sampler':TPESampler(seed=123)}`
        Only applies to engine='optuna'.
            Keyword arguments (key, value mappings) to pass to optuna.create_study.

    kwargs_study_optimize : dict, default `{}`
        Only applies to engine='optuna'.
            Other keyword arguments (key, value mappings) to pass to study.optimize().

    kwargs_gp_minimize : dict, default `{}`
        Only applies to engine='skopt'.
            Other keyword arguments (key, value mappings) to pass to skopt.gp_minimize().
            **Deprecated in version 0.7.0**

    Returns 
    -------
    results : pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    results_opt_best : optuna object (optuna), scipy object (skopt)   
        If optuna engine:
            The best optimization result returned as a FrozenTrial optuna object.

        If skopt engine:
            The best optimization result returned as a OptimizeResult object.
            **Deprecated in version 0.7.0**
    
    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            f'`exog` must have same number of samples as `y`. '
            f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
        )
    
    if engine == 'skopt':
        warnings.warn(
            ("The engine 'skopt' for `bayesian_search_forecaster` is deprecated "
             "in favor of 'optuna' engine. To continue using it, use skforecast "
             "0.6.0. The optimization will be performed using the 'optuna' engine.")
        )
        engine = 'optuna'

    if engine not in ['optuna']:
        raise ValueError(
            f"""`engine` only allows 'optuna', got {engine}."""
        )

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

    return results, results_opt_best


def _bayesian_search_optuna(
    forecaster,
    y: pd.Series,
    search_space: Callable,
    steps: int,
    metric: Union[str, Callable, list],
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
    """
    Bayesian optimization for a Forecaster object using time series backtesting 
    and optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forcaster model.
        
    y : pandas Series
        Training time series. 
        
    search_space : Callable
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.

    steps : int
        Number of steps to predict.
        
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
    
        If Callable:
            Function with arguments y_true, y_pred that returns a float.

        If list:
            List containing several strings and/or Callable.

    initial_train_size : int 
        Number of samples in the initial train split.
 
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
        
    refit : bool, default `False`
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.

    random_state : int, default `123`
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
    results : pandas DataFrame
        Results for each combination of parameters.
            column lags = predictions.
            column params = lower bound of the interval.
            column metric = metric value estimated for the combination of parameters.
            additional n columns with param = value.

    results_opt_best : optuna object
        The best optimization result returned as a FrozenTrial optuna object.

    """

    if type(forecaster).__name__ == 'ForecasterAutoregCustom':
        if lags_grid is not None:
            warnings.warn(
                '`lags_grid` ignored if forecaster is an instance of `ForecasterAutoregCustom`.'
            )
        lags_grid = ['custom predictors']
        
    elif lags_grid is None:
        lags_grid = [forecaster.lags]
   
    lags_list = []
    params_list = []
    results_opt_best = None
    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            'When `metric` is a `list`, each metric name must be unique.'
        )

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
        
        forecaster.set_params(search_space(trial))
        
        metrics, _ = backtesting_forecaster(
                         forecaster         = forecaster,
                         y                  = y,
                         exog               = exog,
                         steps              = steps,
                         metric             = metric,
                         initial_train_size = initial_train_size,
                         fixed_train_size   = fixed_train_size,
                         refit              = refit,
                         verbose            = verbose,
                         show_progress      = False
                     )
        # Store metrics in the variable metric_values defined outside _objective.
        nonlocal metric_values
        metric_values.append(metrics)

        return abs(metrics[0])

    print(
        f"""Number of models compared: {n_trials*len(lags_grid)},
         {n_trials} bayesian search in each lag configuration."""
    )

    for lags in tqdm(lags_grid, desc='lags grid', position=0): #ncols=90
                
        metric_values = [] # This variable will be modified inside _objective function. 
        # It is a trick to extract multiple values from _objective function since
        # only the optimized value can be returned.

        if type(forecaster).__name__ in ['ForecasterAutoreg', 'ForecasterAutoregDirect']:
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        if 'sampler' in kwargs_create_study.keys():
            kwargs_create_study['sampler']._rng = np.random.RandomState(random_state)
            kwargs_create_study['sampler']._random_sampler = RandomSampler(seed=random_state)    

        study = optuna.create_study(**kwargs_create_study)

        if 'sampler' not in kwargs_create_study.keys():
            study.sampler = TPESampler(seed=random_state)

        study.optimize(_objective, n_trials=n_trials, **kwargs_study_optimize)

        best_trial = study.best_trial

        if search_space(best_trial).keys() != best_trial.params.keys():
            raise ValueError(
                f"""Some of the key values do not match the search_space key names.
                Dict keys     : {list(search_space(best_trial).keys())}
                Trial objects : {list(best_trial.params.keys())}."""
            )
        
        for i, trial in enumerate(study.get_trials()):
            params_list.append(trial.params)
            lags_list.append(lags)

            for m, m_values in zip(metric, metric_values[i]):
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(m_values)
        
        if results_opt_best is None:
            results_opt_best = best_trial
        else:
            if best_trial.value < results_opt_best.value:
                results_opt_best = best_trial
        
    results = pd.DataFrame({
                'lags'  : lags_list,
                'params': params_list,
                **metric_dict})

    results = results.sort_values(by=list(metric_dict.keys())[0], ascending=True)
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results['lags'].iloc[0]
        best_params = results['params'].iloc[0]
        best_metric = results[list(metric_dict.keys())[0]].iloc[0]
        
        if type(forecaster).__name__ in ['ForecasterAutoreg', 'ForecasterAutoregDirect']:
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)
        forecaster.fit(y=y, exog=exog)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results, results_opt_best