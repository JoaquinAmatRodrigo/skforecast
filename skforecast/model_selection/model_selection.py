################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import warnings
import logging
from copy import deepcopy
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
import sklearn.pipeline
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import optuna
from optuna.samplers import TPESampler, RandomSampler

from ..exceptions import LongTrainingWarning
from ..exceptions import IgnoredArgumentWarning
from ..utils import check_backtesting_input
from ..utils import select_n_jobs_backtesting

optuna.logging.set_verbosity(optuna.logging.WARNING) # disable optuna logs

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def _backtesting_forecaster_verbose(
    index_values: pd.Index,
    steps: int,
    initial_train_size: int,
    folds: int,
    remainder: int,
    refit: Optional[Union[bool, int]]=False,
    fixed_train_size: bool=True,
    differentiation: Optional[int]=None
) -> None:
    """
    Verbose for backtesting_forecaster functions.
    
    Parameters
    ----------
    index_values : pandas Index
        Values of the index of the series.
    steps : int
        Number of steps to predict.
    initial_train_size : int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
    folds : int
        Number of backtesting stages.
    remainder : int
        Number of observations in the last backtesting stage. 
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.

    Returns
    -------
    None
    
    """

    print("Information of backtesting process")
    print("----------------------------------")
    print(f"Number of observations used for initial training: {initial_train_size}")
    if differentiation is not None:
        print(f"Number of observations used for initial differentiation: {differentiation}")
    print(f"Number of observations used for backtesting: {len(index_values) - initial_train_size}")
    print(f"    Number of folds: {folds}")
    print(f"    Number of steps per fold: {steps}")
    if remainder != 0:
        print(f"    Last fold only includes {remainder} observations.")
    print("")
    for i in range(folds):
        if refit:
            # if fixed_train_size the train size doesn't increase but moves by
            # `steps` in each iteration. if false the train size increases by `steps`
            # in each iteration.
            train_idx_start = i * steps if fixed_train_size else 0
            train_idx_end = initial_train_size + i * steps
        else:
            # The train size doesn't increase and doesn't move
            train_idx_start = 0
            train_idx_end = initial_train_size
        last_window_end = initial_train_size + i * steps
        print(f"Data partition in fold: {i}")
        if i < folds - 1:
            print(
                f"    Training:   {index_values[train_idx_start]} -- {index_values[train_idx_end - 1]}"
                f"  (n={len(index_values[train_idx_start:train_idx_end])})"
            )
            print(
                f"    Validation: {index_values[last_window_end]} -- {index_values[last_window_end + steps - 1]}"
                f"  (n={len(index_values[last_window_end:last_window_end + steps])})"
            )
        else:
            print(
                f"    Training:   {index_values[train_idx_start]} -- {index_values[train_idx_end - 1]}"
                f"  (n={len(index_values[train_idx_start:train_idx_end])})"
            )
            print(
                f"    Validation: {index_values[last_window_end]} -- {index_values[-1]}"
                F"  (n={len(index_values[last_window_end:])})"
            )
    print("")

    return


def _create_backtesting_folds(
    data: Union[pd.Series, pd.DataFrame],
    window_size: int,
    initial_train_size: Union[int, None],
    test_size: int,
    externally_fitted: bool=False,
    refit: Optional[Union[bool, int]]=False,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    return_all_indexes: bool=False,
    differentiation: Optional[int]=None,
    verbose: bool=True
) -> list:
    """
    This function is designed to work after passing the `check_backtesting_input` 
    function from `skforecast.utils`.

    Provides train/test indices (position) to split time series data samples that
    are observed at fixed time intervals, in train/test sets. In each split, test
    indices must be higher than before.

    Four arrays are returned for each fold with the position of train, window size, 
    test including the gap, and test excluding the gap. The gap is the number of
    samples to exclude from the end of each train set before the test set. The
    test excluding the gap is the one that must be used to make evaluate the
    model. The test including the gap is provided for convenience.

    Returned indexes are not the indexes of the original time series, but the
    positional indexes of the samples in the time series. For example, if the   
    original time series is `y = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`, the
    returned indexes for the first fold if  `test_size = 4`, `gap = 1` and 
    `initial_train_size = 2` with `window_size = 2` are: `[[0, 1], [0, 1], 
    [2, 3, 4, 5], [3, 4, 5]]]`. This means that the first fold is using the samples 
    with positional indexes 0 and 1 in the time series as training set, the samples 
    with positional indexes 0 and 1 as last window, and the samples with positional 
    indexes 2, 3, 4 and 5 as test set, but only the samples with positional indexes 
    3, 4 and 5 should be used to evaluate the model since `gap = 1`. The second fold 
    would be `[[0, 1, 2, 3], [2, 3], [4, 5, 6, 7], [5, 6, 7]]`, and so on.

    Each fold also provides information on whether the Forecaster needs to be 
    trained, `True` or `False`. The first fold flag will be always `False` since 
    the first fit is done inside _backtesting_forecaster function.
    
    Parameters
    ----------
    data : pandas Series, pandas DataFrame
        Time series values.
    window_size : int
        Size of the window needed to create the predictors.
    initial_train_size : int, None
        Size of the training set in the first fold. If `None` or 0, the initial
        fold does not include a training set.
    test_size : int
        Size of the test set in each fold.
    externally_fitted : bool, default `False`
        Flag indicating whether the forecaster is already trained. Only used when 
        `initial_train_size` is None and `refit` is False.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    return_all_indexes : bool, default `False`
        If `True`, return all the indexes included in each fold. If `False`, return
        only the first and last index of each partition in each fold.
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.
    verbose : bool, default `True`
        Print information if the folds created.

    Returns
    -------
    folds : list
        List containing the `y` indices (position) for training, last window, test 
        including the gap and test excluding the gap for each fold, and whether to
        fit the Forecaster.
    
    """
    
    idx = range(len(data))
    folds = []
    i = 0
    last_fold_excluded = False

    while initial_train_size + (i * test_size) + gap < len(data):

        if refit:
            # If `fixed_train_size` the train size doesn't increase but moves by 
            # `test_size` positions in each iteration. If `False`, the train size
            # increases by `test_size` in each iteration.
            train_idx_start = i * (test_size) if fixed_train_size else 0
            train_idx_end = initial_train_size + i * (test_size)
            test_idx_start = train_idx_end
        else:
            # The train size doesn't increase and doesn't move.
            train_idx_start = 0
            train_idx_end = initial_train_size
            test_idx_start = initial_train_size + i * (test_size)
        
        last_window_start = test_idx_start - window_size
        test_idx_end = test_idx_start + gap + test_size
    
        partitions = [
            idx[train_idx_start : train_idx_end],
            idx[last_window_start : test_idx_start],
            idx[test_idx_start : test_idx_end],
            idx[test_idx_start + gap : test_idx_end]
        ]
        folds.append(partitions)
        i += 1

    if not allow_incomplete_fold:
        if len(folds[-1][3]) < test_size:
            folds = folds[:-1]
            last_fold_excluded = True

    # Replace partitions inside folds with length 0 with `None`
    folds = [[partition if len(partition) > 0 else None 
              for partition in fold] 
             for fold in folds]

    # Create a flag to know whether to train the forecaster
    if refit == 0:
        refit = False
        
    if isinstance(refit, bool):
        fit_forecaster = [refit]*len(folds)
        fit_forecaster[0] = True
    else:
        fit_forecaster = [False]*len(folds)
        for i in range(0, len(fit_forecaster), refit): 
            fit_forecaster[i] = True
    
    for i in range(len(folds)): 
        folds[i].append(fit_forecaster[i])
        if fit_forecaster[i] is False:
            folds[i][0] = folds[i-1][0]

    # This is done to allow parallelization when `refit` is `False`. The initial 
    # Forecaster fit is outside the auxiliary function.
    folds[0][4] = False
    
    if verbose:
        print("Information of backtesting process")
        print("----------------------------------")
        if externally_fitted:
            print(f"An already trained forecaster is to be used. Window size: {window_size}")
        else:
            if differentiation is None:
                print(f"Number of observations used for initial training: {initial_train_size}")
            else:
                print(f"Number of observations used for initial training: {initial_train_size - differentiation}")
                print(f"    First {differentiation} observation/s in training sets are used for differentiation")
        print(f"Number of observations used for backtesting: {len(data) - initial_train_size}")
        print(f"    Number of folds: {len(folds)}")
        print(f"    Number of steps per fold: {test_size}")
        print(f"    Number of steps to exclude from the end of each train set before test (gap): {gap}")
        if last_fold_excluded:
            print("    Last fold has been excluded because it was incomplete.")
        if len(folds[-1][3]) < test_size:
            print(f"    Last fold only includes {len(folds[-1][3])} observations.")
        print("")

        if differentiation is None:
            differentiation = 0
        for i, fold in enumerate(folds):
            training_start    = data.index[fold[0][0] + differentiation] if fold[0] is not None else None
            training_end      = data.index[fold[0][-1]] if fold[0] is not None else None
            training_length   = len(fold[0]) - differentiation if fold[0] is not None else 0
            validation_start  = data.index[fold[3][0]]
            validation_end    = data.index[fold[3][-1]]
            validation_length = len(fold[3])

            print(f"Fold: {i}")
            if not externally_fitted:
                print(
                    f"    Training:   {training_start} -- {training_end}  (n={training_length})"
                )
            print(
                f"    Validation: {validation_start} -- {validation_end}  (n={validation_length})"
            )
        print("")

    if not return_all_indexes:
        # +1 to prevent iloc pandas from deleting the last observation
        folds = [
            [[fold[0][0], fold[0][-1]+1], 
             [fold[1][0], fold[1][-1]+1], 
             [fold[2][0], fold[2][-1]+1],
             [fold[3][0], fold[3][-1]+1],
             fold[4]] 
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


def _backtesting_forecaster(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int]=None,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: Optional[Union[bool, int]]=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=False,
    show_progress: bool=True
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of forecaster model.

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
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
            - If `string`: {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
            - If `Callable`: Function with arguments y_true, y_pred that returns 
            a float.
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
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`. If `None`, no
        intervals are estimated.
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals.  If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    metrics_value : float, list
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.

            - column pred: predictions.
            - column lower_bound: lower bound of the interval.
            - column upper_bound: upper bound of the interval.
    
    """

    forecaster = deepcopy(forecaster)
    
    if n_jobs == 'auto':
        if isinstance(forecaster.regressor, sklearn.pipeline.Pipeline):
            regressor_name = type(forecaster.regressor[-1]).__name__
        else:
            regressor_name = type(forecaster.regressor).__name__

        n_jobs = select_n_jobs_backtesting(
                     forecaster_name = type(forecaster).__name__,
                     regressor_name  = regressor_name,
                     refit           = refit
                 )
    else:
        n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    if not isinstance(metric, list):
        metrics = [_get_metric(metric=metric) if isinstance(metric, str) else metric]
    else:
        metrics = [_get_metric(metric=m) if isinstance(m, str) else m for m in metric]

    store_in_sample_residuals = False if interval is None else True

    if initial_train_size is not None:
        # First model training, this is done to allow parallelization when `refit` 
        # is `False`. The initial Forecaster fit is outside the auxiliary function.
        exog_train = exog.iloc[:initial_train_size, ] if exog is not None else None
        forecaster.fit(
            y                         = y.iloc[:initial_train_size, ],
            exog                      = exog_train,
            store_in_sample_residuals = store_in_sample_residuals
        )
        window_size = forecaster.window_size
        externally_fitted = False
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        window_size = forecaster.window_size
        initial_train_size = window_size
        externally_fitted = True

    # TODO: remove when all forecaster include differentiation
    if type(forecaster).__name__ != 'ForecasterAutoregDirect':
        differentiation = forecaster.differentiation
    else:
        differentiation = None

    folds = _create_backtesting_folds(
                data                  = y,
                window_size           = window_size,
                initial_train_size    = initial_train_size,
                test_size             = steps,
                externally_fitted     = externally_fitted,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = False,
                differentiation       = differentiation,
                verbose               = verbose  
            )

    if refit:
        n_of_fits = int(len(folds)/refit)
        if type(forecaster).__name__ != 'ForecasterAutoregDirect' and n_of_fits > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits} times. This can take substantial"
                 f" amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )
        elif type(forecaster).__name__ == 'ForecasterAutoregDirect' and n_of_fits*forecaster.steps > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits*forecaster.steps} times "
                 f"({n_of_fits} folds * {forecaster.steps} regressors). This can take "
                 f"substantial amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )
    
    if show_progress:
        folds = tqdm(folds)

    def _fit_predict_forecaster(y, exog, forecaster, interval, fold):
        """
        Fit the forecaster and predict `steps` ahead. This is an auxiliary 
        function used to parallelize the backtesting_forecaster function.
        """

        train_idx_start   = fold[0][0]
        train_idx_end     = fold[0][1]
        last_window_start = fold[1][0]
        last_window_end   = fold[1][1]
        test_idx_start    = fold[2][0]
        test_idx_end      = fold[2][1]

        if fold[4] is False:
            # When the model is not fitted, last_window must be updated to include 
            # the data needed to make predictions.
            last_window_y = y.iloc[last_window_start:last_window_end]
        else:
            # The model is fitted before making predictions. If `fixed_train_size`  
            # the train size doesn't increase but moves by `steps` in each iteration. 
            # If `False` the train size increases by `steps` in each  iteration.
            y_train = y.iloc[train_idx_start:train_idx_end, ]
            exog_train = exog.iloc[train_idx_start:train_idx_end, ] if exog is not None else None
            last_window_y = None
            forecaster.fit(
                y                         = y_train, 
                exog                      = exog_train, 
                store_in_sample_residuals = store_in_sample_residuals
            )

        next_window_exog = exog.iloc[test_idx_start:test_idx_end, ] if exog is not None else None

        steps = len(range(test_idx_start, test_idx_end))
        if type(forecaster).__name__ == 'ForecasterAutoregDirect' and gap > 0:
            # Select only the steps that need to be predicted if gap > 0
            test_idx_start = fold[3][0]
            test_idx_end   = fold[3][1]
            steps = list(np.arange(len(range(test_idx_start, test_idx_end))) + gap + 1)

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
        
        if type(forecaster).__name__ != 'ForecasterAutoregDirect' and gap > 0:
            pred = pred.iloc[gap:, ]

        return pred

    backtest_predictions = (
        Parallel(n_jobs=n_jobs)
        (delayed(_fit_predict_forecaster)
        (y=y, exog=exog, forecaster=forecaster, interval=interval, fold=fold)
         for fold in folds)
    )

    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    metrics_values = [m(
                        y_true = y.loc[backtest_predictions.index],
                        y_pred = backtest_predictions['pred']
                      ) for m in metrics
                     ]
    
    if not isinstance(metric, list):
        metrics_values = metrics_values[0]

    return metrics_values, backtest_predictions


def backtesting_forecaster(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int]=None,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: Optional[Union[bool, int]]=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=False,
    show_progress: bool=True
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of forecaster model.

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
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
            - If `string`: {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
            - If `Callable`: Function with arguments y_true, y_pred that returns 
            a float.
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
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`. If `None`, no
        intervals are estimated.
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals.  If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    metrics_value : float, list
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.

            - column pred: predictions.
            - column lower_bound: lower bound of the interval.
            - column upper_bound: upper bound of the interval.
    
    """
    forecaters_allowed = [
        'ForecasterAutoreg', 
        'ForecasterAutoregCustom', 
        'ForecasterAutoregDirect',
        'ForecasterEquivalentDate'
    ]
    if type(forecaster).__name__ not in forecaters_allowed:
        raise TypeError(
            (f"`forecaster` must be of type {forecaters_allowed}, for all other types of "
             f" forecasters use the functions available in the other `model_selection` "
             f"modules.")
        )
    
    check_backtesting_input(
        forecaster            = forecaster,
        steps                 = steps,
        metric                = metric,
        y                     = y,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        refit                 = refit,
        interval              = interval,
        n_boot                = n_boot,
        random_state          = random_state,
        in_sample_residuals   = in_sample_residuals,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
    )
    
    if type(forecaster).__name__ == 'ForecasterAutoregDirect' and \
       forecaster.steps < steps + gap:
        raise ValueError(
            ("When using a ForecasterAutoregDirect, the combination of steps "
             f"+ gap ({steps+gap}) cannot be greater than the `steps` parameter "
             f"declared when the forecaster is initialized ({forecaster.steps}).")
        )
    
    metrics_values, backtest_predictions = _backtesting_forecaster(
        forecaster            = forecaster,
        y                     = y,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        exog                  = exog,
        refit                 = refit,
        interval              = interval,
        n_boot                = n_boot,
        random_state          = random_state,
        in_sample_residuals   = in_sample_residuals,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
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
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: Optional[Union[bool, int]]=False,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series. 
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
            - If `string`: {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
            - If `Callable`: Function with arguments y_true, y_pred that returns 
            a float.
            - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column lags: lags configuration for each iteration.
            - column params: parameters configuration for each iteration.
            - column metric: metric value estimated for each iteration.
            - additional n columns with param = value.
    
    """

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters(
        forecaster            = forecaster,
        y                     = y,
        param_grid            = param_grid,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        exog                  = exog,
        lags_grid             = lags_grid,
        refit                 = refit,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
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
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: Optional[Union[bool, int]]=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series. 
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
            - If `string`: {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
            - If `Callable`: Function with arguments y_true, y_pred that returns 
            a float.
            - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column lags: lags configuration for each iteration.
            - column params: parameters configuration for each iteration.
            - column metric: metric value estimated for each iteration.
            - additional n columns with param = value.
    
    """

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters(
        forecaster            = forecaster,
        y                     = y,
        param_grid            = param_grid,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        exog                  = exog,
        lags_grid             = lags_grid,
        refit                 = refit,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
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
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: Optional[Union[bool, int]]=False,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series. 
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
            - If `string`: {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
            - If `Callable`: Function with arguments y_true, y_pred that returns 
            a float.
            - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column lags: lags configuration for each iteration.
            - column params: parameters configuration for each iteration.
            - column metric: metric value estimated for each iteration.
            - additional n columns with param = value.

    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f"`exog` must have same number of samples as `y`. "
             f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
        )

    if type(forecaster).__name__ == 'ForecasterAutoregCustom':
        if lags_grid is not None:
            warnings.warn(
                "`lags_grid` ignored if forecaster is an instance of `ForecasterAutoregCustom`.",
                IgnoredArgumentWarning
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
            "When `metric` is a `list`, each metric name must be unique."
        )

    print(f"Number of models compared: {len(param_grid)*len(lags_grid)}.")

    if show_progress:
        lags_grid = tqdm(lags_grid, desc='lags grid', position=0) #ncols=90
        param_grid = tqdm(param_grid, desc='params grid', position=1, leave=False)

    for lags in lags_grid:
        
        if type(forecaster).__name__ in ['ForecasterAutoreg', 'ForecasterAutoregDirect']:
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        for params in param_grid:

            forecaster.set_params(params)
            metrics_values = backtesting_forecaster(
                                 forecaster            = forecaster,
                                 y                     = y,
                                 steps                 = steps,
                                 metric                = metric,
                                 initial_train_size    = initial_train_size,
                                 fixed_train_size      = fixed_train_size,
                                 gap                   = gap,
                                 allow_incomplete_fold = allow_incomplete_fold,
                                 exog                  = exog,
                                 refit                 = refit,
                                 interval              = None,
                                 n_jobs                = n_jobs,
                                 verbose               = verbose,
                                 show_progress         = False
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
    search_space: Callable,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: Optional[Union[bool, int]]=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    show_progress: bool=True,
    engine: str='optuna',
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using time series backtesting and 
    optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series. 
    search_space : Callable (optuna)
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.
    steps : int
        Number of steps to predict.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
            - If `string`: {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
            - If `Callable`: Function with arguments y_true, y_pred that returns 
            a float.
            - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.
    engine : str, default `'optuna'`
        Bayesian optimization runs through the optuna library.
    kwargs_create_study : dict, default `{'direction':'minimize', 'sampler':TPESampler(seed=123)}`
        Only applies to engine='optuna'. Keyword arguments (key, value mappings) 
        to pass to optuna.create_study.
    kwargs_study_optimize : dict, default `{}`
        Only applies to engine='optuna'. Other keyword arguments (key, value mappings) 
        to pass to study.optimize().

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column lags: lags configuration for each iteration.
            - column params: parameters configuration for each iteration.
            - column metric: metric value estimated for each iteration.
            - additional n columns with param = value.
    results_opt_best : optuna object (optuna)  
        The best optimization result returned as a FrozenTrial optuna object.
    
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
                                    gap                   = gap,
                                    allow_incomplete_fold = allow_incomplete_fold,
                                    n_trials              = n_trials,
                                    random_state          = random_state,
                                    return_best           = return_best,
                                    n_jobs                = n_jobs,
                                    verbose               = verbose,
                                    show_progress         = show_progress,
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
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: Optional[Union[bool, int]]=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    show_progress: bool=True,
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using time series backtesting 
    and optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
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
        
            - If `string`: {'mean_squared_error', 'mean_absolute_error',
             'mean_absolute_percentage_error', 'mean_squared_log_error'}
            - If `Callable`: Function with arguments y_true, y_pred that returns 
            a float.
            - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    lags_grid : list of int, lists, numpy ndarray or range, default `None`
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg` or `ForecasterAutoregDirect`.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.
    kwargs_create_study : dict, default `{'direction':'minimize', 'sampler':TPESampler(seed=123)}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study.
    kwargs_study_optimize : dict, default `{}`
        Other keyword arguments (key, value mappings) to pass to study.optimize().

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column lags: lags configuration for each iteration.
            - column params: parameters configuration for each iteration.
            - column metric: metric value estimated for each iteration.
            - additional n columns with param = value.
    results_opt_best : optuna object
        The best optimization result returned as a FrozenTrial optuna object.

    """

    if type(forecaster).__name__ == 'ForecasterAutoregCustom':
        if lags_grid is not None:
            warnings.warn(
                "`lags_grid` ignored if forecaster is an instance of `ForecasterAutoregCustom`.",
                IgnoredArgumentWarning
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
            "When `metric` is a `list`, each metric name must be unique."
        )

    # Objective function using backtesting_forecaster
    def _objective(
        trial,
        forecaster            = forecaster,
        y                     = y,
        exog                  = exog,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        steps                 = steps,
        metric                = metric,
        refit                 = refit,
        n_jobs                = n_jobs,
        verbose               = verbose,
        search_space          = search_space,
    ) -> float:
        
        forecaster.set_params(search_space(trial))
        
        metrics, _ = backtesting_forecaster(
                         forecaster            = forecaster,
                         y                     = y,
                         exog                  = exog,
                         steps                 = steps,
                         metric                = metric,
                         initial_train_size    = initial_train_size,
                         fixed_train_size      = fixed_train_size,
                         gap                   = gap,
                         allow_incomplete_fold = allow_incomplete_fold,
                         refit                 = refit,
                         n_jobs                = n_jobs,
                         verbose               = verbose,
                         show_progress         = False
                     )
        # Store metrics in the variable metric_values defined outside _objective.
        nonlocal metric_values
        metric_values.append(metrics)

        return abs(metrics[0])

    print(
        f"""Number of models compared: {n_trials*len(lags_grid)},
         {n_trials} bayesian search in each lag configuration."""
    )

    if show_progress:
        lags_grid = tqdm(lags_grid, desc='lags grid', position=0)

    for lags in lags_grid:
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
        
    results = pd.DataFrame(
                  {'lags'  : lags_list,
                   'params': params_list,
                   **metric_dict}
              )

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
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results, results_opt_best