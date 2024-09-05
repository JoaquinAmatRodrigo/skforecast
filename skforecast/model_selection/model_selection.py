################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import re
import os
from copy import deepcopy
import logging
from typing import Union, Tuple, Optional, Callable
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

from ..metrics import add_y_train_argument, _get_metric
from ..exceptions import LongTrainingWarning
from ..exceptions import IgnoredArgumentWarning
from ..utils import check_backtesting_input
from ..utils import initialize_lags_grid
from ..utils import initialize_lags
from ..utils import select_n_jobs_backtesting

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def _create_backtesting_folds(
    data: Union[pd.Series, pd.DataFrame],
    window_size: int,
    initial_train_size: Union[int, None],
    test_size: int,
    externally_fitted: bool = False,
    refit: Union[bool, int] = False,
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    return_all_indexes: bool = False,
    differentiation: Optional[int] = None,
    verbose: bool = True
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
    returned indexes for the first fold if `test_size = 4`, `gap = 1` and 
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
    skip_folds : int, list, default `None`
        Folds to skip.
        If `skip_folds` is an integer, every 'skip_folds'-th is returned. If `skip_folds`
        is a list, the folds in the list are skipped. For example, if `skip_folds = 3`,
        and there are 10 folds, the folds returned will be [0, 3, 6, 9]. If `skip_folds`
        is a list [1, 2, 3], the folds returned will be [0, 4, 5, 6, 7, 8, 9].
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

    if isinstance(data, pd.Index):
        data = pd.Series(index=data)
    
    idx = range(len(data))
    folds = []
    i = 0
    last_fold_excluded = False

    while initial_train_size + (i * test_size) + gap < len(data):

        if refit:
            # If `fixed_train_size` the train size doesn't increase but moves by 
            # `test_size` positions in each iteration. If `False`, the train size
            # increases by `test_size` in each iteration.
            train_iloc_start = i * (test_size) if fixed_train_size else 0
            train_iloc_end = initial_train_size + i * (test_size)
            test_iloc_start = train_iloc_end
        else:
            # The train size doesn't increase and doesn't move.
            train_iloc_start = 0
            train_iloc_end = initial_train_size
            test_iloc_start = initial_train_size + i * (test_size)
        
        last_window_iloc_start = test_iloc_start - window_size
        test_iloc_end = test_iloc_start + gap + test_size
    
        partitions = [
            idx[train_iloc_start : train_iloc_end],
            idx[last_window_iloc_start : test_iloc_start],
            idx[test_iloc_start : test_iloc_end],
            idx[test_iloc_start + gap : test_iloc_end]
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
        fit_forecaster = [refit] * len(folds)
        fit_forecaster[0] = True
    else:
        fit_forecaster = [False] * len(folds)
        for i in range(0, len(fit_forecaster), refit): 
            fit_forecaster[i] = True
    
    for i in range(len(folds)): 
        folds[i].append(fit_forecaster[i])
        if fit_forecaster[i] is False:
            folds[i][0] = folds[i - 1][0]

    # This is done to allow parallelization when `refit` is `False`. The initial 
    # Forecaster fit is outside the auxiliary function.
    folds[0][4] = False

    index_to_skip = []
    if skip_folds is not None:
        if isinstance(skip_folds, int) and skip_folds > 0:
            index_to_keep = np.arange(0, len(folds), skip_folds)
            index_to_skip = np.setdiff1d(np.arange(0, len(folds)), index_to_keep, assume_unique=True)
            index_to_skip = [int(x) for x in index_to_skip] # Required since numpy 2.0
        if isinstance(skip_folds, list):
            index_to_skip = [i for i in skip_folds if i < len(folds)]        
    
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
        print(f"    Number skipped folds: {len(index_to_skip)} {index_to_skip if index_to_skip else ''}")
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
            is_fold_skipped   = i in index_to_skip
            has_training      = fold[-1] if i != 0 else True
            training_start    = data.index[fold[0][0] + differentiation] if fold[0] is not None else None
            training_end      = data.index[fold[0][-1]] if fold[0] is not None else None
            training_length   = len(fold[0]) - differentiation if fold[0] is not None else 0
            validation_start  = data.index[fold[3][0]]
            validation_end    = data.index[fold[3][-1]]
            validation_length = len(fold[3])

            print(f"Fold: {i}")
            if is_fold_skipped:
                print("    Fold skipped")
            elif not externally_fitted and has_training:
                print(
                    f"    Training:   {training_start} -- {training_end}  (n={training_length})"
                )
                print(
                    f"    Validation: {validation_start} -- {validation_end}  (n={validation_length})"
                )
            else:
                print("    Training:   No training in this fold")
                print(
                    f"    Validation: {validation_start} -- {validation_end}  (n={validation_length})"
                )

        print("")

    folds = [fold for i, fold in enumerate(folds) if i not in index_to_skip]
    if not return_all_indexes:
        # +1 to prevent iloc pandas from deleting the last observation
        folds = [
            [[fold[0][0], fold[0][-1] + 1], 
             [fold[1][0], fold[1][-1] + 1], 
             [fold[2][0], fold[2][-1] + 1],
             [fold[3][0], fold[3][-1] + 1],
             fold[4]] 
            for fold in folds
        ]

    return folds


def _backtesting_forecaster(
    forecaster: object,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int] = None,
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    refit: Union[bool, int] = False,
    interval: Optional[list] = None,
    n_boot: int = 250,
    random_state: int = 123,
    in_sample_residuals: bool = True,
    binned_residuals: bool = False,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = False,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        error to create prediction intervals. If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    binned_residuals : bool, default `False`
            If `True`, residuals used in each bootstrapping iteration are selected
            conditioning on the predicted values. If `False`, residuals are selected
            randomly without conditioning on the predicted values.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.

        - column pred: predictions.
        - column lower_bound: lower bound of the interval.
        - column upper_bound: upper bound of the interval.
    
    """

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

    if initial_train_size is not None:
        # First model training, this is done to allow parallelization when `refit`
        # is `False`. The initial Forecaster fit is outside the auxiliary function.
        exog_train = exog.iloc[:initial_train_size, ] if exog is not None else None
        forecaster.fit(
            y                         = y.iloc[:initial_train_size, ],
            exog                      = exog_train,
            store_in_sample_residuals = store_in_sample_residuals
        )
        window_size = forecaster.window_size_diff
        externally_fitted = False
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        window_size = forecaster.window_size_diff
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
                skip_folds            = skip_folds,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = False,
                differentiation       = differentiation,
                verbose               = verbose
            )

    if refit:
        n_of_fits = int(len(folds) / refit)
        if type(forecaster).__name__ != 'ForecasterAutoregDirect' and n_of_fits > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits} times. This can take substantial"
                 f" amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )
        elif type(forecaster).__name__ == 'ForecasterAutoregDirect' and n_of_fits * forecaster.steps > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits * forecaster.steps} times "
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

        train_iloc_start       = fold[0][0]
        train_iloc_end         = fold[0][1]
        last_window_iloc_start = fold[1][0]
        last_window_iloc_end   = fold[1][1]
        test_iloc_start        = fold[2][0]
        test_iloc_end          = fold[2][1]

        if fold[4] is False:
            # When the model is not fitted, last_window must be updated to include
            # the data needed to make predictions.
            last_window_y = y.iloc[last_window_iloc_start:last_window_iloc_end]
        else:
            # The model is fitted before making predictions. If `fixed_train_size`
            # the train size doesn't increase but moves by `steps` in each iteration.
            # If `False` the train size increases by `steps` in each iteration.
            y_train = y.iloc[train_iloc_start:train_iloc_end, ]
            exog_train = (
                exog.iloc[train_iloc_start:train_iloc_end,] if exog is not None else None
            )
            last_window_y = None
            forecaster.fit(
                y                         = y_train, 
                exog                      = exog_train, 
                store_in_sample_residuals = store_in_sample_residuals
            )

        next_window_exog = exog.iloc[test_iloc_start:test_iloc_end, ] if exog is not None else None

        steps = len(range(test_iloc_start, test_iloc_end))
        if type(forecaster).__name__ == 'ForecasterAutoregDirect' and gap > 0:
            # Select only the steps that need to be predicted if gap > 0
            test_no_gap_iloc_start = fold[3][0]
            test_no_gap_iloc_end   = fold[3][1]
            steps = list(
                np.arange(len(range(test_no_gap_iloc_start, test_no_gap_iloc_end)))
                + gap
                + 1
            )

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
                       in_sample_residuals = in_sample_residuals,
                       binned_residuals    = binned_residuals,
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

    train_indexes = []
    for i, fold in enumerate(folds):
        fit_fold = fold[-1]
        if i == 0 or fit_fold:
            train_iloc_start = fold[0][0]
            train_iloc_end = fold[0][1]
            train_indexes.append(np.arange(train_iloc_start, train_iloc_end))
    
    train_indexes = np.unique(np.concatenate(train_indexes))
    y_train = y.iloc[train_indexes]

    metric_values = [
        m(
            y_true = y.loc[backtest_predictions.index],
            y_pred = backtest_predictions['pred'],
            y_train = y_train
        ) 
        for m in metrics
    ]

    metric_values = pd.DataFrame(
        data    = [metric_values],
        columns = [m.__name__ for m in metrics]
    )
    
    return metric_values, backtest_predictions


def backtesting_forecaster(
    forecaster: object,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int] = None,
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    refit: Union[bool, int] = False,
    interval: Optional[list] = None,
    n_boot: int = 250,
    random_state: int = 123,
    in_sample_residuals: bool = True,
    binned_residuals: bool = False,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = False,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        error to create prediction intervals. If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    binned_residuals : bool, default `False`
            If `True`, residuals used in each bootstrapping iteration are selected
            conditioning on the predicted values. If `False`, residuals are selected
            randomly without conditioning on the predicted values.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    metric_values : pandas DataFrame
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
        skip_folds            = skip_folds,
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
            (f"When using a ForecasterAutoregDirect, the combination of steps "
             f"+ gap ({steps + gap}) cannot be greater than the `steps` parameter "
             f"declared when the forecaster is initialized ({forecaster.steps}).")
        )
    
    metric_values, backtest_predictions = _backtesting_forecaster(
        forecaster            = forecaster,
        y                     = y,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        skip_folds            = skip_folds,
        allow_incomplete_fold = allow_incomplete_fold,
        exog                  = exog,
        refit                 = refit,
        interval              = interval,
        n_boot                = n_boot,
        random_state          = random_state,
        in_sample_residuals   = in_sample_residuals,
        binned_residuals      = binned_residuals,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
    )

    return metric_values, backtest_predictions


def grid_search_forecaster(
    forecaster: object,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    method: str = 'backtesting',
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    refit: Union[bool, int] = False,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
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
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
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
    initial_train_size : int 
        Number of samples in the initial train split.
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try. Ignored 
        if the forecaster is an instance of `ForecasterAutoregCustom` or 
        `ForecasterAutoregMultiSeriesCustom`.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    
    """
    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )

    param_grid = list(ParameterGrid(param_grid))

    if method == 'backtesting':
        results = _evaluate_grid_hyperparameters(
            forecaster            = forecaster,
            y                     = y,
            param_grid            = param_grid,
            steps                 = steps,
            metric                = metric,
            initial_train_size    = initial_train_size,
            fixed_train_size      = fixed_train_size,
            gap                   = gap,
            skip_folds            = skip_folds,
            allow_incomplete_fold = allow_incomplete_fold,
            exog                  = exog,
            lags_grid             = lags_grid,
            refit                 = refit,
            return_best           = return_best,
            n_jobs                = n_jobs,
            verbose               = verbose,
            show_progress         = show_progress,
            output_file           = output_file
        )
    else:
        results = _evaluate_grid_hyperparameters_one_step_ahead(
            forecaster            = forecaster,
            y                     = y,
            param_grid            = param_grid,
            metric                = metric,
            initial_train_size    = initial_train_size,
            exog                  = exog,
            lags_grid             = lags_grid,
            return_best           = return_best,
            verbose               = verbose,
            show_progress         = show_progress,
            output_file           = output_file
        )

    return results


def random_search_forecaster(
    forecaster: object,
    y: pd.Series,
    param_distributions: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    method: str = 'backtesting',
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    refit: Union[bool, int] = False,
    n_iter: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
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
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try. Ignored 
        if the forecaster is an instance of `ForecasterAutoregCustom` or 
        `ForecasterAutoregMultiSeriesCustom`.
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
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    
    """
    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    if method == 'backtesting':
        results = _evaluate_grid_hyperparameters(
            forecaster            = forecaster,
            y                     = y,
            param_grid            = param_grid,
            steps                 = steps,
            metric                = metric,
            initial_train_size    = initial_train_size,
            fixed_train_size      = fixed_train_size,
            gap                   = gap,
            skip_folds            = skip_folds,
            allow_incomplete_fold = allow_incomplete_fold,
            exog                  = exog,
            lags_grid             = lags_grid,
            refit                 = refit,
            return_best           = return_best,
            n_jobs                = n_jobs,
            verbose               = verbose,
            show_progress         = show_progress,
            output_file           = output_file
        )
    else:
        results = _evaluate_grid_hyperparameters_one_step_ahead(
            forecaster            = forecaster,
            y                     = y,
            param_grid            = param_grid,
            metric                = metric,
            initial_train_size    = initial_train_size,
            exog                  = exog,
            lags_grid             = lags_grid,
            return_best           = return_best,
            verbose               = verbose,
            show_progress         = show_progress,
            output_file           = output_file
        )

    return results


def _evaluate_grid_hyperparameters(
    forecaster: object,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    refit: Union[bool, int] = False,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
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
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try. Ignored 
        if the forecaster is an instance of `ForecasterAutoregCustom` or 
        `ForecasterAutoregMultiSeriesCustom`.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.

    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f"`exog` must have same number of samples as `y`. "
             f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
        )

    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
   
    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )

    print(f"Number of models compared: {len(param_grid) * len(lags_grid)}.")

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
    for lags_k, lags_v in lags_grid_tqdm:
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(lags_v)
            lags_v = forecaster.lags.copy()
            if lags_label == 'values':
                lags_k = lags_v
        
        for params in param_grid:

            forecaster.set_params(params)
            metric_values = backtesting_forecaster(
                                forecaster            = forecaster,
                                y                     = y,
                                steps                 = steps,
                                metric                = metric,
                                initial_train_size    = initial_train_size,
                                fixed_train_size      = fixed_train_size,
                                gap                   = gap,
                                skip_folds            = skip_folds,
                                allow_incomplete_fold = allow_incomplete_fold,
                                exog                  = exog,
                                refit                 = refit,
                                interval              = None,
                                n_jobs                = n_jobs,
                                verbose               = verbose,
                                show_progress         = False
                            )[0]
            metric_values = metric_values.iloc[0, :].to_list()
            warnings.filterwarnings(
                'ignore',
                category = RuntimeWarning, 
                message  = "The forecaster will be fit.*"
            )
            
            lags_list.append(lags_v)
            lags_label_list.append(lags_k)
            params_list.append(params)
            for m, m_value in zip(metric, metric_values):
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(m_value)
        
            if output_file is not None:
                header = ['lags', 'lags_label', 'params', 
                          *metric_dict.keys(), *params.keys()]
                row = [lags_v, lags_k, params, 
                       *metric_values, *params.values()]
                if not os.path.isfile(output_file):
                    with open(output_file, 'w', newline='') as f:
                        f.write('\t'.join(header) + '\n')
                        f.write('\t'.join([str(r) for r in row]) + '\n')
                else:
                    with open(output_file, 'a', newline='') as f:
                        f.write('\t'.join([str(r) for r in row]) + '\n')
    
    results = pd.DataFrame({
                  'lags': lags_list,
                  'lags_label': lags_label_list,
                  'params': params_list,
                  **metric_dict
              })
    
    results = (
        results
        .sort_values(by=list(metric_dict.keys())[0], ascending=True)
        .reset_index(drop=True)
    )
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results.loc[0, 'lags']
        best_params = results.loc[0, 'params']
        best_metric = results.loc[0, list(metric_dict.keys())[0]]
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results


def _evaluate_grid_hyperparameters_one_step_ahead(
    forecaster: object,
    y: pd.Series,
    param_grid: dict,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    return_best: bool = True,
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object by assessing its error in
    one-step-ahead predictions. This method is faster than backtesting, which involves
    multi-step predictions, and allows for quick comparisons between different models
    and hyperparameters. However, while it is efficient for initial evaluations, the
    results may not fully reflect the model's performance in multi-step predictions.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try. Ignored 
        if the forecaster is an instance of `ForecasterAutoregCustom` or 
        `ForecasterAutoregMultiSeriesCustom`.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.

    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f"`exog` must have same number of samples as `y`. "
             f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
        )

    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
   
    if not isinstance(metric, list):
        metric = [metric] 
    metric = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]
    metric_names = [m if isinstance(m, str) else m.__name__ for m in metric]
    
    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
    
    warnings.warn(
        "One-step-ahead predictions are used for faster model comparison, but they may "
        "not fully represent multi-step prediction performance. It is recommended to "
        "backtest the final model for a more accurate multi-step performance estimate."
    )

    if verbose:
        print(f"Number of models compared: {len(param_grid) * len(lags_grid)}.")

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
    metric_list = []
    for lags_k, lags_v in lags_grid_tqdm:
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(lags_v)
            lags_v = forecaster.lags.copy()
            if lags_label == 'values':
                lags_k = lags_v

        train_size = initial_train_size - forecaster.window_size
        X_all, y_all = forecaster.create_train_X_y(y=y, exog=exog)
        X_train = X_all.iloc[:train_size, :]
        X_test  = X_all.iloc[train_size:, :]

        if type(forecaster).__name__ == 'ForecasterAutoreg': 
            y_train = y_all.iloc[:train_size]
            y_test  = y_all.iloc[train_size:]

        if type(forecaster).__name__ == 'ForecasterAutoregDirect':
            y_train = {k: v.iloc[:train_size] for k, v in y_all.items()}
            y_test  = {k: v.iloc[train_size:] for k, v in y_all.items()}

        for params in param_grid:

            forecaster.set_params(params)

            if type(forecaster).__name__ == 'ForecasterAutoreg':
                forecaster.regressor.fit(X_train, y_train)
                pred = forecaster.regressor.predict(X_test)

                metric_values = []
                for m in metric:
                    metric_values.append(
                        m(y_true=y_test, y_pred=pred, y_train=y_train)
                    )

            if type(forecaster).__name__ == 'ForecasterAutoregDirect':
                pred_per_step = {}
                steps = range(1, forecaster.steps + 1)
                for step in steps:
                    X_train_step, y_train_step = forecaster.filter_train_X_y_for_step(
                                                    step    = step,
                                                    X_train = X_train,
                                                    y_train = y_train
                                                  )
                    X_test_step, y_test_step = forecaster.filter_train_X_y_for_step(
                                                    step    = step,  
                                                    X_train = X_test,
                                                    y_train = y_test
                                                )
                    forecaster.regressors_[step].fit(X_train_step, y_train_step)
                    pred = forecaster.regressors_[step].predict(X_test_step)
                    pred_per_step[step] = pred

                metric_values = []
                for m in metric:
                    metric_values_per_step = []
                    for step in steps:
                        metric_values_per_step.append(
                            m(y_true=y_test_step, y_pred=pred_per_step[step], y_train=y_train[step])
                        )
                    metric_values.append(np.mean(metric_values_per_step))


            warnings.filterwarnings(
                'ignore',
                category = RuntimeWarning, 
                message  = "The forecaster will be fit.*"
            )
            
            lags_list.append(lags_v)
            lags_label_list.append(lags_k)
            params_list.append(params)
            metric_list.append(metric_values)
        
            if output_file is not None:
                header = ['lags', 'lags_label', 'params', 
                          *metric_names, *params.keys()]
                row = [lags_v, lags_k, params, 
                       *metric_values, *params.values()]
                if not os.path.isfile(output_file):
                    with open(output_file, 'w', newline='') as f:
                        f.write('\t'.join(header) + '\n')
                        f.write('\t'.join([str(r) for r in row]) + '\n')
                else:
                    with open(output_file, 'a', newline='') as f:
                        f.write('\t'.join([str(r) for r in row]) + '\n')
    
    metric_results = pd.DataFrame(metric_list, columns=metric_names)
    results = pd.DataFrame({
                  'lags': lags_list,
                  'lags_label': lags_label_list,
                  'params': params_list,
                  **metric_results
              })
    
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
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results


def bayesian_search_forecaster(
    forecaster: object,
    y: pd.Series,
    search_space: Callable,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    method: str = 'backtesting',
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    refit: Union[bool, int] = False,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None,
    engine: str = 'optuna',
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {}
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
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output. When a new sampler 
        is passed in `kwargs_create_study`, the seed must be set within the 
        sampler. For example `{'sampler': TPESampler(seed=145)}`.
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
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**
    engine : str, default `'optuna'`
        Bayesian optimization runs through the optuna library.
    kwargs_create_study : dict, default `{}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study().
        If default, the direction is set to 'minimize' and a TPESampler(seed=123) 
        sampler is used during optimization.
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
    best_trial : optuna object
        The best optimization result returned as a FrozenTrial optuna object.
    
    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f"`exog` must have same number of samples as `y`. "
             f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
        )

    # TODO: remove argument engine?
    if engine not in ['optuna']:
        raise ValueError(
            f"`engine` only allows 'optuna', got {engine}."
        )
    
    if method not in ['backtesting', 'one_step_ahead']:
        raise ValueError(
            f"`method` must be 'backtesting' or 'one_step_ahead'. Got {method}."
        )
    
    if method == 'backtesting':
            
        results, best_trial = _bayesian_search_optuna(
                                    forecaster            = forecaster,
                                    y                     = y,
                                    exog                  = exog,
                                    search_space          = search_space,
                                    steps                 = steps,
                                    metric                = metric,
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
                                    output_file           = output_file,
                                    kwargs_create_study   = kwargs_create_study,
                                    kwargs_study_optimize = kwargs_study_optimize
                                )
    else:

        results, best_trial = _bayesian_search_optuna_one_step_ahead(
                                    forecaster            = forecaster,
                                    y                     = y,
                                    exog                  = exog,
                                    search_space          = search_space,
                                    metric                = metric,
                                    initial_train_size    = initial_train_size,
                                    n_trials              = n_trials,
                                    random_state          = random_state,
                                    return_best           = return_best,
                                    verbose               = verbose,
                                    show_progress         = show_progress,
                                    output_file           = output_file,
                                    kwargs_create_study   = kwargs_create_study,
                                    kwargs_study_optimize = kwargs_study_optimize
                                )

    return results, best_trial


def _bayesian_search_optuna(
    forecaster: object,
    y: pd.Series,
    search_space: Callable,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool = True,
    gap: int = 0,
    skip_folds: Optional[Union[int, list]] = None,
    allow_incomplete_fold: bool = True,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    refit: Union[bool, int] = False,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {}
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
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output. When a new sampler 
        is passed in `kwargs_create_study`, the seed must be set within the 
        sampler. For example `{'sampler': TPESampler(seed=145)}`.
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

        - column lags: lags configuration for each iteration.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    best_trial : optuna object
        The best optimization result returned as an optuna FrozenTrial object.

    """
    
    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
        
    # Objective function using backtesting_forecaster
    def _objective(
        trial,
        search_space          = search_space,
        forecaster            = forecaster,
        y                     = y,
        exog                  = exog,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        skip_folds            = skip_folds,
        allow_incomplete_fold = allow_incomplete_fold,
        refit                 = refit,
        n_jobs                = n_jobs,
        verbose               = verbose,
    ) -> float:
        
        sample = search_space(trial)
        sample_params = {k: v for k, v in sample.items() if k != 'lags'}
        forecaster.set_params(sample_params)
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            if "lags" in sample:
                forecaster.set_lags(sample['lags'])
        
        metrics, _ = backtesting_forecaster(
                         forecaster            = forecaster,
                         y                     = y,
                         exog                  = exog,
                         steps                 = steps,
                         metric                = metric,
                         initial_train_size    = initial_train_size,
                         fixed_train_size      = fixed_train_size,
                         skip_folds            = skip_folds,
                         gap                   = gap,
                         allow_incomplete_fold = allow_incomplete_fold,
                         refit                 = refit,
                         n_jobs                = n_jobs,
                         verbose               = verbose,
                         show_progress         = False
                     )
        metrics = metrics.iloc[0, :].to_list()
        
        # Store metrics in the variable `metric_values` defined outside _objective.
        nonlocal metric_values
        metric_values.append(metrics)

        return metrics[0]
       
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

    # `metric_values` will be modified inside _objective function. 
    # It is a trick to extract multiple values from _objective since
    # only the optimized value can be returned.
    metric_values = []

    warnings.filterwarnings(
        "ignore",
        category = UserWarning,
        message  = "Choices for a categorical distribution should be*"
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
             f"  Trial objects keys : {list(best_trial.params.keys())}.")
        )
    warnings.filterwarnings('default')
    
    lags_list = []
    params_list = []
    for i, trial in enumerate(study.get_trials()):
        regressor_params = {k: v for k, v in trial.params.items() if k != 'lags'}
        lags = trial.params.get(
                   'lags',
                   forecaster.lags if hasattr(forecaster, 'lags') else None
               )
        params_list.append(regressor_params)
        lags_list.append(lags)
        for m, m_values in zip(metric, metric_values[i]):
            m_name = m if isinstance(m, str) else m.__name__
            metric_dict[m_name].append(m_values)
    
    if type(forecaster).__name__ != 'ForecasterAutoregCustom':
        lags_list = [
            initialize_lags(forecaster_name=type(forecaster).__name__, lags = lag)
            for lag in lags_list
        ]
    else:
        lags_list = [
            f"custom function: {forecaster.fun_predictors.__name__}"
            for _
            in lags_list
        ]

    results = pd.DataFrame({
                  'lags': lags_list,
                  'params': params_list,
                  **metric_dict
              })
    
    results = (
        results
        .sort_values(by=list(metric_dict.keys())[0], ascending=True)
        .reset_index(drop=True)
    )
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        best_lags = results.loc[0, 'lags']
        best_params = results.loc[0, 'params']
        best_metric = results.loc[0, list(metric_dict.keys())[0]]
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results, best_trial


def _bayesian_search_optuna_one_step_ahead(
    forecaster: object,
    y: pd.Series,
    search_space: Callable,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object by assessing its error in 
    one-step-ahead predictions. This method is faster than backtesting, which involves
    multi-step predictions, and allows for quick comparisons between different models
    and hyperparameters. However, while it is efficient for initial evaluations, the
    results may not fully reflect the model's performance in multi-step predictions.
    
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
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    initial_train_size : int 
        Number of samples in the initial train split.
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output. When a new sampler 
        is passed in `kwargs_create_study`, the seed must be set within the 
        sampler. For example `{'sampler': TPESampler(seed=145)}`.
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

        - column lags: lags configuration for each iteration.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    best_trial : optuna object
        The best optimization result returned as an optuna FrozenTrial object.

    """
    
    if not isinstance(metric, list):
        metric = [metric] 
    metric = [
        _get_metric(metric=m)
        if isinstance(m, str)
        else add_y_train_argument(m) 
        for m in metric
    ]
    metric_names = [m if isinstance(m, str) else m.__name__ for m in metric]
    
    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
    
    warnings.warn(
        "One-step-ahead predictions are used for faster model comparison, but they may "
        "not fully represent multi-step prediction performance. It is recommended to "
        "backtest the final model for a more accurate multi-step performance estimate."
    )
        
    # Objective function using backtesting_forecaster
    def _objective(
        trial,
        search_space          = search_space,
        forecaster            = forecaster,
        y                     = y,
        exog                  = exog,
        metric                = metric,
        initial_train_size    = initial_train_size,
    ) -> float:
        
        sample = search_space(trial)
        sample_params = {k: v for k, v in sample.items() if k != 'lags'}
        forecaster.set_params(sample_params)
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            if "lags" in sample:
                forecaster.set_lags(sample['lags'])

        train_size = initial_train_size - forecaster.window_size
        X_all, y_all = forecaster.create_train_X_y(y=y, exog=exog)
        X_train = X_all.iloc[:train_size, :]
        X_test  = X_all.iloc[train_size:, :]
        if type(forecaster).__name__ == 'ForecasterAutoreg': 
            y_train = y_all.iloc[:train_size]
            y_test  = y_all.iloc[train_size:]

            forecaster.regressor.fit(X_train, y_train)
            pred = forecaster.regressor.predict(X_test)
            
            metrics = []
            for m in metric:
                metrics.append(
                    m(y_true=y_test, y_pred=pred, y_train=y_train)
                )

        if type(forecaster).__name__ == 'ForecasterAutoregDirect':
            y_train = {k: v.iloc[:train_size] for k, v in y_all.items()}
            y_test  = {k: v.iloc[train_size:] for k, v in y_all.items()}

            pred_per_step = {}
            steps = range(1, forecaster.steps + 1)
            for step in steps:
                X_train_step, y_train_step = forecaster.filter_train_X_y_for_step(
                                                    step    = step,
                                                    X_train = X_train,
                                                    y_train = y_train
                                                  )
                X_test_step, y_test_step = forecaster.filter_train_X_y_for_step(
                                                step    = step,  
                                                X_train = X_test,
                                                y_train = y_test
                                            )
                forecaster.regressors_[step].fit(X_train_step, y_train_step)
                pred = forecaster.regressors_[step].predict(X_test_step)
                pred_per_step[step] = pred

            metrics = []
            for m in metric:
                metric_values_per_step = []
                for step in steps:
                    metric_values_per_step.append(
                        m(y_true=y_test_step, y_pred=pred_per_step[step], y_train=y_train[step])
                    )
                metrics.append(np.mean(metric_values_per_step))

        # Store all metrics in the variable `metric_values` defined outside _objective.
        nonlocal metric_values
        metric_values.append(metrics)

        return metrics[0]
       
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

    # `metric_values` will be modified inside _objective function. 
    # It is a trick to extract multiple values from _objective since
    # only the optimized value can be returned.
    metric_values = []

    warnings.filterwarnings(
        "ignore",
        category = UserWarning,
        message  = "Choices for a categorical distribution should be*"
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
             f"  Trial objects keys : {list(best_trial.params.keys())}.")
        )
    warnings.filterwarnings('default')
    
    lags_list = []
    params_list = []
    for i, trial in enumerate(study.get_trials()):
        regressor_params = {k: v for k, v in trial.params.items() if k != 'lags'}
        lags = trial.params.get(
                   'lags',
                   forecaster.lags if hasattr(forecaster, 'lags') else None
               )
        params_list.append(regressor_params)
        lags_list.append(lags)
    
    if type(forecaster).__name__ != 'ForecasterAutoregCustom':
        lags_list = [
            initialize_lags(forecaster_name=type(forecaster).__name__, lags = lag)
            for lag in lags_list
        ]
    else:
        lags_list = [
            f"custom function: {forecaster.fun_predictors.__name__}"
            for _
            in lags_list
        ]

    metric_results = pd.DataFrame(metric_values, columns=metric_names)
    results = pd.DataFrame({
                  'lags': lags_list,
                  'params': params_list,
                  **metric_results
              })
    
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
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results, best_trial


def select_features(
    forecaster: object,
    selector: object,
    y: Union[pd.Series, pd.DataFrame],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    select_only: Optional[str] = None,
    force_inclusion: Optional[Union[list, str]] = None,
    subsample: Union[int, float] = 0.5,
    random_state: int = 123,
    verbose: bool = True
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
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom
        Forecaster model.
    selector : object
        A feature selector from sklearn.feature_selection.
    y : pandas Series, pandas DataFrame
        Target time series to which the feature selection will be applied.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
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
        'ForecasterAutoreg',
        'ForecasterAutoregCustom'
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
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)

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
    exog_cols = [col for col in X_train.columns if col not in autoreg_cols]

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
        X_train = X_train.drop(columns=exog_cols)
    elif select_only == 'exog':
        X_train = X_train.drop(columns=autoreg_cols)

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
        print(f"Number of features available: {X_train.shape[1]}") 
        print(f"    Autoreg (n={len(autoreg_cols)})")
        print(f"    Exog    (n={len(exog_cols)})")
        print(f"Number of features selected: {len(selected_features)}")
        print(f"    Autoreg (n={len(selected_autoreg)}) : {selected_autoreg}")
        print(f"    Exog    (n={len(selected_exog)}) : {selected_exog}")

    return selected_autoreg, selected_exog
