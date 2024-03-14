################################################################################
#                  skforecast.model_selection_multiseries                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import warnings
import logging
import re
from copy import deepcopy
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import optuna
from optuna.samplers import TPESampler, RandomSampler

from ..exceptions import LongTrainingWarning
from ..exceptions import IgnoredArgumentWarning
from ..model_selection.model_selection import _get_metric
from ..model_selection.model_selection import _create_backtesting_folds
from ..utils import check_backtesting_input
from ..utils import initialize_lags_grid
from ..utils import select_n_jobs_backtesting

optuna.logging.set_verbosity(optuna.logging.WARNING) # disable optuna logs

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def _initialize_levels_model_selection_multiseries(
    forecaster, 
    series: pd.DataFrame,
    levels: Optional[Union[str, list]]=None
) -> list:
    """
    Initialize levels for model_selection_multiseries functions.

    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
            levels = list(series.columns) 
        elif isinstance(levels, str):
            levels = [levels]

    return levels


def _backtesting_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int]=None,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: Union[bool, int]=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=False,
    show_progress: bool=True
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
    series : pandas DataFrame
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
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction
        error to create prediction intervals. If `False`, out_sample_residuals 
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
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.
    backtest_predictions : pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`. 
        If there is more than one level, this structure will be repeated for each of them.

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
    elif not isinstance(refit, bool) and refit != 1:
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
        metrics = [_get_metric(metric=metric) if isinstance(metric, str) else metric]
    else:
        metrics = [_get_metric(metric=m) if isinstance(m, str) else m 
                   for m in metric]

    store_in_sample_residuals = False if interval is None else True

    if initial_train_size is not None:
        # First model training, this is done to allow parallelization when `refit` 
        # is `False`. The initial Forecaster fit is outside the auxiliary function.
        exog_train = exog.iloc[:initial_train_size, ] if exog is not None else None
        forecaster.fit(
            series                    = series.iloc[:initial_train_size, ],
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

    folds = _create_backtesting_folds(
                data                  = series,
                window_size           = window_size,
                initial_train_size    = initial_train_size,
                test_size             = steps,
                externally_fitted     = externally_fitted,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = False,
                verbose               = verbose  
            )

    if refit:
        n_of_fits = int(len(folds)/refit)
        if type(forecaster).__name__ != 'ForecasterAutoregMultiVariate' and n_of_fits > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits} times. This can take substantial "
                 f"amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )
        elif type(forecaster).__name__ == 'ForecasterAutoregMultiVariate' and n_of_fits*forecaster.steps > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits*forecaster.steps} times "
                 f"({n_of_fits} folds * {forecaster.steps} regressors). This can take "
                 f"substantial amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )

    if show_progress:
        folds = tqdm(folds)

    def _fit_predict_forecaster(series, exog, forecaster, interval, fold):
        """
        Fit the forecaster and predict `steps` ahead. This is an auxiliary 
        function used to parallelize the backtesting_forecaster_multiseries
        function.
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
            last_window_series = series.iloc[last_window_start:last_window_end, ]
        else:
            # The model is fitted before making predictions. If `fixed_train_size`  
            # the train size doesn't increase but moves by `steps` in each iteration. 
            # If `False` the train size increases by `steps` in each  iteration.

            # TODO: Review when series and exog are dict
            series_train = series.iloc[train_idx_start:train_idx_end, ]
            exog_train = exog.iloc[train_idx_start:train_idx_end, ] if exog is not None else None
            last_window_series = None
            forecaster.fit(
                series                    = series_train, 
                exog                      = exog_train,
                store_in_sample_residuals = store_in_sample_residuals
            )
        
        next_window_exog = exog.iloc[test_idx_start:test_idx_end, ] if exog is not None else None

        steps = len(range(test_idx_start, test_idx_end))
        if type(forecaster).__name__ == 'ForecasterAutoregMultiVariate' and gap > 0:
            # Select only the steps that need to be predicted if gap > 0
            test_idx_start = fold[3][0]
            test_idx_end   = fold[3][1]
            steps = list(np.arange(len(range(test_idx_start, test_idx_end))) + gap + 1)
        
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

        if type(forecaster).__name__ != 'ForecasterAutoregMultiVariate' and gap > 0:
            pred = pred.iloc[gap:, ]
        
        return pred
    
    backtest_predictions = (
        Parallel(n_jobs=n_jobs)
        (delayed(_fit_predict_forecaster)
        (series=series, exog=exog, forecaster=forecaster, interval=interval, fold=fold)
         for fold in folds)
    )

    backtest_predictions = pd.concat(backtest_predictions)

    metrics_levels = [[m(
                         y_true = series[level].loc[backtest_predictions.index],
                         y_pred = backtest_predictions[level]
                        ) for m in metrics
                      ] for level in levels]

    metrics_levels = pd.concat(
                         [pd.DataFrame({'levels': levels}), 
                          pd.DataFrame(
                              data    = metrics_levels,
                              columns = [m if isinstance(m, str) else m.__name__ 
                                         for m in metrics]
                          )],
                         axis=1
                     )

    return metrics_levels, backtest_predictions


def backtesting_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int],
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: Union[bool, int]=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=False,
    show_progress: bool=True
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
    series : pandas DataFrame
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
    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
        If there is more than one level, this structure will be repeated for each of them.

        - column pred: predictions.
        - column lower_bound: lower bound of the interval.
        - column upper_bound: upper bound of the interval.
    
    """

    if type(forecaster).__name__ not in ['ForecasterAutoregMultiSeries', 
                                         'ForecasterAutoregMultiSeriesCustom', 
                                         'ForecasterAutoregMultiVariate']:
        raise TypeError(
            ("`forecaster` must be of type `ForecasterAutoregMultiSeries`, "
             "`ForecasterAutoregMultiSeriesCustom` or `ForecasterAutoregMultiVariate`, "
             "for all other types of forecasters use the functions available in "
             f"the `model_selection` module. Got {type(forecaster).__name__}")
        )
    
    check_backtesting_input(
        forecaster            = forecaster,
        steps                 = steps,
        metric                = metric,
        series                = series,
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

    metrics_levels, backtest_predictions = _backtesting_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series,
        steps                 = steps,
        levels                = levels,
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

    return metrics_levels, backtest_predictions


def grid_search_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

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
    
    """

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster            = forecaster,
                  series                = series,
                  param_grid            = param_grid,
                  steps                 = steps,
                  metric                = metric,
                  initial_train_size    = initial_train_size,
                  fixed_train_size      = fixed_train_size,
                  gap                   = gap,
                  allow_incomplete_fold = allow_incomplete_fold,
                  levels                = levels,
                  exog                  = exog,
                  lags_grid             = lags_grid,
                  refit                 = refit,
                  n_jobs                = n_jobs,
                  return_best           = return_best,
                  verbose               = verbose,
                  show_progress         = show_progress
              )

    return results


def random_search_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_distributions: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
        Training time series.
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions or 
        lists of parameters to try.
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

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
    
    """

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, 
                                       random_state=random_state))

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster            = forecaster,
                  series                = series,
                  param_grid            = param_grid,
                  steps                 = steps,
                  metric                = metric,
                  initial_train_size    = initial_train_size,
                  fixed_train_size      = fixed_train_size,
                  gap                   = gap,
                  allow_incomplete_fold = allow_incomplete_fold,
                  levels                = levels,
                  exog                  = exog,
                  lags_grid             = lags_grid,
                  refit                 = refit,
                  return_best           = return_best,
                  n_jobs                = n_jobs,
                  verbose               = verbose,
                  show_progress         = show_progress
              )

    return results


def _evaluate_grid_hyperparameters_multiseries(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

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
    
    """

    if return_best and exog is not None and (len(exog) != len(series)):
        raise ValueError(
            (f"`exog` must have same number of samples as `series`. "
             f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
        )
    
    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster,
                 series     = series,
                 levels     = levels
             )

    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
   
    lags_list = []
    params_list = []
    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )

    print(
        f"{len(param_grid)*len(lags_grid)} models compared for {len(levels)} level(s). "
        f"Number of iterations: {len(param_grid)*len(lags_grid)}."
    )

    if show_progress:
        lags_grid_tqdm = tqdm(lags_grid.items(), desc='lags grid', position=0) #ncols=90
        param_grid = tqdm(param_grid, desc='params grid', position=1, leave=False)
    else:
        lags_grid_tqdm = lags_grid.items()

    for lags_k, lags_v in lags_grid_tqdm:

        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(lags_v)
            lags_v = lags_k if lags_label == 'keys' else forecaster.lags.copy()
        
        for params in param_grid:

            forecaster.set_params(params)
            metrics_levels = backtesting_forecaster_multiseries(
                                 forecaster            = forecaster,
                                 series                = series,
                                 exog                  = exog,
                                 steps                 = steps,
                                 levels                = levels,
                                 metric                = metric,
                                 initial_train_size    = initial_train_size,
                                 fixed_train_size      = fixed_train_size,
                                 gap                   = gap,
                                 allow_incomplete_fold = allow_incomplete_fold,
                                 refit                 = refit,
                                 interval              = None,
                                 verbose               = verbose,
                                 n_jobs                = n_jobs,
                                 show_progress         = False
                             )[0]
            warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                    message= "The forecaster will be fit.*")

            lags_list.append(lags_v)
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

        if lags_label == 'keys':
            best_lags = lags_grid[best_lags]
        
        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(best_lags)
            best_lags = forecaster.lags
        else:
            best_lags = 'custom_predictors'
        forecaster.set_params(best_params)

        forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
            f"  Levels: {results['levels'].iloc[0]}\n"
        )
            
    return results


def bayesian_search_forecaster_multiseries(
    forecaster,
    series: pd.DataFrame,
    search_space: Callable,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True,
    engine: str='optuna',
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using multi-series backtesting 
    and optuna library.
    **New in version 0.12.0**
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
    show_progress: bool, default `True`
        Whether to show a progress bar.
    engine : str, default `'optuna'`
        Bayesian optimization runs through the optuna library.
    kwargs_create_study : dict, default `{'direction': 'minimize', 'sampler': TPESampler(seed=123)}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study.
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
    results_opt_best : optuna object
        The best optimization result returned as a FrozenTrial optuna object.
    
    """

    if return_best and exog is not None and (len(exog) != len(series)):
        raise ValueError(
            (f"`exog` must have same number of samples as `series`. "
             f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
        )

    if engine not in ['optuna']:
        raise ValueError(
            f"`engine` only allows 'optuna', got {engine}."
        )

    results, results_opt_best = _bayesian_search_optuna_multiseries(
                                    forecaster            = forecaster,
                                    series                = series,
                                    exog                  = exog,
                                    levels                = levels, 
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


def _bayesian_search_optuna_multiseries(
    forecaster,
    series: pd.DataFrame,
    search_space: Callable,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True,
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using multi-series backtesting 
    and optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
    show_progress: bool, default `True`
        Whether to show a progress bar.
    kwargs_create_study : dict, default `{'direction': 'minimize', 'sampler': TPESampler(seed=123)}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study.
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
    results_opt_best : optuna object
        The best optimization result returned as a FrozenTrial optuna object.

    """
    
    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster,
                 series     = series,
                 levels     = levels
             )

    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
   
    lags_list = []
    params_list = []
    results_opt_best = None
    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )

    # Objective function using backtesting_forecaster_multiseries
    def _objective(
        trial,
        search_space          = search_space,
        forecaster            = forecaster,
        series                = series,
        exog                  = exog,
        steps                 = steps,
        levels                = levels,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        refit                 = refit,
        n_jobs                = n_jobs,
        verbose               = verbose
    ) -> float:
        
        forecaster.set_params(search_space(trial))
        
        metrics_levels = backtesting_forecaster_multiseries(
                             forecaster            = forecaster,
                             series                = series,
                             exog                  = exog,
                             steps                 = steps,
                             levels                = levels,
                             metric                = metric,
                             initial_train_size    = initial_train_size,
                             fixed_train_size      = fixed_train_size,
                             gap                   = gap,
                             allow_incomplete_fold = allow_incomplete_fold,
                             refit                 = refit,
                             n_jobs                = n_jobs,
                             verbose               = verbose,
                             show_progress         = False
                         )[0]
        
        # Store metrics in the variable `metric_values` defined outside _objective.
        nonlocal metric_values
        metric_values.append(metrics_levels)

        return metrics_levels.iloc[:, 1].mean()

    print(
        f"""Number of models compared: {n_trials*len(lags_grid)},
         {n_trials} bayesian search in each lag configuration."""
    )

    if show_progress:
        lags_grid_tqdm = tqdm(lags_grid.items(), desc='lags grid', position=0)
    else:
        lags_grid_tqdm = lags_grid.items()

    for lags_k, lags_v in lags_grid_tqdm:

        # `metric_values` will be modified inside _objective function. 
        # It is a trick to extract multiple values from _objective since
        # only the optimized value can be returned.
        metric_values = []

        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(lags_v)
            lags_v = lags_k if lags_label == 'keys' else forecaster.lags.copy()
        
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
            lags_list.append(lags_v)
            m_values = metric_values[i]
            for m in metric:
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(m_values[m_name].mean())
        
        if results_opt_best is None:
            results_opt_best = best_trial
        else:
            if best_trial.value < results_opt_best.value:
                results_opt_best = best_trial

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

        if lags_label == 'keys':
            best_lags = lags_grid[best_lags]
        
        if type(forecaster).__name__ != 'ForecasterAutoregMultiSeriesCustom':
            forecaster.set_lags(best_lags)
            best_lags = forecaster.lags
        else:
            best_lags = 'custom_predictors'
        forecaster.set_params(best_params)

        forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
            f"  Levels: {results['levels'].iloc[0]}\n"
        )
            
    return results, results_opt_best


# Alias MultiVariate
# ==============================================================================
def backtesting_forecaster_multivariate(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: Optional[int],
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: Union[bool, int]=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=False,
    show_progress: bool=True
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
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
    levels : str, list, default `None`
        Time series to be predicted. If `None` all levels will be predicted.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
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
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
        If there is more than one level, this structure will be repeated for each of them.

        - column pred: predictions.
        - column lower_bound: lower bound of the interval.
        - column upper_bound: upper bound of the interval.
    
    """

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        levels                = levels,
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

    return metrics_levels, backtest_predictions


def grid_search_forecaster_multivariate(
    forecaster,
    series: pd.DataFrame,
    param_grid: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    This function is an alias of grid_search_forecaster_multiseries.

    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using multi-series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

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
    
    """

    results = grid_search_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series,
        param_grid            = param_grid,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        levels                = levels,
        exog                  = exog,
        lags_grid             = lags_grid,
        refit                 = refit,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
    )

    return results


def random_search_forecaster_multivariate(
    forecaster,
    series: pd.DataFrame,
    param_distributions: dict,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    This function is an alias of random_search_forecaster_multiseries.

    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using multi-series backtesting.

    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
        Training time series.
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions or 
        lists of parameters to try.
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

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

    """

    results = random_search_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series,
        param_distributions   = param_distributions,
        steps                 = steps,
        metric                = metric,
        initial_train_size    = initial_train_size,
        fixed_train_size      = fixed_train_size,
        gap                   = gap,
        allow_incomplete_fold = allow_incomplete_fold,
        levels                = levels,
        exog                  = exog,
        lags_grid             = lags_grid,
        refit                 = refit,
        n_iter                = n_iter,
        random_state          = random_state,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
    ) 

    return results


def bayesian_search_forecaster_multivariate(
    forecaster,
    series: pd.DataFrame,
    search_space: Callable,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    levels: Optional[Union[str, list]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[Union[list, dict]]=None,
    refit: Union[bool, int]=False,
    n_trials: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Union[int, str]='auto',
    verbose: bool=True,
    show_progress: bool=True,
    engine: str='optuna',
    kwargs_create_study: dict={},
    kwargs_study_optimize: dict={}
) -> Tuple[pd.DataFrame, object]:
    """
    This function is an alias of bayesian_search_forecaster_multiseries.

    Bayesian optimization for a Forecaster object using multi-series backtesting 
    and optuna library.
    **New in version 0.12.0**
    
    Parameters
    ----------
    forecaster : ForecasterAutoregMultiSeries, ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate
        Forecaster model.
    series : pandas DataFrame
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
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.
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
    kwargs_create_study : dict, default `{'direction': 'minimize', 'sampler': TPESampler(seed=123)}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study.
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
    results_opt_best : optuna object
        The best optimization result returned as a FrozenTrial optuna object.
    
    """

    results, results_opt_best = bayesian_search_forecaster_multiseries(
                                    forecaster            = forecaster,
                                    series                = series,
                                    exog                  = exog,
                                    levels                = levels, 
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
                                    engine                = engine,
                                    kwargs_create_study   = kwargs_create_study,
                                    kwargs_study_optimize = kwargs_study_optimize
                                )

    return results, results_opt_best


def select_features_multiseries(
    forecaster: object,
    selector: object,
    series: pd.DataFrame,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    select_only: Optional[str]=None,
    force_inclusion: Optional[Union[list, str]]=None,
    subsample: Union[int, float]=0.5,
    random_state: int=123,
    verbose: bool=True
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
        X_train = X_train.drop(columns=exog_cols)
    elif select_only == 'exog':
        X_train = X_train.drop(columns=autoreg_cols)

    if isinstance(subsample, float):
        subsample = int(len(X_train)*subsample)

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
            ("No autoregressive features has been selected. Since a Forecaster "
             "cannot be created without them, be sure to include at least one "
             "using the `force_inclusion` parameter.")
        )
    else:
        if hasattr(forecaster, 'lags'):
            selected_autoreg = [int(feature.replace('lag_', '')) 
                                for feature in selected_autoreg]

    if exog_cols:
        exog_cols = [col for col in exog_cols if col not in encoding_cols]   

    if verbose:
        print(f"Recursive feature elimination ({selector.__class__.__name__})")
        print("--------------------------------" + "-"*len(selector.__class__.__name__))
        print(f"Total number of records available: {X_train.shape[0]}")
        print(f"Total number of records used for feature selection: {X_train_sample.shape[0]}")
        print(f"Number of features available: {len(autoreg_cols) + len(exog_cols)}") 
        print(f"    Autoreg (n={len(autoreg_cols)})")
        print(f"    Exog    (n={len(exog_cols)})")
        print(f"Number of features selected: {len(selected_features)}")
        print(f"    Autoreg (n={len(selected_autoreg)}) : {selected_autoreg}")
        print(f"    Exog    (n={len(selected_exog)}) : {selected_exog}")

    return selected_autoreg, selected_exog