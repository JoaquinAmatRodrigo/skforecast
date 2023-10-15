################################################################################
#                      skforecast.model_selection_sarimax                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable
import pandas as pd
import warnings
import logging
from copy import deepcopy
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

from ..exceptions import LongTrainingWarning
from ..model_selection.model_selection import _get_metric
from ..model_selection.model_selection import _create_backtesting_folds
from ..utils import check_backtesting_input
from ..utils import select_n_jobs_backtesting

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def _backtesting_sarimax(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: Optional[Union[bool, int]]=False,
    alpha: Optional[float]=None,
    interval: Optional[list]=None,
    n_jobs: Optional[Union[int, str]]='auto',
    suppress_warnings_fit: bool=False,
    verbose: bool=False,
    show_progress: bool=True,
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of ForecasterSarimax.

    - If `refit` is `False`, the model will be trained only once using the 
    `initial_train_size` first observations. 
    - If `refit` is `True`, the model is trained on each iteration, increasing
    the training set. 
    - If `refit` is an `integer`, the model will be trained every that number 
    of iterations.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
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
    initial_train_size : int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
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
    alpha : float, default `0.05`
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
    interval : list, default `None`
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    suppress_warnings_fit : bool, default `False`
        If `True`, warnings generated during fitting will be ignored.
        **New in version 0.10.0**
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
    
    if refit == False:
        n_jobs = 1
    else:
        if n_jobs == 'auto':        
            n_jobs = select_n_jobs_backtesting(
                         forecaster_name = type(forecaster).__name__,
                         regressor_name  = type(forecaster.regressor).__name__,
                         refit           = refit
                     )
        else:
            n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    if not isinstance(metric, list):
        metrics = [_get_metric(metric=metric) if isinstance(metric, str) else metric]
    else:
        metrics = [_get_metric(metric=m) if isinstance(m, str) else m for m in metric]

    # initial_train_size cannot be None because of append method in Sarimax
    # First model training, this is done to allow parallelization when `refit` 
    # is `False`. The initial Forecaster fit is outside the auxiliary function.
    exog_train = exog.iloc[:initial_train_size, ] if exog is not None else None
    forecaster.fit(
        y                 = y.iloc[:initial_train_size, ],
        exog              = exog_train,
        suppress_warnings = suppress_warnings_fit
    )
    window_size = forecaster.window_size
    externally_fitted = False
    
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
                verbose               = verbose  
            )
    
    if refit:
        n_of_fits = int(len(folds)/refit)
        if n_of_fits > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits} times. This can take substantial "
                f"amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )
       
    folds_tqdm = tqdm(folds) if show_progress else folds

    def _fit_predict_forecaster(y, exog, forecaster, alpha, interval, fold, steps):
        """
        Fit the forecaster and predict `steps` ahead. This is an auxiliary 
        function used to parallelize the backtesting_forecaster function.
        """

        # In each iteration the model is fitted before making predictions. 
        # if fixed_train_size the train size doesn't increase but moves by `steps` 
        # in each iteration. if False the train size increases by `steps` in each 
        # iteration.
        train_idx_start = fold[0][0]
        train_idx_end   = fold[0][1]
        test_idx_start  = fold[2][0]
        test_idx_end    = fold[2][1]

        if refit:
            last_window_start = fold[0][1] # Same as train_idx_end
            last_window_end   = fold[1][1]
        else:
            last_window_end   = fold[2][0] # test_idx_start
            last_window_start = last_window_end - steps

        if fold[4] is False:
            # When the model is not fitted, last_window and last_window_exog must 
            # be updated to include the data needed to make predictions.
            last_window_y = y.iloc[last_window_start:last_window_end]
            last_window_exog = exog.iloc[last_window_start:last_window_end] if exog is not None else None 
        else:
            # The model is fitted before making predictions. If `fixed_train_size`  
            # the train size doesn't increase but moves by `steps` in each iteration. 
            # If `False` the train size increases by `steps` in each  iteration.
            y_train = y.iloc[train_idx_start:train_idx_end, ]
            exog_train = exog.iloc[train_idx_start:train_idx_end, ] if exog is not None else None
            
            last_window_y = None
            last_window_exog = None

            forecaster.fit(y=y_train, exog=exog_train, suppress_warnings=suppress_warnings_fit)

        next_window_exog = exog.iloc[test_idx_start:test_idx_end, ] if exog is not None else None

        # After the first fit, ARIMA must use the last windows stored in the model
        if fold == folds[0]:
            last_window_y = None
            last_window_exog = None

        steps = len(range(test_idx_start, test_idx_end))
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

        pred = pred.iloc[gap:, ]            
        
        return pred

    backtest_predictions = (
        Parallel(n_jobs=n_jobs)
        (delayed(_fit_predict_forecaster)
        (
            y=y, 
            exog=exog, 
            forecaster=forecaster, 
            alpha=alpha, 
            interval=interval, 
            fold=fold, 
            steps=steps
        )
        for fold in folds_tqdm)
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


def backtesting_sarimax(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, Callable, list],
    initial_train_size: int,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: Optional[Union[bool, int]]=False,
    alpha: Optional[float]=None,
    interval: Optional[list]=None,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=False,
    suppress_warnings_fit: bool=False,
    show_progress: bool=True
) -> Tuple[Union[float, list], pd.DataFrame]:
    """
    Backtesting of ForecasterSarimax.

    - If `refit` is `False`, the model will be trained only once using the 
    `initial_train_size` first observations. 
    - If `refit` is `True`, the model is trained on each iteration, increasing
    the training set. 
    - If `refit` is an `integer`, the model will be trained every that number 
    of iterations.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.

    Parameters
    ----------
    forecaster : ForecasterSarimax
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
    initial_train_size : int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
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
    alpha : float, default `0.05`
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
    interval : list, default `None`
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**     
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    suppress_warnings_fit : bool, default `False`
        If `True`, warnings generated during fitting will be ignored.
        **New in version 0.10.0**
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
    
    if type(forecaster).__name__ not in ['ForecasterSarimax']:
        raise TypeError(
            ("`forecaster` must be of type `ForecasterSarimax`, for all other "
             "types of forecasters use the functions available in the other "
             "`model_selection` modules.")
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
        alpha                 = alpha,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress
    )
    
    metrics_values, backtest_predictions = _backtesting_sarimax(
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
        alpha                 = alpha,
        interval              = interval,
        n_jobs                = n_jobs,
        verbose               = verbose,
        suppress_warnings_fit = suppress_warnings_fit,
        show_progress         = show_progress
    )

    return metrics_values, backtest_predictions


def grid_search_sarimax(
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
    refit: Optional[Union[bool, int]]=False,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    suppress_warnings_fit: bool=False,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a ForecasterSarimax object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
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
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
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
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    suppress_warnings_fit : bool, default `False`
        If `True`, warnings generated during fitting will be ignored.
        **New in version 0.10.0**
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column params: parameters configuration for each iteration.
            - column metric: metric value estimated for each iteration.
            - additional n columns with param = value.
    
    """

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters_sarimax(
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
        refit                 = refit,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        suppress_warnings_fit = suppress_warnings_fit,
        show_progress         = show_progress
    )

    return results


def random_search_sarimax(
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
    refit: Optional[Union[bool, int]]=False,
    n_iter: int=10,
    random_state: int=123,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    suppress_warnings_fit: bool=False,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
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
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
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
    n_iter : int, default `10`
        Number of parameter settings that are sampled. 
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
    suppress_warnings_fit : bool, default `False`
        If `True`, warnings generated during fitting will be ignored.
        **New in version 0.10.0**
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column params: parameters configuration for each iteration.
            - column metric: metric value estimated for each iteration.
            - additional n columns with param = value.
    
    """

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters_sarimax(
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
        refit                 = refit,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        suppress_warnings_fit = suppress_warnings_fit,
        show_progress         = show_progress
    )

    return results


def _evaluate_grid_hyperparameters_sarimax(
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
    refit: Optional[Union[bool, int]]=False,
    return_best: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=True,
    suppress_warnings_fit: bool=False,
    show_progress: bool=True
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
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
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
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
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
        **New in version 0.9.0**
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    suppress_warnings_fit : bool, default `False`
        If `True`, warnings generated during fitting will be ignored.
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

            - column params: lower bound of the interval.
            - column metric: metric value estimated for the combination of parameters.
            - additional n columns with param = value.

    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f'`exog` must have same number of samples as `y`. '
             f'length `exog`: ({len(exog)}), length `y`: ({len(y)})')
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

    if show_progress:
        param_grid = tqdm(param_grid, desc='params grid', position=0)
  
    for params in param_grid:

        forecaster.set_params(params)
        metrics_values = backtesting_sarimax(
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
                            alpha                 = None,
                            interval              = None,
                            n_jobs                = n_jobs,
                            verbose               = verbose,
                            suppress_warnings_fit = suppress_warnings_fit,
                            show_progress         = False
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
        
        forecaster.set_params(best_params)
        forecaster.fit(y=y, exog=exog, suppress_warnings=suppress_warnings_fit)
        
        print(
            f"`Forecaster` refitted using the best-found parameters, and the whole data set: \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results