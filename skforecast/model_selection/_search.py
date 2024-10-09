################################################################################
#                             skforecast._search                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import os
from copy import copy
import logging
from typing import Union, Tuple, Optional, Callable
import warnings
import pandas as pd
from tqdm.auto import tqdm
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid, ParameterSampler
from ..model_selection._split import TimeSeriesFold, OneStepAheadFold
from ..model_selection._validation import backtesting_forecaster
from ..metrics import add_y_train_argument, _get_metric
from ..utils import (
    initialize_lags_grid,
    initialize_lags
)


def grid_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    param_grid: dict,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
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
    forecaster : ForecasterAutoreg, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try.
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
    
    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters(
        forecaster            = forecaster,
        y                     = y,
        cv                    = cv,
        param_grid            = param_grid,
        metric                = metric,
        exog                  = exog,
        lags_grid             = lags_grid,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress,
        output_file           = output_file
    )

    return results


def random_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    param_distributions: dict,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
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
    forecaster : ForecasterAutoreg, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try.
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

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters(
        forecaster            = forecaster,
        y                     = y,
        cv                    = cv,
        param_grid            = param_grid,
        metric                = metric,
        exog                  = exog,
        lags_grid             = lags_grid,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress,
        output_file           = output_file
    )

    return results


def _calculate_metrics_one_step_ahead(
    forecaster: object,
    y: pd.Series,
    metrics: list,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, dict],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, dict]
) -> list:
    """
    Calculate metrics when predictions are one-step-ahead. When forecaster is
    of type ForecasterAutoregDirect only the regressor for step 1 is used.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    y : pandas Series
        Time series data used to train and test the model.
    metrics : list
        List of metrics.
    X_train : pandas DataFrame
        Predictor values used to train the model.
    y_train : pandas Series
        Target values related to each row of `X_train`.
    X_test : pandas DataFrame
        Predictor values used to test the model.
    y_test : pandas Series
        Target values related to each row of `X_test`.

    Returns
    -------
    metric_values : list
        List with metric values.
    
    """

    if type(forecaster).__name__ == 'ForecasterAutoregDirect':

        step = 1  # Only model for step 1 is optimized.
        X_train, y_train = forecaster.filter_train_X_y_for_step(
                               step    = step,
                               X_train = X_train,
                               y_train = y_train
                           )
        X_test, y_test = forecaster.filter_train_X_y_for_step(
                             step    = step,  
                             X_train = X_test,
                             y_train = y_test
                         )
        forecaster.regressors_[step].fit(X_train, y_train)
        pred = forecaster.regressors_[step].predict(X_test)

    else:
        forecaster.regressor.fit(X_train, y_train)
        pred = forecaster.regressor.predict(X_test)

    pred = pred.ravel()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    if forecaster.differentiation is not None:
        differentiator = copy(forecaster.differentiator)
        differentiator.initial_values = (
            [y.iloc[forecaster.window_size - forecaster.differentiation]]
        )
        pred = differentiator.inverse_transform_next_window(pred)
        y_test = differentiator.inverse_transform_next_window(y_test)
        y_train = differentiator.inverse_transform(y_train)

    if forecaster.transformer_y is not None:
        pred = forecaster.transformer_y.inverse_transform(pred.reshape(-1, 1))
        y_test = forecaster.transformer_y.inverse_transform(y_test.reshape(-1, 1))
        y_train = forecaster.transformer_y.inverse_transform(y_train.reshape(-1, 1))

    metric_values = []
    for m in metrics:
        metric_values.append(
            m(y_true=y_test.ravel(), y_pred=pred.ravel(), y_train=y_train.ravel())
        )

    return metric_values


def _evaluate_grid_hyperparameters(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    param_grid: dict,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
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
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try.
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

    cv_name = type(cv).__name__

    if cv_name not in ['TimeSeriesFold', 'OneStepAheadFold']:
        raise ValueError(
           (f"`cv` must be an instance of `TimeSeriesFold` or `OneStepAheadFold`. "
            f"Got {type(cv)}.")
        )
    
    if cv_name == 'OneStepAheadFold':
        warnings.warn(
            ("One-step-ahead predictions are used for faster model comparison, but they "
             "may not fully represent multi-step prediction performance. It is recommended "
             "to backtest the final model for a more accurate multi-step performance "
             "estimate.")
        )

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
    metric_dict = {
        (m if isinstance(m, str) else m.__name__): [] 
        for m in metric
    }
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
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
    for lags_k, lags_v in lags_grid_tqdm:
        
        forecaster.set_lags(lags_v)
        lags_v = forecaster.lags.copy()
        if lags_label == 'values':
            lags_k = lags_v

        if cv_name == 'OneStepAheadFold':

            (
                X_train,
                y_train,
                X_test,
                y_test
            ) = forecaster._train_test_split_one_step_ahead(
                y=y, initial_train_size=cv.initial_train_size, exog=exog
            )

        for params in param_grid:

            forecaster.set_params(params)

            if cv_name == 'TimeSeriesFold':

                metric_values = backtesting_forecaster(
                                    forecaster    = forecaster,
                                    y             = y,
                                    cv            = cv,
                                    metric        = metric,
                                    exog          = exog,
                                    interval      = None,
                                    n_jobs        = n_jobs,
                                    verbose       = verbose,
                                    show_progress = False
                                )[0]
                metric_values = metric_values.iloc[0, :].to_list()

            else:

                metric_values = _calculate_metrics_one_step_ahead(
                                    forecaster = forecaster,
                                    y          = y,
                                    metrics    = metric,
                                    X_train    = X_train,
                                    y_train    = y_train,
                                    X_test     = X_test,
                                    y_test     = y_test
                                )

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
        
        forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  {'Backtesting' if cv_name == 'TimeSeriesFold' else 'One-step-ahead'} "
            f"metric: {best_metric}"
        )
            
    return results


def bayesian_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    search_space: Callable,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
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
    Bayesian search for hyperparameters of a Forecaster object.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
    search_space : Callable (optuna)
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
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
        The best optimization result returned as a FrozenTrial optuna object.
    
    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f"`exog` must have same number of samples as `y`. "
             f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
        )
            
    results, best_trial = _bayesian_search_optuna(
                                forecaster            = forecaster,
                                y                     = y,
                                cv                    = cv,
                                exog                  = exog,
                                search_space          = search_space,
                                metric                = metric,
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

    return results, best_trial


def _bayesian_search_optuna(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    search_space: Callable,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
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
    Bayesian search for hyperparameters of a Forecaster object using optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series. 
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.
    search_space : Callable
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
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
    
    cv_name = type(cv).__name__

    if cv_name not in ['TimeSeriesFold', 'OneStepAheadFold']:
        raise ValueError(
           (f"`cv` must be an instance of `TimeSeriesFold` or `OneStepAheadFold`. "
            f"Got {type(cv)}.")
        )
    
    if cv_name == 'OneStepAheadFold':
        warnings.warn(
            ("One-step-ahead predictions are used for faster model comparison, but they "
             "may not fully represent multi-step prediction performance. It is recommended "
             "to backtest the final model for a more accurate multi-step performance "
             "estimate.")
        )
    
    if not isinstance(metric, list):
        metric = [metric]
    metric = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
        
    # Objective function using backtesting_forecaster
    if cv_name == 'TimeSeriesFold':

        def _objective(
            trial,
            search_space          = search_space,
            forecaster            = forecaster,
            y                     = y,
            cv                    = cv,
            exog                  = exog,
            metric                = metric,
            n_jobs                = n_jobs,
            verbose               = verbose,
        ) -> float:
            
            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != 'lags'}
            forecaster.set_params(sample_params)
            if "lags" in sample:
                forecaster.set_lags(sample['lags'])
            
            metrics, _ = backtesting_forecaster(
                             forecaster            = forecaster,
                             y                     = y,
                             cv                    = cv,
                             exog                  = exog,
                             metric                = metric,
                             n_jobs                = n_jobs,
                             verbose               = verbose,
                             show_progress         = False
                         )
            metrics = metrics.iloc[0, :].to_list()
            
            # Store metrics in the variable `metric_values` defined outside _objective.
            nonlocal metric_values
            metric_values.append(metrics)

            return metrics[0]
        
    else:

        def _objective(
            trial,
            search_space          = search_space,
            forecaster            = forecaster,
            y                     = y,
            cv                    = cv,
            exog                  = exog,
            metric                = metric
        ) -> float:
            
            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != 'lags'}
            forecaster.set_params(sample_params)
            if "lags" in sample:
                forecaster.set_lags(sample['lags'])

            (
                X_train,
                y_train,
                X_test,
                y_test
            ) = forecaster._train_test_split_one_step_ahead(
                y=y, initial_train_size=cv.initial_train_size, exog=exog
            )

            metrics = _calculate_metrics_one_step_ahead(
                          forecaster = forecaster,
                          y          = y,
                          metrics    = metric,
                          X_train    = X_train,
                          y_train    = y_train,
                          X_test     = X_test,
                          y_test     = y_test
                      )

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
        for m, m_values in zip(metric, metric_values[i]):
            m_name = m if isinstance(m, str) else m.__name__
            metric_dict[m_name].append(m_values)
    
    lags_list = [
        initialize_lags(forecaster_name=type(forecaster).__name__, lags = lag)[0]
        for lag in lags_list
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
        
        forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  {'Backtesting' if cv_name == 'TimeSeriesFold' else 'One-step-ahead'} "
            f"metric: {best_metric}"
        )
            
    return results, best_trial
