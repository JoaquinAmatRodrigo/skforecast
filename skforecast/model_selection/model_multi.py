################################################################################
#                        skforecast.model_selection                            #
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
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import optuna
from optuna.samplers import TPESampler, RandomSampler
optuna.logging.set_verbosity(optuna.logging.WARNING) # disable optuna logs
from skopt.utils import use_named_args
from skopt import gp_minimize

from ..ForecasterAutoreg import ForecasterAutoreg
from ..ForecasterAutoregCustom import ForecasterAutoregCustom
from ..ForecasterAutoregDirect import ForecasterAutoregDirect
from ..ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from ..ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

from ..model_selection.model_selection import _get_metric

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)

def _backtesting_forecaster_verbose(
    index_values: Union[pd.DatetimeIndex, pd.RangeIndex],
    steps: int,
    refit: bool,
    initial_train_size: int,
    fixed_train_size: bool,
    folds: int,
    remainder: int,
) -> None:
    print(f"Information of backtesting process")
    print(f"----------------------------------")
    print(f"Number of observations used for initial training: {initial_train_size}")
    print(f"Number of observations used for backtesting: {len(index_values) - initial_train_size}")
    print(f"    Number of folds: {folds}")
    print(f"    Number of steps per fold: {steps}")
    if remainder != 0:
        print(f"    Last fold only includes {remainder} observations.")
    print("")
    for i in range(folds):
        if refit:
            if fixed_train_size:
                # The train size doesn't increase but moves by `steps` in each iteration.
                train_idx_start = i * steps
            else:
                # The train size increases by `steps` in each iteration.
                train_idx_start = 0
            train_idx_end = initial_train_size + i * steps
        else:
            # The train size doesn't increase and doesn't move
            train_idx_start = 0
            train_idx_end = initial_train_size
        last_window_end = initial_train_size + i * steps
        print(f"Data partition in fold: {i}")
        if i < folds - 1:
            print(f"    Training:   {index_values[train_idx_start]} -- {index_values[train_idx_end - 1]}  (n={len(index_values[train_idx_start:train_idx_end])})")
            print(f"    Validation: {index_values[last_window_end]} -- {index_values[last_window_end + steps - 1]}  (n={len(index_values[last_window_end:last_window_end + steps])})")
        else:
            print(f"    Training:   {index_values[train_idx_start]} -- {index_values[train_idx_end - 1]}  (n={len(index_values[train_idx_start:train_idx_end])})")
            print(f"    Validation: {index_values[last_window_end]} -- {index_values[-1]}  (n={len(index_values[last_window_end:])})")
    print("")

    return


def _backtesting_forecaster_refit_multiseries(
    forecaster,
    series: pd.DataFrame,
    steps: int,
    level: str,
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
    
    forecaster = deepcopy(forecaster)
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
    backtest_predictions = []
    
    folds = int(np.ceil((len(forecaster.index_values) - initial_train_size) / steps))
    remainder = (len(forecaster.index_values) - initial_train_size) % steps
    
    if verbose:
        _backtesting_forecaster_verbose(
            index_values       = forecaster.index_values,
            steps              = steps,
            refit              = True,
            initial_train_size = initial_train_size,
            fixed_train_size   = fixed_train_size,
            folds              = folds,
            remainder          = remainder
        )
        
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
                    forecaster.fit(series=series.iloc[train_idx_start:train_idx_end, ])
                    pred = forecaster.predict(steps=steps, level=level)
                else:
                    forecaster.fit(
                        series = series.iloc[train_idx_start:train_idx_end, ], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ]
                    )
                    pred = forecaster.predict(steps=steps, level=level, exog=next_window_exog)
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(series=series.iloc[train_idx_start:train_idx_end, ])
                        pred = forecaster.predict(steps=steps, level=level)
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end, ], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict(steps=steps, level=level, exog=next_window_exog)
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(series=series.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict(steps=steps, level=level)
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict(steps=steps, level=level, exog=next_window_exog)
        else:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(series=series.iloc[train_idx_start:train_idx_end])
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                else:
                    forecaster.fit(
                        series = series.iloc[train_idx_start:train_idx_end], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ]
                    )
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(series=series.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
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
                        forecaster.fit(series=series.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            series = series.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                level        = level,
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
                    y_true = series[level].iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return metric_value, backtest_predictions