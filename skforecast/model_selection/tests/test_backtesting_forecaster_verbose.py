# Unit test _backtesting_forecaster_verbose
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection.model_selection import _backtesting_forecaster_verbose


def test_backtesting_forecaster_verbose_refit_False(capfd):
    """
    Test print _backtesting_forecaster_verbose refit=False.
    """
    index_values = pd.DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
                                     '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
                                     '2022-01-09', '2022-01-10', '2022-01-11', '2022-01-12',
                                     '2022-01-13', '2022-01-14', '2022-01-15', '2022-01-16',
                                     '2022-01-17', '2022-01-18', '2022-01-19', '2022-01-20'],
                                     dtype='datetime64[ns]', name='date', freq='D')
    
    steps = 3
    refit = False
    initial_train_size = 5
    fixed_train_size = False
    folds = int(np.ceil((len(index_values) - initial_train_size) / steps))
    remainder = (len(index_values) - initial_train_size) % steps

    _backtesting_forecaster_verbose(
        index_values       = index_values,
        steps              = steps,
        refit              = refit,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        folds              = folds,
        remainder          = remainder
    )

    out, _ = capfd.readouterr()

    assert out == 'Information of backtesting process\n----------------------------------\nNumber of observations used for initial training: 5\nNumber of observations used for backtesting: 15\n    Number of folds: 5\n    Number of steps per fold: 3\n\nData partition in fold: 0\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-06 00:00:00 -- 2022-01-08 00:00:00  (n=3)\nData partition in fold: 1\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-09 00:00:00 -- 2022-01-11 00:00:00  (n=3)\nData partition in fold: 2\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-12 00:00:00 -- 2022-01-14 00:00:00  (n=3)\nData partition in fold: 3\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-15 00:00:00 -- 2022-01-17 00:00:00  (n=3)\nData partition in fold: 4\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-18 00:00:00 -- 2022-01-20 00:00:00  (n=3)\n\n'


def test_backtesting_forecaster_verbose_refit_True_and_fixed_train_size_True(capfd):
    """
    Test print _backtesting_forecaster_verbose refit=True, fixed_train_size=True.
    """
    index_values = pd.DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
                                     '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
                                     '2022-01-09', '2022-01-10', '2022-01-11', '2022-01-12',
                                     '2022-01-13', '2022-01-14', '2022-01-15', '2022-01-16',
                                     '2022-01-17', '2022-01-18', '2022-01-19', '2022-01-20'],
                                     dtype='datetime64[ns]', name='date', freq='D')
    
    steps = 3
    refit = True
    initial_train_size = 5
    fixed_train_size = True
    folds = int(np.ceil((len(index_values) - initial_train_size) / steps))
    remainder = (len(index_values) - initial_train_size) % steps

    _backtesting_forecaster_verbose(
        index_values       = index_values,
        steps              = steps,
        refit              = refit,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        folds              = folds,
        remainder          = remainder
    )

    out, _ = capfd.readouterr()

    assert out == 'Information of backtesting process\n----------------------------------\nNumber of observations used for initial training: 5\nNumber of observations used for backtesting: 15\n    Number of folds: 5\n    Number of steps per fold: 3\n\nData partition in fold: 0\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-06 00:00:00 -- 2022-01-08 00:00:00  (n=3)\nData partition in fold: 1\n    Training:   2022-01-04 00:00:00 -- 2022-01-08 00:00:00  (n=5)\n    Validation: 2022-01-09 00:00:00 -- 2022-01-11 00:00:00  (n=3)\nData partition in fold: 2\n    Training:   2022-01-07 00:00:00 -- 2022-01-11 00:00:00  (n=5)\n    Validation: 2022-01-12 00:00:00 -- 2022-01-14 00:00:00  (n=3)\nData partition in fold: 3\n    Training:   2022-01-10 00:00:00 -- 2022-01-14 00:00:00  (n=5)\n    Validation: 2022-01-15 00:00:00 -- 2022-01-17 00:00:00  (n=3)\nData partition in fold: 4\n    Training:   2022-01-13 00:00:00 -- 2022-01-17 00:00:00  (n=5)\n    Validation: 2022-01-18 00:00:00 -- 2022-01-20 00:00:00  (n=3)\n\n'


def test_backtesting_forecaster_verbose_refit_True_and_fixed_train_size_False(capfd):
    """
    Test print _backtesting_forecaster_verbose refit=True, fixed_train_size=False.
    """
    index_values = pd.DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
                                     '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
                                     '2022-01-09', '2022-01-10', '2022-01-11', '2022-01-12',
                                     '2022-01-13', '2022-01-14', '2022-01-15', '2022-01-16',
                                     '2022-01-17', '2022-01-18', '2022-01-19', '2022-01-20'],
                                     dtype='datetime64[ns]', name='date', freq='D')
    
    steps = 3
    refit = True
    initial_train_size = 5
    fixed_train_size = False
    folds = int(np.ceil((len(index_values) - initial_train_size) / steps))
    remainder = (len(index_values) - initial_train_size) % steps

    _backtesting_forecaster_verbose(
        index_values       = index_values,
        steps              = steps,
        refit              = refit,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        folds              = folds,
        remainder          = remainder
    )

    out, _ = capfd.readouterr()

    assert out == 'Information of backtesting process\n----------------------------------\nNumber of observations used for initial training: 5\nNumber of observations used for backtesting: 15\n    Number of folds: 5\n    Number of steps per fold: 3\n\nData partition in fold: 0\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-06 00:00:00 -- 2022-01-08 00:00:00  (n=3)\nData partition in fold: 1\n    Training:   2022-01-01 00:00:00 -- 2022-01-08 00:00:00  (n=8)\n    Validation: 2022-01-09 00:00:00 -- 2022-01-11 00:00:00  (n=3)\nData partition in fold: 2\n    Training:   2022-01-01 00:00:00 -- 2022-01-11 00:00:00  (n=11)\n    Validation: 2022-01-12 00:00:00 -- 2022-01-14 00:00:00  (n=3)\nData partition in fold: 3\n    Training:   2022-01-01 00:00:00 -- 2022-01-14 00:00:00  (n=14)\n    Validation: 2022-01-15 00:00:00 -- 2022-01-17 00:00:00  (n=3)\nData partition in fold: 4\n    Training:   2022-01-01 00:00:00 -- 2022-01-17 00:00:00  (n=17)\n    Validation: 2022-01-18 00:00:00 -- 2022-01-20 00:00:00  (n=3)\n\n'