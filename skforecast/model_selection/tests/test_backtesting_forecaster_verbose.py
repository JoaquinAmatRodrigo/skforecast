# Unit test _create_backtesting_folds
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection.model_selection import _backtesting_forecaster_verbose
from skforecast.model_selection.model_selection import _create_backtesting_folds


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


def test_backtesting_forecaster_verbose_refit_False_with_remainder(capfd):
    """
    Test print _backtesting_forecaster_verbose refit=False with remainder.
    """
    index_values = pd.DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
                                     '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
                                     '2022-01-09', '2022-01-10', '2022-01-11', '2022-01-12',
                                     '2022-01-13', '2022-01-14', '2022-01-15', '2022-01-16',
                                     '2022-01-17', '2022-01-18', '2022-01-19'],
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

    assert out == 'Information of backtesting process\n----------------------------------\nNumber of observations used for initial training: 5\nNumber of observations used for backtesting: 14\n    Number of folds: 5\n    Number of steps per fold: 3\n    Last fold only includes 2 observations.\n\nData partition in fold: 0\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-06 00:00:00 -- 2022-01-08 00:00:00  (n=3)\nData partition in fold: 1\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-09 00:00:00 -- 2022-01-11 00:00:00  (n=3)\nData partition in fold: 2\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-12 00:00:00 -- 2022-01-14 00:00:00  (n=3)\nData partition in fold: 3\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-15 00:00:00 -- 2022-01-17 00:00:00  (n=3)\nData partition in fold: 4\n    Training:   2022-01-01 00:00:00 -- 2022-01-05 00:00:00  (n=5)\n    Validation: 2022-01-18 00:00:00 -- 2022-01-19 00:00:00  (n=2)\n\n'


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


def test_create_backtesting_folds_no_refit_no_gap_no_remainder(capfd):
    """
    Test _create_backtesting_folds output when refit is False, gap=0 and not 
    remainder
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    test_size = 10
    refit = False

    folds = _create_backtesting_folds(
                y                     = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                gap                   = 0,
                refit                 = refit,
                fixed_train_size      = False,
                allow_incomplete_fold = True,
                return_all_indexes    = True,
                verbose               = True
            )
    
    expected = [[range(0, 70), range(70, 80), range(70, 80)],
                [range(0, 70), range(80, 90), range(80, 90)],
                [range(0, 70), range(90, 100), range(90, 100)]]
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    
    
def test_create_backtesting_folds_no_refit_no_gap_allow_incomplete_fold_False(capfd):
    """
    Test _create_backtesting_folds output when refit is False, gap=0, 
    remainder and allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 65
    test_size = 10
    refit = False
    allow_incomplete_fold = False

    folds = _create_backtesting_folds(
                y                     = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                gap                   = 0,
                refit                 = refit,
                fixed_train_size      = False,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = True,
                verbose               = True
            )
    
    expected = [[range(0, 65), range(65, 75), range(65, 75)],
                [range(0, 65), range(75, 85), range(75, 85)],
                [range(0, 65), range(85, 95), range(85, 95)]]
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 65\n"
        "Number of observations used for backtesting: 35\n"
        "    Number of folds: 3\n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-06 00:00:00  (n=65)\n"
        "    Validation: 2022-03-07 00:00:00 -- 2022-03-16 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-06 00:00:00  (n=65)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-26 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-06 00:00:00  (n=65)\n"
        "    Validation: 2022-03-27 00:00:00 -- 2022-04-05 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    
    
def test_create_backtesting_folds_no_refit_gap_allow_incomplete_fold_True_return_all_indexes_False(capfd):
    """
    Test _create_backtesting_folds output when refit is False, gap=5, 
    remainder, allow_incomplete_fold=True and return_all_indexes=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = False
    allow_incomplete_fold = True
    return_all_indexes = False

    folds = _create_backtesting_folds(
                y                     = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                gap                   = gap,
                refit                 = refit,
                fixed_train_size      = False,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
    
    expected = [[[0, 69], [70, 81], [75, 81]],
                [[0, 69], [77, 88], [82, 88]],
                [[0, 69], [84, 95], [89, 95]],
                [[0, 69], [91, 99], [96, 99]]]
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 4\n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold only includes 4 observations.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n"
        "Fold: 3\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-04-07 00:00:00 -- 2022-04-10 00:00:00  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected