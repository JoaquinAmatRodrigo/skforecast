# Unit test _create_backtesting_folds
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection.model_selection import _create_backtesting_folds


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 80), range(70, 80), True],
                                  [range(0, 70), range(80, 90), range(80, 90), False],
                                  [range(0, 70), range(90, 100), range(90, 100), False]]),
                          (False, [[[0, 70], [70, 80], [70, 80], True],
                                   [[0, 70], [80, 90], [80, 90], False],
                                   [[0, 70], [90, 100], [90, 100], False]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_no_refit_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is False, gap=0 and not 
    remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    test_size = 10
    refit = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = False,
                gap                   = 0,
                allow_incomplete_fold = True,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
    
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
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 65), range(65, 75), range(65, 75), True],
                                  [range(0, 65), range(75, 85), range(75, 85), False],
                                  [range(0, 65), range(85, 95), range(85, 95), False]]),
                          (False, [[[0, 65], [65, 75], [65, 75], True],
                                   [[0, 65], [75, 85], [75, 85], False],
                                   [[0, 65], [85, 95], [85, 95], False]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_no_refit_no_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is 0 (False), gap=0, 
    remainder and allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 65
    test_size = 10
    refit = 0
    allow_incomplete_fold = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = False,
                gap                   = 0,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
    
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
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(77, 89), range(82, 89), False],
                                  [range(0, 70), range(84, 96), range(89, 96), False],
                                  [range(0, 70), range(91, 100), range(96, 100), False]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [77, 89], [82, 89], False],
                                   [[0, 70], [84, 96], [89, 96], False],
                                   [[0, 70], [91, 100], [96, 100], False]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_no_refit_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is False, gap=5, 
    remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = False
    allow_incomplete_fold = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = False,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
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
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 15), range(15, 30), range(20, 30), False],
                                  [range(0, 15), range(25, 40), range(30, 40), False],
                                  [range(0, 15), range(35, 50), range(40, 50), False],
                                  [range(0, 15), range(45, 60), range(50, 60), False],
                                  [range(0, 15), range(55, 70), range(60, 70), False],
                                  [range(0, 15), range(65, 80), range(70, 80), False],
                                  [range(0, 15), range(75, 90), range(80, 90), False],
                                  [range(0, 15), range(85, 100), range(90, 100), False]]),
                          (False, [[[0, 15], [15, 30], [20, 30], False],
                                   [[0, 15], [25, 40], [30, 40], False],
                                   [[0, 15], [35, 50], [40, 50], False],
                                   [[0, 15], [45, 60], [50, 60], False],
                                   [[0, 15], [55, 70], [60, 70], False],
                                   [[0, 15], [65, 80], [70, 80], False],
                                   [[0, 15], [75, 90], [80, 90], False],
                                   [[0, 15], [85, 100], [90, 100], False]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_no_refit_no_initial_train_size(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is False, gap=5, 
    already trained, no remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    initial_train_size = 15 # window_size
    gap = 5
    test_size = 10
    refit = False
    externally_fitted = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = externally_fitted,
                refit                 = refit,
                fixed_train_size      = False,
                gap                   = gap,
                allow_incomplete_fold = True,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "An already trained forecaster is to be used. Window size: 15\n"
        "Number of observations used for backtesting: 85\n"
        "    Number of folds: 8\n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n\n"
        "Fold: 0\n"
        "    Validation: 20 -- 29  (n=10)\n"
        "Fold: 1\n"
        "    Validation: 30 -- 39  (n=10)\n"
        "Fold: 2\n"
        "    Validation: 40 -- 49  (n=10)\n"
        "Fold: 3\n"
        "    Validation: 50 -- 59  (n=10)\n"
        "Fold: 4\n"
        "    Validation: 60 -- 69  (n=10)\n"
        "Fold: 5\n"
        "    Validation: 70 -- 79  (n=10)\n"
        "Fold: 6\n"
        "    Validation: 80 -- 89  (n=10)\n"
        "Fold: 7\n"
        "    Validation: 90 -- 99  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 80), range(70, 80), True],
                                  [range(0, 80), range(80, 90), range(80, 90), True],
                                  [range(0, 90), range(90, 100), range(90, 100), True]]),
                          (False, [[[0, 70], [70, 80], [70, 80], True],
                                   [[0, 80], [80, 90], [80, 90], True],
                                   [[0, 90], [90, 100], [90, 100], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_no_fixed_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is True, fixed_train_size is 
    False, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    test_size = 10
    refit = True
    fixed_train_size = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = 0,
                allow_incomplete_fold = True,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
    
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
        "    Training:   2022-01-01 00:00:00 -- 2022-03-21 00:00:00  (n=80)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-31 00:00:00  (n=90)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 80), range(70, 80), True],
                                  [range(10, 80), range(80, 90), range(80, 90), True],
                                  [range(20, 90), range(90, 100), range(90, 100), True]]),
                          (False, [[[0, 70], [70, 80], [70, 80], True],
                                   [[10, 80], [80, 90], [80, 90], True],
                                   [[20, 90], [90, 100], [90, 100], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_fixed_train_size_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is 1 (True), fixed_train_size 
    is True, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    test_size = 10
    refit = 1
    fixed_train_size = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = 0,
                allow_incomplete_fold = True,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
    
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
        "    Training:   2022-01-11 00:00:00 -- 2022-03-21 00:00:00  (n=70)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-21 00:00:00 -- 2022-03-31 00:00:00  (n=70)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 77), range(77, 89), range(82, 89), True],
                                  [range(0, 84), range(84, 96), range(89, 96), True],
                                  [range(0, 91), range(91, 100), range(96, 100), True]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[0, 77], [77, 89], [82, 89], True],
                                   [[0, 84], [84, 96], [89, 96], True],
                                   [[0, 91], [91, 100], [96, 100], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_no_fixed_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is True, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = True
    fixed_train_size = False
    allow_incomplete_fold = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
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
        "    Training:   2022-01-01 00:00:00 -- 2022-03-18 00:00:00  (n=77)\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-25 00:00:00  (n=84)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n"
        "Fold: 3\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-04-01 00:00:00  (n=91)\n"
        "    Validation: 2022-04-07 00:00:00 -- 2022-04-10 00:00:00  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(7, 77), range(77, 89), range(82, 89), True],
                                  [range(14, 84), range(84, 96), range(89, 96), True],
                                  [range(21, 91), range(91, 100), range(96, 100), True]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[7, 77], [77, 89], [82, 89], True],
                                   [[14, 84], [84, 96], [89, 96], True],
                                   [[21, 91], [91, 100], [96, 100], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_fixed_train_size_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is True, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = True
    fixed_train_size = True
    allow_incomplete_fold = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
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
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   7 -- 76  (n=70)\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   14 -- 83  (n=70)\n"
        "    Validation: 89 -- 95  (n=7)\n"
        "Fold: 3\n"
        "    Training:   21 -- 90  (n=70)\n"
        "    Validation: 96 -- 99  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 77), range(77, 89), range(82, 89), True],
                                  [range(0, 84), range(84, 96), range(89, 96), True]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[0, 77], [77, 89], [82, 89], True],
                                   [[0, 84], [84, 96], [89, 96], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_no_fixed_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is True, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = True
    fixed_train_size = False
    allow_incomplete_fold = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-18 00:00:00  (n=77)\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-25 00:00:00  (n=84)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(7, 77), range(77, 89), range(82, 89), True],
                                  [range(14, 84), range(84, 96), range(89, 96), True]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[7, 77], [77, 89], [82, 89], True],
                                   [[14, 84], [84, 96], [89, 96], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_fixed_train_size_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is True, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = True
    fixed_train_size = True
    allow_incomplete_fold = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   7 -- 76  (n=70)\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   14 -- 83  (n=70)\n"
        "    Validation: 89 -- 95  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 60), range(60, 70), range(60, 70), True],
                                  [range(0, 60), range(70, 80), range(70, 80), False],
                                  [range(0, 80), range(80, 90), range(80, 90), True],
                                  [range(0, 80), range(90, 100), range(90, 100), False]]),
                          (False, [[[0, 60], [60, 70], [60, 70], True],
                                   [[0, 60], [70, 80], [70, 80], False],
                                   [[0, 80], [80, 90], [80, 90], True],
                                   [[0, 80], [90, 100], [90, 100], False]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_int_no_fixed_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is 2, fixed_train_size is 
    False, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 60
    test_size = 10
    refit = 2
    fixed_train_size = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = 0,
                allow_incomplete_fold = True,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 60\n"
        "Number of observations used for backtesting: 40\n"
        "    Number of folds: 4\n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-01 00:00:00  (n=60)\n"
        "    Validation: 2022-03-02 00:00:00 -- 2022-03-11 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-01 00:00:00  (n=60)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-21 00:00:00  (n=80)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 3\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-21 00:00:00  (n=80)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 60), range(60, 70), range(60, 70), True],
                                  [range(0, 60), range(70, 80), range(70, 80), False],
                                  [range(0, 60), range(80, 90), range(80, 90), False],
                                  [range(30, 90), range(90, 100), range(90, 100), True]]),
                          (False, [[[0, 60], [60, 70], [60, 70], True],
                                   [[0, 60], [70, 80], [70, 80], False],
                                   [[0, 60], [80, 90], [80, 90], False],
                                   [[30, 90], [90, 100], [90, 100], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_int_fixed_train_size_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is 3, fixed_train_size is 
    True, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 60
    test_size = 10
    refit = 3
    fixed_train_size = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = 0,
                allow_incomplete_fold = True,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 60\n"
        "Number of observations used for backtesting: 40\n"
        "    Number of folds: 4\n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-01 00:00:00  (n=60)\n"
        "    Validation: 2022-03-02 00:00:00 -- 2022-03-11 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-01 00:00:00  (n=60)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-01 00:00:00  (n=60)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 3\n"
        "    Training:   2022-01-31 00:00:00 -- 2022-03-31 00:00:00  (n=60)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(77, 89), range(82, 89), False],
                                  [range(0, 84), range(84, 96), range(89, 96), True],
                                  [range(0, 84), range(91, 100), range(96, 100), False]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [77, 89], [82, 89], False],
                                   [[0, 84], [84, 96], [89, 96], True],
                                   [[0, 84], [91, 100], [96, 100], False]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_int_no_fixed_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is 2, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = 2
    fixed_train_size = False
    allow_incomplete_fold = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
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
        "    Training:   2022-01-01 00:00:00 -- 2022-03-25 00:00:00  (n=84)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n"
        "Fold: 3\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-25 00:00:00  (n=84)\n"
        "    Validation: 2022-04-07 00:00:00 -- 2022-04-10 00:00:00  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(77, 89), range(82, 89), False],
                                  [range(0, 70), range(84, 96), range(89, 96), False],
                                  [range(21, 91), range(91, 100), range(96, 100), True]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [77, 89], [82, 89], False],
                                   [[0, 70], [84, 96], [89, 96], False],
                                   [[21, 91], [91, 100], [96, 100], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_int_fixed_train_size_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is 3, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = 3
    fixed_train_size = True
    allow_incomplete_fold = True

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
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
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 89 -- 95  (n=7)\n"
        "Fold: 3\n"
        "    Training:   21 -- 90  (n=70)\n"
        "    Validation: 96 -- 99  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(77, 89), range(82, 89), False],
                                  [range(0, 70), range(84, 96), range(89, 96), False]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [77, 89], [82, 89], False],
                                   [[0, 70], [84, 96], [89, 96], False]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_int_no_fixed_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is 3, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = 3
    fixed_train_size = False
    allow_incomplete_fold = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(77, 89), range(82, 89), False],
                                  [range(14, 84), range(84, 96), range(89, 96), True]]),
                          (False, [[[0, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [77, 89], [82, 89], False],
                                   [[14, 84], [84, 96], [89, 96], True]])], 
                         ids = lambda argument : f'{argument}' )
def test_create_backtesting_folds_refit_int_fixed_train_size_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test _create_backtesting_folds output when refit is True, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    initial_train_size = 70
    gap = 5
    test_size = 7
    refit = 2
    fixed_train_size = True
    allow_incomplete_fold = False

    folds = _create_backtesting_folds(
                data                  = y,
                initial_train_size    = initial_train_size,
                test_size             = test_size,
                externally_fitted     = False,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = return_all_indexes,
                verbose               = True
            )
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of backtesting process\n"
        "----------------------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   14 -- 83  (n=70)\n"
        "    Validation: 89 -- 95  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected