# Unit test initialize_lags_grid
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.utils import initialize_lags_grid
from skforecast.exceptions import IgnoredArgumentWarning

def create_predictors(y):  # pragma: no cover
    """
    Create first 4 lags of a time series, used in ForecasterAutoregCustom.
    """

    lags = y[-1:-5:-1]

    return lags


def test_TypeError_initialize_lags__rid_when_not_list_dict_or_None():
    """
    Test TypeError is raised when lags_grid is not a list, dict or None.
    """
    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=2)
    lags_grid = 'not_valid_type'

    err_msg = re.escape(
        (f"`lags_grid` argument must be a list, dict or None. "
         f"Got {type(lags_grid)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags_grid(forecaster, lags_grid)


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregCustom(
                              regressor      = Ridge(random_state=123),
                              fun_predictors = create_predictors,
                              window_size    = 4),
                          ForecasterAutoregMultiSeriesCustom(
                              regressor      = Ridge(random_state=123),
                              fun_predictors = create_predictors,
                              window_size    = 4)], 
                         ids=lambda fn: f'forecaster_name: {type(fn).__name__}')
def test_IgnoredArgumentWarning_initialize_lags_grid_when_forecaster_has_custom_predictors(forecaster):
    """
    Test IgnoredArgumentWarning is raised when forecaster is `ForecasterAutoregCustom`
    or `ForecasterAutoregMultiSeriesCustom`.
    """
    lags_grid = [1, 2, 3]

    warn_msg = re.escape(
        (f"`lags_grid` ignored if forecaster is an instance of "
         f"`{type(forecaster).__name__}`."),
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        initialize_lags_grid(forecaster, lags_grid)


def test_initialize_lags_grid_when_lags_grid_is_a_list():
    """
    Test initialize_lags_grid when lags_grid is a list.
    """
    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=2)
    lags_grid = [1, [2, 4], range(3, 5), np.array([3, 7])]
    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)

    lags_grid_expected = {
        '1': 1, 
        '[2, 4]': [2, 4], 
        'range(3, 5)': range(3, 5), 
        '[3 7]': np.array([3, 7])
    }

    assert lags_label == 'values'
    assert lags_grid.keys() == lags_grid_expected.keys()
    for v, v_expected in zip(lags_grid.values(), lags_grid_expected.values()):
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, v_expected)
        elif isinstance(v, range):
            assert list(v) == list(v_expected)
        else:
            assert v == v_expected


@pytest.mark.parametrize("lags, lags_grid_expected",
                         [(3, {'[1, 2, 3]': [1, 2, 3]}), 
                          ([1, 2, 3], {'[1, 2, 3]': [1, 2, 3]}),
                          (range(1, 4), {'[1, 2, 3]': [1, 2, 3]}),
                          (np.array([1, 2, 3]), {'[1, 2, 3]': [1, 2, 3]})],
                         ids=lambda lags: f'lags, lags_grid_expected: {lags}')
def test_initialize_lags_grid_when_lags_grid_is_None(lags, lags_grid_expected):
    """
    Test initialize_lags_grid when lags_grid is None.
    """
    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=lags)
    lags_grid = None
    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)

    assert lags_label == 'values'
    assert lags_grid.keys() == lags_grid_expected.keys()
    for v, v_expected in zip(lags_grid.values(), lags_grid_expected.values()):
        assert v == v_expected


def test_initialize_lags_grid_when_lags_grid_is_a_dict():
    """
    Test initialize_lags_grid when lags_grid is a dict.
    """
    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=2)
    lags_grid = {'1': 1, '[2, 4]': [2, 4], 'range(3, 5)': range(3, 5), '[3 7]': np.array([3, 7])}
    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)

    lags_grid_expected = {
        '1': 1, 
        '[2, 4]': [2, 4], 
        'range(3, 5)': range(3, 5), 
        '[3 7]': np.array([3, 7])
    }
    
    assert lags_label == 'keys'
    assert lags_grid.keys() == lags_grid_expected.keys()
    for v, v_expected in zip(lags_grid.values(), lags_grid_expected.values()):
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, v_expected)
        elif isinstance(v, range):
            assert list(v) == list(v_expected)
        else:
            assert v == v_expected