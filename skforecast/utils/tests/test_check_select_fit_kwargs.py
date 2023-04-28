# Unit test check_select_fit_kwargs ForecasterAutoreg
# ==============================================================================
from skforecast.utils import check_select_fit_kwargs
from lightgbm import LGBMRegressor
import pytest


def test_check_select_fit_kwargs_ignores_sample_weight():
    """
    Test check_select_fit_kwargs method ignores sample_weight argument.
    """
    results = check_select_fit_kwargs(
                regressor = LGBMRegressor(),
                fit_kwargs = {'sample_weight': [1,2,3]}
              )

    expected = {}
    assert results == expected


def test_check_select_fit_kwargs_ignores_arguments_not_in_regressor_fit():
    """
    Test check_select_fit_kwargs method ignores arguments not in regressor fit method.
    """
    results = check_select_fit_kwargs(
                regressor = LGBMRegressor(),
                fit_kwargs = {'feature_name':'auto', 'no_valid_argument': 10}
               )

    expected = {'feature_name':'auto'}
    assert results == expected


def test_check_select_fit_kwargs_raise_error_when_fit_kwargs_is_not_dict():
    """
    Test check_select_fit_kwargs raises error when fit_kwargs is not a dictionary.
    """
    with pytest.raises(TypeError):
        check_select_fit_kwargs(
            regressor = LGBMRegressor(),
            fit_kwargs = 'auto'
        )


def test_check_select_fit_kwargs_warning_when_fit_kwargs_has_sample_weight():
    """
    Test check_select_fit_kwargs issues warning when fit_kwargs has sample_weight argument.
    """
    with pytest.warns(UserWarning):
        check_select_fit_kwargs(
            regressor = LGBMRegressor(),
            fit_kwargs = {'sample_weight': [1,2,3]}
        )


def test_check_select_fit_kwargs_warning_when_fit_kwargs_has_arguments_not_in_fit():
    """
    Test check_select_fit_kwargs issues warning when fit_kwargs has arguments not in
    regressor fit method.
    """
    with pytest.warns(UserWarning):
        check_select_fit_kwargs(
            regressor = LGBMRegressor(),
            fit_kwargs = {'no_valid_argument': 10}
        )