# Unit test set_fit_kwargs ForecasterAutoreg
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from lightgbm import LGBMRegressor
import pytest


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterAutoreg(
                    regressor = LGBMRegressor(),
                    lags = 3,
                    fit_kwargs = {'categorical_feature': 'auto'}
                 )
    new_fit_kwargs = {'categorical_feature': ['exog']}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    expected = {'categorical_feature': ['exog']}
    results = forecaster.fit_kwargs
    assert results == expected


def test_set_fit_kwargs_ignores_sample_weight():
    """
    Test set_fit_kwargs method ignores sample_weight argument.
    """
    forecaster = ForecasterAutoreg(
                    regressor = LGBMRegressor(),
                    lags = 3,
                    fit_kwargs = {'sample_weight': [1,2,3]}
                 )

    expected = {}
    results = forecaster.fit_kwargs
    assert results == expected


def test_set_fit_kwargs_ignores_arguments_not_in_regressor_fit():
    """
    Test set_fit_kwargs method ignores arguments not in regressor fit method.
    """
    forecaster = ForecasterAutoreg(
                    regressor = LGBMRegressor(),
                    lags = 3,
                    fit_kwargs = {'feature_name':'auto', 'no_valid_argument': 10}
                 )

    expected = {'feature_name':'auto'}
    results = forecaster.fit_kwargs
    assert results == expected


def test_set_fit_kwargs_raise_error_when_fit_kwargs_is_not_dict():
    """
    Test set_fit_kwargs raises error when fit_kwargs is not a dictionary.
    """
    with pytest.raises(TypeError):
        ForecasterAutoreg(
            regressor = LGBMRegressor(),
            lags = 3,
            fit_kwargs = 'auto'
        )


def test_set_fit_kwargs_warning_when_fit_kwargs_has_sample_weight():
    """
    Test set_fit_kwargs issues warning when fit_kwargs has sample_weight argument.
    """
    with pytest.warns(UserWarning):
        ForecasterAutoreg(
            regressor = LGBMRegressor(),
            lags = 3,
            fit_kwargs = {'sample_weight': [1,2,3]}
        )


def test_set_fit_kwargs_warning_when_fit_kwargs_has_arguments_not_in_fit():
    """
    Test set_fit_kwargs issues warning when fit_kwargs has arguments not in
    regressor fit method.
    """
    with pytest.warns(UserWarning):
        ForecasterAutoreg(
            regressor = LGBMRegressor(),
            lags = 3,
            fit_kwargs = {'no_valid_argument': 10}
        )