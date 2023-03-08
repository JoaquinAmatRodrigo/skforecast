# Unit test set_out_sample_residuals ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

# Fixtures
series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                       'l2': pd.Series(np.arange(10))})

def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]

    return lags


@pytest.mark.parametrize("residuals", [[1, 2, 3], {'1': [1,2,3,4]}], 
                         ids=lambda residuals: f'residuals: {residuals}')
def test_set_out_sample_residuals_TypeError_when_residuals_is_not_a_dict_of_numpy_ndarray(residuals):
    """
    Test TypeError is raised when residuals is not a dict of numpy ndarrays.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                )
    err_msg = re.escape(
                f"`residuals` argument must be a dict of numpy ndarrays in the form "
                "`{level: residuals}`. " 
                f"Got {type(residuals)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                )
    residuals = {'l1': np.array([1, 2, 3, 4, 5]), 
                 'l2': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `set_out_sample_residuals()`.")
            )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_warning_when_residuals_not_for_all_levels():
    """
    Test Warning is raised when residuals does not contain a residue for all levels.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor      = LinearRegression(fit_intercept=True),
                    fun_predictors = create_predictors,
                    window_size    = 5
                 )
                
    forecaster.fit(series=series)
    residuals = {'l1': np.array([1, 2, 3])}

    err_msg = re.escape(
                f"""
                Only residuals of levels 
                {set(forecaster.series_col_names).intersection(set(residuals.keys()))} 
                are updated.
                """
            )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_False():
    """
    Test Warning is raised when forcaster has a transformer_series and transform=False.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor          = LinearRegression(),
                    fun_predictors     = create_predictors,
                    window_size        = 5,
                    transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    residuals = {'l1': np.array([1, 2, 3]), 'l2': np.array([1, 2, 3])}
    level = 'l1'

    err_msg = re.escape(
                    ("Argument `transform` is set to `False` but forecaster was "
                    f"trained using a transformer {forecaster.transformer_series_[level]} "
                    f"for level {level}. Ensure that the new residuals are "
                     "already transformed or set `transform=True`.")
                )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=False)


def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_True():
    """
    Test Warning is raised when forcaster has a transformer_y and transform=True.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor          = LinearRegression(),
                    fun_predictors     = create_predictors,
                    window_size        = 5,
                    transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    residuals = {'l1': np.array([1, 2, 3]), 'l2': np.array([1, 2, 3])}
    level = 'l1'

    err_msg = re.escape(
                ("Residuals will be transformed using the same transformer used "
                f"when training the forecaster for level {level} : "
                f"({forecaster.transformer_series_[level]}). Ensure that the new "
                 "residuals are on the same scale as the original time series.")
            )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=True)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor         = LinearRegression(),
                    fun_predictors    = create_predictors,
                    window_size       = 5,
                )
    forecaster.fit(series=series)
    residuals = {'l1': np.arange(10), 'l2': np.arange(10)}
    new_residuals = {'l1': np.arange(20), 'l2': np.arange(20)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=new_residuals, append=False)
    expected = {'l1': np.arange(20), 'l2': np.arange(20)}
    results = forecaster.out_sample_residuals

    assert expected.keys() == results.keys()
    assert all(all(expected[k] == results[k]) for k in expected.keys())


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                )
    forecaster.fit(series=series)
    residuals = {'l1': np.arange(10), 'l2': np.arange(10)}
    
    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=residuals, append=True)
    expected = {'l1': np.append(np.arange(10), np.arange(10)),
                'l2': np.append(np.arange(10), np.arange(10))}
    results = forecaster.out_sample_residuals

    assert expected.keys() == results.keys()
    assert all(all(expected[k] == results[k]) for k in expected.keys())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    """
    Test len residuals stored when its length is greater than 1000.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                )
    forecaster.fit(series=series)

    residuals = {'l1': np.arange(2000), 'l2': np.arange(2000)}
    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals

    assert list(results.keys()) == ['l1', 'l2']
    assert all(len(value)==1000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_length_is_more_than_1000_and_append():
    """
    Test residuals stored when new residuals length is more than 1000 and append is True.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                )
    forecaster.fit(series=series)
    residuals = {'l1': np.arange(10), 'l2': np.arange(10)}
    residuals_2 = {'l1': np.arange(1200), 'l2': np.arange(1200)}

    forecaster.set_out_sample_residuals(residuals = residuals)
    forecaster.set_out_sample_residuals(residuals = residuals_2,
                                        append    = True)

    expected = {}
    for key, value in residuals_2.items():
        rng = np.random.default_rng(seed=123)
        expected_2 = rng.choice(a=value, size=1000, replace=False)
        expected[key] = np.hstack([np.arange(10), expected_2])[:1000]

    results = forecaster.out_sample_residuals

    assert expected.keys() == results.keys()
    assert all(all(expected[k] == results[k]) for k in expected.keys())
    assert all(len(value)==1000 for value in results.values())