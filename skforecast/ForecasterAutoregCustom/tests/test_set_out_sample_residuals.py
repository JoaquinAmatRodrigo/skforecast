# Unit test set_out_sample_residuals ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """

    lags = y[-1:-6:-1]

    return lags  


def test_set_out_sample_residuals_TypeError_when_residuals_is_not_numpy_ndarray():
    """
    Test TypeError is raised when residuals argument is not numpy ndarray.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    residuals=pd.Series([1, 2, 3])

    err_msg = re.escape(
                f"`residuals` argument must be `numpy ndarray`. Got {type(residuals)}."
            )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_False():
    """
    Test Warning is raised when forecaster has a transformer_y and transform=False.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5,
                     transformer_y  = StandardScaler()
                 )
    residuals = np.arange(10)

    err_msg = re.escape(
                (f"Argument `transform` is set to `False` but forecaster was trained "
                 f"using a transformer {forecaster.transformer_y}. Ensure that the new residuals "
                 f"are already transformed or set `transform=True`.")
            )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=False)


def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_True():
    """
    Test Warning is raised when forecaster has a transformer_y and transform=True.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5,
                     transformer_y  = StandardScaler()
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))
    residuals = np.arange(10)

    err_msg = re.escape(
                (f"Residuals will be transformed using the same transformer used "
                 f"when training the forecaster ({forecaster.transformer_y}). Ensure that the "
                 f"new residuals are on the same scale as the original time series.")
            )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=True)


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    """
    Test len residuals stored when its length is greater than 1000.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))

    assert len(forecaster.out_sample_residuals) == 1000


def test_same_out_sample_residuals_stored_when_residuals_length_is_greater_than_1000():
    """
    Test same out sample residuals are stored when its length is greater than 1000
    executing function two times.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))
    out_sample_residuals_1 = forecaster.out_sample_residuals
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))
    out_sample_residuals_2 = forecaster.out_sample_residuals

    assert (out_sample_residuals_1 == out_sample_residuals_2).all()


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    forecaster.set_out_sample_residuals(residuals=np.arange(20))
    forecaster.set_out_sample_residuals(residuals=np.arange(10), append=False)
    expected = np.arange(10)
    results = forecaster.out_sample_residuals

    assert (results == expected).all()


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    forecaster.set_out_sample_residuals(residuals=np.arange(10))
    forecaster.set_out_sample_residuals(residuals=np.arange(10), append=True)
    expected = np.hstack([np.arange(10), np.arange(10)])
    results = forecaster.out_sample_residuals

    assert (results == expected).all()


def test_set_out_sample_residuals_when_residuals_length_is_more_than_1000_and_append():
    """
    Test residuals stored when new residuals length is more than 1000 and append is True.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    forecaster.set_out_sample_residuals(residuals=np.arange(10))
    forecaster.set_out_sample_residuals(residuals=np.arange(1000), append=True)
    expected = np.hstack([np.arange(10), np.arange(1200)])[:1000]
    results = forecaster.out_sample_residuals

    assert (results == expected).all()


def test_set_out_sample_residuals_when_transform_is_True():
    """
    Test residuals stored when using a StandardScaler() as transformer.
    """

    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                    transformer_y=StandardScaler()
                    
                )
    y = pd.Series(
            np.array([
                12.5, 10.3,  9.9, 10.4,  9.9,  8.5, 10.6, 11.4, 10. ,  9.5, 10.1,
                11.5, 11.4, 11.3, 10.5,  9.6, 10.4, 11.7,  8.7, 10.6])
        )
    forecaster.fit(y=y)
    new_residuals = np.random.normal(size=100)
    new_residuals_transformed = forecaster.transformer_y.transform(new_residuals.reshape(-1, 1))
    new_residuals_transformed = new_residuals_transformed.flatten()
    forecaster.set_out_sample_residuals(residuals=new_residuals, transform=True)

    np.testing.assert_array_equal(new_residuals_transformed, forecaster.out_sample_residuals)