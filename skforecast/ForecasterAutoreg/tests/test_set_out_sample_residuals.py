# Unit test set_out_sample_residuals ForecasterAutoreg
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures_ForecasterAutoreg import y

if pd.__version__ < '2.2.0':
    freq = "H"
else:
    freq = "h"


def test_set_out_sample_residuals_TypeError_when_residuals_is_not_numpy_array_or_pandas_series():
    """
    Test TypeError is raised when residuals argument is not numpy ndarray.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    residuals = [1, 2, 3]

    err_msg = re.escape(
        (f"`residuals` argument must be `numpy ndarray` or `pandas Series`, "
         f"but found {type(residuals)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_TypeError_when_y_pred_is_not_numpy_array_or_pandas_series():
    """
    Test TypeError is raised when y_pred argument is not numpy ndarray.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    residuals = pd.Series([1, 2, 3])
    y_pred = [1, 2, 3]

    err_msg = re.escape(
        f"`y_pred` argument must be `numpy ndarray`, `pandas Series` or `None`, "
        f"but found {type(y_pred)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_residuals_and_y_pred_have_different_lenght():
    """
    Test ValueError is raised when residuals and y_pred have different lenght.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    residuals = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 2, 3, 4])

    err_msg = re.escape(
        f"`residuals` and `y_pred` must have the same length, but found "
        f"{len(residuals)} and {len(y_pred)}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_residuals_and_y_pred_have_different_index():
    """
    Test ValueError is raised when residuals and y_pred have different index.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    residuals = pd.Series([1, 2, 3], index=[1, 2, 3])
    y_pred = pd.Series([1, 2, 3], index=[1, 2, 4])

    err_msg = re.escape(
        f"`residuals` and `y_pred` must have the same index, but found "
        f"{residuals.index} and {y_pred.index}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred)


def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_False():
    """
    Test Warning is raised when forecaster has a transformer_y and transform=False.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, transformer_y=StandardScaler())
    residuals = np.arange(10)

    warn_msg = re.escape(
        (f"Argument `transform` is set to `False` but forecaster was trained "
         f"using a transformer {forecaster.transformer_y}. Ensure that the new residuals "
         f"are already transformed or set `transform=True`.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=False)


def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_True():
    """
    Test Warning is raised when forecaster has a transformer_y and transform=True.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, transformer_y=StandardScaler())
    forecaster.fit(y=pd.Series(np.arange(10)))
    residuals = np.arange(10)

    warn_msg = re.escape(
        (f"Residuals will be transformed using the same transformer used "
         f"when training the forecaster ({forecaster.transformer_y}). Ensure that the "
         f"new residuals are on the same scale as the original time series.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=True)


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    """
    Test len residuals stored when its length is greater than 1000.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))

    assert len(forecaster.out_sample_residuals_) == 1000


def test_same_out_sample_residuals_stored_when_residuals_length_is_greater_than_1000():
    """
    Test same out sample residuals are stored when its length is greater than 1000
    executing function two times.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))
    out_sample_residuals_1 = forecaster.out_sample_residuals_
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))
    out_sample_residuals_2 = forecaster.out_sample_residuals_

    np.testing.assert_almost_equal(out_sample_residuals_1, out_sample_residuals_2)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(20))
    forecaster.set_out_sample_residuals(residuals=np.arange(10), append=False)
    expected = np.arange(10)
    results = forecaster.out_sample_residuals_

    np.testing.assert_almost_equal(results, expected)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(10))
    forecaster.set_out_sample_residuals(residuals=np.arange(10), append=True)
    expected = np.hstack([np.arange(10), np.arange(10)])
    results = forecaster.out_sample_residuals_

    np.testing.assert_almost_equal(results, expected)


def test_set_out_sample_residuals_when_residuals_length_is_more_than_1000_and_append():
    """
    Test residuals stored when new residuals length is more than 1000 and append is True.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(10))
    forecaster.set_out_sample_residuals(residuals=np.arange(1000), append=True)
    expected = np.hstack([np.arange(10), np.arange(1200)])[:1000]
    results = forecaster.out_sample_residuals_

    np.testing.assert_almost_equal(results, expected)


def test_set_out_sample_residuals_when_transform_is_True():
    """
    Test residuals stored when using a StandardScaler() as transformer.
    """

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, transformer_y=StandardScaler())
    y = pd.Series(
            np.array([
                12.5, 10.3,  9.9, 10.4,  9.9,  8.5, 10.6, 11.4, 10.,  9.5, 10.1,
                11.5, 11.4, 11.3, 10.5,  9.6, 10.4, 11.7,  8.7, 10.6])
        )
    forecaster.fit(y=y)
    new_residuals = np.random.normal(size=100)
    new_residuals_transformed = forecaster.transformer_y.transform(new_residuals.reshape(-1, 1))
    new_residuals_transformed = new_residuals_transformed.ravel()
    forecaster.set_out_sample_residuals(residuals=new_residuals, transform=True)

    np.testing.assert_array_equal(new_residuals_transformed, forecaster.out_sample_residuals_)


def test_same_out_sample_residuals_by_bin_stored_when_y_pred_is_provided():
    """
    Test out sample residuals by bin are stored when y_pred is provided.
    """
    # The in sample residuals and predictions are passed to the method
    # set_out_sample_residuals so out_sample_residuals_by_bin and
    # out_sample_residuals_by_bin must be equal
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags = 5,
                     binner_kwargs = {'n_bins': 3}
                 )
    forecaster.fit(y)
    X_train, y_train = forecaster.create_train_X_y(y)
    forecaster.regressor.fit(X_train, y_train)
    predictions = forecaster.regressor.predict(X_train)
    residuals = y_train - predictions

    forecaster.set_out_sample_residuals(
        residuals=residuals,
        y_pred=predictions
    )

    for k in forecaster.out_sample_residuals_by_bin_.keys():
        np.testing.assert_almost_equal(
            forecaster.in_sample_residuals_by_bin_[k],
            forecaster.out_sample_residuals_by_bin_[k]
        )


def test_set_out_sample_residuals_stores_maximum_200_residuals_per_bin():
    """
    Test that set_out_sample_residuals stores a maximum of 200 residuals per bin.
    """
    y = pd.Series(
        data=np.random.normal(loc=10, scale=1, size=1000),
        index=pd.date_range(start="01-01-2000", periods=1000, freq="h"),
    )
    residuals = np.random.normal(loc=0, scale=1, size=1000)
    y_pred = y
    forecaster = ForecasterAutoreg(
        regressor=LinearRegression(), lags=5, binner_kwargs={"n_bins": 2}
    )
    forecaster.fit(y)
    forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred)

    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 200

    np.testing.assert_array_almost_equal(
        forecaster.out_sample_residuals_,
        np.concatenate(list(forecaster.out_sample_residuals_by_bin_.values())),
    )


def test_set_out_sample_residuals_append_new_residuals_per_bin():
    """
    Test that set_out_sample_residuals append residuals per bin and stores
    maximum 200 residuals per bin.
    """
    rng = np.random.default_rng(12345)
    y = pd.Series(
        data=rng.normal(loc=10, scale=1, size=100),
        index=pd.date_range(start="01-01-2000", periods=100, freq="h"),
    )
    residuals = rng.normal(loc=0, scale=1, size=100)
    y_pred = y

    forecaster = ForecasterAutoreg(
        regressor=LinearRegression(), lags=5, binner_kwargs={"n_bins": 2}
    )
    forecaster.fit(y)   
    forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred, append=True)
    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 50

    forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred, append=True)
    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 100

    forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred, append=True)
    forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred, append=True)
    forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred, append=True)
    forecaster.set_out_sample_residuals(residuals=residuals, y_pred=y_pred, append=True)
    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 200


def test_set_out_sample_residuals_when_there_are_no_residuals_for_some_bins():
    """
    Test that set_out_sample_residuals works when there are no residuals for some bins.
    """
    rng = np.random.default_rng(12345)
    y = pd.Series(
        data=rng.normal(loc=10, scale=1, size=100),
        index=pd.date_range(start="01-01-2000", periods=100, freq="h"),
    )
    y_pred = y.loc[y > 10]
    residuals = rng.normal(loc=0, scale=1, size=len(y_pred))

    forecaster = ForecasterAutoreg(
        regressor=LinearRegression(), lags=5, binner_kwargs={"n_bins": 3}
    )
    forecaster.fit(y)

    warn_msg = re.escape(
        (
            f"The following bins have no out of sample residuals: [0]. "
            f"No predicted values fall in the interval "
            f"[{forecaster.binner_intervals_[0]}]. "
            f"Empty bins will be filled with a random sample of residuals from "
            f"the other bins."
        )
    )
    with pytest.warns(UserWarning, match=warn_msg):
        forecaster.set_out_sample_residuals(
            residuals=residuals, y_pred=y_pred, append=True
        )

    assert len(forecaster.out_sample_residuals_by_bin_[0]) == 200
