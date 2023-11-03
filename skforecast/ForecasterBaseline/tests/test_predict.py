# Unit test predict ForecasterEquivalentDate
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.baseline import ForecasterEquivalentDate

# Fixtures
from .fixtures_ForecasterAutoregCustom import y

import pytest
from skforecast.baseline import ForecasterEquivalentDate


def test_predict_exception_with_unfitted_forecaster():
    """
    Test exception when predict method is called before fitting the forecaster.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    with pytest.raises(
        Exception,
        match="This Forecaster instance is not fitted yet. Call `fit` with appropriate arguments before using predict."
        ):
        forecaster.predict(steps=3)


def test_predict_with_int_offset_offset_1_n_offsets_1():
    """
    Test predict method with int offset, offset=1 and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(
        offset=1,
        n_offsets=1
    )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected_predictions = pd.Series(
        data = np.array([0.61289453, 0.61289453, 0.61289453, 0.61289453, 0.61289453]),
        index = pd.date_range(start='2000-02-20', periods=5, freq='D'),
        name = 'pred'
    )
    pd.testing.assert_series_equal(predictions, expected_predictions)


def test_predict_with_int_offset_offset_7_n_offsets_1():
    """
    Test predict method with int offset, offset=7 and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(
        offset=7,
        n_offsets=1
    )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected_predictions = pd.Series(
        data = np.array([0.41482621, 0.86630916, 0.25045537, 0.48303426, 0.98555979]),
        index = pd.date_range(start='2000-02-20', periods=5, freq='D'),
        name = 'pred'
    )
    pd.testing.assert_series_equal(predictions, expected_predictions)


def test_predict_with_int_offset_offset_7_n_offsets_1():
    """
    Test predict method with int offset, offset=7 and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(
        offset=7,
        n_offsets=2,
        agg_func=np.mean
    )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=7)
    expected_predictions = pd.Series(
        data = np.array([0.42058876, 0.87984916, 0.5973077 , 0.49243547, 0.80475637,
                         0.31755176, 0.46509001]),
        index = pd.date_range(start='2000-02-20', periods=7, freq='D'),
        name = 'pred'
    )
    pd.testing.assert_series_equal(predictions, expected_predictions)


def test_predict_with_int_offset_offset_7_n_offsets_1_using_last_window():
    """
    Test predict method with int offset, offset=1 and n_offsets=1, using last_window.
    """
    last_window = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759]),
        index = pd.date_range(start='2000-08-10', periods=5, freq='D'),
        name = 'y'
    )
    forecaster = ForecasterEquivalentDate(
                    offset    = 5,
                    n_offsets = 1
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=7, last_window=last_window)
    expected_predictions = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.73799541, 0.18249173]),
        index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
        name = 'pred'
    )
    pd.testing.assert_series_equal(predictions, expected_predictions)


def test_predict_with_int_offset_offset_7_n_offsets_2_using_last_window():
    """
    Test predict method with int offset, offset=1 and n_offsets=1, using last_window.
    """
    last_window = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
        index = pd.date_range(start='2000-08-05', periods=10, freq='D'),
        name = 'y'
    )
    forecaster = ForecasterEquivalentDate(
                    offset    = 5,
                    n_offsets = 2
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=7, last_window=last_window)
    expected_predictions = pd.Series(
        data = np.array([0.49422539, 0.332763, 0.58050578, 0.52551824, 0.57236106,
                         0.49422539, 0.332763]),
        index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
        name = 'pred'
    )
    pd.testing.assert_series_equal(predictions, expected_predictions)


