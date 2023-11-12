# Unit test predict ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterBaseline import ForecasterEquivalentDate

# Fixtures
from .fixtures_ForecasterEquivalentDate import y


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)

    err_msg = re.escape(
                ('This Forecaster instance is not fitted yet. Call `fit` with '
                 'appropriate arguments before using predict.')
              )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=3)


def test_predict_5_steps_with_int_offset_offset_1_n_offsets_1():
    """
    Test predict method with int offset, offset=1 and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = 1,
                     n_offsets = 1
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)

    expected = pd.Series(
        data  = np.array([0.61289453, 0.61289453, 0.61289453, 0.61289453, 0.61289453]),
        index = pd.date_range(start='2000-02-20', periods=5, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_5_steps_with_int_offset_7_n_offsets_1():
    """
    Test predict method with int offset, offset=7 and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = 7,
                     n_offsets = 1
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)

    expected = pd.Series(
        data  = np.array([0.41482621, 0.86630916, 0.25045537, 0.48303426, 0.98555979]),
        index = pd.date_range(start='2000-02-20', periods=5, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_7_steps_with_int_offset_7_n_offsets_2():
    """
    Test predict method with int offset, offset=7 and n_offsets=2.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = 7,
                     n_offsets = 2,
                     agg_func  = np.mean
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=7)

    expected = pd.Series(
        data  = np.array([0.42058876, 0.87984916, 0.5973077 , 0.49243547, 0.80475637,
                          0.31755176, 0.46509001]),
        index = pd.date_range(start='2000-02-20', periods=7, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_7_steps_with_int_offset_5_n_offsets_1_using_last_window():
    """
    Test predict method with int offset, offset=1 and n_offsets=1, using last_window.
    """
    last_window = pd.Series(
        data  = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759]),
        index = pd.date_range(start='2000-08-10', periods=5, freq='D'),
        name  = 'y'
    )

    forecaster = ForecasterEquivalentDate(
                     offset    = 5,
                     n_offsets = 1
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=7, last_window=last_window)

    expected = pd.Series(
        data  = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                          0.73799541, 0.18249173]),
        index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_7_steps_with_int_offset_5_n_offsets_2_using_last_window():
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
    
    expected = pd.Series(
        data  = np.array([0.49422539, 0.332763, 0.58050578, 0.52551824, 0.57236106,
                          0.49422539, 0.332763]),
        index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_series_equal(predictions, expected)


# def test_predict_ValueError_when_all_equivalent_values_are_missing():
#     """
#     Test ValueError when equivalent values are missing because the offset is
#     too large.
#     """    
#     forecaster = ForecasterEquivalentDate(
#                      offset    = pd.offsets.Day(500),
#                      n_offsets = 1
#                  )
#     forecaster.fit(y=y)

#     err_msg = re.escape(
#                 (f"All equivalent values are missing. This is caused by using an offset "
#                  f"({forecaster.offset}) larger than the data available. Try decrease the "
#                  f"size of the offset ({forecaster.offset}), the number of offsets "
#                  f"({forecaster.n_offsets}) or increase the size of `last_window`. In "
#                  f"backtesing this error may be caused by using an too small "
#                  f"`initial_train_size`.")
#               )
#     with pytest.raises(ValueError, match = err_msg):
#         forecaster.predict(steps=3)


# def test_predict_ValueError_when_any_equivalent_values_are_missing():
#     """
#     Test ValueError when equivalent values are missing because the offset is
#     too large.
#     """    
#     forecaster = ForecasterEquivalentDate(
#                      offset    = pd.offsets.Day(26),
#                      n_offsets = 2
#                  )
#     forecaster.fit(y=y)

#     warn_msg = re.escape(
#                 (f"Steps: ['2000-12-31'] "
#                  f"are calculated with less than {forecaster.n_offsets} offsets. To avoid "
#                  f"this increase `last_window` size or decrease the number "
#                  f"of offsets. The current configuration needs a total offset "
#                  f"of {forecaster.offset * forecaster.n_offsets}.")
#               )
#     with pytest.warns(UserWarning, match = warn_msg):
#         forecaster.predict(steps=3)


def test_predict_7_steps_with_offset_Day_5_n_offsets_1_using_last_window():
    """
    Test predict method with offset pd.offsets.Day() and 
    n_offsets=1, using last_window.
    """
    last_window = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759]),
        index = pd.date_range(start='2000-08-10', periods=5, freq='D'),
        name = 'y'
    )

    forecaster = ForecasterEquivalentDate(
                     offset    = pd.offsets.Day(5),
                     n_offsets = 1
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=7, last_window=last_window)

    expected = pd.Series(
        data  = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                          0.73799541, 0.18249173]),
        index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_7_steps_with_offset_Day_5_n_offsets_2_using_last_window():
    """
    Test predict method with offset pd.offsets.Day(5) and n_offsets=2,
    using last_window.
    """
    last_window = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
        index = pd.date_range(start='2000-08-05', periods=10, freq='D'),
        name = 'y'
    )
    forecaster = ForecasterEquivalentDate(
                    offset    = pd.offsets.Day(5),
                    n_offsets = 2
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=7, last_window=last_window)

    expected = pd.Series(
        data  = np.array([0.49422539, 0.332763, 0.58050578, 0.52551824, 0.57236106,
                          0.49422539, 0.332763]),
        index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_series_equal(predictions, expected)