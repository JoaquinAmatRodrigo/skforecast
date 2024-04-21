# Unit test series_long_to_dict
# ==============================================================================
import pandas as pd
import numpy as np
import pytest
from ..preprocessing import series_long_to_dict

# Fixtures
n_A = 10
n_B = 5
n_C = 15
series_id = np.repeat(["A", "B", "C"], [n_A, n_B, n_C])
index_A = pd.date_range("2020-01-01", periods=n_A, freq="D")
index_B = pd.date_range("2020-01-01", periods=n_B, freq="D")
index_C = pd.date_range("2020-01-01", periods=n_C, freq="D")
index = index_A.append(index_B).append(index_C)
values_A = np.arange(n_A)
values_B = np.arange(n_B)
values_C = np.arange(n_C)
values = np.concatenate([values_A, values_B, values_C])

series_long = pd.DataFrame(
    {
        "series_id": series_id,
        "datetime": index,
        "values": values,
    }
)


def test_check_output_series_long_to_dict():
    """
    Check output of series_long_to_dict.
    """

    expected = {
        "A": pd.Series(values_A, index=index_A, name="A"),
        "B": pd.Series(values_B, index=index_B, name="B"),
        "C": pd.Series(values_C, index=index_C, name="C"),
    }

    results = series_long_to_dict(
        data=series_long,
        series_id="series_id",
        index="datetime",
        values="values",
        freq="D",
    )

    for k in expected.keys():
        pd.testing.assert_series_equal(results[k], expected[k])


def test_TypeError_when_data_is_not_dataframe():
    """
    Raise TypeError if data is not a pandas DataFrame.
    """
    err_msg = "`data` must be a pandas DataFrame."
    with pytest.raises(TypeError, match=err_msg):
        series_long_to_dict(
            data='not_a_dataframe',
            series_id="series_id",
            index="datetime",
            values="values",
            freq="D",
        )


def test_ValueError_when_series_id_not_in_data():
    """
    Raise ValueError if series_id is not in data.
    """
    series_id = "series_id_not_in_data"
    err_msg = f"Column '{series_id}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        series_long_to_dict(
            data=series_long,
            series_id=series_id,
            index="datetime",
            values="values",
            freq="D",
        )


def test_ValueError_when_index_not_in_data():
    """
    Raise ValueError if index is not in data.
    """
    index = "series_id_not_in_data"
    err_msg = f"Column '{index}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        series_long_to_dict(
            data=series_long,
            series_id="series_id",
            index=index,
            values="values",
            freq="D",
        )


def test_ValueError_when_values_not_in_data():
    """
    Raise ValueError if values is not in data.
    """
    values = "values_not_in_data"
    err_msg = f"Column '{values}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        series_long_to_dict(
            data=series_long,
            series_id="series_id",
            index="datetime",
            values=values,
            freq="D",
        )