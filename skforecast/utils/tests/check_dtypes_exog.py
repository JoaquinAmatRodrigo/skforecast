# Unit test check_dtypes_exog
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_dtypes_exog
from skforecast.exceptions import ValueTypeWarning


def test_check_y_exception_when_exog_is_not_pandas_series():
    """
    Check exception is raised when exog is not a pandas Series.
    """
    err_msg = re.escape('`exog` must be `pd.Series` or `pd.DataFrame`.')
    with pytest.raises(TypeError, match = err_msg):
        check_dtypes_exog([1, 2, 3])


def test_check_y_warning_when_exog_is_Series_with_str_values():
    """
    Check warning is issued when exog is pandas Series with missing values.
    """
    warn_msg = re.escape(
        ('`exog` must contain only int, float or category dtypes. Most '
         'machine learning models do not allow other types of values values . '
         'Fitting the forecaster may fail.')
    )
    with pytest.warns(ValueTypeWarning, match = warn_msg):
        check_dtypes_exog(pd.Series(['A', 'B', 'C']))


def test_check_y_warning_when_exog_is_DataFrame_with_str_values():
    """
    Check warning is issued when exog is pandas DataFrame with missing values.
    """
    warn_msg = re.escape(
        ('`exog` must contain only int, float or category dtypes. Most '
         'machine learning models do not allow other types of values values . '
         'Fitting the forecaster may fail.')
    )
    with pytest.warns(ValueTypeWarning, match = warn_msg):
        check_dtypes_exog(pd.Series(['A', 'B', 'C']).to_frame())


def test_check_y_exception_when_exog_is_Series_with_no_int_categories():
    """
    Check exception is raised when exog is pandas Series with no integer
    categories.
    """
    err_msg = re.escape(
        ("If exog is of type category, it must contain only integer values. "
        "See skforecast docs for more info about how to include categorical "
        "features https://joaquinamatrodrigo.github.io/skforecast/"
        "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_dtypes_exog(pd.Series(['A', 'B', 'C'], dtype='category'))


def test_check_y_exception_when_exog_is_DataFrame_with_no_int_categories():
    """
    Check exception is raised when exog is pandas DataFrame with no integer
    categories.
    """
    err_msg = re.escape(
        ("Categorical columns in exog must contain only integer values. "
        "See skforecast docs for more info about how to include categorical "
        "features https://joaquinamatrodrigo.github.io/skforecast/"
        "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_dtypes_exog(pd.Series(['A', 'B', 'C'], dtype='category').to_frame())