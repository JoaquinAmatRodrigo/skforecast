# Unit test check_y
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_y


def test_check_y_exception_when_y_not_series():
    """
    Check exception is raised when y is not pandas Series
    """
    with pytest.raises(Exception):
        check_y(y=10)
    
    with pytest.raises(Exception):
        check_y(y=[1, 2, 3])
    
    with pytest.raises(Exception):
        check_y(y=np.arange(10).reshape(-1, 1))


def test_check_y_exception_when_y_has_missing_values():
    """
    Check exception is raised when y has missing values
    """
    with pytest.raises(Exception):
        check_y(pd.Series([0, 1, None]))