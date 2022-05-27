# Unit test check_exog
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_exog


def test_check_exog_exception_when_exog_not_series_or_dataframe():
    '''
    Check exception is raised when y is not pandas Series
    '''
    with pytest.raises(Exception):
        check_exog(y=10)
    
    with pytest.raises(Exception):
        check_exog(y=[1, 2, 3])
    
    with pytest.raises(Exception):
        check_exog(y=np.arange(10).reshape(-1, 2))


def test_check_exog_exception_when_exog_has_missing_values():
    '''
    Check exception is raised when y has missing values
    '''
    with pytest.raises(Exception):
        check_exog(pd.Series([0, 1, None]))
    with pytest.raises(Exception):
        check_exog(pd.DataFrame([[1,2], [3, None], [5, 6]]))