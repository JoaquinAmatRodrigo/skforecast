# Unit test check_exog
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_exog


@pytest.mark.parametrize("exog", 
                         [10, [1, 2, 3], np.arange(10).reshape(-1, 1)], 
                         ids = lambda exog : f'exog: {exog}'
                        )
def test_check_exog_exception_when_exog_not_series_or_dataframe(exog):
    """
    Check exception is raised when exog is not a pandas Series or a
    pandas DataFrame
    """
    err_msg = re.escape('`exog` must be `pd.Series` or `pd.DataFrame`.')
    with pytest.raises(TypeError, match = err_msg):
        check_exog(exog=exog)


@pytest.mark.parametrize("exog", 
                         [pd.Series([0, 1, None]),
                          pd.DataFrame([[1,2], [3, None], [5, 6]])], 
                         ids = lambda exog : f'exog: {exog}'
                        )
def test_check_exog_exception_when_exog_has_missing_values(exog):
    """
    Check exception is raised when y has missing values
    """
    err_msg = re.escape('`exog` has missing values.')
    with pytest.raises(ValueError, match = err_msg):
        check_exog(exog)