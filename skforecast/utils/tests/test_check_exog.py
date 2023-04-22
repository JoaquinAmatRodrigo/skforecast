# Unit test check_exog
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_exog
from skforecast.exceptions import MissingValuesExogWarning


@pytest.mark.parametrize("exog", 
                         [10, [1, 2, 3], np.arange(10).reshape(-1, 1)], 
                         ids = lambda exog : f'exog: {exog}'
                        )
def test_check_exog_TypeError_when_exog_not_series_or_dataframe(exog):
    """
    Check TypeError is raised when exog is not a pandas Series or a
    pandas DataFrame
    """
    err_msg = re.escape("`exog` must be a pandas Series or DataFrame.")
    with pytest.raises(TypeError, match = err_msg):
        check_exog(exog=exog)


@pytest.mark.parametrize("exog", 
                         [pd.Series([0, 1, None]),
                          pd.DataFrame([[1, 2], [3, None], [5, 6]])], 
                         ids = lambda exog : f'exog type: {type(exog)}'
                        )
def test_check_exog_MissingValuesExogWarning_when_exog_has_missing_values_and_allow_nan_False(exog):
    """
    Check MissingValuesExogWarning is issued when y has missing values
    """
    warn_msg = re.escape(
                ("`exog` has missing values. Most machine learning models do "
                 "not allow missing values. Fitting the forecaster may fail.")
               )
    with pytest.warns(MissingValuesExogWarning, match = warn_msg):
        check_exog(exog, allow_nan=False)