# Unit test preprocess_levels
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import preprocess_levels


def test_output_preprocess_levels_when_col_names_can_transform_to_int():
    '''
    '''

    df = pd.DataFrame({'1': pd.Series(np.arange(3)), 
                       '2': pd.Series(np.arange(3))
                      })

    results = preprocess_levels(df)
    expected = {'1': 1, '2': 2}

    assert results == expected


def test_output_preprocess_levels_when_col_names_can_not_transform_to_int():
    '''
    '''

    df = pd.DataFrame({'y_1': pd.Series(np.arange(3)), 
                       'y_2': pd.Series(np.arange(3))
                      })

    results = preprocess_levels(df)
    expected = {'y_1': 0, 'y_2': 1}

    assert results == expected