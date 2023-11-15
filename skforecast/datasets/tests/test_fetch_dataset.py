# Unit test fetch_dataset
# ==============================================================================
import re
import pytest
import pandas as pd
from skforecast.datasets import fetch_dataset


def test_fetch_dataset():
    """
    Test function `fetch_dataset`.
    """
    
    # Test fetching the 'h2o' dataset
    df = fetch_dataset('h2o', version='latest', raw=False, verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (204, 1)
    assert df.index.freq == 'MS'
    assert df.index[0].strftime('%Y-%m-%d') == '1991-07-01'
    assert df.index[-1].strftime('%Y-%m-%d') == '2008-06-01'

    # Test fetching the 'items_sales' dataset
    df = fetch_dataset('items_sales', version='latest', raw=False, verbose=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1097, 3)
    assert df.index.freq == 'D'
    assert df.index[0] == pd.Timestamp('2012-01-01')
    assert df.index[-1] == pd.Timestamp('2015-01-01')

    # Test fetching a non-existent dataset
    with pytest.raises(ValueError):
        fetch_dataset('non_existent_dataset', version='latest', raw=False, verbose=False)

    # Test fetching a dataset with a non-existent version
    with pytest.raises(ValueError):
        fetch_dataset('h2o', version='non_existent_version', raw=False, verbose=False)