# Unit test initialize_transformer_series
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skforecast.utils import initialize_transformer_series
from skforecast.exceptions import IgnoredArgumentWarning


def test_initialize_transformer_series_when_transformer_series_is_None():
    """
    Test initialize_transformer_series when `transformer_series` is None.
    """
    series_col_names = ['col1', 'col2']
    transformer_series = None
    
    transformer_series_ = initialize_transformer_series(
                              series_col_names = series_col_names,
                              transformer_series = transformer_series
                          )

    assert transformer_series_ == {'col1': None, 'col2': None}


def test_initialize_transformer_series_when_transformer_series_is_StandardScaler():
    """
    Test initialize_transformer_series when `transformer_series` is a StandardScaler.
    """
    series_col_names = ['col1', 'col2']
    transformer_series = StandardScaler()
    
    transformer_series_ = initialize_transformer_series(
                              series_col_names = series_col_names,
                              transformer_series = transformer_series
                          )

    assert list(transformer_series_.keys()) == ['col1', 'col2']
    for col in series_col_names:
        assert isinstance(transformer_series_[col], StandardScaler)


def test_initialize_transformer_series_when_transformer_series_is_dict():
    """
    Test initialize_transformer_series when `transformer_series` is a dict.
    """
    series_col_names = ['col1', 'col2']
    transformer_series = {'col1': StandardScaler(), 
                          'col2': StandardScaler()}
    
    transformer_series_ = initialize_transformer_series(
                              series_col_names = series_col_names,
                              transformer_series = transformer_series
                          )

    assert list(transformer_series_.keys()) == ['col1', 'col2']
    for col in series_col_names:
        assert isinstance(transformer_series_[col], StandardScaler)


def test_initialize_transformer_series_IgnoredArgumentWarning_when_levels_of_transformer_series_not_equal_to_series_col_names():
    """
    Test IgnoredArgumentWarning is raised when `transformer_series` is a dict and its keys 
    are not the same as series_col_names.
    """
    series_col_names = ['col1', 'col2']
    transformer_series = {'col1': StandardScaler(), 
                          'col3': StandardScaler()}
    
    series_not_in_transformer_series = set(['col2'])
    
    warn_msg = re.escape(
        (f"{series_not_in_transformer_series} not present in `transformer_series`."
         f" No transformation is applied to these series.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        transformer_series_ = initialize_transformer_series(
                                  series_col_names = series_col_names,
                                  transformer_series = transformer_series
                              )

    assert list(transformer_series_.keys()) == ['col1', 'col2']
    assert isinstance(transformer_series_['col1'], StandardScaler)
    assert isinstance(transformer_series_['col2'], type(None))