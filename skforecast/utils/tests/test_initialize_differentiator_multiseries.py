# Unit test initialize_differentiator_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.utils import initialize_differentiator_multiseries


def test_initialize_differentiator_multiseries_when_differentiator_is_None():
    """
    Test initialize_differentiator_multiseries when `differentiator` is None.
    """
    series_col_names = ['col1', 'col2']
    
    differentiator_ = initialize_differentiator_multiseries(
                          series_col_names = series_col_names,
                          fitted           = False,
                          differentiator   = None
                      )

    assert differentiator_ == {'col1': None, 'col2': None}


def test_initialize_differentiator_multiseries_when_differentiator_is_TimeSeriesDifferentiator():
    """
    Test initialize_differentiator_multiseries when `differentiator` is a 
    skforecast TimeSeriesDifferentiator.
    """
    series_col_names = ['col1', 'col2']
    differentiator = TimeSeriesDifferentiator()
    
    differentiator_ = initialize_differentiator_multiseries(
                          series_col_names = series_col_names,
                          fitted           = False,
                          differentiator   = differentiator
                      )

    assert list(differentiator_.keys()) == ['col1', 'col2']
    for col in series_col_names:
        assert isinstance(differentiator_[col], TimeSeriesDifferentiator)