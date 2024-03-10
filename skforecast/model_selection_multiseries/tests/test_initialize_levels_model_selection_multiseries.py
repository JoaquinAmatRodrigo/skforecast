# Unit test _initialize_levels_model_selection_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries.model_selection_multiseries import _initialize_levels_model_selection_multiseries
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from .fixtures_model_selection_multiseries import series

def create_predictors(y): # pragma: no cover
    """
    Create first 4 lags of a time series, used in ForecasterAutoregCustom.
    """

    lags = y[-1:-5:-1]

    return lags


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(
                              regressor = Ridge(random_state=123),
                              lags      = 4),
                          ForecasterAutoregMultiSeriesCustom(
                              regressor      = Ridge(random_state=123),
                              fun_predictors = create_predictors,
                              window_size    = 4)], 
                         ids=lambda fn: f'forecaster_name: {type(fn).__name__}')
def test_initialize_levels_model_selection_multiseries_TypeError_when_levels_not_list_str_None(forecaster):
    """
    Test TypeError is raised in _initialize_levels_model_selection_multiseries when 
    `levels` is not a `list`, `str` or `None`.
    """
    levels = 5

    err_msg = re.escape(
        ("`levels` must be a `list` of column names, a `str` of a column "
         "name or `None` when using a forecaster of type "
         "['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiSeriesCustom', "
         "'ForecasterRnn']. If the forecaster is of type "
         "`ForecasterAutoregMultiVariate`, this argument is ignored.")
    )
    with pytest.raises(TypeError, match = err_msg):
        _initialize_levels_model_selection_multiseries(
            forecaster = forecaster, 
            series     = series,
            levels     = levels
        )


def test_initialize_levels_model_selection_multiseries_IgnoredArgumentWarning_forecaster_multivariate_and_levels():
    """
    Test IgnoredArgumentWarning is raised when levels is not forecaster.level or 
    None in ForecasterAutoregMultiVariate.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    
    levels = 'not_l1_or_None'
    
    warn_msg = re.escape(
        ("`levels` argument have no use when the forecaster is of type "
         "`ForecasterAutoregMultiVariate`. The level of this forecaster is "
         "'l1', to predict another level, change the `level` "
         "argument when initializing the forecaster.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        levels = _initialize_levels_model_selection_multiseries(
                     forecaster = forecaster, 
                     series     = series,
                     levels     = levels
                 )
        
    assert levels == ['l1']


@pytest.mark.parametrize("levels, levels_expected",
                         [(None, ['l1', 'l2']), 
                          ('l1', ['l1']),
                          (['l1', 'l2'], ['l1', 'l2'])],
                         ids=lambda lags: f'lags, lags_grid_expected: {lags}')
def test_initialize_levels_model_selection_multiseries_when_for_all_inputs(levels, levels_expected):
    """
    Test initialize_levels_model_selection_multiseries when levels is None, 
    str or list.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster, 
                 series     = series,
                 levels     = levels
             )
    
    assert levels == levels_expected