# Unit test backtesting_forecaster
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection import backtesting_forecaster

# Fixtures
from .fixtures_model_selection import y
from .fixtures_model_selection import exog


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    forecaters_allowed = [
        'ForecasterAutoreg', 
        'ForecasterAutoregCustom', 
        'ForecasterAutoregDirect',
        'ForecasterEquivalentDate'
    ]

    err_msg = re.escape(
        f"`forecaster` must be of type {forecaters_allowed}, for all other types of "
        f" forecasters use the functions available in the other `model_selection` "
        f"modules.")
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_ValueError_when_ForecasterAutoregDirect_not_enough_steps():
    """
    Test ValueError is raised in backtesting_forecaster when there is not enough 
    steps to predict steps+gap in a ForecasterAutoregDirect.
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = Ridge(random_state=123),
                    lags      = 5,
                    steps     = 5
                 )
    
    gap = 5
    steps = 1
    
    err_msg = re.escape(
            ("When using a ForecasterAutoregDirect, the combination of steps "
             f"+ gap ({steps+gap}) cannot be greater than the `steps` parameter "
             f"declared when the forecaster is initialized ({forecaster.steps}).")
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = steps,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = gap,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )