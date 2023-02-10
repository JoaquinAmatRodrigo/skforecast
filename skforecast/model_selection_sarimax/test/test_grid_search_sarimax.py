# Unit test grid_search_sarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
from pmdarima.arima import ARIMA

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import y_datetime


def test_output_grid_search_sarimax_sarimax_with_mocked():
    """
    Test output of grid_search_sarimax in ForecasterSarimax with mocked
    (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))

    param_grid = [{'order' : [(1,1,1), (2,2,2)],
                   'method': ['powell', 'lbfgs']}]

    results = grid_search_sarimax(
                  forecaster         = forecaster,
                  y                  = y_datetime,
                  param_grid         = param_grid,
                  steps              = 3,
                  refit              = False,
                  metric             = 'mean_absolute_error',
                  initial_train_size = len(y_datetime)-12,
                  fixed_train_size   = False,
                  return_best        = False,
                  verbose            = True
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'             : [{'method': 'lbfgs', 'order': (1,1,1)}, 
                                         {'method': 'powell', 'order': (1,1,1)}, 
                                         {'method': 'powell', 'order': (2,2,2)}, 
                                         {'method': 'lbfgs', 'order': (2,2,2)}],
                 'mean_absolute_error': np.array([0.2168435 , 0.21686885, 0.25012153, 0.25015256]),
                 'method'             : ['lbfgs', 'powell', 'powell', 'lbfgs'],
                 'order'              : [(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 2, 2)]},
        index = np.array([2, 0, 1, 3])
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.001)