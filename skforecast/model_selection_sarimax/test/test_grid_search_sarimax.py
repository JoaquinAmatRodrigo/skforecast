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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))

    param_grid = [{'order' : [(1,1,1), (2,2,2)],
                    'method': ['lbfgs']}]

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
                  verbose            = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'             : [{'method': 'lbfgs', 'order': (1, 1, 1)}, 
                                        {'method': 'lbfgs', 'order': (2, 2, 2)}],
                    'mean_absolute_error': np.array([0.21691209090287925, 0.25017244915790254]),
                    'method'             : ['lbfgs', 'lbfgs'],
                    'order'              : [(1, 1, 1), (2, 2, 2)]},
        index = pd.Index(np.array([0, 1]), dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)