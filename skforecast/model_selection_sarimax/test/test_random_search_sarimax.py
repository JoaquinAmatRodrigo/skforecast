# Unit test random_search_sarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import random_search_sarimax
from pmdarima.arima import ARIMA

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import y_datetime


def test_output_random_search_sarimax_sarimax_with_mocked():
    """
    Test output of random_search_sarimax in ForecasterSarimax with mocked
    (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))

    # Generate 15 random `order`
    np.random.seed(123)
    values = [(p,d,q) for p,d,q in zip(np.random.randint(0, high=4, size=3, dtype=int), 
                                       np.random.randint(0, high=4, size=3, dtype=int),
                                       np.random.randint(0, high=4, size=3, dtype=int))]

    param_distributions = {'order': values}

    results = random_search_sarimax(
                  forecaster          = forecaster,
                  y                   = y_datetime,
                  param_distributions = param_distributions,
                  n_iter              = 3,
                  random_state        = 123,
                  steps               = 3,
                  refit               = False,
                  metric              = 'mean_absolute_error',
                  initial_train_size  = len(y_datetime)-12,
                  fixed_train_size    = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params': np.array([{'order': (1, 0, 1)}, {'order': (2, 2, 3)}, {'order': (2, 2, 2)}],
                                    dtype=object),
                'mean_absolute_error': np.array([0.20379714167503413, 0.24736436822367514, 0.2501673978304879]),
                'order'              : [(1, 0, 1), (2, 2, 3), (2, 2, 2)]},
        index = pd.Index(np.array([1, 2, 0]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.001)
