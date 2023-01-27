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
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))

    # Generate 15 random `order`
    np.random.seed(123)
    values = [(p,d,q) for p,d,q in zip(np.random.randint(0, high=4, size=15, dtype=int), 
                                       np.random.randint(0, high=4, size=15, dtype=int),
                                       np.random.randint(0, high=4, size=15, dtype=int))]

    param_distributions = {'order' : values}

    results = random_search_sarimax(
                  forecaster          = forecaster,
                  y                   = y_datetime,
                  param_distributions = param_distributions,
                  n_iter              = 10,
                  random_state        = 123,
                  steps               = 3,
                  refit               = False,
                  metric              = 'mean_absolute_error',
                  initial_train_size  = len(y_datetime)-12,
                  fixed_train_size    = False,
                  return_best         = False,
                  verbose             = True
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'             : np.array([{'order': (0, 0, 2)}, {'order': (2, 1, 3)}, {'order': (1, 1, 1)},
                                                  {'order': (3, 1, 0)}, {'order': (2, 1, 0)}, {'order': (1, 2, 1)},
                                                  {'order': (2, 2, 0)}, {'order': (3, 2, 0)}, {'order': (2, 3, 3)},
                                                  {'order': (1, 3, 2)}], dtype=object),
                 'mean_absolute_error': np.array([0.20445157, 0.21545791, 0.2168435 , 0.24189445, 0.24504986,
                                                  0.27556196, 0.28612558, 0.29589668, 0.31338047, 0.31517517]),
                 'order'              : [(0, 0, 2), (2, 1, 3), (1, 1, 1), (3, 1, 0), (2, 1, 0), 
                                         (1, 2, 1), (2, 2, 0), (3, 2, 0), (2, 3, 3), (1, 3, 2)]},
        index = np.array([2, 8, 7, 6, 3, 9, 4, 1, 5, 0])
    )

    pd.testing.assert_frame_equal(results, expected_results)