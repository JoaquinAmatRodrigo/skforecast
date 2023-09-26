# Unit test random_search_forecaster
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
from .fixtures_model_selection import y


def test_output_random_search_forecaster_ForecasterAutoreg_with_mocked():
    """
    Test output of random_search_forecaster in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3)
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_distributions = {'alpha':np.logspace(-5, 3, 10)}
    n_iter = 3

    results = random_search_forecaster(
                  forecaster   = forecaster,
                  y            = y,
                  lags_grid    = lags_grid,
                  param_distributions  = param_distributions,
                  steps        = steps,
                  refit        = False,
                  metric       = 'mean_squared_error',
                  initial_train_size = len(y_train),
                  fixed_train_size   = False,
                  n_iter       = n_iter,
                  random_state = 123,
                  return_best  = False,
                  verbose      = False
              )
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2]],
            'params':[{'alpha': 1e-05}, {'alpha': 0.03593813663804626}, {'alpha': 1e-05},
                      {'alpha': 0.03593813663804626}, {'alpha': 16.681005372000556}, {'alpha': 16.681005372000556}],
            'mean_squared_error':np.array([0.06460234, 0.06475887, 0.06776596, 
                               0.06786132, 0.0713478, 0.07161]),                                                               
            'alpha' :np.array([1.00000000e-05, 3.59381366e-02, 1.00000000e-05, 
                               3.59381366e-02, 1.66810054e+01, 1.66810054e+01])
                                     },
            index=pd.Index([1, 0, 4, 3, 5, 2], dtype='int64')
                                   )

    pd.testing.assert_frame_equal(results, expected_results)