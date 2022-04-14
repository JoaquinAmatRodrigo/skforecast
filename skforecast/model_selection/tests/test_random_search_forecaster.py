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

# Fixtures _backtesting_forecaster_refit Series (skforecast==0.4.2)
# np.random.seed(123)
# y = np.random.rand(50)

y = pd.Series(
    np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
              0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
              0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
              0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
              0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
              0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
              0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
              0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
              0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
              0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]))

def test_output_random_search_forecaster_ForecasterAutoreg_with_mocked():
    '''
    Test output of random_search_forecaster in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3)
    '''
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
            'metric':np.array([0.06460234, 0.06475887, 0.06776596, 
                               0.06786132, 0.0713478, 0.07161]),                                                               
            'alpha' :np.array([1.00000000e-05, 3.59381366e-02, 1.00000000e-05, 
                               3.59381366e-02, 1.66810054e+01, 1.66810054e+01])
                                     },
            index=[1, 0, 4, 3, 5, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)