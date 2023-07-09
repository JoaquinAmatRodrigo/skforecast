# Unit test select_n_jobs_fit_forecaster
# ==============================================================================
import pytest
from joblib import cpu_count
from skforecast.utils.utils import select_n_jobs_fit_forecaster

@pytest.mark.parametrize("forecaster_name, regressor_name, n_jobs_expected", 
                         [('ForecasterAutoregDirect', 'LinearRegression', 1),
                          ('ForecasterAutoregDirect', 'RandomForestRegressor', cpu_count()),
                          ('ForecasterAutoregMultiVariate', 'LinearRegression', 1),
                          ('ForecasterAutoregMultiVariate', 'RandomForestRegressor', cpu_count()),
                          ('ForecasterAutoreg', 'LinearRegression', 1),
                          ('ForecasterAutoreg', 'RandomForestRegressor', 1)
                          ], 
                         ids=lambda info: f'info: {info}')
def test_select_n_jobs_fit_forecaster(forecaster_name, regressor_name, n_jobs_expected):
    """
    Test select_n_jobs_fit_forecaster
    """
    n_jobs = select_n_jobs_fit_forecaster(
                 forecaster_name = forecaster_name, 
                 regressor_name  = regressor_name
             )
    
    assert n_jobs == n_jobs_expected