# Unit test select_n_jobs_backtesting
# ==============================================================================
import pytest
from joblib import cpu_count
from skforecast.utils.utils import select_n_jobs_backtesting


@pytest.mark.parametrize("forecaster_name, regressor_name, refit, n_jobs_expected", 
                         [('ForecasterAutoreg', 'LinearRegression', True, 1),
                          ('ForecasterAutoreg', 'LinearRegression', 2, 1),
                          ('ForecasterAutoreg', 'LinearRegression', False, 1),
                          ('ForecasterAutoreg', 'HistGradientBoostingRegressor', 1, cpu_count()),
                          ('ForecasterAutoreg', 'HistGradientBoostingRegressor', 2, 1),
                          ('ForecasterAutoreg', 'HistGradientBoostingRegressor', 0, 1),
                          ('ForecasterAutoregDirect', 'LinearRegression', True, 1),
                          ('ForecasterAutoregDirect', 'LinearRegression', 2, 1),
                          ('ForecasterAutoregDirect', 'LinearRegression', False, 1),
                          ('ForecasterAutoregDirect', 'HistGradientBoostingRegressor', 1, 1),
                          ('ForecasterAutoregDirect', 'HistGradientBoostingRegressor', 2, 1),
                          ('ForecasterAutoregDirect', 'HistGradientBoostingRegressor', 0, 1),
                          ('ForecasterAutoregMultiseries', 'LinearRegression', True, cpu_count()),
                          ('ForecasterAutoregMultiseries', 'LinearRegression', 2, 1),
                          ('ForecasterAutoregMultiseries', 'LinearRegression', False, cpu_count()),
                          ('ForecasterAutoregMultiseries', 'HistGradientBoostingRegressor', 1, cpu_count()),
                          ('ForecasterAutoregMultiseries', 'HistGradientBoostingRegressor', 2, 1),
                          ('ForecasterAutoregMultiseries', 'HistGradientBoostingRegressor', 0, cpu_count()),
                          ('ForecasterSarimax', 'ARIMA', True, 1),
                          ('ForecasterSarimax', 'ARIMA', 2, 1),
                          ('ForecasterSarimax', 'ARIMA', False, 1),
                          ('ForecasterEquivalentDate', None, True, 1),
                          ('ForecasterEquivalentDate', None, 2, 1),
                          ('ForecasterEquivalentDate', None, False, 1),
                          ('Forecaster', 'regressor', False, 1)
                          ], 
                         ids=lambda info: f'info: {info}')
def test_select_n_jobs_backtesting(forecaster_name, regressor_name, refit, n_jobs_expected):
    """
    Test select_n_jobs_backtesting
    """
    n_jobs = select_n_jobs_backtesting(
                 forecaster_name = forecaster_name, 
                 regressor_name  = regressor_name,
                 refit           = refit
             )
    
    assert n_jobs == n_jobs_expected