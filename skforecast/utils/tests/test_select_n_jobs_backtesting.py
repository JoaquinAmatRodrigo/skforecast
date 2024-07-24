# Unit test select_n_jobs_backtesting
# ==============================================================================
import pytest
from joblib import cpu_count
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from skforecast.utils.utils import select_n_jobs_backtesting
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.ForecasterBaseline import ForecasterEquivalentDate


@pytest.mark.parametrize("forecaster, refit, n_jobs_expected", 
    [(ForecasterAutoreg(LinearRegression(), lags=2), True, 1),
    (ForecasterAutoreg(make_pipeline(StandardScaler(), LinearRegression()), lags=2), True, 1),
    (ForecasterAutoreg(LinearRegression(), lags=2), 2, 1),
    (ForecasterAutoreg(LinearRegression(), lags=2), False, 1),
    (ForecasterAutoreg(HistGradientBoostingRegressor(), lags=2), 1, cpu_count() - 1),
    (ForecasterAutoreg(HistGradientBoostingRegressor(), lags=2), 2, 1),
    (ForecasterAutoreg(HistGradientBoostingRegressor(), lags=2), 0, 1),
    (ForecasterAutoreg(LGBMRegressor(), lags=2), True, 1),
    (ForecasterAutoreg(LGBMRegressor(), lags=2), False, 1),
    (ForecasterAutoregDirect(LinearRegression(), steps=3, lags=2), True, 1),
    (ForecasterAutoregDirect(LinearRegression(), steps=3, lags=2), 2, 1),
    (ForecasterAutoregDirect(LinearRegression(), steps=3, lags=2), False, 1),
    (ForecasterAutoregDirect(HistGradientBoostingRegressor(), steps=3, lags=2), 1, 1),
    (ForecasterAutoregDirect(HistGradientBoostingRegressor(), steps=3, lags=2), 2, 1),
    (ForecasterAutoregDirect(HistGradientBoostingRegressor(), steps=3, lags=2), 0, 1),
    (ForecasterAutoregDirect(LGBMRegressor(), steps=3, lags=2), True, 1),
    (ForecasterAutoregDirect(LGBMRegressor(), steps=3, lags=2), False, 1),
    (ForecasterAutoregMultiSeries(LinearRegression(), lags=2), True, cpu_count() - 1),
    (ForecasterAutoregMultiSeries(LinearRegression(), lags=2), 2, 1),
    (ForecasterAutoregMultiSeries(LinearRegression(), lags=2), False, cpu_count() - 1),
    (ForecasterAutoregMultiSeries(HistGradientBoostingRegressor(), lags=2), 1, cpu_count() - 1),
    (ForecasterAutoregMultiSeries(HistGradientBoostingRegressor(), lags=2), 2, 1),
    (ForecasterAutoregMultiSeries(HistGradientBoostingRegressor(), lags=2), 0, cpu_count() - 1),
    (ForecasterAutoregMultiSeries(LGBMRegressor(), lags=2), True, 1),
    (ForecasterAutoregMultiSeries(LGBMRegressor(), lags=2), False, 1),
    (ForecasterSarimax(Sarimax((1, 0, 1))), True, 1),
    (ForecasterSarimax(Sarimax((1, 0, 1))), 2, 1),
    (ForecasterSarimax(Sarimax((1, 0, 1))), False, 1),
    (ForecasterEquivalentDate(2), True, 1),
    (ForecasterEquivalentDate(2), 2, 1),
    (ForecasterEquivalentDate(2), False, 1)
], 
ids=lambda info: f'info: {info}')
def test_select_n_jobs_backtesting(forecaster, refit, n_jobs_expected):
    """
    Test select_n_jobs_backtesting
    """
    n_jobs = select_n_jobs_backtesting(
                 forecaster = forecaster, 
                 refit      = refit
             )
    
    assert n_jobs == n_jobs_expected