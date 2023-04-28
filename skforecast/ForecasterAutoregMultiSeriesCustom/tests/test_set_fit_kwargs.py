# Unit test set_fit_kwargs ForecasterAutoregMultiSeriesCustom
# ==============================================================================
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from lightgbm import LGBMRegressor

def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """

    lags = y[-1:-6:-1]

    return lags


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LGBMRegressor(),
                     fun_predictors = create_predictors,
                     window_size    = 5,
                     fit_kwargs     = {'categorical_feature': 'auto'}
                 )
    
    new_fit_kwargs = {'categorical_feature': ['exog']}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'categorical_feature': ['exog']}

    assert results == expected