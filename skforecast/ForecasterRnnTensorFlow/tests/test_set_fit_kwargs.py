# Unit test set_fit_kwargs ForecasterAutoregMultiVariate
# ==============================================================================
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from lightgbm import LGBMRegressor


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor  = LGBMRegressor(),
                     steps      = 3,
                     level      = 'l1',
                     lags       = 3,
                     fit_kwargs = {'categorical_feature': 'auto'}
                 )
    
    new_fit_kwargs = {'categorical_feature': ['exog']}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'categorical_feature': ['exog']}

    assert results == expected