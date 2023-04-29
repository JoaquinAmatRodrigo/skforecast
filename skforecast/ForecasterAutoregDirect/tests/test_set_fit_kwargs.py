# Unit test set_fit_kwargs ForecasterAutoregDirect
# ==============================================================================
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from lightgbm import LGBMRegressor


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterAutoregDirect(
                     regressor  = LGBMRegressor(),
                     steps      = 3,
                     lags       = 3,
                     fit_kwargs = {'categorical_feature': 'auto'}
                 )
    
    new_fit_kwargs = {'categorical_feature': ['exog']}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'categorical_feature': ['exog']}

    assert results == expected