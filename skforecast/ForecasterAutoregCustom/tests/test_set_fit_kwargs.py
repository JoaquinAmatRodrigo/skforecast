# Unit test set_fit_kwargs ForecasterAutoregCustom
# ==============================================================================
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from lightgbm import LGBMRegressor

def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor = LGBMRegressor(),
                    lags = 3,
                    fit_kwargs = {'categorical_feature': 'auto'}
                 )
    new_fit_kwargs = {'categorical_feature': ['exog']}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    expected = {'categorical_feature': ['exog']}
    results = forecaster.fit_kwargs
    assert results == expected