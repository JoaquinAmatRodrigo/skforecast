# Unit test get_feature_importance
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

def test_output_get_feature_importance_when_regressor_is_RandomForest():
    forecaster = ForecasterAutoreg(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.94766355, 0., 0.05233645])
                })
    results = forecaster.get_feature_importance()
    assert (results['feature'] == expected['feature']).all()
    assert results['importance'].values == approx(expected['importance'].values)
    
    
def test_output_get_feature_importance_when_regressor_is_linear_model():
    forecaster = ForecasterAutoreg(Lasso(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = None
    results = forecaster.get_feature_importance()
    assert results is expected
    