# Test method get_feature_importance()
# ==============================================================================

import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from sklearn.ensemble import RandomForestRegressor


def test_get_feature_importance_when_regressor_is_RandomForestRegressor_lags_3_step_1():
    '''
    Test output of get_feature_importance for step 1, when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    '''
    forecaster = ForecasterAutoregMultiOutput(
                    RandomForestRegressor(random_state=123),
                    lags = 3,
                    steps = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)))
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'importance': np.array([0.3902439024390244, 0.3170731707317073, 0.2926829268292683])
                })
    pd.testing.assert_frame_equal(results, expected)
  
    
def test_get_feature_importance_when_regressor_is_RandomForestRegressor_lags_3_step_1_exog_included():
    '''
    Test output of get_feature_importance for step 1, when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(5)) and
    exog=pd.Series(np.arange(5), name='exog').
    '''
    forecaster = ForecasterAutoregMultiOutput(
                    RandomForestRegressor(random_state=123),
                    lags = 3,
                    steps = 1
                 )
    forecaster.fit(y=pd.Series(np.arange(5)), exog=pd.Series(np.arange(5), name='exog'))
    results = forecaster.get_feature_importance(step=1)
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3', 'exog'],
                    'importance': np.array([0.1951219512195122, 0.0975609756097561,
                                            0.36585365853658536, 0.34146341463414637])
                })
    pd.testing.assert_frame_equal(results, expected)