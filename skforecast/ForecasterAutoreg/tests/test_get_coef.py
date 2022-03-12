# Unit test get_coef ForecasterAutoreg
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def test_output_get_coef_when_regressor_is_LinearRegression():
    '''
    Test output of get_coef when regressor is LinearRegression with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'coef': np.array([0.33333333, 0.33333333, 0.33333333])
                })
    results = forecaster.get_coef()
    assert (results['feature'] == expected['feature']).all()
    assert results['coef'].values == approx(expected['coef'].values)


def test_output_get_coef_when_regressor_is_RandomForest():
    '''
    Test output of get_coef when regressor is RandomForestRegressor with lags=3
    and it is trained with y=pd.Series(np.arange(5)).
    '''
    forecaster = ForecasterAutoreg(RandomForestRegressor(n_estimators=1, max_depth=2), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = None
    results = forecaster.get_coef()
    assert results is expected


def test_output_get_coef_when_pipeline():
    '''
    Test output of get_coef when regressor is pipeline,
    (StandardScaler() + LinearRegression with lags=3),
    it is trained with y=pd.Series(np.arange(5)).
    '''
    forecaster = ForecasterAutoreg(
                    regressor = make_pipeline(StandardScaler(), LinearRegression()),
                    lags      = 3
                    )
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = pd.DataFrame({
                    'feature': ['lag_1', 'lag_2', 'lag_3'],
                    'coef': np.array([0.166667, 0.166667, 0.166667])
                })
    results = forecaster.get_coef()
    pd.testing.assert_frame_equal(expected, results)