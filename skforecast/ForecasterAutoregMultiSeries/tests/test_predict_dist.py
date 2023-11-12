# Unit test predict_dist ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Fixtures
from .fixtures_ForecasterAutoregMultiSeries import series
from .fixtures_ForecasterAutoregMultiSeries import exog
from .fixtures_ForecasterAutoregMultiSeries import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


@pytest.mark.parametrize("level", 
                         ['1', ['1']], 
                         ids=lambda lvl: f'level: {lvl}')
def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(level):
    """
    Test output of predict_dist when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed. Single level.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_dist(
                  steps               = 2,
                  distribution        = norm,
                  levels              = level,
                  exog                = exog_predict,
                  n_boot              = 4,
                  in_sample_residuals = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.38172712, 0.13932345],
                                       [0.22554853, 0.16819172]]),
                   columns = ['1_loc', '1_scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("levels", 
                         [['1', '2'], None], 
                         ids=lambda lvl: f'levels: {lvl}')
def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer(levels):
    """
    Test output of predict_dist when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed. Multiple levels.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    
    forecaster.fit(series=series, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_dist(
                  steps               = 2,
                  distribution        = norm,
                  levels              = levels,
                  exog                = exog_predict,
                  n_boot              = 4,
                  in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.38172712, 0.13932345, 0.54404355, 0.20840904],
                                       [0.22554853, 0.16819172, 0.42133164, 0.21519488]]),
                   columns = ['1_loc', '1_scale', '2_loc', '2_scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(results, expected)