# Unit test predict_interval ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    
    lags = y[-1:-6:-1]
    
    return lags 
   

def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = pd.DataFrame(
                   np.array([[10., 20., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index = pd.RangeIndex(start=10, stop=11, step=1)
               )
    results = forecaster.predict_interval(steps=1, in_sample_residuals=True, n_boot=2)  
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )    
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = pd.DataFrame(
                   np.array([[10., 20., 20.],
                         [11., 23., 23.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index = pd.RangeIndex(start=10, stop=12, step=1)
               )
    results = forecaster.predict_interval(steps=2, in_sample_residuals=True, n_boot=2)  
    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals, fill_value=10))
    expected = pd.DataFrame(
                   np.array([[10., 20., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index = pd.RangeIndex(start=10, stop=11, step=1)
               )
    results = forecaster.predict_interval(steps=1, in_sample_residuals=False, n_boot=2)  
    pd.testing.assert_frame_equal(results, expected)

    
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals, fill_value=10))
    expected = pd.DataFrame(
                   np.array([[10., 20., 20.],
                         [11., 23., 23.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index = pd.RangeIndex(start=10, stop=12, step=1)
               )
    results = forecaster.predict_interval(steps=2, in_sample_residuals=False)  
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    transformer_y = StandardScaler()
    forecaster = ForecasterAutoregCustom(
                    regressor = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                    transformer_y = transformer_y,
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5)
    expected = pd.DataFrame(
                data = np.array([[-0.1578203 , -2.02104471,  1.55076389],
                                 [-0.18459942, -1.85211385,  1.49693466],
                                 [-0.13711051, -1.82918043,  1.37837389],
                                 [-0.01966358, -1.7522189 ,  1.31940346],
                                 [-0.03228613, -1.77074022,  1.56056698]]),
                index = pd.RangeIndex(start=20, stop=25, step=1),
                columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59, 0.02, -0.9, 1.09, -3.61, 0.72, -0.11, -0.4])
        )
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
               'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']}
           )
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )
    forecaster = ForecasterAutoregCustom(
                    regressor = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5,
                    transformer_y = transformer_y,
                    transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(steps=5, exog=exog_predict)
    expected = pd.DataFrame(
                data = np.array([[ 0.50619336,  0.50619336,  0.50619336],
                                 [-0.09630298, -0.09630298, -0.09630298],
                                 [ 0.05254973,  0.05254973,  0.05254973],
                                 [ 0.12281153,  0.12281153,  0.12281153],
                                 [ 0.00221741,  0.00221741,  0.00221741]]),
                index = pd.RangeIndex(start=8, stop=13, step=1),
                columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)