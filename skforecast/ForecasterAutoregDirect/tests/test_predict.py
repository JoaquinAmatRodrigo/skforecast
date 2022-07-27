# Unit test predict ForecasterAutoregDirect
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

        
def test_predict_output_when_regressor_is_LinearRegression():
    '''
    Test predict output when using LinearRegression as regressor.
    '''
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(50)))
    results = forecaster.predict()
    expected = pd.Series(
                data = np.array([50., 51., 52.]),
                index = pd.RangeIndex(start=50, stop=53, step=1),
                name = 'pred'
               )
    
    pd.testing.assert_series_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    transformer_y = StandardScaler()
    forecaster = ForecasterAutoregDirect(
                    regressor = LinearRegression(),
                    lags = 5,
                    steps = 5,
                    transformer_y = transformer_y,
                )
    forecaster.fit(y=y)
    predictions = forecaster.predict()
    expected = pd.Series(
                data = np.array([0.27498792, 0.1134674 , 0.3824246 , 0.62852197, 0.44001725]),
                index = pd.RangeIndex(start=20, stop=25, step=1),
                name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.skip(reason="work in progress")
def test_predict_output_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4])
        )
    exog = pd.DataFrame({
                'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']}
           )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregDirect(
                    regressor = LinearRegression(),
                    lags = 5,
                    steps = 5,
                    transformer_y = transformer_y,
                    transformer_exog = transformer_exog,
                 )
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict(exog=exog)
    expected = pd.Series(
                data = np.array([0.50619336, -0.09630298,  0.05254973,  0.12281153,  0.00221741]),
                index = pd.RangeIndex(start=8, stop=13, step=1),
                name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
    