# Unit test create_predict_X ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# Fixtures
from .fixtures_ForecasterAutoregCustom import y as y_categorical
from .fixtures_ForecasterAutoregCustom import exog as exog_categorical
from .fixtures_ForecasterAutoregCustom import data # to test results when using differentiation

def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    
    lags = y[-1:-6:-1]
    
    return lags

def create_predictors_1_5(y): # pragma: no cover
    """
    Create first 15 lags of a time series.
    """
    
    lags = y[[-1, -5]]
    
    return lags 


def test_create_predict_X_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.create_predict_X(steps=5)


def test_create_predict_X_when_regressor_is_LinearRegression():
    """
    Test create_predict_X when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float), name='y'))
    results = forecaster.create_predict_X(steps=5)

    expected = pd.DataFrame(
        data = {
            'custom_predictor_0': [49., 50., 51., 52., 53.],
            'custom_predictor_1': [48., 49., 50., 51., 52.],
            'custom_predictor_2': [47., 48., 49., 50., 51.],
            'custom_predictor_3': [46., 47., 48., 49., 50.],
            'custom_predictor_4': [45., 46., 47., 48., 49.]
        },
        index = pd.RangeIndex(start=50, stop=55, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test create_predict_X when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88]),
            name = 'y'
        )
    
    forecaster = ForecasterAutoregCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 5,
                     name_predictors = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                     transformer_y   = StandardScaler()
                 )
    forecaster.fit(y=y)
    results = forecaster.create_predict_X(steps=5)

    expected = pd.DataFrame(
        data = np.array([
            [-0.52297655, -0.28324197, -0.45194408, -0.30987914, -0.1056608 ],
            [-0.1578203 , -0.52297655, -0.28324197, -0.45194408, -0.30987914],
            [-0.18459942, -0.1578203 , -0.52297655, -0.28324197, -0.45194408],
            [-0.13711051, -0.18459942, -0.1578203 , -0.52297655, -0.28324197],
            [-0.01966358, -0.13711051, -0.18459942, -0.1578203 , -0.52297655]]),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        index = pd.RangeIndex(start=20, stop=25, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog_series():
    """
    Test create_predict_X when using LinearRegression as regressor, StandardScaler
    as transformer_y and StandardScaler as transformer_exog.
    """
    y = pd.Series(np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4]))
    exog = pd.Series(np.array([7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4]), name='exog')
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    forecaster = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 5,
                     name_predictors  = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler()
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.create_predict_X(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
        data = np.array([
            [ 0.0542589 ,  0.27129451,  0.89246539, -2.34810076,  1.16937289, -1.76425513],
            [ 0.50619336,  0.0542589 ,  0.27129451,  0.89246539, -2.34810076, -1.00989936],
            [-0.09630298,  0.50619336,  0.0542589 ,  0.27129451,  0.89246539,  0.59254869],
            [ 0.05254973, -0.09630298,  0.50619336,  0.0542589 ,  0.27129451,  0.45863938],
            [ 0.12281153,  0.05254973, -0.09630298,  0.50619336,  0.0542589 ,  0.1640389 ]]
        ),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        index = pd.RangeIndex(start=8, stop=13, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog_df():
    """
    Test create_predict_X when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4])
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
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 5,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.create_predict_X(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
        data = np.array([[ 0.0542589 ,  0.27129451,  0.89246539, -2.34810076,  1.16937289,
                           -1.76425513,  1.        ,  0.        ],
                         [ 0.50619336,  0.0542589 ,  0.27129451,  0.89246539, -2.34810076,
                           -1.00989936,  1.        ,  0.        ],
                         [-0.09630298,  0.50619336,  0.0542589 ,  0.27129451,  0.89246539,
                           0.59254869,  1.        ,  0.        ],
                         [ 0.05254973, -0.09630298,  0.50619336,  0.0542589 ,  0.27129451,
                           0.45863938,  1.        ,  0.        ],
                         [ 0.12281153,  0.05254973, -0.09630298,  0.50619336,  0.0542589 ,
                           0.1640389 ,  0.        ,  1.        ]]),
        columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2',
                   'custom_predictor_3', 'custom_predictor_4', 'col_1', 'col_2_a', 'col_2_b'],
        index = pd.RangeIndex(start=8, stop=13, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test create_predict_X when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog_categorical,
         'exog_2': ['a', 'b', 'c', 'd', 'e']*10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J']*10)}
    )
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    categorical_features = df_exog.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer_exog = make_column_transformer(
                           (
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               categorical_features
                           ),
                           remainder="passthrough",
                           verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    forecaster = ForecasterAutoregCustom(
                     regressor        = HistGradientBoostingRegressor(
                                            categorical_features = categorical_features,
                                            random_state         = 123
                                        ),
                     fun_predictors   = create_predictors,
                     window_size      = 5,
                     transformer_y    = None,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_categorical, exog=df_exog)
    results = forecaster.create_predict_X(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
        data = np.array([
                [0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.        , 0.        , 0.12062867],
                [0.61187012, 0.61289453, 0.51948512, 0.98555979, 0.48303426,
                    1.        , 1.        , 0.8263408 ],
                [0.42274801, 0.61187012, 0.61289453, 0.51948512, 0.98555979,
                    2.        , 2.        , 0.60306013],
                [0.43214802, 0.42274801, 0.61187012, 0.61289453, 0.51948512,
                    3.        , 3.        , 0.54506801],
                [0.4923281 , 0.43214802, 0.42274801, 0.61187012, 0.61289453,
                    4.        , 4.        , 0.34276383],
                [0.53073262, 0.4923281 , 0.43214802, 0.42274801, 0.61187012,
                    0.        , 0.        , 0.30412079],
                [0.48004443, 0.53073262, 0.4923281 , 0.43214802, 0.42274801,
                    1.        , 1.        , 0.41702221],
                [0.56731689, 0.48004443, 0.53073262, 0.4923281 , 0.43214802,
                    2.        , 2.        , 0.68130077],
                [0.53956024, 0.56731689, 0.48004443, 0.53073262, 0.4923281 ,
                    3.        , 3.        , 0.87545684],
                [0.47670124, 0.53956024, 0.56731689, 0.48004443, 0.53073262,
                    4.        , 4.        , 0.51042234]]),
        columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2',
                   'custom_predictor_3', 'custom_predictor_4', 'exog_2', 'exog_3', 'exog_1'],
        index = pd.RangeIndex(start=50, stop=60, step=1)
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_create_predict_X_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_y():
    """
    Test create_predict_X when using LinearRegression as regressor and differentiation=1,
    and transformer_y is StandardScaler.
    """

    end_train = '2003-03-01 23:59:00'

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    steps = 4

    forecaster = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors_1_5,
                     window_size      = 5,
                     differentiation  = 1
                )
    forecaster.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    results = forecaster.create_predict_X(steps=steps, exog=exog.loc[end_train:])

    expected = pd.DataFrame(
        data = np.array([
            [ 0.07503713, -0.01018012,  1.16172882],
            [ 2.04678814,  0.10597971,  0.29468848],
            [ 2.0438193 , -0.01463081, -0.4399757 ],
            [ 2.06140742, -0.48984868,  1.25008389]]
        ),
        columns = ['custom_predictor_0', 'custom_predictor_1', 'exog'],
        index = pd.date_range(start='2003-04-01', periods=steps, freq='MS')
    )
    
    pd.testing.assert_frame_equal(results, expected)