# Unit test create_predict_X ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

# Fixtures
from .fixtures_ForecasterAutoregMultiSeries import series
from .fixtures_ForecasterAutoregMultiSeries import exog
from .fixtures_ForecasterAutoregMultiSeries import exog_predict

THIS_DIR = Path(__file__).parent
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}

series_2 = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50, dtype=float)), 
                         '2': pd.Series(np.arange(start=50, stop=100, dtype=float))})


def test_create_predict_X_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.create_predict_X(steps=5)


def test_output_create_predict_X_when_regressor_is_LinearRegression():
    """
    Test output create_predict_X when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.fit(series=series_2)
    results = forecaster.create_predict_X(steps=5)

    expected = {
        '1': pd.DataFrame(
                 data = {
                     'lag_1': [49., 50., 51., 52., 53.],
                     'lag_2': [48., 49., 50., 51., 52.],
                     'lag_3': [47., 48., 49., 50., 51.],
                     'lag_4': [46., 47., 48., 49., 50.],
                     'lag_5': [45., 46., 47., 48., 49.],
                     '_level_skforecast': [0., 0., 0., 0., 0.]
                 },
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             ),
        '2': pd.DataFrame(
                 data = {
                     'lag_1': [99., 100., 101., 102., 103.],
                     'lag_2': [98., 99., 100., 101., 102.],
                     'lag_3': [97., 98., 99., 100., 101.],
                     'lag_4': [96., 97., 98., 99., 100.],
                     'lag_5': [95., 96., 97., 98., 99.],
                     '_level_skforecast': [1., 1., 1., 1., 1.]
                 },
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test create_predict_X output when using LinearRegression as regressor 
    and StandardScaler.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'onehot',
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster.create_predict_X(steps=5, levels='1')

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835,
                       1.        ,  0.        ],
                     [ 0.52791431,  0.47762884,  0.07582436,  2.08066403, -0.08097053,
                       1.        ,  0.        ],
                     [ 0.44509712,  0.52791431,  0.47762884,  0.07582436,  2.08066403,
                       1.        ,  0.        ],
                     [ 0.42176045,  0.44509712,  0.52791431,  0.47762884,  0.07582436,
                       1.        ,  0.        ],
                     [ 0.48087237,  0.42176045,  0.44509712,  0.52791431,  0.47762884,
                       1.        ,  0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', '1', '2'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_predict_X_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog(transformer_series):
    """
    Test create_predict_X output when using LinearRegression as regressor, 
    StandardScaler as transformer_series and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'ordinal',
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.create_predict_X(steps=5, levels='1', exog=exog_predict)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835,
                       0.        , -0.09362908,  1.        ,  0.        ],
                     [ 0.53267333,  0.47762884,  0.07582436,  2.08066403, -0.08097053,
                       0.        ,  0.45144522,  1.        ,  0.        ],
                     [ 0.44478046,  0.53267333,  0.47762884,  0.07582436,  2.08066403,
                       0.        , -1.53968887,  1.        ,  0.        ],
                     [ 0.52579563,  0.44478046,  0.53267333,  0.47762884,  0.07582436,
                       0.        , -1.45096055,  1.        ,  0.        ],
                     [ 0.57391142,  0.52579563,  0.44478046,  0.53267333,  0.47762884,
                       0.        , -0.77240468,  1.        ,  0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_1', 'exog_2_a', 'exog_2_b'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_create_predict_X_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test create_predict_X output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog with series 
    of different lengths.
    """
    new_series = series.copy()
    new_series.iloc[:10, 1] = np.nan

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=new_series, exog=exog)
    results = forecaster.create_predict_X(steps=5, exog=exog_predict)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835,
                       0.        , -0.09575486,  1.        ,  0.        ],
                     [ 0.52343425,  0.47762884,  0.07582436,  2.08066403, -0.08097053,
                       0.        ,  0.44024531,  1.        ,  0.        ],
                     [ 0.42697026,  0.52343425,  0.47762884,  0.07582436,  2.08066403,
                       0.        , -1.51774136,  1.        ,  0.        ],
                     [ 0.48906322,  0.42697026,  0.52343425,  0.47762884,  0.07582436,
                       0.        , -1.43049014,  1.        ,  0.        ],
                     [ 0.55395581,  0.48906322,  0.42697026,  0.52343425,  0.47762884,
                       0.        , -0.76323054,  1.        ,  0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_1', 'exog_2_a', 'exog_2_b'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             ),
        '2': pd.DataFrame(
                 data = np.array([
                     [-0.57388468, -0.96090271, -0.36483376, -1.29936753,  0.73973379,
                       1.        , -0.09575486,  1.        ,  0.        ],
                     [ 0.49560898, -0.57388468, -0.96090271, -0.36483376, -1.29936753,
                       1.        ,  0.44024531,  1.        ,  0.        ],
                     [ 0.53289813,  0.49560898, -0.57388468, -0.96090271, -0.36483376,
                       1.        , -1.51774136,  1.        ,  0.        ],
                     [ 0.61304224,  0.53289813,  0.49560898, -0.57388468, -0.96090271,
                       1.        , -1.43049014,  1.        ,  0.        ],
                     [ 0.61509108,  0.61304224,  0.53289813,  0.49560898, -0.57388468,
                       1.        , -0.76323054,  1.        ,  0.        ]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_1', 'exog_2_a', 'exog_2_b'],
                 index = pd.RangeIndex(start=50, stop=55, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test create_predict_X output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog['exog_1'],
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
    
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = HistGradientBoostingRegressor(
                                              categorical_features = categorical_features,
                                              random_state         = 123
                                          ),
                     lags               = 5,
                     transformer_series = None,
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=df_exog)
    results = forecaster.create_predict_X(steps=10, exog=exog_predict)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                      0.        , 0.        , 0.        , 0.51312815],
                     [0.55674244, 0.61289453, 0.51948512, 0.98555979, 0.48303426,
                      0.        , 1.        , 1.        , 0.66662455],
                     [0.39860559, 0.55674244, 0.61289453, 0.51948512, 0.98555979,
                      0.        , 2.        , 2.        , 0.10590849],
                     [0.53719969, 0.39860559, 0.55674244, 0.61289453, 0.51948512,
                      0.        , 3.        , 3.        , 0.13089495],
                     [0.7759386 , 0.53719969, 0.39860559, 0.55674244, 0.61289453,
                      0.        , 4.        , 4.        , 0.32198061],
                     [0.56654221, 0.7759386 , 0.53719969, 0.39860559, 0.55674244,
                      0.        , 0.        , 0.        , 0.66156434],
                     [0.43373393, 0.56654221, 0.7759386 , 0.53719969, 0.39860559,
                      0.        , 1.        , 1.        , 0.84650623],
                     [0.49776999, 0.43373393, 0.56654221, 0.7759386 , 0.53719969,
                      0.        , 2.        , 2.        , 0.55325734],
                     [0.47797067, 0.49776999, 0.43373393, 0.56654221, 0.7759386 ,
                      0.        , 3.        , 3.        , 0.85445249],
                     [0.49149785, 0.47797067, 0.49776999, 0.43373393, 0.56654221,
                      0.        , 4.        , 4.        , 0.38483781]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_2', 'exog_3', 'exog_1'],
                 index = pd.RangeIndex(start=50, stop=60, step=1)
             ),
        '2': pd.DataFrame(
                 data = np.array([
                     [0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                      1.        , 0.        , 0.        , 0.51312815],
                     [0.54564144, 0.34345601, 0.2408559 , 0.39887629, 0.15112745,
                      1.        , 1.        , 1.        , 0.66662455],
                     [0.53774558, 0.54564144, 0.34345601, 0.2408559 , 0.39887629,
                      1.        , 2.        , 2.        , 0.10590849],
                     [0.63668042, 0.53774558, 0.54564144, 0.34345601, 0.2408559 ,
                      1.        , 3.        , 3.        , 0.13089495],
                     [0.66871809, 0.63668042, 0.53774558, 0.54564144, 0.34345601,
                      1.        , 4.        , 4.        , 0.32198061],
                     [0.80858451, 0.66871809, 0.63668042, 0.53774558, 0.54564144,
                      1.        , 0.        , 0.        , 0.66156434],
                     [0.39998025, 0.80858451, 0.66871809, 0.63668042, 0.53774558,
                      1.        , 1.        , 1.        , 0.84650623],
                     [0.64896715, 0.39998025, 0.80858451, 0.66871809, 0.63668042,
                      1.        , 2.        , 2.        , 0.55325734],
                     [0.43578494, 0.64896715, 0.39998025, 0.80858451, 0.66871809,
                      1.        , 3.        , 3.        , 0.85445249],
                     [0.37087107, 0.43578494, 0.64896715, 0.39998025, 0.80858451,
                      1.        , 4.        , 4.        , 0.38483781]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                            '_level_skforecast', 'exog_2', 'exog_3', 'exog_1'],
                 index = pd.RangeIndex(start=50, stop=60, step=1)
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict():
    """
    Test output ForecasterAutoregMultiSeries create_predict_X method when 
    series and exog are dictionaries.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    results = forecaster.create_predict_X(
        steps=5, exog=exog_dict_test, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([
                           [-1.77580056e+00, -1.07810218e+00, -4.96184646e-01,
                            -4.57091003e-01, -3.52586104e-01,  0.00000000e+00,
                             8.21644127e-03,  1.42962482e+00,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03, -1.77580056e+00, -1.07810218e+00,
                            -4.96184646e-01, -4.57091003e-01,  0.00000000e+00,
                             1.11220226e+00,  8.96343746e-01,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03,  1.44369849e+03, -1.77580056e+00,
                            -1.07810218e+00, -4.96184646e-01,  0.00000000e+00,
                             1.38486425e+00, -3.01927953e-01,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03,  1.44369849e+03,  1.44369849e+03,
                            -1.77580056e+00, -1.07810218e+00,  0.00000000e+00,
                             6.20882352e-01, -1.26286725e+00,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03,  1.44369849e+03,  1.44369849e+03,
                             1.44369849e+03, -1.77580056e+00,  0.00000000e+00,
                            -6.04449469e-01, -1.26286725e+00,          np.nan,
                                     np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([
                           [ 1.15231370e+00,  1.16332175e+00,  6.78968143e-01,
                             3.69366454e-01,  2.18005286e-01,  1.00000000e+00,
                             8.21644127e-03,  1.42962482e+00,  1.11141113e+00,
                            -8.79435260e-01],
                           [ 2.14116223e+03,  1.15231370e+00,  1.16332175e+00,
                             6.78968143e-01,  3.69366454e-01,  1.00000000e+00,
                             1.11220226e+00,  8.96343746e-01,  1.13275580e+00,
                             5.89479621e-03],
                           [ 2.02558287e+03,  2.14116223e+03,  1.15231370e+00,
                             1.16332175e+00,  6.78968143e-01,  1.00000000e+00,
                             1.38486425e+00, -3.01927953e-01,  1.17758690e+00,
                            -3.53258397e-01],
                           [ 2.02558287e+03,  2.02558287e+03,  2.14116223e+03,
                             1.15231370e+00,  1.16332175e+00,  1.00000000e+00,
                             6.20882352e-01, -1.26286725e+00,  1.04283370e+00,
                             8.42872836e-01],
                           [ 2.02558287e+03,  2.02558287e+03,  2.02558287e+03,
                             2.14116223e+03,  1.15231370e+00,  1.00000000e+00,
                            -6.04449469e-01, -1.26286725e+00,  1.00599776e+00,
                            -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([
                           [ 3.39980134e+00, -3.78377384e-01, -1.03823091e+00,
                            -7.46859590e-01, -6.26519755e-01,  3.00000000e+00,
                             8.21644127e-03,          np.nan,  1.11141113e+00,
                            -8.79435260e-01],
                           [ 2.20078503e+03,  3.39980134e+00, -3.78377384e-01,
                            -1.03823091e+00, -7.46859590e-01,  3.00000000e+00,
                             1.11220226e+00,          np.nan,  1.13275580e+00,
                             5.89479621e-03],
                           [ 2.13316531e+03,  2.20078503e+03,  3.39980134e+00,
                            -3.78377384e-01, -1.03823091e+00,  3.00000000e+00,
                             1.38486425e+00,          np.nan,  1.17758690e+00,
                            -3.53258397e-01],
                           [ 2.09026933e+03,  2.13316531e+03,  2.20078503e+03,
                             3.39980134e+00, -3.78377384e-01,  3.00000000e+00,
                             6.20882352e-01,          np.nan,  1.04283370e+00,
                             8.42872836e-01],
                           [ 2.09026933e+03,  2.09026933e+03,  2.13316531e+03,
                             2.20078503e+03,  3.39980134e+00,  3.00000000e+00,
                            -6.04449469e-01,          np.nan,  1.00599776e+00,
                            -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([
                           [-7.50737618e-01, -2.43098121e-01,  6.71989902e-01,
                             8.32297096e-01,  6.14957530e-01,  4.00000000e+00,
                             8.21644127e-03,  1.42962482e+00,  1.11141113e+00,
                            -8.79435260e-01],
                           [ 7.31805276e+03, -7.50737618e-01, -2.43098121e-01,
                             6.71989902e-01,  8.32297096e-01,  4.00000000e+00,
                             1.11220226e+00,  8.96343746e-01,  1.13275580e+00,
                             5.89479621e-03],
                           [ 7.31805276e+03,  7.31805276e+03, -7.50737618e-01,
                            -2.43098121e-01,  6.71989902e-01,  4.00000000e+00,
                             1.38486425e+00, -3.01927953e-01,  1.17758690e+00,
                            -3.53258397e-01],
                           [ 7.31805276e+03,  7.31805276e+03,  7.31805276e+03,
                            -7.50737618e-01, -2.43098121e-01,  4.00000000e+00,
                             6.20882352e-01, -1.26286725e+00,  1.04283370e+00,
                             8.42872836e-01],
                           [ 7.31805276e+03,  7.31805276e+03,  7.31805276e+03,
                             7.31805276e+03, -7.50737618e-01,  4.00000000e+00,
                            -6.04449469e-01, -1.26286725e+00,  1.00599776e+00,
                            -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_series():
    """
    Test create_predict_X output when using LinearRegression as regressor and differentiation=1,
    and transformer_series is StandardScaler.
    """
    end_train = '2003-01-30 23:59:00'

    # Data scaled and differentiated
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2003-01-01', periods=len(series), freq='D')
    series_dict_datetime = {
        "1": series_datetime['1'].loc[:end_train],
        "2": series_datetime['2'].loc[:end_train]
    }
    
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(series)), name='exog'
    )
    exog.index = pd.date_range(start='2003-01-01', periods=len(series), freq='D')
    exog_dict_datetime = {
        "1": exog.loc[:end_train],
        "2": exog.loc[:end_train]
    }
    exog_pred = {
        '1': exog.loc[end_train:],
        '2': exog.loc[end_train:]
    }

    steps = len(series_datetime.loc[end_train:])

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(), 
                     lags               = 3, 
                     transformer_series = StandardScaler(),    
                     differentiation    = 1
                 )
    forecaster.fit(series=series_dict_datetime, exog=exog_dict_datetime)
    results = forecaster.create_predict_X(steps=steps, exog=exog_pred)

    expected = {
        '1': pd.DataFrame(
                 data = np.array([
                     [ 1.51888875,  0.2947634 , -0.60134326,  0.        , -0.78916779],
                     [ 0.459864  ,  1.51888875,  0.2947634 ,  0.        ,  0.07769117],
                     [ 0.39973381,  0.459864  ,  1.51888875,  0.        ,  0.27283009],
                     [ 0.44179464,  0.39973381,  0.459864  ,  0.        ,  0.35075195],
                     [ 0.49113681,  0.44179464,  0.39973381,  0.        ,  0.31597336],
                     [ 0.44897147,  0.49113681,  0.44179464,  0.        ,  0.14091339],
                     [ 0.45293977,  0.44897147,  0.49113681,  0.        ,  1.28618144],
                     [ 0.39137413,  0.45293977,  0.44897147,  0.        ,  0.09164865],
                     [ 0.46273278,  0.39137413,  0.45293977,  0.        , -0.50744682],
                     [ 0.4989439 ,  0.46273278,  0.39137413,  0.        ,  0.01522573],
                     [ 0.47172223,  0.4989439 ,  0.46273278,  0.        ,  0.82813767],
                     [ 0.41566484,  0.47172223,  0.4989439 ,  0.        ,  0.15290495],
                     [ 0.4727283 ,  0.41566484,  0.47172223,  0.        ,  0.61532804],
                     [ 0.44813237,  0.4727283 ,  0.41566484,  0.        , -0.06287136],
                     [ 0.47772115,  0.44813237,  0.4727283 ,  0.        , -1.1896156 ],
                     [ 0.55631152,  0.47772115,  0.44813237,  0.        ,  0.28674823],
                     [ 0.49080023,  0.55631152,  0.47772115,  0.        , -0.64581528],
                     [ 0.55415958,  0.49080023,  0.55631152,  0.        ,  0.20879998],
                     [ 0.52705955,  0.55415958,  0.49080023,  0.        ,  1.80029302],
                     [ 0.42909467,  0.52705955,  0.55415958,  0.        ,  0.14269745]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog'], 
                 index = pd.date_range(start='2003-01-31', periods=steps, freq='D')
             ),
        '2': pd.DataFrame(
                 data = np.array([
                     [ 2.09399597, -0.02273313, -1.53568303,  1.        , -0.78916779],
                     [ 0.48504726,  2.09399597, -0.02273313,  1.        ,  0.07769117],
                     [ 0.34947472,  0.48504726,  2.09399597,  1.        ,  0.27283009],
                     [ 0.39318927,  0.34947472,  0.48504726,  1.        ,  0.35075195],
                     [ 0.47754113,  0.39318927,  0.34947472,  1.        ,  0.31597336],
                     [ 0.41441454,  0.47754113,  0.39318927,  1.        ,  0.14091339],
                     [ 0.40016222,  0.41441454,  0.47754113,  1.        ,  1.28618144],
                     [ 0.32596074,  0.40016222,  0.41441454,  1.        ,  0.09164865],
                     [ 0.4092334 ,  0.32596074,  0.40016222,  1.        , -0.50744682],
                     [ 0.44214484,  0.4092334 ,  0.32596074,  1.        ,  0.01522573],
                     [ 0.40184777,  0.44214484,  0.4092334 ,  1.        ,  0.82813767],
                     [ 0.33172964,  0.40184777,  0.44214484,  1.        ,  0.15290495],
                     [ 0.39305831,  0.33172964,  0.40184777,  1.        ,  0.61532804],
                     [ 0.35718608,  0.39305831,  0.33172964,  1.        , -0.06287136],
                     [ 0.38513205,  0.35718608,  0.39305831,  1.        , -1.1896156 ],
                     [ 0.47101043,  0.38513205,  0.35718608,  1.        ,  0.28674823],
                     [ 0.38843927,  0.47101043,  0.38513205,  1.        , -0.64581528],
                     [ 0.45586888,  0.38843927,  0.47101043,  1.        ,  0.20879998],
                     [ 0.41797769,  0.45586888,  0.38843927,  1.        ,  1.80029302],
                     [ 0.29753778,  0.41797769,  0.45586888,  1.        ,  0.14269745]]),
                 columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog'], 
                 index = pd.date_range(start='2003-01-31', periods=steps, freq='D')
             )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict_encoding_None():
    """
    Test output ForecasterAutoregMultiSeries create_predict_X method when 
    series and exog are dictionaries and encoding is None.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = None,
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler()
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    results = forecaster.create_predict_X(
        steps=5, exog=exog_dict_test, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([
                                [-9.83270685e-01, -8.82486054e-01, -7.98426315e-01,
                                    -7.92779121e-01, -7.77683073e-01,  8.21644127e-03,
                                    1.42962482e+00,             np.nan,             np.nan],
                                [ 2.84684766e+03, -9.83270685e-01, -8.82486054e-01,
                                    -7.98426315e-01, -7.92779121e-01,  1.11220226e+00,
                                    8.96343746e-01,             np.nan,             np.nan],
                                [ 2.84684766e+03,  2.84684766e+03, -9.83270685e-01,
                                    -8.82486054e-01, -7.98426315e-01,  1.38486425e+00,
                                    -3.01927953e-01,             np.nan,            np.nan],
                                [ 2.95894007e+03,  2.84684766e+03,  2.84684766e+03,
                                    -9.83270685e-01, -8.82486054e-01,  6.20882352e-01,
                                    -1.26286725e+00,             np.nan,             np.nan],
                                [ 2.95894007e+03,  2.95894007e+03,  2.84684766e+03,
                                    2.84684766e+03, -9.83270685e-01, -6.04449469e-01,
                                    -1.26286725e+00,             np.nan,             np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([[-1.00754379e-01, -9.72790502e-02, -2.50193273e-01,
                                            -3.47936948e-01, -3.95722853e-01,  8.21644127e-03,
                                            1.42962482e+00,  1.11141113e+00, -8.79435260e-01],
                                        [ 3.16462868e+03, -1.00754379e-01, -9.72790502e-02,
                                            -2.50193273e-01, -3.47936948e-01,  1.11220226e+00,
                                            8.96343746e-01,  1.13275580e+00,  5.89479621e-03],
                                        [ 2.95894007e+03,  3.16462868e+03, -1.00754379e-01,
                                            -9.72790502e-02, -2.50193273e-01,  1.38486425e+00,
                                            -3.01927953e-01,  1.17758690e+00, -3.53258397e-01],
                                        [ 3.16462868e+03,  2.95894007e+03,  3.16462868e+03,
                                            -1.00754379e-01, -9.72790502e-02,  6.20882352e-01,
                                            -1.26286725e+00,  1.04283370e+00,  8.42872836e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  2.95894007e+03,
                                            3.16462868e+03, -1.00754379e-01, -6.04449469e-01,
                                            -1.26286725e+00,  1.00599776e+00, -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([[ 2.27390539e-01, -5.17576412e-01, -6.47683828e-01,
                                            -5.90232336e-01, -5.66504184e-01,  8.21644127e-03,
                                            np.nan,  1.11141113e+00, -8.79435260e-01],
                                        [ 2.84684766e+03,  2.27390539e-01, -5.17576412e-01,
                                            -6.47683828e-01, -5.90232336e-01,  1.11220226e+00,
                                            np.nan,  1.13275580e+00,  5.89479621e-03],
                                        [ 3.16462868e+03,  2.84684766e+03,  2.27390539e-01,
                                            -5.17576412e-01, -6.47683828e-01,  1.38486425e+00,
                                            np.nan,  1.17758690e+00, -3.53258397e-01],
                                        [ 2.95894007e+03,  3.16462868e+03,  2.84684766e+03,
                                            2.27390539e-01, -5.17576412e-01,  6.20882352e-01,
                                            np.nan,  1.04283370e+00,  8.42872836e-01],
                                        [ 3.16462868e+03,  2.95894007e+03,  3.16462868e+03,
                                            2.84684766e+03,  2.27390539e-01, -6.04449469e-01,
                                            np.nan,  1.00599776e+00, -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([[ 1.22899343e+00,  1.65819658e+00,  2.43189260e+00,
                                            2.56743042e+00,  2.38367241e+00,  8.21644127e-03,
                                            1.42962482e+00,  1.11141113e+00, -8.79435260e-01],
                                        [ 3.42642783e+03,  1.22899343e+00,  1.65819658e+00,
                                            2.43189260e+00,  2.56743042e+00,  1.11220226e+00,
                                            8.96343746e-01,  1.13275580e+00,  5.89479621e-03],
                                        [ 3.16462868e+03,  3.42642783e+03,  1.22899343e+00,
                                            1.65819658e+00,  2.43189260e+00,  1.38486425e+00,
                                            -3.01927953e-01,  1.17758690e+00, -3.53258397e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  3.42642783e+03,
                                            1.22899343e+00,  1.65819658e+00,  6.20882352e-01,
                                            -1.26286725e+00,  1.04283370e+00,  8.42872836e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  3.16462868e+03,
                                            3.42642783e+03,  1.22899343e+00, -6.04449469e-01,
                                            -1.26286725e+00,  1.00599776e+00, -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict_unknown_level():
    """
    Test output ForecasterAutoregMultiSeries create_predict_X method when 
    series and exog are dictionaries and unknown level.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler()
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    results = forecaster.create_predict_X(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([
                           [-1.77580056e+00, -1.07810218e+00, -4.96184646e-01,
                            -4.57091003e-01, -3.52586104e-01,  0.00000000e+00,
                             8.21644127e-03,  1.42962482e+00,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03, -1.77580056e+00, -1.07810218e+00,
                            -4.96184646e-01, -4.57091003e-01,  0.00000000e+00,
                             1.11220226e+00,  8.96343746e-01,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03,  1.44369849e+03, -1.77580056e+00,
                            -1.07810218e+00, -4.96184646e-01,  0.00000000e+00,
                             1.38486425e+00, -3.01927953e-01,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03,  1.44369849e+03,  1.44369849e+03,
                            -1.77580056e+00, -1.07810218e+00,  0.00000000e+00,
                             6.20882352e-01, -1.26286725e+00,          np.nan,
                                     np.nan],
                           [ 1.44369849e+03,  1.44369849e+03,  1.44369849e+03,
                             1.44369849e+03, -1.77580056e+00,  0.00000000e+00,
                            -6.04449469e-01, -1.26286725e+00,          np.nan,
                                     np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([
                           [ 1.15231370e+00,  1.16332175e+00,  6.78968143e-01,
                             3.69366454e-01,  2.18005286e-01,  1.00000000e+00,
                             8.21644127e-03,  1.42962482e+00,  1.11141113e+00,
                            -8.79435260e-01],
                           [ 2.14116223e+03,  1.15231370e+00,  1.16332175e+00,
                             6.78968143e-01,  3.69366454e-01,  1.00000000e+00,
                             1.11220226e+00,  8.96343746e-01,  1.13275580e+00,
                             5.89479621e-03],
                           [ 2.02558287e+03,  2.14116223e+03,  1.15231370e+00,
                             1.16332175e+00,  6.78968143e-01,  1.00000000e+00,
                             1.38486425e+00, -3.01927953e-01,  1.17758690e+00,
                            -3.53258397e-01],
                           [ 2.02558287e+03,  2.02558287e+03,  2.14116223e+03,
                             1.15231370e+00,  1.16332175e+00,  1.00000000e+00,
                             6.20882352e-01, -1.26286725e+00,  1.04283370e+00,
                             8.42872836e-01],
                           [ 2.02558287e+03,  2.02558287e+03,  2.02558287e+03,
                             2.14116223e+03,  1.15231370e+00,  1.00000000e+00,
                            -6.04449469e-01, -1.26286725e+00,  1.00599776e+00,
                            -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([
                           [ 3.39980134e+00, -3.78377384e-01, -1.03823091e+00,
                            -7.46859590e-01, -6.26519755e-01,  3.00000000e+00,
                             8.21644127e-03,          np.nan,  1.11141113e+00,
                            -8.79435260e-01],
                           [ 2.20078503e+03,  3.39980134e+00, -3.78377384e-01,
                            -1.03823091e+00, -7.46859590e-01,  3.00000000e+00,
                             1.11220226e+00,          np.nan,  1.13275580e+00,
                             5.89479621e-03],
                           [ 2.13316531e+03,  2.20078503e+03,  3.39980134e+00,
                            -3.78377384e-01, -1.03823091e+00,  3.00000000e+00,
                             1.38486425e+00,          np.nan,  1.17758690e+00,
                            -3.53258397e-01],
                           [ 2.09026933e+03,  2.13316531e+03,  2.20078503e+03,
                             3.39980134e+00, -3.78377384e-01,  3.00000000e+00,
                             6.20882352e-01,          np.nan,  1.04283370e+00,
                             8.42872836e-01],
                           [ 2.09026933e+03,  2.09026933e+03,  2.13316531e+03,
                             2.20078503e+03,  3.39980134e+00,  3.00000000e+00,
                            -6.04449469e-01,          np.nan,  1.00599776e+00,
                            -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([
                           [-7.50737618e-01, -2.43098121e-01,  6.71989902e-01,
                             8.32297096e-01,  6.14957530e-01,  4.00000000e+00,
                             8.21644127e-03,  1.42962482e+00,  1.11141113e+00,
                            -8.79435260e-01],
                           [ 7.31805276e+03, -7.50737618e-01, -2.43098121e-01,
                             6.71989902e-01,  8.32297096e-01,  4.00000000e+00,
                             1.11220226e+00,  8.96343746e-01,  1.13275580e+00,
                             5.89479621e-03],
                           [ 7.31805276e+03,  7.31805276e+03, -7.50737618e-01,
                            -2.43098121e-01,  6.71989902e-01,  4.00000000e+00,
                             1.38486425e+00, -3.01927953e-01,  1.17758690e+00,
                            -3.53258397e-01],
                           [ 7.31805276e+03,  7.31805276e+03,  7.31805276e+03,
                            -7.50737618e-01, -2.43098121e-01,  4.00000000e+00,
                             6.20882352e-01, -1.26286725e+00,  1.04283370e+00,
                             8.42872836e-01],
                           [ 7.31805276e+03,  7.31805276e+03,  7.31805276e+03,
                             7.31805276e+03, -7.50737618e-01,  4.00000000e+00,
                            -6.04449469e-01, -1.26286725e+00,  1.00599776e+00,
                            -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1005': pd.DataFrame(
                       data = np.array([
                           [1.22899343e+00, 1.65819658e+00, 2.43189260e+00, 2.56743042e+00,
                            2.38367241e+00,         np.nan,         np.nan,         np.nan,
                                    np.nan,         np.nan],
                            [3.37437763e+03, 1.22899343e+00, 1.65819658e+00, 2.43189260e+00,
                             2.56743042e+00,         np.nan,         np.nan,         np.nan,
                                     np.nan,         np.nan],
                            [3.00828166e+03, 3.37437763e+03, 1.22899343e+00, 1.65819658e+00,
                             2.43189260e+00,         np.nan,         np.nan,         np.nan,
                                     np.nan,         np.nan],
                            [3.00828166e+03, 3.00828166e+03, 3.37437763e+03, 1.22899343e+00,
                             1.65819658e+00,         np.nan,         np.nan,         np.nan,
                                     np.nan,         np.nan],
                            [3.00828166e+03, 3.00828166e+03, 3.00828166e+03, 3.37437763e+03,
                             1.22899343e+00,         np.nan,         np.nan,         np.nan,
                                     np.nan,         np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  '_level_skforecast', 'sin_day_of_week', 
                                  'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_create_predict_X_output_when_series_and_exog_dict_encoding_None_unknown_level():
    """
    Test output ForecasterAutoregMultiSeries create_predict_X method when 
    series and exog are dictionaries and encoding is None and unknown level.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = None,
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler()
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1004']
    results = forecaster.create_predict_X(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_test_2, suppress_warnings=True
    )

    expected = {
        'id_1000': pd.DataFrame(
                       data = np.array([[-9.83270685e-01, -8.82486054e-01, -7.98426315e-01,
                                    -7.92779121e-01, -7.77683073e-01,  8.21644127e-03,
                                    1.42962482e+00,             np.nan,             np.nan],
                                [ 2.84684766e+03, -9.83270685e-01, -8.82486054e-01,
                                    -7.98426315e-01, -7.92779121e-01,  1.11220226e+00,
                                    8.96343746e-01,             np.nan,             np.nan],
                                [ 2.84684766e+03,  2.84684766e+03, -9.83270685e-01,
                                    -8.82486054e-01, -7.98426315e-01,  1.38486425e+00,
                                    -3.01927953e-01,             np.nan,            np.nan],
                                [ 2.95894007e+03,  2.84684766e+03,  2.84684766e+03,
                                    -9.83270685e-01, -8.82486054e-01,  6.20882352e-01,
                                    -1.26286725e+00,             np.nan,             np.nan],
                                [ 2.95894007e+03,  2.95894007e+03,  2.84684766e+03,
                                    2.84684766e+03, -9.83270685e-01, -6.04449469e-01,
                                    -1.26286725e+00,             np.nan,             np.nan]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1001': pd.DataFrame(
                       data = np.array([[-1.00754379e-01, -9.72790502e-02, -2.50193273e-01,
                                            -3.47936948e-01, -3.95722853e-01,  8.21644127e-03,
                                            1.42962482e+00,  1.11141113e+00, -8.79435260e-01],
                                        [ 3.16462868e+03, -1.00754379e-01, -9.72790502e-02,
                                            -2.50193273e-01, -3.47936948e-01,  1.11220226e+00,
                                            8.96343746e-01,  1.13275580e+00,  5.89479621e-03],
                                        [ 2.95894007e+03,  3.16462868e+03, -1.00754379e-01,
                                            -9.72790502e-02, -2.50193273e-01,  1.38486425e+00,
                                            -3.01927953e-01,  1.17758690e+00, -3.53258397e-01],
                                        [ 3.16462868e+03,  2.95894007e+03,  3.16462868e+03,
                                            -1.00754379e-01, -9.72790502e-02,  6.20882352e-01,
                                            -1.26286725e+00,  1.04283370e+00,  8.42872836e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  2.95894007e+03,
                                            3.16462868e+03, -1.00754379e-01, -6.04449469e-01,
                                            -1.26286725e+00,  1.00599776e+00, -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1003': pd.DataFrame(
                       data = np.array([[ 2.27390539e-01, -5.17576412e-01, -6.47683828e-01,
                                            -5.90232336e-01, -5.66504184e-01,  8.21644127e-03,
                                            np.nan,  1.11141113e+00, -8.79435260e-01],
                                        [ 2.84684766e+03,  2.27390539e-01, -5.17576412e-01,
                                            -6.47683828e-01, -5.90232336e-01,  1.11220226e+00,
                                            np.nan,  1.13275580e+00,  5.89479621e-03],
                                        [ 3.16462868e+03,  2.84684766e+03,  2.27390539e-01,
                                            -5.17576412e-01, -6.47683828e-01,  1.38486425e+00,
                                            np.nan,  1.17758690e+00, -3.53258397e-01],
                                        [ 2.95894007e+03,  3.16462868e+03,  2.84684766e+03,
                                            2.27390539e-01, -5.17576412e-01,  6.20882352e-01,
                                            np.nan,  1.04283370e+00,  8.42872836e-01],
                                        [ 3.16462868e+03,  2.95894007e+03,  3.16462868e+03,
                                            2.84684766e+03,  2.27390539e-01, -6.04449469e-01,
                                            np.nan,  1.00599776e+00, -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1004': pd.DataFrame(
                       data = np.array([[ 1.22899343e+00,  1.65819658e+00,  2.43189260e+00,
                                            2.56743042e+00,  2.38367241e+00,  8.21644127e-03,
                                            1.42962482e+00,  1.11141113e+00, -8.79435260e-01],
                                        [ 3.42642783e+03,  1.22899343e+00,  1.65819658e+00,
                                            2.43189260e+00,  2.56743042e+00,  1.11220226e+00,
                                            8.96343746e-01,  1.13275580e+00,  5.89479621e-03],
                                        [ 3.16462868e+03,  3.42642783e+03,  1.22899343e+00,
                                            1.65819658e+00,  2.43189260e+00,  1.38486425e+00,
                                            -3.01927953e-01,  1.17758690e+00, -3.53258397e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  3.42642783e+03,
                                            1.22899343e+00,  1.65819658e+00,  6.20882352e-01,
                                            -1.26286725e+00,  1.04283370e+00,  8.42872836e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  3.16462868e+03,
                                            3.42642783e+03,  1.22899343e+00, -6.04449469e-01,
                                            -1.26286725e+00,  1.00599776e+00, -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   ),
        'id_1005': pd.DataFrame(
                       data = np.array([[ 1.22899343e+00,  1.65819658e+00,  2.43189260e+00,
                                            2.56743042e+00,  2.38367241e+00,  8.21644127e-03,
                                            1.42962482e+00,  1.11141113e+00, -8.79435260e-01],
                                        [ 3.42642783e+03,  1.22899343e+00,  1.65819658e+00,
                                            2.43189260e+00,  2.56743042e+00,  1.11220226e+00,
                                            8.96343746e-01,  1.13275580e+00,  5.89479621e-03],
                                        [ 3.16462868e+03,  3.42642783e+03,  1.22899343e+00,
                                            1.65819658e+00,  2.43189260e+00,  1.38486425e+00,
                                            -3.01927953e-01,  1.17758690e+00, -3.53258397e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  3.42642783e+03,
                                            1.22899343e+00,  1.65819658e+00,  6.20882352e-01,
                                            -1.26286725e+00,  1.04283370e+00,  8.42872836e-01],
                                        [ 3.16462868e+03,  3.16462868e+03,  3.16462868e+03,
                                            3.42642783e+03,  1.22899343e+00, -6.04449469e-01,
                                            -1.26286725e+00,  1.00599776e+00, -6.23146331e-01]]),
                       columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                  'sin_day_of_week', 'cos_day_of_week', 'air_temperature', 'wind_speed'],
                       index = pd.date_range(start='2016-08-01', periods=5, freq='D')
                   )
    }

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])
