# Unit test create_predict_inputs ForecasterAutoregMultiSeriesCustom
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

from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom

# Fixtures
from .fixtures_ForecasterAutoregMultiSeriesCustom import series
from .fixtures_ForecasterAutoregMultiSeriesCustom import exog
from .fixtures_ForecasterAutoregMultiSeriesCustom import exog_predict

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

def create_predictors_5(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]

    return lags


def create_predictors_15(y): # pragma: no cover
    """
    Create first 15 lags of a time series.
    """
    lags = y[-1:-16:-1]

    return lags


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_5,
                     window_size    = 5
                 )

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.create_predict_inputs(steps=5)


def test_output_create_predict_inputs_when_regressor_is_LinearRegression():
    """
    Test output create_predict_inputs when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_5,
                     window_size        = 5,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2)
    results = forecaster.create_predict_inputs(steps=5)

    expected = (
        {'1': pd.Series(
                  data = np.array([45., 46., 47., 48., 49.]),
                  index = pd.RangeIndex(start=45, stop=50, step=1),
                  name = '1'
              ),
         '2': pd.Series(
                  data = np.array([95., 96., 97., 98., 99.]),
                  index = pd.RangeIndex(start=45, stop=50, step=1),
                  name = '2'
              )
        },
        {'1': None, '2': None},
        ['1', '2'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    for k in expected[0].keys():
        pd.testing.assert_series_equal(results[0][k], expected[0][k])
    for k in expected[1].keys():
        assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test create_predict_inputs output when using LinearRegression as regressor 
    and StandardScaler.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_5,
                     window_size        = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster.create_predict_inputs(steps=5, levels='1')
    
    expected = (
        {'1': pd.Series(
                  data = np.array([-1.0814183452563133, -0.08097053361357116, 2.0806640273868244, 
                                   0.07582436346198025, 0.4776288428854555]),
                  index = pd.RangeIndex(start=45, stop=50, step=1),
                  name = '1'
              )
        },
        {'1': None},
        ['1'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    for k in expected[0].keys():
        pd.testing.assert_series_equal(results[0][k], expected[0][k])
    for k in expected[1].keys():
        assert results[1] == expected[1]
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog(transformer_series):
    """
    Test create_predict_inputs output when using LinearRegression as regressor, 
    StandardScaler as transformer_series and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_5,
                     window_size        = 5,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.create_predict_inputs(steps=5, levels='1', 
                                               exog=exog_predict, output_type='numpy')
    
    expected = (
        {'1': np.array([-1.0814183452563133, -0.08097053361357116, 2.0806640273868244, 
                         0.07582436346198025, 0.4776288428854555])
        },
        {'1': np.array([[-0.09362908,  1., 0.],
                        [ 0.45144522,  1., 0.],
                        [-1.53968887,  1., 0.],
                        [-1.45096055,  1., 0.],
                        [-0.77240468,  1., 0.]])
        },
        ['1'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    for k in expected[0].keys():
        np.testing.assert_array_almost_equal(results[0][k], expected[0][k])
    for k in expected[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test create_predict_inputs output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog with series 
    of different lengths.
    """
    new_series = series.copy()
    new_series['2'].iloc[:10] = np.nan

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_5,
                     window_size        = 5,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=new_series, exog=exog)
    results = forecaster.create_predict_inputs(steps=5, exog=exog_predict)
    
    expected = (
        {'1': pd.Series(
                  data = np.array([-1.08141835, -0.08097053, 2.08066403, 0.07582436, 0.47762884]),
                  index = pd.RangeIndex(start=45, stop=50, step=1),
                  name = '1'
              ),
         '2': pd.Series(
                  data = np.array([0.73973379, -1.29936753, -0.36483376, -0.96090271, -0.57388468]),
                  index = pd.RangeIndex(start=45, stop=50, step=1),
                  name = '2'
              )
        },
        {'1': pd.DataFrame({
                  'exog_1': [-0.09575486, 0.44024531, -1.51774136, -1.43049014, -0.76323054],
                  'exog_2_a': [1., 1., 1., 1., 1.],
                  'exog_2_b': [0., 0., 0., 0., 0.]},
                  index=pd.RangeIndex(start=50, stop=55, step=1)
              )
        },
        ['1', '2'],
        pd.RangeIndex(start=50, stop=55, step=1),
        None
    )

    for k in expected[0].keys():
        pd.testing.assert_series_equal(results[0][k], expected[0][k])
    for k in expected[1].keys():
        pd.testing.assert_frame_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test create_predict_inputs output when using HistGradientBoostingRegressor and categorical variables.
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
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = HistGradientBoostingRegressor(
                                              categorical_features = categorical_features,
                                              random_state         = 123
                                          ),
                     fun_predictors     = create_predictors_5,
                     window_size        = 5,
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=df_exog)
    results = forecaster.create_predict_inputs(steps=10, exog=exog_predict)
    
    expected = (
        {'1': pd.Series(
                  data = np.array([0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
                  index = pd.RangeIndex(start=45, stop=50, step=1),
                  name = '1'
              ),
         '2': pd.Series(
                  data = np.array([0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]),
                  index = pd.RangeIndex(start=45, stop=50, step=1),
                  name = '2'
              )
        },
        {'1': pd.DataFrame({
                  'exog_2': np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int),
                  'exog_3': np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int),
                  'exog_1': [0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061,
                             0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781]},
                  index=pd.RangeIndex(start=50, stop=60, step=1)
              ),
         '2': pd.DataFrame({
                  'exog_2': np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int),
                  'exog_3': np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int),
                  'exog_1': [0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061,
                             0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781]},
                  index=pd.RangeIndex(start=50, stop=60, step=1)
              )
        },
        ['1', '2'],
        pd.RangeIndex(start=50, stop=60, step=1),
        None
    )

    for k in expected[0].keys():
        pd.testing.assert_series_equal(results[0][k], expected[0][k])
    for k in expected[1].keys():
        pd.testing.assert_frame_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_series_and_exog_dict():
    """
    Test output ForecasterAutoregMultiSeries create_predict_inputs method when 
    series and exog are dictionaries.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        fun_predictors     = create_predictors_5,
        window_size        = 5,
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    results = forecaster.create_predict_inputs(
        steps=5, exog=exog_dict_test, suppress_warnings=True
    )
    
    expected = (
        {'id_1000': pd.Series(
                        data  = np.array([-0.3525861, -0.457091, -0.49618465, -1.07810218, -1.77580056]),
                        index = pd.date_range(start='2016-07-27', periods=5, freq='D'),
                        name  = 'id_1000'
                    ),
         'id_1001': pd.Series(
                        data  = np.array([0.21800529, 0.36936645, 0.67896814, 1.16332175, 1.1523137]),
                        index = pd.date_range(start='2016-07-27', periods=5, freq='D'),
                        name  = 'id_1001'
                    ),
         'id_1003': pd.Series(
                        data  = np.array([-0.62651976, -0.74685959, -1.03823091, -0.37837738, 3.39980134]),
                        index = pd.date_range(start='2016-07-27', periods=5, freq='D'),
                        name  = 'id_1003'
                    ),
         'id_1004': pd.Series(
                        data  = np.array([ 0.61495753, 0.8322971, 0.6719899 , -0.24309812, -0.75073762]),
                        index = pd.date_range(start='2016-07-27', periods=5, freq='D'),
                        name  = 'id_1004'
                    )
        },
        {'id_1000': pd.DataFrame({
                        'sin_day_of_week': [0.00821644, 1.11220226, 1.38486425, 0.62088235, -0.60444947],
                        'cos_day_of_week': [1.42962482, 0.89634375, -0.30192795, -1.26286725, -1.26286725],
                        'air_temperature': 5*[np.nan],
                        'wind_speed': 5*[np.nan]},
                        index=pd.date_range(start='2016-08-01', periods=5, freq='D')
                    ),
         'id_1001': pd.DataFrame({
                        'sin_day_of_week': [0.00821644, 1.11220226, 1.38486425, 0.62088235, -0.60444947],
                        'cos_day_of_week': [1.42962482, 0.89634375, -0.30192795, -1.26286725, -1.26286725],
                        'air_temperature': [1.11141113002944, 1.1327558043882977, 1.1775868983168865, 1.0428336959958195, 1.0059977577330665],
                        'wind_speed': [-0.8794352596577425, 0.005894796212821553, -0.35325839699185807, 0.842872836428647, -0.6231463305262476]},
                        index=pd.date_range(start='2016-08-01', periods=5, freq='D')
                    ),
         'id_1003': pd.DataFrame({
                        'sin_day_of_week': [0.00821644, 1.11220226, 1.38486425, 0.62088235, -0.60444947],
                        'cos_day_of_week': 5*[np.nan],
                        'air_temperature': [1.11141113002944, 1.1327558043882977, 1.1775868983168865, 1.0428336959958195, 1.0059977577330665],
                        'wind_speed': [-0.8794352596577425, 0.005894796212821553, -0.35325839699185807, 0.842872836428647, -0.6231463305262476]},
                        index=pd.date_range(start='2016-08-01', periods=5, freq='D')
                    ),
         'id_1004': pd.DataFrame({
                        'sin_day_of_week': [0.00821644, 1.11220226, 1.38486425, 0.62088235, -0.60444947],
                        'cos_day_of_week': [1.42962482, 0.89634375, -0.30192795, -1.26286725, -1.26286725],
                        'air_temperature': [1.11141113002944, 1.1327558043882977, 1.1775868983168865, 1.0428336959958195, 1.0059977577330665],
                        'wind_speed': [-0.8794352596577425, 0.005894796212821553, -0.35325839699185807, 0.842872836428647, -0.6231463305262476]},
                        index=pd.date_range(start='2016-08-01', periods=5, freq='D')
                    ),
        },
        ['id_1000', 'id_1001', 'id_1003', 'id_1004'],
        pd.date_range(start='2016-08-01', periods=5, freq='D'),
        None
    )

    for k in expected[0].keys():
        expected[0][k].index.name = 'timestamp'
        pd.testing.assert_series_equal(results[0][k], expected[0][k])
    for k in expected[1].keys():
        pd.testing.assert_frame_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    assert results[4] == expected[4]


def test_create_predict_inputs_output_when_regressor_is_LinearRegression_with_exog_differentiation_is_1_and_transformer_series():
    """
    Test create_predict_inputs output when using LinearRegression as regressor and differentiation=1,
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
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_15,
                     window_size        = 15,
                     transformer_series = StandardScaler(),
                     differentiation    = 1,
                 )
    forecaster.fit(series=series_dict_datetime, exog=exog_dict_datetime)
    results = forecaster.create_predict_inputs(steps=steps, exog=exog_pred,
                                               predict_boot=True)
    
    expected = (
        {'1': pd.Series(
                  data = np.array([np.nan, 1.53099926e+00, -2.50175863e+00, -3.17051107e-02,
                                   1.60372524e+00, 1.24398054e-03, 4.61947999e-01, 9.68409847e-01,
                                   -5.62842287e-01, -5.10849917e-01, 5.01788973e-01, -1.79911269e+00,
                                   1.74873121e-01, -6.01343264e-01, 2.94763400e-01, 1.51888875e+00]),
                  index = pd.date_range(start='2003-01-15', periods=16, freq='D'),
                  name = '1'
              ),
         '2': pd.Series(
                  data = np.array([np.nan, -2.93045496, 2.6268138, -2.00736345, -0.19086106,
                                   1.46005589, -1.8403251 , 3.04806289, -0.99623011, 0.3712246,
                                   -2.73026424, 2.23235938, -0.14532345, -1.53568303, -0.02273313,
                                   2.09399597]),
                  index = pd.date_range(start='2003-01-15', periods=16, freq='D'),
                  name = '2'
              )
        },
        {'1': pd.DataFrame({
                  'exog': np.array([
                              -0.78916779,  0.07769117,  0.27283009,  0.35075195,  0.31597336,
                              0.14091339,  1.28618144,  0.09164865, -0.50744682,  0.01522573,
                              0.82813767,  0.15290495,  0.61532804, -0.06287136, -1.1896156 ,
                              0.28674823, -0.64581528,  0.20879998,  1.80029302,  0.14269745
                          ])
                  },
                  index=pd.date_range(start='2003-01-31', periods=steps, freq='D')
              ),
         '2': pd.DataFrame({
                  'exog': np.array([
                              -0.78916779,  0.07769117,  0.27283009,  0.35075195,  0.31597336,
                              0.14091339,  1.28618144,  0.09164865, -0.50744682,  0.01522573,
                              0.82813767,  0.15290495,  0.61532804, -0.06287136, -1.1896156 ,
                              0.28674823, -0.64581528,  0.20879998,  1.80029302,  0.14269745
                          ])
                  },
                  index=pd.date_range(start='2003-01-31', periods=steps, freq='D')
              )
        },
        ['1', '2'],
        pd.date_range(start='2003-01-31', periods=steps, freq='D'),
        {'1': np.array([0.11250873, -0.45215401,  0.50931909, -1.2515849 ,  0.32346492,
                        -0.16591436,  0.61386783,  0.49003355,  0.32481062, -0.050827  ,
                        -0.46246282, -0.20198051, -0.22783566,  0.43875451]),
         '2': np.array([-0.11905336, -0.96448195,  0.20875269,  0.7334418 , -0.45287812,
                        0.20410656,  0.27482526,  0.12415931, -1.10570249, -0.08341689,
                        0.83327134,  0.37868888, -0.69352672,  0.66181368])}
    )

    for k in expected[0].keys():
        pd.testing.assert_series_equal(results[0][k], expected[0][k])
    for k in expected[1].keys():
        pd.testing.assert_frame_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    pd.testing.assert_index_equal(results[3], expected[3])
    for k in expected[4].keys():
        np.testing.assert_array_almost_equal(results[4][k], expected[4][k])

    