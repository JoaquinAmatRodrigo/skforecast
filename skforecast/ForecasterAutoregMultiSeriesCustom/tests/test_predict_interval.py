# Unit test predict_interval ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
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
series_2 = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                         '2': pd.Series(np.arange(10))})

def create_predictors(y):  # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags

def create_predictors_14_lags(y):  # pragma: no cover
    """
    Create first 14 lags of a time series.
    """
    lags = y[-1:-15:-1]

    return lags


@pytest.fixture(params=[('1'  , np.array([[10., 20., 20.]])), 
                        (['2'], np.array([[10., 30., 30.]])),
                        (['1', '2'], np.array([[10., 20., 20., 10., 30., 30.]]))
                        ],
                        ids=lambda d: f'levels: {d[0]}, preds: {d[1]}')
def expected_pandas_dataframe(request):
    """
    This is a pytest fixture. It's a function that can be passed to a
    test so that we have a single block of code that can generate testing
    examples.

    We're using `params` in the call to declare that we want multiple versions
    to be generated. This is similar to the parametrize decorator, but it's difference
    because we can re-use `pd.Series` in multiple tests.
    """
    levels = request.param[0]
    levels_names = [levels] if isinstance(levels, str) else levels

    column_names = []
    for level in levels_names:
        _ = [f'{level}', f'{level}_lower_bound', f'{level}_upper_bound']
        column_names.append(_)
        
    expected_df = pd.DataFrame(
                      data    = request.param[1],
                      columns = np.concatenate(column_names),
                      index   = pd.RangeIndex(start=10, stop=11, step=1)
                  )

    return levels, expected_df


def test_predict_output_when_regressor_is_LinearRegression_steps_is_1_in_sample_residuals_is_True_with_fixture(expected_pandas_dataframe):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals. This test is equivalent to the next one.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor           = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)

    predictions = forecaster.predict_interval(steps=1, levels=expected_pandas_dataframe[0], 
                                              in_sample_residuals=True, suppress_warnings=True)
    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor           = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = pd.DataFrame(
                    data = np.array([[10., 20., 20.]]),
                    columns = ['1', '1_lower_bound', '1_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_1 = forecaster.predict_interval(steps=1, levels='1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = pd.DataFrame(
                    data = np.array([[10., 30., 30.]]),
                    columns = ['2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_2 = forecaster.predict_interval(steps=1, levels=['2'], in_sample_residuals=True)

    expected_3 = pd.DataFrame(
                    data = np.array([[10., 20., 20., 10., 30., 30.]]),
                    columns = ['1', '1_lower_bound', '1_upper_bound', '2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_3 = forecaster.predict_interval(steps=1, levels=None, in_sample_residuals=True)

    pd.testing.assert_frame_equal(results_1, expected_1)
    pd.testing.assert_frame_equal(results_2, expected_2)
    pd.testing.assert_frame_equal(results_3, expected_3)


@pytest.fixture(params=[('1'  , np.array([[10., 20.        , 20.],
                                          [11., 24.33333333, 24.33333333]])), 
                        (['2'], np.array([[10., 30.              , 30.],
                                          [11., 37.66666666666667, 37.66666666666667]])),
                        (['1', '2'], np.array([[10., 20.        , 20.        , 10., 30.              , 30.],
                                               [11., 24.33333333, 24.33333333, 11., 37.66666666666667, 37.66666666666667]]))
                        ],
                        ids=lambda d: f'levels: {d[0]}, preds: {d[1]}')
def expected_pandas_dataframe_2(request):
    """
    This is a pytest fixture. It's a function that can be passed to a
    test so that we have a single block of code that can generate testing
    examples.

    We're using `params` in the call to declare that we want multiple versions
    to be generated. This is similar to the parametrize decorator, but it's difference
    because we can re-use `pd.Series` in multiple tests.
    """
    levels = request.param[0]
    levels_names = [levels] if isinstance(levels, str) else levels

    column_names = []
    for level in levels_names:
        _ = [f'{level}', f'{level}_lower_bound', f'{level}_upper_bound']
        column_names.append(_)
        
    expected_df = pd.DataFrame(
                      data    = request.param[1],
                      columns = np.concatenate(column_names),
                      index   = pd.RangeIndex(start=10, stop=12, step=1)
                  )

    return levels, expected_df


def test_predict_output_when_regressor_is_LinearRegression_steps_is_2_in_sample_residuals_is_True_with_fixture(expected_pandas_dataframe_2):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals. This test is equivalent to the next one.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor           = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)

    predictions = forecaster.predict_interval(steps=2, levels=expected_pandas_dataframe_2[0], in_sample_residuals=True)
    expected = expected_pandas_dataframe_2[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor           = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = pd.DataFrame(
                    np.array([[10., 20.        , 20.],
                              [11., 24.33333333, 24.33333333]
                             ]),
                    columns = ['1', '1_lower_bound', '1_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_1 = forecaster.predict_interval(steps=2, levels='1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = pd.DataFrame(
                    np.array([[10., 30.              , 30.],
                              [11., 37.66666666666667, 37.66666666666667]
                             ]),
                    columns = ['2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_2 = forecaster.predict_interval(steps=2, levels='2', in_sample_residuals=True)

    expected_3 = pd.DataFrame(
                    data = np.array([[10., 20.        , 20.        , 10., 30.              , 30.              ],
                                     [11., 24.33333333, 24.33333333, 11., 37.66666666666667, 37.66666666666667]]),
                    columns = ['1', '1_lower_bound', '1_upper_bound', '2', '2_lower_bound', '2_upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_3 = forecaster.predict_interval(steps=2, levels=['1', '2'], in_sample_residuals=True)

    pd.testing.assert_frame_equal(results_1, expected_1)
    pd.testing.assert_frame_equal(results_2, expected_2)
    pd.testing.assert_frame_equal(results_3, expected_3)


def test_predict_output_when_regressor_is_LinearRegression_steps_is_1_in_sample_residuals_is_False_with_fixture(expected_pandas_dataframe):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor           = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    residuals = {'1': np.full_like(forecaster.in_sample_residuals['1'], fill_value=10), 
                 '2': np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)}
    forecaster.set_out_sample_residuals(residuals=residuals)

    predictions = forecaster.predict_interval(steps=1, levels=expected_pandas_dataframe[0], in_sample_residuals=False)
    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_steps_is_2_in_sample_residuals_is_False_with_fixture(expected_pandas_dataframe_2):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals. This test is equivalent to the next one.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor           = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)

    residuals = {'1': np.full_like(forecaster.in_sample_residuals['1'], fill_value=10), 
                 '2': np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)}
    forecaster.set_out_sample_residuals(residuals=residuals)

    predictions = forecaster.predict_interval(steps=2, levels=expected_pandas_dataframe_2[0], in_sample_residuals=False)
    expected = expected_pandas_dataframe_2[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test predict_interval output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    predictions = forecaster.predict_interval(steps=5, levels='1')
    expected = pd.DataFrame(
                   data = np.array([[0.49300446, 0.1127329 , 0.8957496 ],
                                    [0.50358145, 0.06968912, 0.90112753],
                                    [0.50306954, 0.06633492, 0.88968806],
                                    [0.50796191, 0.13082997, 0.90510251],
                                    [0.50781568, 0.11536063, 0.93464907]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict_interval output when using LinearRegression as regressor and transformer_series
    is a dict with 2 different transformers.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = {'1': StandardScaler(), '2': MinMaxScaler(), '_unknown_level': StandardScaler()}
                 )
    forecaster.fit(series=series)
    predictions = forecaster.predict_interval(steps=5, levels=['1'])
    expected = pd.DataFrame(
                   data = np.array([[0.5567357 , 0.15168829, 0.93889146],
                                    [0.50694567, 0.08199289, 0.90753719],
                                    [0.51127353, 0.08587232, 0.90581568],
                                    [0.51327383, 0.14090782, 0.91811281],
                                    [0.50978027, 0.10608631, 0.93405271]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog():
    """
    Test predict_interval output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['exog_1']),
                             ('onehot', OneHotEncoder(), ['exog_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict_interval(steps=5, levels=['1', '2'], exog=exog_predict)
    expected = pd.DataFrame(
                   data = np.array([[0.50201669, 0.10891904, 0.90089875, 0.52531076, 0.10696675, 0.9156779 ],
                                    [0.49804821, 0.09776535, 0.88356798, 0.51683613, 0.08385857, 0.92154261],
                                    [0.59201747, 0.1880649 , 0.96148336, 0.61509778, 0.18270673, 1.02114822],
                                    [0.60179565, 0.216096  , 0.9880778 , 0.60446614, 0.18396533, 0.98997181],
                                    [0.56736867, 0.17944505, 0.97036278, 0.56524059, 0.13232921, 0.97872883]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound', '2', '2_lower_bound', '2_upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_series_and_exog_dict():
    """
    Test output ForecasterAutoregMultiSeries predict_interval method when series and 
    exog are dictionaries.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        fun_predictors=create_predictors_14_lags,
        window_size=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    predictions = forecaster.predict_interval(
        steps=5, exog=exog_dict_test, suppress_warnings=True, n_boot=10, interval=[5, 95]
    )
    expected = pd.DataFrame(
        data=np.array([
            [1438.14154717,  984.89302371, 1866.71388628, 2090.79352613,
            902.35566771, 2812.9492079 , 2166.9832933 , 1478.23569823,
            2519.27378966, 7285.52781428, 4241.35466074, 8500.84534052],
            [1438.14154717, 1368.43204887, 1893.86659752, 2089.11038884,
            900.67253042, 2812.29873034, 2074.55994929, 1414.59564291,
            3006.64848384, 7488.18398744, 5727.33006697, 8687.00097656],
            [1438.14154717, 1051.65779678, 1863.99279153, 2089.11038884,
            933.86328276, 2912.07131797, 2035.99448247, 1455.42927055,
            3112.84247337, 7488.18398744, 4229.48737525, 8490.6467214 ],
            [1403.93625654, 1130.69596556, 1740.72910642, 2089.11038884,
            900.67253042, 2912.07131797, 2035.99448247, 1522.39553008,
            1943.48913692, 7488.18398744, 5983.32345864, 9175.53503076],
            [1403.93625654, 1075.26038028, 1851.5525839 , 2089.11038884,
            933.86328276, 2909.31686272, 2035.99448247, 1732.93979512,
            2508.99760524, 7488.18398744, 5073.26873656, 8705.7520813 ]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=[
            'id_1000',
            'id_1000_lower_bound',
            'id_1000_upper_bound',
            'id_1001',
            'id_1001_lower_bound',
            'id_1001_upper_bound',
            'id_1003',
            'id_1003_lower_bound',
            'id_1003_upper_bound',
            'id_1004',
            'id_1004_lower_bound',
            'id_1004_upper_bound'
        ]
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_series_and_exog_dict_unknown_level():
    """
    Test output ForecasterAutoregMultiSeries predict_interval method when series and 
    exog are dictionaries and unknown_level.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LGBMRegressor(
                         n_estimators=30, random_state=123, verbose=-1, max_depth=4
                     ),
                     fun_predictors     = create_predictors_14_lags,
                     window_size        = 14,
                     encoding           = 'ordinal',
                     dropna_from_series = False,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                 )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    predictions = forecaster.predict_interval(
        steps=5, levels=levels, exog=exog_dict_test_2, last_window=last_window,
        suppress_warnings=True, n_boot=10, interval=[5, 95]
    )
    expected = pd.DataFrame(
        data=np.array([
            [1330.53853595, 1234.54638554, 1466.01101346, 2655.95253058,
             1806.16587168, 2980.80800441, 2645.09087689, 2436.33118532,
             2896.8958089 , 7897.51938494, 6475.8689303 , 8520.39336408,
             4890.22840888, 2714.04801977, 6113.89932062],
            [1401.63085157, 1297.33464356, 1604.49157285, 2503.75247961,
             1390.65424338, 2780.38027935, 2407.17525054, 2218.88594845,
             3064.37772365, 8577.09840856, 7096.41496521, 8763.40553896,
             4756.81020006, 3777.92035635, 5477.79419038],
            [1387.26572882, 1225.39562421, 1530.74271605, 2446.28038665,
             1274.09842569, 3059.12744785, 2314.08602238, 2067.62137868,
             2805.34679097, 8619.98311729, 6304.39688754, 9186.66428031,
             4947.44052717, 3826.16081797, 5253.17164831],
            [1310.82275942, 1228.56216954, 1517.4560568 , 2389.3764241 ,
             1613.21336702, 3084.63006301, 2245.05149747, 1759.35788172,
             2358.47886659, 8373.80334337, 7640.47624518, 8833.97112851,
             4972.50918694, 3180.5809587 , 7426.8270528 ],
            [1279.37274512, 1236.62366009, 1452.44787801, 2185.06104284,
             1240.87596129, 2946.07346052, 2197.45288166, 1622.570817  ,
             2074.46153095, 8536.31820994, 7357.82403861, 8997.25758537,
             5213.7612468 , 3840.69918874, 6307.8333356 ]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=[
            'id_1000',
            'id_1000_lower_bound',
            'id_1000_upper_bound',
            'id_1001',
            'id_1001_lower_bound',
            'id_1001_upper_bound',
            'id_1003',
            'id_1003_lower_bound',
            'id_1003_upper_bound',
            'id_1004',
            'id_1004_lower_bound',
            'id_1004_upper_bound',
            'id_1005',
            'id_1005_lower_bound',
            'id_1005_upper_bound'
        ]
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_series_and_exog_dict_encoding_None_unknown_level():
    """
    Test output ForecasterAutoregMultiSeries predict_interval method when series and 
    exog are dictionaries, encoding is None and unknown_level.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LGBMRegressor(
                         n_estimators=30, random_state=123, verbose=-1, max_depth=4
                     ),
                     fun_predictors     = create_predictors_14_lags,
                     window_size        = 14,
                     encoding           = None,
                     differentiation    = 1,
                     dropna_from_series = False,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                 )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004'] * 0.9
    exog_dict_test_2 = exog_dict_test.copy()
    exog_dict_test_2['id_1005'] = exog_dict_test_2['id_1001']
    predictions = forecaster.predict_interval(
        steps=5, levels=levels, exog=exog_dict_test_2, last_window=last_window,
        suppress_warnings=True, n_boot=10, interval=[5, 95]
    )
    expected = pd.DataFrame(
        data=np.array([
            [1261.93265537,  784.41616115, 1503.55050672, 3109.36774743,
             2631.85125321, 3350.98559877, 3565.43804407, 3087.92154985,
             3807.05589541, 7581.0124551 , 7103.49596088, 7822.63030645,
             6929.60563584, 6452.08914162, 7171.22348719],
            [1312.20749816,  637.9184331 , 1456.55230372, 3370.63276557,
             2696.34370051, 3514.97757113, 3486.84974947, 2795.52389009,
             3631.19455503, 7877.71418945, 7243.41987007, 8022.05899501,
             7226.30737019, 6592.01305081, 7370.65217575],
            [1269.60061174,  336.37560258, 1749.69442106, 3451.58214186,
             2469.45506604, 3906.21421331, 3265.50308765, 2392.65631312,
             3745.59689697, 7903.88998388, 6991.82265856, 8383.9837932 ,
             7211.07145676, 6299.00413144, 7691.16526608],
            [1216.71296132, -803.46566696, 2451.85576454, 3420.93162585,
             1399.08558266, 4684.62426649, 3279.93748551, 1338.36404222,
             4737.69647581, 7895.69977262, 5978.62410616, 9257.08365774,
             7260.90982474, 5309.65908504, 8564.26513062],
            [1199.80671909, -341.36057715, 2287.58311122, 3410.88134138,
             1737.91607302, 4481.53218781, 3385.66459202, 1835.69033892,
             4698.25050698, 7915.94534006, 6423.67830994, 9122.23338527,
             7281.15539217, 5770.52244805, 8429.41485815]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=[
            'id_1000',
            'id_1000_lower_bound',
            'id_1000_upper_bound',
            'id_1001',
            'id_1001_lower_bound',
            'id_1001_upper_bound',
            'id_1003',
            'id_1003_lower_bound',
            'id_1003_upper_bound',
            'id_1004',
            'id_1004_lower_bound',
            'id_1004_upper_bound',
            'id_1005',
            'id_1005_lower_bound',
            'id_1005_upper_bound'
        ]
    )

    pd.testing.assert_frame_equal(predictions, expected)