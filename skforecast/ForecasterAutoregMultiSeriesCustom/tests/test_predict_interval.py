# Unit test predict_interval ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

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

def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags

def create_predictors_14_lags(y): # pragma: no cover
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

    predictions = forecaster.predict_interval(steps=1, levels=expected_pandas_dataframe[0], in_sample_residuals=True)
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
                     transformer_series = {'1': StandardScaler(), '2': MinMaxScaler()}
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