# Unit test predict_interval ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# Fixtures
# np.random.seed(123)
# series_1 = np.random.rand(50)
# series_2 = np.random.rand(50)
# exog = np.random.rand(50)
series = pd.DataFrame({'1': pd.Series(np.array(
                                [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                                 0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                                 0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                                 0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                                 0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                                 0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                                 0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                                 0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                                 0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                                 0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]
                                )
                            ), 
                       '2': pd.Series(np.array(
                                [0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                                 0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                                 0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                                 0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                                 0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                                 0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                                 0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                                 0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                                 0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                                 0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]
                                )
                            )
                      }
         )
    
exog = pd.DataFrame({'col_1': pd.Series(np.array(
                                [0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061,
                                 0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781,
                                 0.3167879 , 0.35426468, 0.17108183, 0.82911263, 0.33867085,
                                 0.55237008, 0.57855147, 0.52153306, 0.00268806, 0.98834542,
                                 0.90534158, 0.20763586, 0.29248941, 0.52001015, 0.90191137,
                                 0.98363088, 0.25754206, 0.56435904, 0.80696868, 0.39437005,
                                 0.73107304, 0.16106901, 0.60069857, 0.86586446, 0.98352161,
                                 0.07936579, 0.42834727, 0.20454286, 0.45063649, 0.54776357,
                                 0.09332671, 0.29686078, 0.92758424, 0.56900373, 0.457412  ,
                                 0.75352599, 0.74186215, 0.04857903, 0.7086974 , 0.83924335]
                                )
                              ),
                     'col_2': ['a']*25 + ['b']*25}
       )


def test_predict_interval_exception_when_forecaster_in_sample_residuals_are_not_stored():
    """
    Test exception is raised when forecaster.in_sample_residuals are not stored 
    during fit method.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=False)

    err_msg = re.escape(
                ("`forecaster.in_sample_residuals['1']` contains `None` values. "
                 'Try using `fit` method with `in_sample_residuals=True` or set in '
                 '`predict_interval` method `in_sample_residuals=False` and use '
                 '`out_sample_residuals` (see `set_out_sample_residuals()`).')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=1, levels='1', in_sample_residuals=True)


def test_predict_interval_exception_when_out_sample_residuals_is_None():
    """
    Test exception is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
                ('`forecaster.out_sample_residuals` is `None`. Use '
                 '`in_sample_residuals=True` or method `set_out_sample_residuals()` '
                 'before `predict_interval()`.')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=1, levels='1', in_sample_residuals=False)


def test_predict_interval_exception_when_not_out_sample_residuals_for_all_levels():
    """
    Test exception is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is not available for all levels.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    residuals = pd.DataFrame({'2': [1, 2, 3, 4, 5], 
                              '3': [1, 2, 3, 4, 5]})
    forecaster.set_out_sample_residuals(residuals = residuals)
    levels = '1'

    err_msg = re.escape(
                (f'Not `forecaster.out_sample_residuals` for levels: {set(levels) - set(forecaster.out_sample_residuals.keys())}. '
                 f'Use method `set_out_sample_residuals()`.')
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=1, levels=levels, in_sample_residuals=False)


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
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)
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
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

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
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)
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
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

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
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    residuals = pd.DataFrame({'1': np.full_like(forecaster.in_sample_residuals['1'], fill_value=10), 
                              '2': np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)})
    forecaster.set_out_sample_residuals(residuals=residuals)

    predictions = forecaster.predict_interval(steps=1, levels=expected_pandas_dataframe[0], in_sample_residuals=False)
    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_steps_is_2_in_sample_residuals_is_False_with_fixture(expected_pandas_dataframe_2):
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals. This test is equivalent to the next one.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    residuals = pd.DataFrame({'1': np.full_like(forecaster.in_sample_residuals['1'], fill_value=10), 
                              '2': np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)})
    forecaster.set_out_sample_residuals(residuals=residuals)

    predictions = forecaster.predict_interval(steps=2, levels=expected_pandas_dataframe_2[0], in_sample_residuals=False)
    expected = expected_pandas_dataframe_2[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test predict_interval output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = LinearRegression(),
                     lags = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series = series)
    predictions = forecaster.predict_interval(steps=5, levels='1')
    expected = pd.DataFrame(
                   data = np.array([[0.52791431, 0.18396317, 0.95754082],
                                    [0.44509712, 0.0128976 , 0.84417134],
                                    [0.42176045, 0.05538845, 0.82860157],
                                    [0.48087237, 0.13819232, 0.89557796],
                                    [0.48268008, 0.1309113 , 0.90553765]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict_interval output when using LinearRegression as regressor and transformer_series
    is a dict with 2 diferent transformers.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = LinearRegression(),
                     lags = 5,
                     transformer_series = {'1': StandardScaler(), '2': MinMaxScaler()}
                 )
    forecaster.fit(series = series)
    predictions = forecaster.predict_interval(steps=5, levels=['1'])
    expected = pd.DataFrame(
                   data = np.array([[0.59619193, 0.18343628, 1.00239295],
                                    [0.46282914, 0.05746129, 0.85226151],
                                    [0.41738496, 0.01752287, 0.8300486 ],
                                    [0.48522676, 0.11948742, 0.90879137],
                                    [0.47525733, 0.117253  , 0.88886001]]),
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
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = LinearRegression(),
                     lags = 5,
                     transformer_series = StandardScaler(),
                     transformer_exog = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict_interval(steps=5, levels=['1', '2'], exog=exog)
    expected = pd.DataFrame(
                   data = np.array([[0.53267333, 0.17691231, 0.9399491 , 0.55496412, 0.14714165, 0.9246417],
                                    [0.44478046, 0.04641456, 0.83203647, 0.57787982, 0.14450512, 0.93373637],
                                    [0.52579563, 0.13671047, 0.92765308, 0.66389117, 0.23054494, 1.02657915],
                                    [0.57391142, 0.21373477, 0.97709097, 0.65789846, 0.23499484, 1.02002429],
                                    [0.54633594, 0.1725495 , 0.94995973, 0.5841187 , 0.18811899, 0.9513278 ]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1', '1_lower_bound', '1_upper_bound', '2', '2_lower_bound', '2_upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)