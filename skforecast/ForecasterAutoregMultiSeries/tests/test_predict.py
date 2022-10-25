# Unit test predict ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
from pytest import approx
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


@pytest.fixture(params=[('1'  , [50., 51., 52., 53., 54.]), 
                        (['2'], [100., 101., 102., 103., 104.]),
                        (['1', '2'], [[50., 100.],
                                      [51., 101.],
                                      [52., 102.],
                                      [53., 103.],
                                      [54., 104.]])
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
        
    expected_df = pd.DataFrame(
                      data    = request.param[1],
                      columns = levels_names,
                      index   = pd.RangeIndex(start=50, stop=55, step=1)
                  )

    return levels, expected_df


def test_predict_output_when_regressor_is_LinearRegression_with_fixture(expected_pandas_dataframe):
    """
    Test predict output when using LinearRegression as regressor with pytest fixture.
    This test is equivalent to the next one.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=5, levels=expected_pandas_dataframe[0])
    expected = expected_pandas_dataframe[1]

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression():
    """
    Test predict output when using LinearRegression as regressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series)
    predictions_1 = forecaster.predict(steps=5, levels='1')
    expected_1 = pd.DataFrame(
                     data    = np.array([50., 51., 52., 53., 54.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1']
                 )

    predictions_2 = forecaster.predict(steps=5, levels=['2'])
    expected_2 = pd.DataFrame(
                     data    = np.array([100., 101., 102., 103., 104.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['2']
                 )

    predictions_3 = forecaster.predict(steps=5, levels=['1', '2'])
    expected_3 = pd.DataFrame(
                     data    = np.array([[50., 100.],
                                         [51., 101.],
                                         [52., 102.],
                                         [53., 103.],
                                         [54., 104.]]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1', '2']
                 )

    pd.testing.assert_frame_equal(predictions_1, expected_1)
    pd.testing.assert_frame_equal(predictions_2, expected_2)
    pd.testing.assert_frame_equal(predictions_3, expected_3)


def test_predict_output_when_regressor_is_LinearRegression_with_last_window():
    """
    Test predict output when using LinearRegression as regressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(start=0, stop=50)), 
                           '2': pd.Series(np.arange(start=50, stop=100))
                          })
    last_window = pd.DataFrame(
                      {'1': [45, 46, 47, 48, 49], 
                       '2': [95, 96, 97, 98, 99], 
                       '3': [1, 2, 3, 4, 5]}, 
                      index = pd.RangeIndex(start=45, stop=50, step=1)
                  )

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series)
    predictions_1 = forecaster.predict(steps=5, levels='1', last_window=last_window)
    expected_1 = pd.DataFrame(
                     data    = np.array([50., 51., 52., 53., 54.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1']
                 )

    predictions_2 = forecaster.predict(steps=5, levels=['2'], last_window=last_window)
    expected_2 = pd.DataFrame(
                     data    = np.array([100., 101., 102., 103., 104.]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['2']
                 )

    predictions_3 = forecaster.predict(steps=5, levels=['1', '2'], last_window=last_window)
    expected_3 = pd.DataFrame(
                     data    = np.array([[50., 100.],
                                         [51., 101.],
                                         [52., 102.],
                                         [53., 103.],
                                         [54., 104.]]),
                     index   = pd.RangeIndex(start=50, stop=55, step=1),
                     columns = ['1', '2']
                 )

    pd.testing.assert_frame_equal(predictions_1, expected_1)
    pd.testing.assert_frame_equal(predictions_2, expected_2)
    pd.testing.assert_frame_equal(predictions_3, expected_3)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = LinearRegression(),
                     lags = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series = series)
    predictions = forecaster.predict(steps=5, levels='1')
    expected = pd.DataFrame(
                   data    = np.array([0.52791431, 0.44509712, 0.42176045, 0.48087237, 0.48268008]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict output when using LinearRegression as regressor and transformer_series
    is a dict with 2 different transformers.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = LinearRegression(),
                     lags = 5,
                     transformer_series = {'1': StandardScaler(), '2': MinMaxScaler()}
                 )
    forecaster.fit(series = series)
    predictions = forecaster.predict(steps=5, levels=['1'])
    expected = pd.DataFrame(
                   data    = np.array([0.59619193, 0.46282914, 0.41738496, 0.48522676, 0.47525733]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
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
    predictions = forecaster.predict(steps=5, levels='1', exog=exog)
    expected = pd.DataFrame(
                   data    = np.array([0.53267333, 0.44478046, 0.52579563, 0.57391142, 0.54633594]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)