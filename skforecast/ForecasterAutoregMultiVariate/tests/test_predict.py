# Unit test predict ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# Fixtures
# np.random.seed(123)
# l1 = np.random.rand(50)
# l2 = np.random.rand(50)
# exog_1 = np.random.rand(50)
series = pd.DataFrame({'l1': pd.Series(np.array(
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
                       'l2': pd.Series(np.array(
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
    
exog = pd.DataFrame({'exog_1': pd.Series(np.array(
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
                     'exog_2': ['a']*25 + ['b']*25}
       )

exog_predict = exog.copy()
exog_predict.index = pd.RangeIndex(start=50, stop=100)


@pytest.mark.parametrize("steps", [[1, 2.0, 3], [1, 4.]], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_exception_when_steps_list_contain_floats(steps):
    """
    Test predict exception when steps is a list with floats.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
                    f"`steps` argument must be an int, a list of ints or `None`. "
                    f"Got {type(steps)}."
                )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.predict(steps=steps)


@pytest.mark.parametrize("steps", [3, [1, 2, 3], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_output_when_regressor_is_LinearRegression(steps):
    """
    Test predict output when using LinearRegression as regressor.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=steps)
    expected = pd.DataFrame(
                   data    = np.array([0.63114259, 0.3800417, 0.33255977]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_list_interspersed():
    """
    Test predict output when using LinearRegression as regressor and steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=3, steps=5)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=[1, 4])
    expected = pd.DataFrame(
                   data    = np.array([0.61048324, 0.53962565]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1)[[0, 3]],
                   columns = ['l2']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_different_lags():
    """
    Test predict output when using LinearRegression as regressor and different
    lags configuration for each series.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': [1, 7]}, steps=3)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=3)
    expected = pd.DataFrame(
                   data    = np.array([0.58053278, 0.43052971, 0.60582844]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l2']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_using_last_window():
    """
    Test predict output when using LinearRegression as regressor and last_window.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)
    last_window = pd.DataFrame(
                      data    = np.array([[0.98555979, 0.39887629],
                                          [0.51948512, 0.2408559 ],
                                          [0.61289453, 0.34345601]]),
                      index   = pd.RangeIndex(start=47, stop=50, step=1),
                      columns = ['l1', 'l2']
                  )
    results = forecaster.predict(steps=[1, 2], last_window=last_window)
    expected = pd.DataFrame(
                   data    = np.array([0.63114259, 0.3800417]),
                   index   = pd.RangeIndex(start=50, stop=52, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_using_exog():
    """
    Test predict output when using LinearRegression as regressor and exog.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series.iloc[:40,], exog=exog.iloc[:40, 0])
    results = forecaster.predict(steps=None, exog=exog.iloc[40:43, 0])
    expected = pd.DataFrame(
                   data    = np.array([0.57243323, 0.56121924, 0.38879785]),
                   index   = pd.RangeIndex(start=40, stop=43, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster.predict()
    expected = pd.DataFrame(
                   data    = np.array([0.60056539, 0.42924504, 0.34173573, 0.44231236, 0.40133213]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_as_dict():
    """
    Test predict output when using LinearRegression as regressor and transformer_series
    is a dict with 2 different transformers.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l2',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = {'l1': StandardScaler(), 'l2': MinMaxScaler()}
                 )
    forecaster.fit(series=series)
    results = forecaster.predict()
    expected = pd.DataFrame(
                   data    = np.array([0.65049981, 0.57548048, 0.64278726, 0.54421867, 0.7851753 ]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['l2']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_series and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['exog_1']),
                             ('onehot', OneHotEncoder(), ['exog_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                       )
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict(steps=[1, 2, 3, 4, 5], exog=exog_predict)
    expected = pd.DataFrame(
                   data    = np.array([0.61043227, 0.46658137, 0.54994519, 0.52561227, 0.46596527]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)