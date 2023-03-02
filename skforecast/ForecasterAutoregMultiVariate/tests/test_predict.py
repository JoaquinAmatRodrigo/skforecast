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
from skforecast.ForecasterAutoregMultiVariate.tests.fixtures_ForecasterAutoregMultiVariate import series
from skforecast.ForecasterAutoregMultiVariate.tests.fixtures_ForecasterAutoregMultiVariate import exog
from skforecast.ForecasterAutoregMultiVariate.tests.fixtures_ForecasterAutoregMultiVariate import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


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


def test_predict_output_when_regressor_is_LinearRegression_and_weight_func():
    """
    Test predict output when using LinearRegression as regressor and custom_weights.
    """
    def custom_weights(index):
        """
        Return 1 for all elements in index
        """
        weights = np.ones_like(index)

        return weights
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3, weight_func=custom_weights)
    forecaster.fit(series=series)
    results = forecaster.predict(steps=3)
    expected = pd.DataFrame(
                   data    = np.array([0.63114259, 0.3800417, 0.33255977]),
                   index   = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(results, expected)