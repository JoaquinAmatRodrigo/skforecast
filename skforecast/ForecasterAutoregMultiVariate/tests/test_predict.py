# Unit test predict ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

# Fixtures
from .fixtures_ForecasterAutoregMultiVariate import series
from .fixtures_ForecasterAutoregMultiVariate import exog
from .fixtures_ForecasterAutoregMultiVariate import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


@pytest.mark.parametrize("steps", [[1, 2.0, 3], [1, 4.]], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_TypeError_when_steps_list_contain_floats(steps):
    """
    Test predict TypeError when steps is a list with floats.
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


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)

    err_msg = re.escape(
                ("This Forecaster instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using predict.")
              )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


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


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_predict_output_when_regressor_is_LinearRegression_with_transform_series_and_transform_exog(n_jobs):
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
                     transformer_exog   = transformer_exog,
                     n_jobs             = n_jobs
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


def test_predict_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test predict output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog['exog_1'],
                            'exog_2': ['a', 'b', 'c', 'd', 'e']*10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J']*10)})
    
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
    
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = HistGradientBoostingRegressor(
                                              categorical_features = categorical_features,
                                              random_state         = 123
                                          ),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 10,
                     transformer_series = None,
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.4668392 , 
                                    0.47754412, 0.47360906, 0.47749396, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor():
    """
    Test predict output when using LGBMRegressor and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog['exog_1'],
                            'exog_2': ['a', 'b', 'c', 'd', 'e']*10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J']*10)})
    
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
    
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LGBMRegressor(random_state=123),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 10,
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                     fit_kwargs         = {'categorical_feature': categorical_features}
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.46683919,
                                    0.47754412, 0.47360906, 0.47749395, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_categorical_features_native_implementation_LGBMRegressor_auto():
    """
    Test predict output when using LGBMRegressor and categorical variables with 
    categorical_features='auto'.
    """
    df_exog = pd.DataFrame({'exog_1': exog['exog_1'],
                            'exog_2': ['a', 'b', 'c', 'd', 'e']*10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J']*10)})
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    pipeline_categorical = make_pipeline(
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               FunctionTransformer(
                                   func=lambda x: x.astype('category'),
                                   feature_names_out= 'one-to-one'
                               )
                           )
    transformer_exog = make_column_transformer(
                            (
                                pipeline_categorical,
                                make_column_selector(dtype_exclude=np.number)
                            ),
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LGBMRegressor(random_state=123),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 10,
                     transformer_series = None,
                     transformer_exog   = transformer_exog,
                     fit_kwargs         = {'categorical_feature': 'auto'}
                 )
    forecaster.fit(series=series, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([0.50131059, 0.49276926, 0.47433929, 0.46683919,
                                    0.47754412, 0.47360906, 0.47749395, 0.48461923, 
                                    0.48686681, 0.50223394]),
                   index   = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['l1']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)