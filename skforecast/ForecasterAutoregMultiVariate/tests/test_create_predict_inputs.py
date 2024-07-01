# Unit test _create_predict_inputs ForecasterAutoregMultiVariate
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
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

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
def test_create_predict_inputs_TypeError_when_steps_list_contain_floats(steps):
    """
    Test _create_predict_inputs TypeError when steps is a list with floats.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        (f"`steps` argument must be an int, a list of ints or `None`. "
         f"Got {type(steps)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_predict_inputs(steps=steps)


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
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
        forecaster._create_predict_inputs(steps=5)


@pytest.mark.parametrize("steps", [3, [1, 2, 3], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_create_predict_inputs_output(steps):
    """
    Test _create_predict_inputs output.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=steps)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]])],
        [1, 2, 3],
        pd.RangeIndex(start=47, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_when_list_interspersed():
    """
    Test _create_predict_inputs output when steps is
    a list with interspersed steps.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=3, steps=5, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=[1, 4])

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]])],
        [1, 4],
        pd.RangeIndex(start=47, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_when_different_lags():
    """
    Test _create_predict_inputs output when different
    lags configuration for each series.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': [1, 7]}, 
                                               steps=3, transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=3)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.34345601, 0.76254781]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.34345601, 0.76254781]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537,
                    0.34345601, 0.76254781]])],
        [1, 2, 3],
        pd.RangeIndex(start=43, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_7']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_when_lags_dict_with_None_in_level_lags():
    """
    Test _create_predict_inputs output when lags is a 
    dict and level has None lags configuration.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 5, 'l2': None}, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=3)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]])],
        [1, 2, 3],
        pd.RangeIndex(start=45, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_when_lags_dict_with_None_but_no_in_level():
    """
    Test _create_predict_inputs output when lags is a 
    dict with None values.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags={'l1': 5, 'l2': None}, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=3)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537]])],
        [1, 2, 3],
        pd.RangeIndex(start=45, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_when_last_window():
    """
    Test _create_predict_inputs output when external last_window.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series)
    last_window = pd.DataFrame(
        data    = np.array([[0.98555979, 0.39887629],
                            [0.51948512, 0.2408559 ],
                            [0.61289453, 0.34345601]]),
        index   = pd.RangeIndex(start=47, stop=50, step=1),
        columns = ['l1', 'l2']
    )
    results = forecaster._create_predict_inputs(steps=[1, 2], last_window=last_window)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 
                    0.34345601, 0.2408559, 0.39887629]])],
        [1, 2],
        pd.RangeIndex(start=47, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_when_exog():
    """
    Test _create_predict_inputs output when exog.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3,
                                               transformer_series=None)
    forecaster.fit(series=series.iloc[:40,], exog=exog.iloc[:40, 0])
    results = forecaster._create_predict_inputs(steps=None, exog=exog.iloc[40:43, 0])

    expected = (
        [np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.09332671]]),
         np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.29686078]]),
         np.array([[0.50183668, 0.94416002, 0.89338916, 
                    0.39818568, 0.30476807, 0.04359146, 0.92758424]])],
        [1, 2, 3],
        pd.RangeIndex(start=37, stop=40, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'exog_1']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_with_transform_series():
    """
    Test _create_predict_inputs output when StandardScaler.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     lags               = 5,
                     steps              = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589]])],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=45, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_with_transform_series_as_dict():
    """
    Test _create_predict_inputs output when transformer_series
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
    results = forecaster._create_predict_inputs()

    expected = (
        [np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    0.33426983,  0.22949344,  0.39086564,  0.13786173,  0.68990237]])],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=45, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_create_predict_inputs_output_with_transform_series_and_transform_exog(n_jobs):
    """
    Test _create_predict_inputs output when StandardScaler
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
    results = forecaster._create_predict_inputs(steps=[1, 2, 3, 4, 5], exog=exog_predict)

    expected = (
        [np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -0.02551075,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                     0.51927383,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -1.470801945,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -1.38212079,  1.,  0.]]),
         np.array([[0.47762884,  0.07582436,  2.08066403, -0.08097053, -1.08141835, 
                    -0.61148712, -1.00971677, -0.39638015, -1.35798666,  0.74018589,
                    -0.70392558,  1.,  0.]])],
        [1, 2, 3, 4, 5],
        pd.RangeIndex(start=45, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1', 'exog_2_a', 'exog_2_b']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test _create_predict_inputs output when using HistGradientBoostingRegressor 
    and categorical variables.
    """
    df_exog = pd.DataFrame({
        'exog_1': exog['exog_1'],
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
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)

    expected = (
        [np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    0.        , 0.        , 0.51312815]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    1.        , 1.        , 0.66662455]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    2.        , 2.        , 0.10590849]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    3.        , 3.        , 0.13089495]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    4.        , 4.        , 0.32198061]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    0.        , 0.        , 0.66156434]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    1.        , 1.        , 0.84650623]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    2.        , 2.        , 0.55325734]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    3.        , 3.        , 0.85445249]]),
         np.array([[0.61289453, 0.51948512, 0.98555979, 0.48303426, 0.25045537, 
                    0.34345601, 0.2408559 , 0.39887629, 0.15112745, 0.6917018 ,
                    4.        , 4.        , 0.38483781]])],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        pd.RangeIndex(start=45, stop=50, step=1),
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_2', 'exog_3', 'exog_1']
    )
    
    for step in range(len(expected[0])):
        np.testing.assert_almost_equal(results[0][step], expected[0][step])
    assert results[1] == expected[1]
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]