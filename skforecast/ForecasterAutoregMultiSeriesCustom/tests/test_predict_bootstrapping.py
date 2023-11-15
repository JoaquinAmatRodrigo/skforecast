# Unit test predict_bootstrapping ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Fixtures
from .fixtures_ForecasterAutoregMultiSeriesCustom import series
from .fixtures_ForecasterAutoregMultiSeriesCustom import exog
from .fixtures_ForecasterAutoregMultiSeriesCustom import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )

def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags


def test_predict_bootstrapping_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )

    err_msg = re.escape(
                ("This Forecaster instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using predict.")
              )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


def test_predict_bootstrapping_ValueError_when_not_in_sample_residuals_for_any_level():
    """
    Test ValueError is raised when in_sample_residuals=True but there is no
    residuals for any level.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    
    forecaster.fit(series=series)
    forecaster.in_sample_residuals = {2: np.array([1, 2, 3])}

    err_msg = re.escape(
                (f"Not `forecaster.in_sample_residuals` for levels: "
                 f"{set(['1', '2']) - set(forecaster.in_sample_residuals.keys())}.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels=None, in_sample_residuals=True)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is None.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series)

    err_msg = re.escape(
                ("`forecaster.out_sample_residuals` is `None`. Use "
                 "`in_sample_residuals=True` or method "
                 "`set_out_sample_residuals()` before `predict_interval()`, "
                 "`predict_bootstrapping()`,`predict_quantiles()` or "
                 "`predict_dist()`.")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels='1', in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_not_out_sample_residuals_for_all_levels():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is not available for all levels.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series)
    residuals = {'2': np.array([1, 2, 3, 4, 5]), 
                 '3': np.array([1, 2, 3, 4, 5])}
    forecaster.out_sample_residuals = residuals
    levels = ['1']

    err_msg = re.escape(
                (f"Not `forecaster.out_sample_residuals` for levels: "
                 f"{set(levels) - set(forecaster.out_sample_residuals.keys())}. "
                 f"Use method `set_out_sample_residuals()`.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels=levels, in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_level_out_sample_residuals_value_is_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals has a level with a None.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series)
    residuals = {'1': np.array([1, 2, 3, 4, 5])}
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
                    (f"forecaster residuals for level '2' are `None`. Check `forecaster.out_sample_residuals`.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_step_out_sample_residuals_value_contains_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals has a step with a None value.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series)
    residuals = {'1': np.array([1, 2, 3, 4, 5]),
                 '2': np.array([1, 2, 3, 4, None])}
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
                    (f"forecaster residuals for level '2' contains `None` values. Check `forecaster.out_sample_residuals`.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, in_sample_residuals=False)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, exog=exog['exog_1'])
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict['exog_1'], in_sample_residuals=True)

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.24898667, 0.69862572, 0.72572763, 0.45672811]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1 ,'2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, exog=exog['exog_1'])
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict['exog_1'], in_sample_residuals=True)

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729],
                                         [0.04906615, 0.45019329, 0.07478854, 0.222437  ]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.24898667, 0.69862572, 0.72572763, 0.45672811],
                                         [0.29374176, 0.40758093, 0.74577937, 0.11883889]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )    

    expected = {'1': expected_1 ,'2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, exog=exog['exog_1'])
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict['exog_1'], in_sample_residuals=False)

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.24898667, 0.69862572, 0.72572763, 0.45672811]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1 ,'2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3
                 )
    forecaster.fit(series=series, exog=exog['exog_1'])
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict['exog_1'], in_sample_residuals=False)

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729],
                                         [0.04906615, 0.45019329, 0.07478854, 0.222437  ]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.24898667, 0.69862572, 0.72572763, 0.45672811],
                                         [0.29374176, 0.40758093, 0.74577937, 0.11883889]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )  

    expected = {'1': expected_1 ,'2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer_fixed_to_zero():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=0)
    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=0)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.50201669, 0.50201669, 0.50201669, 0.50201669],
                                         [0.49804821, 0.49804821, 0.49804821, 0.49804821]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.52531076, 0.52531076, 0.52531076, 0.52531076],
                                         [0.51683613, 0.51683613, 0.51683613, 0.51683613]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )  

    expected = {'1': expected_1 ,'2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors,
                     window_size        = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16958001, 0.55887507, 0.41716867, 0.38128471],
                                         [0.08114962, 0.48871107, 0.07673561, 0.25559782]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.24448676, 0.70066373, 0.77244769, 0.45857602],
                                         [0.33048491, 0.44364526, 0.75067101, 0.16052538]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )  

    expected = {'1': expected_1 ,'2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])