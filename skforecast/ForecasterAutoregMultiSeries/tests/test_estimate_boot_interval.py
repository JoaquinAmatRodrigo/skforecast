# Unit test predict_bootstrapping ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression

# Fixtures
from .fixtures_ForecasterAutoregMultiSeries import series
from .fixtures_ForecasterAutoregMultiSeries import exog
from .fixtures_ForecasterAutoregMultiSeries import exog_predict


def test_predict_bootstrapping_ValueError_when_forecaster_in_sample_residuals_contains_None_values():
    """
    Test ValueError is raised when forecaster.in_sample_residuals are not stored 
    during fit method.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=False)

    err_msg = re.escape(
                (f"`forecaster.in_sample_residuals['1']` contains `None` values. "
                  "Try using `fit` method with `in_sample_residuals=True` or set in "
                  "`predict_interval()`, `predict_bootstrapping()` or "
                  "`predict_dist()` method `in_sample_residuals=False` and use "
                  "`out_sample_residuals` (see `set_out_sample_residuals()`).")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels='1', in_sample_residuals=True)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is None.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=False)

    err_msg = re.escape(
                ('`forecaster.out_sample_residuals` is `None`. Use '
                 '`in_sample_residuals=True` or method `set_out_sample_residuals()` '
                 'before `predict_interval()`, `predict_bootstrapping()` or '
                 '`predict_dist()`.')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels='1', in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_not_out_sample_residuals_for_all_levels():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals is not available for all levels.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    residuals = pd.DataFrame({'2': [1, 2, 3, 4, 5], 
                              '3': [1, 2, 3, 4, 5]})
    forecaster.set_out_sample_residuals(residuals = residuals)
    levels = ['1']

    err_msg = re.escape(
                (f'Not `forecaster.out_sample_residuals` for levels: {set(levels) - set(forecaster.out_sample_residuals.keys())}. '
                 f'Use method `set_out_sample_residuals()`.')
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels=levels, in_sample_residuals=False)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, exog=exog['col_1'])
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict['col_1'], in_sample_residuals=True)

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
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    expected = pd.DataFrame(
                   data    = np.array([[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                       [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict, in_sample_residuals=True)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    expected = pd.DataFrame(
                   data    = np.array([[0.39132925, 0.71907802, 0.45297104, 0.52687879]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=51)
               )
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    expected = pd.DataFrame(
                   data    = np.array([[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                       [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer_fixed_to_zero():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """
    forecaster = ForecasterAutoreg(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=0)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                   data = np.array([[0.67998879, 0.67998879, 0.67998879, 0.67998879],
                                    [0.46135022, 0.46135022, 0.46135022, 0.46135022]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterAutoreg(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                   data    = np.array(
                                 [[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                  [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
                )

    pd.testing.assert_frame_equal(expected, results)




















































def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step is predicted using in-sample residuals.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = np.array([[20., 20.]])
    results_1 = forecaster.predict_bootstrapping(steps=1, level = '1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = np.array([[30., 30.]])
    results_2 = forecaster.predict_bootstrapping(steps=1, level = '2', in_sample_residuals=True)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps are predicted using in-sample residuals.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = np.array([[20.        , 20.],
                           [24.33333333, 24.33333333]])
    results_1 = forecaster.predict_bootstrapping(steps=2, level = '1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = np.array([[30.              , 30.],
                           [37.66666666666667, 37.66666666666667]])
    results_2 = forecaster.predict_bootstrapping(steps=2, level = '2', in_sample_residuals=True)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step is predicted using out sample residuals.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.out_sample_residuals = {'1': np.full_like(forecaster.in_sample_residuals['1'], fill_value=10),
                                       '2': np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)}

    expected_1 = np.array([[20., 20.]])
    results_1 = forecaster.predict_bootstrapping(steps=1, level='1', in_sample_residuals=False)

    expected_2 = np.array([[30., 30.]])
    results_2 = forecaster.predict_bootstrapping(steps=1, level='2', in_sample_residuals=False)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps are predicted using out sample residuals.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.out_sample_residuals = {'1': np.full_like(forecaster.in_sample_residuals['1'], fill_value=10),
                                       '2': np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)}

    expected_1 = np.array([[20.        , 20.        ],
                           [24.33333333, 24.33333333]])
    results_1 = forecaster.predict_bootstrapping(steps=2, level='1', in_sample_residuals=False)

    expected_2 = np.array([[30.              , 30.],
                           [37.66666666666667, 37.66666666666667]])
    results_2 = forecaster.predict_bootstrapping(steps=2, level='2', in_sample_residuals=False)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)