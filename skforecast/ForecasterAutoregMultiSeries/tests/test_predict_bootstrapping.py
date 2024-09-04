# Unit test predict_bootstrapping ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.exceptions import UnknownLevelWarning
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

# Fixtures
from .fixtures_ForecasterAutoregMultiSeries import series
from .fixtures_ForecasterAutoregMultiSeries import exog
from .fixtures_ForecasterAutoregMultiSeries import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


def test_predict_bootstrapping_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels=None)


def test_predict_bootstrapping_IgnoredArgumentWarning_when_not_available_self_last_window_for_some_levels():
    """
    Test IgnoredArgumentWarning is raised when last_window is not available for 
    levels because it was not stored during fit.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'], store_last_window=['1'])

    warn_msg = re.escape(
        ("Levels {'2'} are excluded from prediction "
         "since they were not stored in `last_window_` attribute "
         "during training. If you don't want to retrain the "
         "Forecaster, provide `last_window` as argument.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps               = 1,
                      levels              = ['1', '2'],
                      n_boot              = 4, 
                      exog                = exog_predict['exog_1'], 
                      in_sample_residuals = True
                  )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1}

    pd.testing.assert_frame_equal(results['1'], expected['1'])


@pytest.mark.parametrize("store_last_window",
                         [['1'], False],
                         ids=lambda slw: f"store_last_window: {slw}")
def test_predict_bootstrapping_ValueError_when_not_available_self_last_window_for_levels(store_last_window):
    """
    Test ValueError is raised when last_window is not available for all 
    levels because it was not stored during fit.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series, store_last_window=store_last_window)

    err_msg = re.escape(
        ("No series to predict. None of the series {'2'} are present in "
         "`last_window_` attribute. Provide `last_window` as argument "
         "in predict method.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=5, levels=['2'], last_window=None)


def test_predict_bootstrapping_IgnoredArgumentWarning_when_levels_is_list_and_different_last_index_in_self_last_window_DatetimeIndex():
    """
    Test IgnoredArgumentWarning is raised when levels is a list and have 
    different last index in last_window attribute using a DatetimeIndex.
    """
    series_2 = {
        '1': series['1'].copy(),
        '2': series['2'].iloc[:30].copy()
    }
    series_2['1'].index = pd.date_range(start='2020-01-01', periods=50)
    series_2['2'].index = pd.date_range(start='2020-01-01', periods=30)
    exog_2 = {
        '1': exog['exog_1'].copy(),
        '2': exog['exog_1'].iloc[:30].copy()
    }
    exog_2['1'].index = pd.date_range(start='2020-01-01', periods=50)
    exog_2['2'].index = pd.date_range(start='2020-01-01', periods=30)
    exog_2_pred = {
        '1': exog_predict['exog_1'].copy()
    }
    exog_2_pred['1'].index = pd.date_range(start='2020-02-20', periods=50)

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, exog=exog_2)

    warn_msg = re.escape(
        ("Only series whose last window ends at the same index "
         "can be predicted together. Series that do not reach the "
         "maximum index, '2020-02-19 00:00:00', are excluded "
         "from prediction: {'2'}.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps               = 1,
                      levels              = ['1', '2'],
                      n_boot              = 4, 
                      exog                = exog_2_pred, 
                      in_sample_residuals = True
                  )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.17482136, 0.59548022, 0.40159385, 0.37372285]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.date_range(start='2020-02-20', periods=1)
             )
    }

    for k in results.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_predict_bootstrapping_UnknownLevelWarning_when_not_in_sample_residuals_for_level():
    """
    Test UnknownLevelWarning is raised when predicting an unknown level and 
    in_sample_residuals is True (using _unknown_level residuals).
    """
    forecaster = ForecasterAutoregMultiSeries(
        LGBMRegressor(verbose=-1), lags=3, encoding='ordinal'
    )
    forecaster.fit(series=series)
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1']

    warn_msg = re.escape(
        ("`levels` {'3'} were not included in training. "
         "Unknown levels are encoded as NaN, which may cause the "
         "prediction to fail if the regressor does not accept NaN values.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps               = 1,
                      levels              = ['1', '2', '3'],
                      last_window         = last_window,
                      n_boot              = 4,
                      in_sample_residuals = True
                  )

    warn_msg = re.escape(
        ("`levels` {'3'} are not present in `forecaster.in_sample_residuals_`, "
         "most likely because they were not present in the training data. "
         "A random sample of the residuals from other levels will be used. "
         "This can lead to inaccurate intervals for the unknown levels.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps               = 1,
                      levels              = ['1', '2', '3'],
                      last_window         = last_window,
                      n_boot              = 4,
                      in_sample_residuals = True
                  )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.2557472, 0.45547833, 0.40141629, 0.41792497]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '2': pd.DataFrame(
                 data    = np.array([[0.35308571, 0.39818568, 0.67766497, 0.22103753]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '3': pd.DataFrame(
                 data    = np.array([[0.51830135, 0.20914413, 0.92354139, 0.45547833]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_encoding_None_unknown_level():
    """
    Test UnknownLevelWarning is raised when predicting an unknown level and 
    encoding is None (using _unknown_level residuals).
    """
    forecaster = ForecasterAutoregMultiSeries(
        LGBMRegressor(verbose=-1), lags=3, encoding=None
    )
    forecaster.fit(series=series)
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1']

    results = forecaster.predict_bootstrapping(
                  steps               = 1,
                  levels              = ['1', '2', '3'],
                  last_window         = last_window,
                  n_boot              = 4,
                  in_sample_residuals = True
              )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.46049943, 0.27232912, 0.4773099, 0.51544113]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '2': pd.DataFrame(
                 data    = np.array([[0.53328674, 0.58200333, 0.60314635, 0.41766002]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '3': pd.DataFrame(
                 data    = np.array([[0.52831631, 0.19924718, 0.91194292, 0.42972248]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_encoding_None_unknown_level_differentiation():
    """
    Test UnknownLevelWarning is raised when predicting an unknown level and 
    encoding is None (using _unknown_level residuals) and differentiation=1.
    """
    forecaster = ForecasterAutoregMultiSeries(
        LGBMRegressor(verbose=-1), lags=3, encoding=None, differentiation=1
    )
    forecaster.fit(series=series)
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1'] * 0.9

    results = forecaster.predict_bootstrapping(
                  steps               = 1,
                  levels              = ['1', '2', '3'],
                  last_window         = last_window,
                  n_boot              = 5,
                  in_sample_residuals = True
              )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.85193261, 0.65669636, 0.40904666, 0.48671657, 0.64058698]]),
                 columns = [f"pred_boot_{i}" for i in range(5)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '2': pd.DataFrame(
                 data    = np.array([[0.25888000,  0.35629127, -0.27895615,  0.16318238,  0.16318238]]),
                 columns = [f"pred_boot_{i}" for i in range(5)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '3': pd.DataFrame(
                 data    = np.array([[0.44749540, 0.46893592, 1.08720922, 0.74829265, 0.748292653]]),
                 columns = [f"pred_boot_{i}" for i in range(5)],
                 index   = pd.RangeIndex(50, 51)
             )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals_ is None.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_` is `None`. Use "
         "`in_sample_residuals=True` or the  `set_out_sample_residuals()` "
         "method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels='1', in_sample_residuals=False)


def test_predict_bootstrapping_UnknownLevelWarning_out_sample_residuals_with_encoding_None():
    """
    Test UnknownLevelWarning is raised when encoding is None and 
    out_sample_residuals is set.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3, 
                                              encoding=None)
    forecaster.fit(series=series)

    new_residuals = {
        '1': np.array([1, 2, 3, 4, 5]), 
        '2': np.array([1, 2, 3, 4, 5])
    }
    warn_msg = re.escape(
        ("As `encoding` is set to `None`, no distinction between levels "
         "is made. All residuals are stored in the '_unknown_level' key.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        forecaster.set_out_sample_residuals(new_residuals)
    
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1']
    results = forecaster.predict_bootstrapping(
                  steps               = 1,
                  levels              = ['1', '2', '3'],
                  last_window         = last_window,
                  n_boot              = 4,
                  in_sample_residuals = False
              )

    expected = {
        '1': pd.DataFrame(
              data    = np.array([[4.48307367, 4.48307367, 5.48307367, 2.48307367]]),
              columns = [f"pred_boot_{i}" for i in range(4)],
              index   = pd.RangeIndex(50, 51)
          ),
        '2': pd.DataFrame(
              data    = np.array([[1.5061193, 3.5061193, 3.5061193, 1.5061193]]),
              columns = [f"pred_boot_{i}" for i in range(4)],
              index   = pd.RangeIndex(50, 51)
          ),
        '3': pd.DataFrame(
              data    = np.array([[1.48307367, 2.48307367, 5.48307367, 2.48307367]]),
              columns = [f"pred_boot_{i}" for i in range(4)],
              index   = pd.RangeIndex(50, 51)
          )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_predict_bootstrapping_ValueError_when_not_level_in_out_sample_residuals(transformer_series):
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals_ is missing a level.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    residuals = {'1': np.array([1, 2, 3, 4, 5])}
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
        ("Not available residuals for level '2'. "
         "Check `forecaster.out_sample_residuals_`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, in_sample_residuals=False)


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_predict_bootstrapping_ValueError_when_level_out_sample_residuals_value_contains_None_or_NaNs(transformer_series):
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals_ has a level with a None or NaN value.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    residuals = {'1': np.array([1, 2, 3, 4, 5]),
                 '2': np.array([1, 2, 3, 4, None])}  # StandardScaler() transforms None to NaN
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
        ("forecaster residuals for level '2' contains `None` "
         "or `NaNs` values. Check `forecaster.out_sample_residuals_`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, in_sample_residuals=False)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    results = forecaster.predict_bootstrapping(
                  steps               = 1, 
                  n_boot              = 4, 
                  exog                = exog_predict['exog_1'], 
                  in_sample_residuals = True
              )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.31806650, 0.40236263, 0.73878038, 0.12951421]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    results = forecaster.predict_bootstrapping(
                  steps               = 2, 
                  n_boot              = 4, 
                  exog                = exog_predict['exog_1'], 
                  in_sample_residuals = True
              )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729],
                                         [0.07792903, 0.47905617, 0.10365142, 0.25129988]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.69663465, 1.03514738, 0.72210743, 0.57237401],
                                         [0.47358417, 0.25757726, 0.76135164, 0.32060755]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )    

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
                  steps               = 1, 
                  n_boot              = 4, 
                  exog                = exog_predict['exog_1'], 
                  in_sample_residuals = False,
                  suppress_warnings   = True
              )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.31806650, 0.40236263, 0.73878038, 0.12951421]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
                  steps               = 2, 
                  n_boot              = 4, 
                  exog                = exog_predict['exog_1'], 
                  in_sample_residuals = False
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16386591, 0.56398143, 0.38576231, 0.37782729],
                                         [0.07792903, 0.47905617, 0.10365142, 0.25129988]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.69663465, 1.03514738, 0.72210743, 0.57237401],
                                         [0.47358417, 0.25757726, 0.76135164, 0.32060755]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer_fixed_to_zero():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """    
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)

    forecaster.in_sample_residuals_['1'] = np.full_like(forecaster.in_sample_residuals_['1'], fill_value=0)
    forecaster.in_sample_residuals_['2'] = np.full_like(forecaster.in_sample_residuals_['2'], fill_value=0)
    results = forecaster.predict_bootstrapping(
                  steps               = 2, 
                  n_boot              = 4, 
                  exog                = exog_predict, 
                  in_sample_residuals = True
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([[0.50201669, 0.50201669, 0.50201669, 0.50201669],
                                         [0.52505626, 0.52505626, 0.52505626, 0.52505626]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.52531076, 0.52531076, 0.52531076, 0.52531076],
                                         [0.54676784, 0.54676784, 0.54676784, 0.54676784]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_bootstrapping(
                  steps               = 2, 
                  n_boot              = 4, 
                  exog                = exog_predict, 
                  in_sample_residuals = True
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([[0.16958001, 0.55887507, 0.41716867, 0.38128471],
                                         [0.10815767, 0.51571912, 0.10374366, 0.28260587]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.73391792, 1.07905155, 0.76739951, 0.57081569],
                                         [0.52099221, 0.30211351, 0.81041311, 0.35755142]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_differentiation():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed and differentiation.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                     differentiation    = 1
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_bootstrapping(
                  steps               = 3, 
                  n_boot              = 10, 
                  exog                = exog_predict, 
                  in_sample_residuals = True
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([
                                   [0.47934326, 0.95168238, 0.4030405 , 0.34931421, 0.4030405 ,
                                    0.49328573, 0.38902953, 0.45618806, 0.88232388, 0.612208  ],
                                   [0.35408694, 0.61371682, 0.62401789, 0.55599343, 0.57505757,
                                    0.73604114, 0.36091845, 0.59391637, 0.5984969 , 0.44011073],
                                   [0.19794885, 0.81617238, 0.35911255, 0.75628619, 0.77605324,
                                    0.56566898, 0.21739428, 0.71080749, 0.76868514, 0.26477136]]),
                     columns = [f"pred_boot_{i}" for i in range(10)],
                     index   = pd.RangeIndex(start=50, stop=53)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([
                                   [ 0.84930299, -0.11175337,  0.50405259,  0.14628148,  0.38288212,
                                     0.32115554,  0.07709315,  0.43881105,  0.23392556, -0.21650711],
                                   [ 0.41249056,  0.02020636, -0.11352055,  0.4148467 ,  0.29391619,
                                     0.79821461,  0.57576831,  0.46159357, -0.31856089, -0.26444904],
                                   [ 0.60723463,  0.45673017, -0.16179447,  0.08755587,  0.5958805 ,
                                     0.85290729,  0.19710101,  0.66891429, -0.11672833, -0.17209489]]),
                     columns = [f"pred_boot_{i}" for i in range(10)],
                     index   = pd.RangeIndex(start=50, stop=53)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])