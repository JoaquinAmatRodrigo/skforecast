# Unit test predict_bootstrapping ForecasterAutoreg
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Fixtures
from .fixtures_ForecasterAutoreg import y
from .fixtures_ForecasterAutoreg import exog
from .fixtures_ForecasterAutoreg import exog_predict
from .fixtures_ForecasterAutoreg import data  # to test results when using differentiation


def test_predict_bootstrapping_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ is None.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_` is `None`. Use "
         "`use_in_sample_residuals=True` or the "
         "`set_out_sample_residuals()` method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, use_in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_by_bin_is_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_by_bin_ is None.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_by_bin_` is `None`. Use "
         "`use_in_sample_residuals=True` or the "
         "`set_out_sample_residuals()` method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, use_in_sample_residuals=False,
                                         use_binned_residuals=True)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(
        steps=1, n_boot=4, exog=exog_predict, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array([[0.76205175, 0.52687879, 0.40505505, 0.77278459]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=51)
               )

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(
        steps=2, n_boot=4, exog=exog_predict, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.76205175, 0.52687879, 0.40505505, 0.77278459],
                                  [0.86040207, 0.16150649, 0.11656985, 0.39176265]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
        steps=1, n_boot=4, exog=exog_predict, use_in_sample_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[0.76205175, 0.52687879, 0.40505505, 0.77278459]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=51)
               )

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
        steps=2, n_boot=4, exog=exog_predict, use_in_sample_residuals=False
    )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.76205175, 0.52687879, 0.40505505, 0.77278459],
                                       [0.86040207, 0.16150649, 0.11656985, 0.39176265]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

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
    forecaster.in_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=0)
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict, n_boot=4, use_in_sample_residuals=True
    )
    
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
                     binner_kwargs    = {'n_bins': 15}
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict, n_boot=4, use_in_sample_residuals=True
    )
    
    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.76205175, 0.52687879, 0.40505505, 0.77278459],
                                  [0.86040207, 0.16150649, 0.11656985, 0.39176265]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
                )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_and_differentiation_is_1():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    differentiation is 1.
    """
    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    data_diff = diferenciator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'
    steps = len(data.loc[end_train:])

    forecaster_1 = ForecasterAutoreg(regressor=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    boot_predictions_diff = forecaster_1.predict_bootstrapping(
                                steps=steps,
                                exog=exog_diff.loc[end_train:],
                                n_boot=10
                            )
    last_value_train = data.loc[:end_train].iloc[[-1]]
    boot_predictions_1 = boot_predictions_diff.copy()
    boot_predictions_1.loc[last_value_train.index[0]] = last_value_train.values[0]
    boot_predictions_1 = boot_predictions_1.sort_index()
    boot_predictions_1 = boot_predictions_1.cumsum(axis=0).iloc[1:,]
    boot_predictions_1 = boot_predictions_1.asfreq('MS')
    
    forecaster_2 = ForecasterAutoreg(regressor=LinearRegression(), lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    boot_predictions_2 = forecaster_2.predict_bootstrapping(
                            steps  = steps,
                            exog   = exog_diff.loc[end_train:],
                            n_boot = 10
                        )
    
    pd.testing.assert_frame_equal(boot_predictions_1, boot_predictions_2)


def test_predict_bootstrapping_output_when_window_features():
    """
    Test output of predict_bootstrapping when regressor is LGBMRegressor and
    4 steps ahead are predicted with exog and window features using in-sample residuals.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2001-01-01', periods=len(y), freq='D')
    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2001-01-01', periods=len(exog), freq='D')
    exog_predict_datetime = exog_predict.copy()
    exog_predict_datetime.index = pd.date_range(start='2001-02-20', periods=len(exog_predict), freq='D')
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterAutoreg(
        LGBMRegressor(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.predict_bootstrapping(
        steps=4, n_boot=10, exog=exog_predict_datetime, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.61866004, 0.89697256, 0.14325579, 0.18128852, 0.86846565,
                                   0.55020849, 0.25870679, 0.51670634, 0.29239436, 0.42315364],
                                  [0.26475697, 0.67230222, 0.18926364, 0.45124373, 0.34317802,
                                   0.6900186 , 0.43134277, 0.67230222, 0.20674374, 0.84082826],
                                  [0.40053981, 0.32295891, 0.61577585, 0.15682665, 0.43966009,
                                   0.46585818, 0.5110144 , 0.5225711 , 0.56031677, 0.47247859],
                                  [0.39037942, 0.06805095, 0.86630916, 0.52190308, 0.47247859,
                                   0.43966009, 0.46227478, 0.58525714, 0.5189877 , 0.28131971]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(10)],
                   index   = pd.date_range(start='2001-02-20', periods=4, freq='D')
               )

    pd.testing.assert_frame_equal(expected, results)
