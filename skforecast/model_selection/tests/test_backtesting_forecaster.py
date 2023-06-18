# Unit test backtesting_forecaster
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection import backtesting_forecaster

# Fixtures
from .fixtures_model_selection import y
from .fixtures_model_selection import exog


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    err_msg = re.escape(
        ("`forecaster` must be of type `ForecasterAutoreg`, `ForecasterAutoregCustom` "
         "or `ForecasterAutoregDirect`, for all other types of forecasters "
         "use the functions available in the other `model_selection` modules.")
        )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_ValueError_when_ForecasterAutoregDirect_not_enough_steps():
    """
    Test ValueError is raised in backtesting_forecaster when there is not enough 
    steps to predict steps+gap in a ForecasterAutoregDirect.
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = Ridge(random_state=123),
                    lags      = 5,
                    steps     = 5
                 )
    
    gap = 5
    steps = 1
    
    err_msg = re.escape(
            ("When using a ForecasterAutoregDirect, the combination of steps "
             f"+ gap ({steps+gap}) cannot be greater than the `steps` parameter "
             f"declared when the forecaster is initialized ({forecaster.steps}).")
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = steps,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = gap,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )
        

def test_output_backtesting_forecaster_refit_int_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of backtesting_forecaster refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=0, metric='mean_squared_error',
    'in_sample_residuals = True'. Refit int.
    """
    expected_metric = 0.06119167557650236
    expected_predictions = pd.DataFrame(
        data = np.array([[0.55616986, 0.15288789, 0.89198752],
                         [0.48751797, 0.14866438, 0.83169303],
                         [0.5776799 , 0.1675223 , 1.03338459],
                         [0.50916147, 0.15329105, 0.86187874],
                         [0.47430051, 0.0796644 , 0.82587748],
                         [0.49192271, 0.14609696, 0.95959395],
                         [0.48350642, 0.17102066, 0.82819688],
                         [0.52459152, 0.18622779, 0.99188165],
                         [0.52501537, 0.13641764, 0.86685356],
                         [0.4680474 , 0.08515461, 0.81792677],
                         [0.51236176, 0.1201916 , 0.85743741],
                         [0.52331622, 0.14224683, 0.88195282],
                         [0.4430938 , 0.0509291 , 0.69854202],
                         [0.49911716, 0.1231365 , 0.8711497 ],
                         [0.43652745, 0.0532075 , 0.70011897],
                         [0.46763972, 0.07270655, 0.84965528],
                         [0.46901878, 0.08173407, 0.82098555],
                         [0.55371362, 0.14618224, 0.98199137],
                         [0.59754846, 0.20694109, 0.95302703],
                         [0.53048912, 0.14816008, 0.95027659]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.RangeIndex(start=33, stop=50, step=1)
    )

    forecaster = ForecasterAutoregDirect(
                     regressor = Ridge(random_state=123), 
                     lags      = 3,
                     steps     = 8
                 )

    n_backtest = 20
    y_train = y[:-n_backtest]

    metric, backtest_predictions = backtesting_forecaster(
                                       forecaster            = forecaster,
                                       y                     = y,
                                       exog                  = exog,
                                       refit                 = 2,
                                       initial_train_size    = len(y_train),
                                       fixed_train_size      = False,
                                       gap                   = 0,
                                       allow_incomplete_fold = True,
                                       steps                 = 2,
                                       metric                = 'mean_squared_error',
                                       interval              = [5, 95],
                                       n_boot                = 500,
                                       random_state          = 123,
                                       in_sample_residuals   = True,
                                       verbose               = False,
                                       n_jobs                = 1
                                   )

    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_int_interval_yes_exog_not_allow_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'in_sample_residuals = True', allow_incomplete_fold = False. Refit int.
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = 0.06911974957922452
    expected_predictions = pd.DataFrame(
        data = np.array([[0.51878642, 0.17324039, 0.87771447],
                         [0.49269791, 0.15223278, 0.86847335],
                         [0.49380441, 0.14003517, 0.81226679],
                         [0.51467463, 0.18225936, 0.91506854],
                         [0.52689128, 0.19548148, 0.87248805],
                         [0.4505235 , 0.12652546, 0.76823097],
                         [0.45217559, 0.10842513, 0.77368381],
                         [0.45205878, 0.11824909, 0.7633748 ],
                         [0.45234553, 0.11528922, 0.76300039],
                         [0.45177969, 0.09886977, 0.74916731],
                         [0.45401067, 0.07419957, 0.83718661],
                         [0.461257  , 0.07775626, 0.84405451],
                         [0.45914652, 0.05976913, 0.83641267],
                         [0.51072008, 0.11892485, 0.9041168 ],
                         [0.4961424 , 0.13756431, 0.87439121]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.date_range(start='2022-02-03', periods=15, freq='D')
    )

    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=3)

    metric, backtest_predictions = backtesting_forecaster(
                                       forecaster            = forecaster,
                                       y                     = y_with_index,
                                       exog                  = exog_with_index,
                                       refit                 = 3,
                                       initial_train_size    = len(y_with_index) - 20,
                                       fixed_train_size      = True,
                                       gap                   = 3,
                                       allow_incomplete_fold = False,
                                       steps                 = 5,
                                       metric                = 'mean_squared_error',
                                       interval              = [5, 95],
                                       n_boot                = 500,
                                       random_state          = 123,
                                       in_sample_residuals   = True,
                                       verbose               = False,
                                       n_jobs                = 1
                                   )
    backtest_predictions = backtest_predictions.asfreq('D')

    assert expected_metric == metric
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)