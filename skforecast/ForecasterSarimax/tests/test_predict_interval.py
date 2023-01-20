# Unit test predict_interval ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.utils import expand_index
from pmdarima.arima import ARIMA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Fixtures
from .fixtures_ForecasterSarimax import y
from .fixtures_ForecasterSarimax import exog
from .fixtures_ForecasterSarimax import exog_predict
from .fixtures_ForecasterSarimax import df_exog
from .fixtures_ForecasterSarimax import df_exog_predict
from .fixtures_ForecasterSarimax import y_datetime
from .fixtures_ForecasterSarimax import lw_datetime
from .fixtures_ForecasterSarimax import exog_datetime
from .fixtures_ForecasterSarimax import lw_exog_datetime
from .fixtures_ForecasterSarimax import exog_predict_datetime
from .fixtures_ForecasterSarimax import df_exog_datetime
from .fixtures_ForecasterSarimax import df_lw_exog_datetime
from .fixtures_ForecasterSarimax import df_exog_predict_datetime


def test_predict_interval_ValueError_when_ForecasterSarimax_last_window_exog_is_not_None_and_last_window_is_not_provided():
    """
    Check ValueError is raised when last_window_exog is not None, but 
    last_window is not provided.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y, exog=exog)
    
    err_msg = re.escape(
                ('To make predictions unrelated to the original data, both '
                 '`last_window` and `last_window_exog` must be provided.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05, 
            exog             = exog_predict, 
            last_window      = None, 
            last_window_exog = exog
        )


def test_predict_interval_ValueError_when_ForecasterSarimax_last_window_exog_is_None_and_included_exog_is_true():
    """
    Check ValueError is raised when last_window_exog is None, but included_exog
    is True and last_window is provided.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y, exog=exog)
    lw = pd.Series(np.random.rand(10), index=pd.RangeIndex(start=50, stop=60))
    ex_pred = pd.Series(np.random.rand(10), index=pd.RangeIndex(start=60, stop=70))
    
    err_msg = re.escape(
                ('Forecaster trained with exogenous variable/s. To make predictions '
                 'unrelated to the original data, same variable/s must be provided '
                 'using `last_window_exog`.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05, 
            exog             = ex_pred, 
            last_window      = lw, 
            last_window_exog = None
        )


def test_predict_interval_ValueError_when_interval_is_not_symmetrical():
    """
    Raise ValueError if `interval` is not symmetrical.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y)
    alpha = None
    interval_not_symmetrical = [5, 97.5] 

    err_msg = re.escape(
                (f'When using `interval` in ForecasterSarimax, it must be symmetrical. '
                 f'For example, interval of 95% should be as `interval = [2.5, 97.5]`. '
                 f'Got {interval_not_symmetrical}.')
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps    = 5, 
            alpha    = alpha, 
            interval = interval_not_symmetrical
        )


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.53566226, 0.07599312, 0.99533139],
                                       [0.5235128 , 0.05620538, 0.99082021],
                                       [0.5220779 , 0.05432689, 0.98982891],
                                       [0.52240695, 0.05460468, 0.99020921],
                                       [0.52302639, 0.05521626, 0.99083652]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(steps=5, exog=exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.60497852,  0.15960892, 1.05034811],
                                       [0.44483443, -0.00619962, 0.89586848],
                                       [0.49239874,  0.04107972, 0.94371776],
                                       [0.50520397,  0.05385485, 0.95655309],
                                       [0.54934541,  0.09799221, 1.00069861]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_transform_y(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with a StandardScaler() as transformer_y.
    """
    forecaster = ForecasterSarimax(
                     regressor     = ARIMA(order=(1,1,1)),
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.53572671, 0.07602913, 0.99542429],
                                       [0.52359439, 0.05626217, 0.99092661],
                                       [0.52216565, 0.05439018, 0.98994111],
                                       [0.52249851, 0.05467184, 0.99032517],
                                       [0.52312129, 0.05528678, 0.99095581]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_transform_y_and_transform_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterSarimax(
                     regressor        = ARIMA(order=(1,1,1)),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_interval(steps=5, exog=df_exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.90846704, 0.47796249, 1.33897159],
                                       [0.7647883 , 0.33209942, 1.19747719],
                                       [0.82033368, 0.38757178, 1.25309559],
                                       [0.84058598, 0.40781874, 1.27335322],
                                       [0.89169909, 0.4589314 , 1.32446679]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_ValueError_when_last_window_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.values)
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=lw_datetime.values)
    lw_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_test)
    expected_index = expand_index(forecaster.extended_index, 1)[0]

    err_msg = re.escape(
        (f'To make predictions unrelated to the original data, `last_window` '
         f'has to start at the end of the index seen by the forecaster.\n'
         f'    Series last index         : {forecaster.extended_index[-1]}.\n'
         f'    Expected index            : {expected_index}.\n'
         f'    `last_window` index start : {lw_test.index[0]}.')
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05,
            last_window      = lw_test,
        )


def test_predict_interval_ValueError_when_last_window_exog_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window_exog` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.values)
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=lw_datetime.values)
    lw_test.index = pd.date_range(start='2022-02-20', periods=50, freq='D')

    exog_test = pd.Series(data=exog_datetime.values)
    exog_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_pred_test = pd.Series(data=exog_predict_datetime.values)
    exog_pred_test.index = pd.date_range(start='2022-04-11', periods=10, freq='D')
    lw_exog_test = pd.Series(data=lw_exog_datetime.values)
    lw_exog_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_test, exog=exog_test)
    expected_index = expand_index(forecaster.extended_index, 1)[0]

    err_msg = re.escape(
        (f'To make predictions unrelated to the original data, `last_window_exog` '
         f'has to start at the end of the index seen by the forecaster.\n'
         f'    Series last index              : {forecaster.extended_index[-1]}.\n'
         f'    Expected index                 : {expected_index}.\n'
         f'    `last_window_exog` index start : {lw_exog_test.index[0]}.')
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05,
            exog             = exog_pred_test, 
            last_window      = lw_test, 
            last_window_exog = lw_exog_test
        )


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with `last_window`.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      last_window      = lw_datetime,
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.54929553, 0.09185879, 1.00673228],
                                       [0.53939052, 0.07506382, 1.00371722],
                                       [0.53832513, 0.07369267, 1.00295758],
                                       [0.538715  , 0.07405442, 1.00337559],
                                       [0.53934445, 0.07467982, 1.00400909]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      exog             = exog_predict_datetime, 
                      last_window      = lw_datetime, 
                      last_window_exog = lw_exog_datetime
                  )

    expected = pd.DataFrame(
                   data    = np.array([[0.61391626, 0.17071366, 1.05711886],
                                       [0.45503861, 0.00679555, 0.90328167],
                                       [0.50278237, 0.05435185, 0.95121289],
                                       [0.51561303, 0.06716642, 0.96405964],
                                       [0.55975807, 0.11130937, 1.00820676]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog_amd_transformers(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterSarimax(
                     regressor        = ARIMA(order=(1,1,1)), 
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_datetime, exog=df_exog_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      exog             = df_exog_predict_datetime, 
                      last_window      = lw_datetime, 
                      last_window_exog = df_lw_exog_datetime
                  )

    expected = pd.DataFrame(
                   data    = np.array([[1.088376  , 0.65997541, 1.51677659],
                                       [0.95966559, 0.52943384, 1.38989734],
                                       [1.01645633, 0.58618273, 1.44672992],
                                       [1.03681224, 0.60653591, 1.46708856],
                                       [1.08793397, 0.65765741, 1.51821053]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='A'))], 
                         ids = lambda values : f'y, index: {values}')
def test_predict_interval_ForecasterSarimax_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index is updated when using predict_interval twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_1)
    result_1 = forecaster.extended_index.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index, idx)