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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.53566226, 0.07600754908668045, 0.99533139],
                                       [0.5235128 , 0.05622015027520466, 0.99082021],
                                       [0.5220779 , 0.05434179783202908, 0.98982891],
                                       [0.52240695, 0.05461972628135853, 0.99020921],
                                       [0.52302639, 0.05523144682828113, 0.99083652]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, rtol=0.01)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(steps=5, exog=exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.6051897923832147,  0.15983172025483505, 1.0505478645115942],
                                       [0.44485328798558965, -0.006171887610120108, 0.8958784635812994],
                                       [0.4924928857243766,  0.04118198841225701, 0.9438037830364963],
                                       [0.5053206219772969,  0.05397913000502269, 0.956662113949571],
                                       [0.5495283846598701,  0.09818235692460225, 1.0008744123951379]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, rtol=0.01)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_transform_y(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with a StandardScaler() as transformer_y.
    """
    forecaster = ForecasterSarimax(
                     regressor     = ARIMA(maxiter=1000, order=(1,1,1)),
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.53578347, 0.07614449, 0.99542244],
                                        [0.52363457, 0.05633326, 0.99093588],
                                        [0.52219955, 0.05445244, 0.98994666],
                                        [0.52253151, 0.05473284, 0.99033018],
                                        [0.52315489, 0.0553483 , 0.99096149]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, rtol=0.01)


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
                     regressor        = ARIMA(maxiter=1000, order=(1,1,1)),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_interval(steps=5, exog=df_exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.90819939, 0.47768224, 1.33871653],
                                        [0.76454177, 0.33183966, 1.19724387],
                                        [0.82006953, 0.38729437, 1.2528447 ],
                                        [0.84031253, 0.40753202, 1.27309303],
                                        [0.89140952, 0.45862856, 1.32419049]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, rtol=0.01)


def test_predict_interval_ValueError_when_last_window_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.values)
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=lw_datetime.values)
    lw_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
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

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      last_window      = lw_datetime,
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.5493021 , 0.0918756 , 1.0067286 ],
                                        [0.53939807, 0.07508138, 1.00371476],
                                        [0.53833294, 0.07371045, 1.00295544],
                                        [0.53872299, 0.07407235, 1.00337363],
                                        [0.53935261, 0.07469791, 1.00400731]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, rtol=0.01)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
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
                   data    = np.array([[0.61420452, 0.17100769, 1.05740135],
                                        [0.45514546, 0.0069039 , 0.90338703],
                                        [0.50296609, 0.05453606, 0.95139611],
                                        [0.51581947, 0.06737283, 0.96426612],
                                        [0.56003087, 0.11158168, 1.00848007]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, rtol=0.01)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog_and_transformers(alpha, interval):
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
                     regressor        = ARIMA(maxiter=1000, order=(1,1,1)), 
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
                   data    = np.array([[1.08795272, 0.65953954, 1.51636591],
                                        [0.95925274, 0.52900782, 1.38949766],
                                        [1.01602516, 0.58573837, 1.44631196],
                                        [1.03637173, 0.60608219, 1.46666126],
                                        [1.08747734, 0.65718757, 1.51776712]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, rtol=0.01)


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='A'))], 
                         ids = lambda values : f'y, index: {values}')
def test_predict_interval_ForecasterSarimax_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index is updated when using predict_interval twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_1)
    result_1 = forecaster.extended_index.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index, idx)