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
# np.random.seed(123)
# y = np.random.rand(50)
# exog = np.random.rand(50)
y = pd.Series(
        data = np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                         0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                         0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                         0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                         0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                         0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                         0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                         0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                         0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]
            ),
        name = 'y'
    )

exog = pd.Series(
           data = np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                            0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                            0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                            0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                            0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                            0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                            0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                            0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                            0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                            0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]
               ),
           name = 'exog'
       )


def test_exception_predict_interval_when_interval_is_not_symmetrical():
    """
    Raise exception if `interval` is not symmetrical.
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
        forecaster.predict_interval(steps=5, alpha=alpha, interval=interval_not_symmetrical)


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
    predictions = forecaster.predict_interval(steps=5, exog=exog, alpha=alpha, interval=interval)
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
    df_exog = pd.DataFrame({
                  'exog_1': exog,
                  'exog_2': ['a']*25+['b']*25}
              )
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
    predictions = forecaster.predict_interval(steps=5, exog=df_exog, alpha=alpha, interval=interval)
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


def test_exception_predict_interval_when_last_window_index_does_not_follow_training_set():
    """
    Raise exception if `last_window` index does not start at the end of the training set.
    """
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2022-01-01', periods=50)
    lw_datetime = pd.Series(data=np.random.rand(50))
    lw_datetime.index = pd.date_range(start='2022-03-01', periods=50)

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_datetime)
    expected_index = expand_index(forecaster.last_window.index, 1)[0]

    err_msg = re.escape(
        (f'To make predictions unrelated to the original data, `last_window` '
         f'has to start at the end of the training set.\n'
         f'    Series last index         : {forecaster.last_window.index[-1]}.\n'
         f'    Expected index            : {expected_index}.\n'
         f'    `last_window` index start : {lw_datetime.index[0]}.')
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05,
            last_window      = lw_datetime,
        )


def test_exception_predict_interval_when_last_window_exog_index_does_not_follow_training_set():
    """
    Raise exception if `last_window_exog` index does not start at the end of the training set.
    """
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2022-01-01', periods=50)
    exog_datetime = pd.Series(data=list(exog))
    exog_datetime.index = pd.date_range(start='2022-01-01', periods=50)

    lw_datetime = pd.Series(data=np.random.rand(50))
    lw_datetime.index = pd.date_range(start='2022-02-20', periods=50)
    lw_exog_datetime = pd.Series(data=np.random.rand(50))
    lw_exog_datetime.index = pd.date_range(start='2022-03-01', periods=50)

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    expected_index = expand_index(forecaster.last_window.index, 1)[0]

    err_msg = re.escape(
        (f'To make predictions unrelated to the original data, `last_window_exog` '
         f'has to start at the end of the training set.\n'
         f'    Series last index              : {forecaster.last_window.index[-1]}.\n'
         f'    Expected index                 : {expected_index}.\n'
         f'    `last_window_exog` index start : {lw_exog_datetime.index[0]}.')
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05,
            exog             = exog_datetime, 
            last_window      = lw_datetime, 
            last_window_exog = lw_exog_datetime
        )


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with `last_window`.
    """
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
    lw_datetime = pd.Series(data=list(exog))
    lw_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      last_window      = lw_datetime,
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.50413804, 0.04670129, 0.96157478],
                                       [0.53115634, 0.06682964, 0.99548305],
                                       [0.53616963, 0.07153717, 1.00080209],
                                       [0.53756023, 0.07289965, 1.00222082],
                                       [0.53835444, 0.07368981, 1.00301907]]),
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
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
    lw_datetime = pd.Series(data=list(y))
    lw_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

    exog_datetime = pd.Series(data=list(exog))
    exog_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
    lw_exog_datetime = pd.Series(data=list(exog))
    lw_exog_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      exog             = exog_datetime, 
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


'''
@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog_amd_transformers(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
    lw_datetime = pd.Series(data=list(y))
    lw_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

    df_exog = pd.DataFrame({'exog_1': exog, 'exog_2': ['a']*25+['b']*25})
    df_exog.index = pd.date_range(start='2000', periods=50, freq='A')
    df_lw_exog = pd.DataFrame({'exog_1': exog, 'exog_2': ['a']*25+['b']*25})
    df_lw_exog.index = pd.date_range(start='2050', periods=50, freq='A')

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
    forecaster.fit(y=y_datetime, exog=df_exog)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      exog             = df_exog, 
                      last_window      = lw_datetime, 
                      last_window_exog = df_lw_exog
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
'''