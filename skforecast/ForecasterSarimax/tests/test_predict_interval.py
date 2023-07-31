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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.5357917994, 0.07612415  , 0.9954594488],
                                        [0.5236704382, 0.0563663856, 0.9909744907],
                                        [0.5222455717, 0.0544981184, 0.9899930251],
                                        [0.5225814407, 0.0547827783, 0.9903801031],
                                        [0.523207142, 0.0554006397, 0.9910136443]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(steps=5, exog=exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[ 0.6051979717,  0.1597548469,  1.0506410965],
                                        [ 0.444932999, -0.0061864617,  0.8960524597],
                                        [ 0.4925494003,  0.041144114,  0.9439546866],
                                        [ 0.5053724811,  0.0539370043,  0.9568079579],
                                        [ 0.5495608555,  0.0981213044,  1.0010004067]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_transform_y(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with a StandardScaler() as transformer_y.
    """
    forecaster = ForecasterSarimax(
                     regressor     = ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)),
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
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


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
                     regressor        = ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_interval(steps=5, exog=df_exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.9334520425, 0.5065760981, 1.360327987 ],
                                        [0.8113853881, 0.3821636265, 1.2406071496],
                                        [0.8620528628, 0.4327501715, 1.2913555541],
                                        [0.8819410192, 0.452632205, 1.3112498333],
                                        [0.9286877882, 0.499378447, 1.3579971294]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


def test_predict_interval_ValueError_when_last_window_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.to_numpy())
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=lw_datetime.to_numpy())
    lw_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
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
    y_test = pd.Series(data=y_datetime.to_numpy())
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=lw_datetime.to_numpy())
    lw_test.index = pd.date_range(start='2022-02-20', periods=50, freq='D')

    exog_test = pd.Series(data=exog_datetime.to_numpy())
    exog_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_pred_test = pd.Series(data=exog_predict_datetime.to_numpy())
    exog_pred_test.index = pd.date_range(start='2022-04-11', periods=10, freq='D')
    lw_exog_test = pd.Series(data=lw_exog_datetime.to_numpy())
    lw_exog_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      last_window      = lw_datetime,
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.5495549286, 0.0921199189, 1.0069899383],
                                        [0.5396990973, 0.0753759932, 1.0040222014],
                                        [0.5386471567, 0.0740184686, 1.0032758447],
                                        [0.5390444125, 0.0743876373, 1.0037011876],
                                        [0.5396802185, 0.0750194215, 1.0043410155]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
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
                   data    = np.array([[0.6142681126, 0.1709925946, 1.0575436306],
                                        [0.4552897475, 0.0069627552, 0.9036167398],
                                        [0.5030886553, 0.0545735995, 0.9516037111],
                                        [0.5159376248, 0.0674064456, 0.9644688039],
                                        [0.5601296715, 0.1115964175, 1.0086629255]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


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
                     regressor        = ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)), 
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
                   data    = np.array([[1.1431156427, 0.7183255309, 1.5679057545],
                                        [1.0393416189, 0.6125730792, 1.4661101587],
                                        [1.0916050805, 0.6647897114, 1.5184204496],
                                        [1.1116324828, 0.6848139825, 1.5384509831],
                                        [1.1583914007, 0.7315726345, 1.5852101669]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='A')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='A'))], 
                         ids = lambda values : f'y, index: {values}')
def test_predict_interval_ForecasterSarimax_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index is updated when using predict_interval twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,  order=(1,1,1)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_1)
    result_1 = forecaster.extended_index.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index, idx)