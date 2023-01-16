# Unit test predict ForecasterSarimax
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

exog_predict = pd.Series(
                  data = np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                                   0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234]
                      ),
                  name = 'exog',
                  index = pd.RangeIndex(start=50, stop=60)
              )


def test_predict_ValueError_when_ForecasterSarimax_last_window_exog_is_not_None_and_last_window_is_not_provided():
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
        forecaster.predict(steps=5, exog=exog_predict, last_window=None, last_window_exog=exog)


def test_predict_ValueError_when_ForecasterSarimax_last_window_exog_is_None_and_included_exog_is_true():
    """
    Check ValueError is raised when last_window_exog is None, but included_exog
    is True and last_window is provided.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y, exog=exog)
    lw = pd.Series(np.random.rand(10), index=pd.RangeIndex(start=50, stop=60))
    exog_predict = pd.Series(np.random.rand(10), index=pd.RangeIndex(start=60, stop=70))
    
    err_msg = re.escape(
                ('Forecaster trained with exogenous variable/s. To make predictions '
                 'unrelated to the original data, same variable/s must be provided '
                 'using `last_window_exog`.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(steps=5, exog=exog_predict, last_window=lw, last_window_exog=None)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.5356622558529589, 0.5235127967378337, 0.5220779036724879, 0.5224069466873313, 0.5230263860169242]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.6430690155462436, 0.6050710801113841, 0.6397294193166768, 0.6285198294044027, 0.6656188675900477])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax(kwargs, data):
    """
    Test predict output of ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(**kwargs))
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.6049785151374183, 0.4448344266724246, 0.4923987407544393, 0.5052039736669536, 0.5493454088764397]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.6841313267300797, 0.5141384604507838, 0.5766435826495582, 0.5939182835941423, 0.6536426162184714])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_exog(kwargs, data):
    """
    Test predict output of ForecasterSarimax with exogenous variables.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(**kwargs))
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_predict)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.5357267116006734, 0.5235943872856297, 0.5221656461585646, 0.5224985075944557, 0.5231212947061987]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.6432582412135628, 0.6055273247183961, 0.6402460549654707, 0.6290458388549236, 0.6662003438751096])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_transform_y(kwargs, data):
    """
    Test predict output of ForecasterSarimax with a StandardScaler() as transformer_y.
    """
    forecaster = ForecasterSarimax(
                     regressor     = ARIMA(**kwargs),
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.9084670377251036, 0.7647883047064794, 0.8203336846873355, 0.8405859799961493, 0.8916990937197793]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [1.0095544323406163, 0.8707230817689734, 0.925811215455429, 0.9522848237323704, 1.0168960034047383])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_transform_y_and_transform_exog(kwargs, data):
    """
    Test predict output of ForecasterSarimax, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    df_exog = pd.DataFrame({
                  'exog_1': exog,
                  'exog_2': ['a']*25+['b']*25}
              )
    df_exog_predict = df_exog.copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=100)
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterSarimax(
                     regressor        = ARIMA(**kwargs),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict(steps=5, exog=df_exog_predict)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_exception_predict_when_last_window_index_does_not_follow_training_set():
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
        forecaster.predict(steps=5, last_window=lw_datetime)


def test_exception_predict_when_last_window_exog_index_does_not_follow_training_set():
    """
    Raise exception if `last_window_exog` index does not start at the end of the training set.
    """
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2022-01-01', periods=50)
    exog_datetime = pd.Series(data=list(exog))
    exog_datetime.index = pd.date_range(start='2022-01-01', periods=50)
    exog_pred_datetime = pd.Series(data=list(exog_predict))
    exog_pred_datetime.index = pd.date_range(start='2022-04-11', periods=10)

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
        forecaster.predict(
            steps            = 5, 
            exog             = exog_pred_datetime, 
            last_window      = lw_datetime,
            last_window_exog = lw_exog_datetime
        )


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.5041380371817271, 0.5311563444658294, 0.5361696300771597, 0.5375602349850797, 0.5383544389589756]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.7707974069999539, 0.8983527340212023, 0.8496848647451074, 0.8983347795049217, 0.8896948675413555])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_last_window(kwargs, data):
    """
    Test predict output of ForecasterSarimax with `last_window`.
    """
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
    lw_datetime = pd.Series(data=list(exog))
    lw_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

    forecaster = ForecasterSarimax(regressor=ARIMA(**kwargs))
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict(steps=5, last_window=lw_datetime)
    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='A'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.6139162641758982, 0.455038610445325, 0.502782372051503, 0.5156130317886716, 0.5597580698531707]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.7798489327681167, 0.6278742459100042, 0.6901067862699033, 0.7134687449777593, 0.7801833466513106])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_last_window_and_exog(kwargs, data):
    """
    Test predict output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    y_datetime = pd.Series(data=list(y))
    y_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
    lw_datetime = pd.Series(data=list(y))
    lw_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

    exog_datetime = pd.Series(data=list(exog))
    exog_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
    exog_pred_datetime = pd.Series(data=list(exog_predict))
    exog_pred_datetime.index = pd.date_range(start='2100', periods=10, freq='A')
    lw_exog_datetime = pd.Series(data=list(exog))
    lw_exog_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

    forecaster = ForecasterSarimax(regressor=ARIMA(**kwargs))
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict(
                      steps            = 5, 
                      exog             = exog_pred_datetime, 
                      last_window      = lw_datetime, 
                      last_window_exog = lw_exog_datetime
                  )

    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='A'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


'''
@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.6139162641758982, 0.455038610445325, 0.502782372051503, 0.5156130317886716, 0.5597580698531707]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.7798489327681167, 0.6278742459100042, 0.6901067862699033, 0.7134687449777593, 0.7801833466513106])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_last_window_and_exog_amd_transformers(kwargs, data):
    """
    Test predict output of ForecasterSarimax with exogenous variables and `last_window`.
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
                     regressor        = ARIMA(**kwargs), 
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_datetime, exog=df_exog)
    predictions = forecaster.predict(
                      steps            = 5, 
                      exog             = df_exog, 
                      last_window      = lw_datetime, 
                      last_window_exog = df_lw_exog
                  )

    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='A'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
'''