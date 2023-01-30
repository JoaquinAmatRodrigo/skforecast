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
        forecaster.predict(
            steps            = 5, 
            exog             = exog_predict, 
            last_window      = None, 
            last_window_exog = exog
        )


def test_predict_ValueError_when_ForecasterSarimax_last_window_exog_is_None_and_included_exog_is_true():
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
        forecaster.predict(
            steps            = 5, 
            exog             = ex_pred, 
            last_window      = lw, 
            last_window_exog = None
        )


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


def test_predict_ValueError_when_last_window_index_does_not_follow_training_set():
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

    err_msg = re.escape(
        (f'To make predictions unrelated to the original data, `last_window` '
         f'has to start at the end of the index seen by the forecaster.\n'
         f'    Series last index         : 2022-02-19 00:00:00.\n'
         f'    Expected index            : 2022-02-20 00:00:00.\n'
         f'    `last_window` index start : 2022-03-01 00:00:00.')
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(steps=5, last_window=lw_test)


def test_predict_ValueError_when_last_window_exog_index_does_not_follow_training_set():
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

    err_msg = re.escape(
        (f'To make predictions unrelated to the original data, `last_window_exog` '
         f'has to start at the end of the index seen by the forecaster.\n'
         f'    Series last index              : 2022-02-19 00:00:00.\n'
         f'    Expected index                 : 2022-02-20 00:00:00.\n'
         f'    `last_window_exog` index start : 2022-03-01 00:00:00.')
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(
            steps            = 5, 
            exog             = exog_pred_test, 
            last_window      = lw_test,
            last_window_exog = lw_exog_test
        )


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.5492955335851895, 0.5393905172567712, 0.538325126193933, 0.5387150002224543, 0.5393444541452022]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.8557544588691266, 0.8657142597545064, 0.8937406845687847, 0.8885063451884633, 0.9368087260810022])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_last_window(kwargs, data):
    """
    Test predict output of ForecasterSarimax with `last_window`.
    """
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
    forecaster = ForecasterSarimax(regressor=ARIMA(**kwargs))
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict(
                      steps            = 5, 
                      exog             = exog_predict_datetime, 
                      last_window      = lw_datetime, 
                      last_window_exog = lw_exog_datetime
                  )

    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='A'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [1.0883760008461447, 0.9596655915602446, 1.0164563275894087, 1.0368122358153902, 1.0879339700749853]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [1.1359948592023694, 1.0197787345609874, 1.0563186630823083, 1.0890821402058224, 1.1539621608480948])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_last_window_and_exog_and_transformers(kwargs, data):
    """
    Test predict output of ForecasterSarimax with exogenous variables, `last_window`
    and transformers.
    """
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
    forecaster.fit(y=y_datetime, exog=df_exog_datetime)
    predictions = forecaster.predict(
                      steps            = 5, 
                      exog             = df_exog_predict_datetime, 
                      last_window      = lw_datetime, 
                      last_window_exog = df_lw_exog_datetime
                  )

    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='A'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='A'))], 
                         ids = lambda values : f'y, index: {values}')
def test_predict_ForecasterSarimax_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index is updated when using predict twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict(steps=5, last_window=lw_1)
    result_1 = forecaster.extended_index.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict(steps=5, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index, idx)