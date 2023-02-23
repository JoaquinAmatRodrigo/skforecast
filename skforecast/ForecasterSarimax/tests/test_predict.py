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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19, order=(1,1,1)))
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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19, order=(1,1,1)))
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
                            [0.5357917994, 0.5236704382, 0.5222455717, 0.5225814407, 0.523207142]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.64358976, 0.6057425641, 0.6404405344, 0.6293049695, 0.6664847988])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax(kwargs, data):
    """
    Test predict output of ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19, **kwargs))
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.6051979717, 0.444932999, 0.4925494003, 0.5053724811, 0.5495608555]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.6912741215, 0.5366948079, 0.5966985374, 0.612552592 , 0.6717425605])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_exog(kwargs, data):
    """
    Test predict output of ForecasterSarimax with exogenous variables.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,**kwargs))
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_predict)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.5357919716, 0.5236706695, 0.5222458244, 0.5225817054, 0.5232074166]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.6435918004, 0.6057455572, 0.6404437473, 0.6293084134, 0.6664885996])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_transform_y(kwargs, data):
    """
    Test predict output of ForecasterSarimax with a StandardScaler() as transformer_y.
    """
    forecaster = ForecasterSarimax(
                     regressor     = ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,**kwargs),
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.9334520425, 0.8113853881, 0.8620528628, 0.8819410192, 0.9286877882]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [1.0112299699, 0.8707156188, 0.9284195562, 0.9518789332, 1.0175980197])], 
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
                     regressor        = ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,**kwargs),
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
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


def test_predict_ValueError_when_last_window_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.values)
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=lw_datetime.values)
    lw_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,order=(1,1,1)))
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

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,order=(1,1,1)))
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
                           [0.5495549286, 0.5396990973, 0.5386471567, 0.5390444125, 0.5396802185]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.8589994954, 0.8696286475, 0.8975070653, 0.8923704537, 0.9408528053])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_last_window(kwargs, data):
    """
    Test predict output of ForecasterSarimax with `last_window`.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,**kwargs))
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict(steps=5, last_window=lw_datetime)
    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='A'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [0.6142681126, 0.4552897475, 0.5030886553, 0.5159376248, 0.5601296715]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [0.8965811992, 0.782936975, 0.8376117942, 0.8599394856, 0.929108665])], 
                         ids = lambda values : f'order, seasonal_order: {values}')
def test_predict_output_ForecasterSarimax_with_last_window_and_exog(kwargs, data):
    """
    Test predict output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,**kwargs))
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
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1,1,1), 
                            'seasonal_order': (0,0,0,0)}, 
                            [1.1431156427, 1.0393416189, 1.0916050805, 1.1116324828, 1.1583914007]), 
                          ({'order': (1,1,1), 
                            'seasonal_order': (1,1,1,2)}, 
                            [1.1125228794, 0.9921145554, 1.0325979552, 1.0629749413, 1.1260719217])], 
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
                     regressor        = ARIMA(maxiter=10000, trend=None, method='nm', ftol=1e-19,**kwargs), 
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
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='A'))], 
                         ids = lambda values : f'y, index: {values}')
def test_predict_ForecasterSarimax_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index is updated when using predict twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, trend=None, method='nm', ftol=1e-19,order=(1,1,1)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict(steps=5, last_window=lw_1)
    result_1 = forecaster.extended_index.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict(steps=5, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index, idx)