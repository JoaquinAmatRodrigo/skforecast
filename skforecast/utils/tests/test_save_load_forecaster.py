# Unit test save_forecaster and load_forecaster
# ==============================================================================
import os
import re
import joblib
import pytest
import inspect
import warnings
import numpy as np
import pandas as pd
import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
from skforecast.exceptions import SkforecastVersionWarning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def create_predictors(y): # pragma: no cover
    """
    Create first 2 lags of a time series.
    """
    
    lags = y[-1:-3:-1]
    
    return lags

def custom_weights(y): # pragma: no cover
    """
    """
    return np.ones(len(y))

def custom_weights2(y): # pragma: no cover
    """
    """
    return np.arange(1, len(y)+1)


def test_save_and_load_forecaster_persistence():
    """ 
    Test if a loaded forecaster is exactly the same as the original one.
    """
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3, transformer_y=StandardScaler())
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)
    save_forecaster(forecaster=forecaster, file_name='forecaster.joblib', verbose=True)
    forecaster_loaded = load_forecaster(file_name='forecaster.joblib', verbose=True)
    os.remove('forecaster.joblib')

    for key in vars(forecaster).keys():
        attribute_forecaster = forecaster.__getattribute__(key)
        attribute_forecaster_loaded = forecaster_loaded.__getattribute__(key)

        if key in ['regressor', 'binner', 'transformer_y', 'transformer_exog']:
            assert joblib.hash(attribute_forecaster) == joblib.hash(attribute_forecaster_loaded)
        elif isinstance(attribute_forecaster, (np.ndarray, pd.Series, pd.DataFrame, pd.Index)):
            assert (attribute_forecaster == attribute_forecaster_loaded).all()
        elif isinstance(attribute_forecaster, dict):
            assert attribute_forecaster.keys() == attribute_forecaster_loaded.keys()
            for k in attribute_forecaster.keys():
                if isinstance(attribute_forecaster[k], (np.ndarray, pd.Series, pd.DataFrame, pd.Index)):
                    assert (attribute_forecaster[k] == attribute_forecaster_loaded[k]).all()
                else:
                    assert attribute_forecaster[k] == attribute_forecaster_loaded[k]
        else:
            assert attribute_forecaster == attribute_forecaster_loaded


def test_save_and_load_forecaster_SkforecastVersionWarning():
    """ 
    Test warning used to notify that the skforecast version installed in the 
    environment differs from the version used to create the forecaster.
    """
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=3)
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)
    forecaster.skforecast_version = '0.0.0'
    save_forecaster(forecaster=forecaster, file_name='forecaster.joblib', verbose=False)

    warn_msg = re.escape(
        (f"The skforecast version installed in the environment differs "
         f"from the version used to create the forecaster.\n"
         f"    Installed Version  : {skforecast.__version__}\n"
         f"    Forecaster Version : 0.0.0\n"
         f"This may create incompatibilities when using the library.")
    )
    with pytest.warns(SkforecastVersionWarning, match = warn_msg):
        load_forecaster(file_name='forecaster.joblib', verbose=False)
        os.remove('forecaster.joblib')


@pytest.mark.parametrize("weight_func", 
                         [custom_weights, 
                          {'serie_1': custom_weights, 
                           'serie_2': custom_weights2}], 
                         ids = lambda func : f'type: {type(func)}')
def test_save_forecaster_save_custom_functions(weight_func):
    """ 
    Test if custom functions are saved correctly.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 3,
                     weight_func    = weight_func
                 )
    save_forecaster(forecaster=forecaster, file_name='forecaster.joblib', 
                    save_custom_functions=True)
    os.remove('forecaster.joblib')

    fun_predictors_file = forecaster.fun_predictors.__name__ + '.py'
    assert os.path.exists(fun_predictors_file)
    with open(fun_predictors_file, 'r') as file:
        assert inspect.getsource(forecaster.fun_predictors) == file.read()
    os.remove(fun_predictors_file)
    
    weight_functions = weight_func.values() if isinstance(weight_func, dict) else [weight_func]
    for weight_func in weight_functions:
        weight_func_file = weight_func.__name__ + '.py'
        assert os.path.exists(weight_func_file)
        with open(weight_func_file, 'r') as file:
            assert inspect.getsource(weight_func) == file.read()
        os.remove(weight_func_file)


@pytest.mark.parametrize("weight_func", 
                         [None,
                          custom_weights, 
                          {'serie_1': custom_weights, 
                           'serie_2': custom_weights2}], 
                         ids = lambda func : f'func: {func}')
def test_save_forecaster_warning_dont_save_custom_functions(weight_func):
    """ 
    Test UserWarning when custom functions are not saved.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 3,
                     weight_func    = weight_func
                 )

    warn_msg = re.escape(
        ("Custom functions used to create predictors or weights are not saved. "
         "To save them, set `save_custom_functions` to `True`.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        save_forecaster(forecaster=forecaster, file_name='forecaster.joblib', 
                        save_custom_functions=False)
        os.remove('forecaster.joblib')