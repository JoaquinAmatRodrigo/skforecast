################################################################################
#                             skforecast.utils                                 #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Any, Optional, Tuple, Callable
import warnings
import importlib
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
import inspect
from copy import deepcopy

from ..exceptions import MissingValuesExogWarning
from ..exceptions import DataTypeWarning

optional_dependencies = {
    "sarimax": ['pmdarima>=2.0, <2.1'],
    "plotting": ['matplotlib>=3.3, <3.8', 'seaborn>=0.11, <0.13', 'statsmodels>=0.12, <0.14']
}


def initialize_lags(
    forecaster_name: str,
    lags: Any
) -> np.ndarray:
    """
    Check lags argument input and generate the corresponding numpy ndarray.
    
    Parameters
    ----------
    forecaster_name : str
        Forecaster name. ForecasterAutoreg, ForecasterAutoregCustom, 
        ForecasterAutoregDirect, ForecasterAutoregMultiSeries, 
        ForecasterAutoregMultiVariate.

    lags : Any
        Lags used as predictors.
        
    Returns
    ----------
    lags : numpy ndarray
        Lags used as predictors.
    
    """

    if isinstance(lags, int) and lags < 1:
        raise ValueError('Minimum value of lags allowed is 1.')

    if isinstance(lags, (list, np.ndarray)):
        for lag in lags:
            if not isinstance(lag, (int, np.int64, np.int32)):
                raise TypeError('All values in `lags` must be int.')
        
    if isinstance(lags, (list, range, np.ndarray)) and min(lags) < 1:
        raise ValueError('Minimum value of lags allowed is 1.')

    if isinstance(lags, int):
        lags = np.arange(lags) + 1
    elif isinstance(lags, (list, range)):
        lags = np.array(lags)
    elif isinstance(lags, np.ndarray):
        lags = lags
    else:
        if not forecaster_name == 'ForecasterAutoregMultiVariate':
            raise TypeError(
                ("`lags` argument must be an int, 1d numpy ndarray, range or list. "
                 f"Got {type(lags)}.")
            )
        else:
            raise TypeError(
                ("`lags` argument must be a dict, int, 1d numpy ndarray, range or list. "
                 f"Got {type(lags)}.")
            )

    return lags


def initialize_weights(
    forecaster_name: str,
    regressor: object,
    weight_func: Union[Callable, dict],
    series_weights: dict
) -> Tuple[Union[Callable, dict], Union[Callable, dict], dict]:
    """
    Check weights arguments, `weight_func` and `series_weights` for the different 
    forecasters. Create `source_code_weight_func`, source code of the custom 
    function(s) used to create weights.
    
    Parameters
    ----------
    forecaster_name : str
        Forecaster name. ForecasterAutoreg, ForecasterAutoregCustom, 
        ForecasterAutoregDirect, ForecasterAutoregMultiSeries, 
        ForecasterAutoregMultiVariate, ForecasterAutoregMultiSeriesCustom.

    regressor : regressor or pipeline compatible with the scikit-learn API
        Regressor of the forecaster.

    weight_func : Callable, dict
        Argument `weight_func` of the forecaster.

    series_weights : dict
        Argument `series_weights` of the forecaster.
        
        
    Returns
    ----------
    weight_func : Callable, dict
        Argument `weight_func` of the forecaster.

    source_code_weight_func : str, dict
        Argument `source_code_weight_func` of the forecaster.

    series_weights : dict
        Argument `series_weights` of the forecaster.
    
    """

    source_code_weight_func = None

    if weight_func is not None:

        if forecaster_name in ['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiSeriesCustom']:
            if not isinstance(weight_func, (Callable, dict)):
                raise TypeError(
                    (f"Argument `weight_func` must be a Callable or a dict of "
                     f"Callables. Got {type(weight_func)}.")
                )
        elif not isinstance(weight_func, Callable):
            raise TypeError(
                f"Argument `weight_func` must be a Callable. Got {type(weight_func)}."
            )
        
        if isinstance(weight_func, dict):
            source_code_weight_func = {}
            for key in weight_func:
                source_code_weight_func[key] = inspect.getsource(weight_func[key])
        else:
            source_code_weight_func = inspect.getsource(weight_func)

        if 'sample_weight' not in inspect.signature(regressor.fit).parameters:
            warnings.warn(
                (f"Argument `weight_func` is ignored since regressor {regressor} "
                 f"does not accept `sample_weight` in its `fit` method.")
            )
            weight_func = None
            source_code_weight_func = None

    if series_weights is not None:
        if not isinstance(series_weights, dict):
            raise TypeError(
                (f"Argument `series_weights` must be a dict of floats or ints."
                 f"Got {type(series_weights)}.")
            )
        if 'sample_weight' not in inspect.signature(regressor.fit).parameters:
            warnings.warn(
                (f"Argument `series_weights` is ignored since regressor {regressor} "
                 f"does not accept `sample_weight` in its `fit` method.")
            )
            series_weights = None

    return weight_func, source_code_weight_func, series_weights


def check_select_fit_kwargs(
    regressor: object,
    fit_kwargs: Optional[dict]=None
) -> dict:
    """
    Check if `fit_kwargs` is a dict and select only the keys that are used by
    the `fit` method of the regressor.

    Parameters
    ----------
    regressor : object
        Regressor object.

    fit_kwargs : dict, default `None`
        Dictionary with the arguments to pass to the `fit' method of the 
        forecaster.

    Returns
    -------
    fit_kwargs : dict
        Dictionary with the arguments to be passed to the `fit` method of the 
        regressor after removing the unused keys.
    
    """

    if fit_kwargs is None:
        fit_kwargs = {}
    else:
        if not isinstance(fit_kwargs, dict):
            raise TypeError(
                f"Argument `fit_kwargs` must be a dict. Got {type(fit_kwargs)}."
            )

        # Non used keys
        non_used_keys = [k for k in fit_kwargs.keys()
                        if k not in inspect.signature(regressor.fit).parameters]
        if non_used_keys:
            warnings.warn(
                (f"Argument/s {non_used_keys} ignored since they are not used by the "
                 f"regressor's `fit` method.")
            )

        if 'sample_weight' in fit_kwargs.keys():
            warnings.warn(
                ("The `sample_weight` argument is ignored. Use `weight_func` to pass "
                 "a function that defines the individual weights for each sample "
                 "based on its index.")
            )
            del fit_kwargs['sample_weight']

        # Select only the keyword arguments allowed by the regressor's `fit` method.
        fit_kwargs = {k:v for k, v in fit_kwargs.items()
                      if k in inspect.signature(regressor.fit).parameters}

    return fit_kwargs


def check_y(
    y: Any
) -> None:
    """
    Raise Exception if `y` is not pandas Series or if it has missing values.
    
    Parameters
    ----------        
    y : Any
        Time series values.
        
    Returns
    ----------
    None
    
    """
    
    if not isinstance(y, pd.Series):
        raise TypeError('`y` must be a pandas Series.')
        
    if y.isnull().any():
        raise ValueError('`y` has missing values.')
    
    return
    
    
def check_exog(
    exog: Any,
    allow_nan: bool=True
) -> None:
    """
    Raise Exception if `exog` is not pandas Series or pandas DataFrame.
    If `allow_nan = True`, issue a warning if `exog` contains NaN values.
    
    Parameters
    ----------        
    exog :  Any
        Exogenous variable/s included as predictor/s.
    allow_nan: bool, default True
        If True, allows the presence of NaN values in `exog`. If False (default),
        issue a warning if `exog` contains NaN values.

    Returns
    ----------
    None

    """
        
    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError("`exog` must be a pandas Series or DataFrame.")

    if not allow_nan:
        if exog.isnull().any().any():
            warnings.warn(
                ("`exog` has missing values. Most machine learning models do not allow "
                 "missing values. Fitting the forecaster may fail."), MissingValuesExogWarning
        )
         
    return


def get_exog_dtypes(
    exog: Union[pd.DataFrame, pd.Series]
) -> dict:
    """
    Store dtypes of `exog`.

    Parameters
    ----------
    exog :  pandas DataFrame, pandas Series
        Exogenous variable/s included as predictor/s.

    Returns
    -------
    exog_dtypes : dict
        Dictionary with the dtypes in `exog`.
    
    """

    if isinstance(exog, pd.Series):
        exog_dtypes = {exog.name: exog.dtypes}
    else:
        exog_dtypes = exog.dtypes.to_dict()
    
    return exog_dtypes


def check_exog_dtypes(
    exog: Union[pd.DataFrame, pd.Series]
) -> None:
    """
    Raise Exception if `exog` has categorical columns with non integer values.
    This is needed when using machine learning regressors that allow categorical
    features.
    Rise a Warning if values are not `init`, `float`, or `category`.
    
    Parameters
    ----------        
    exog :  pandas DataFrame, pandas Series
        Exogenous variable/s included as predictor/s.

    Returns
    ----------
    None

    """
    check_exog(exog=exog, allow_nan=False)

    if isinstance(exog, pd.DataFrame):
        if not all([dtype in ['float', 'int', 'category'] for dtype in exog.dtypes]):
            warnings.warn(
                ("`exog` may contain only `int`, `float` or `category` dtypes. Most "
                 "machine learning models do not allow other types of values. "
                 "Fitting the forecaster may fail."), DataTypeWarning
            )
        for col in exog.select_dtypes(include='category'):
            if exog[col].cat.categories.dtype not in [int, np.int32, np.int64]:
                raise TypeError(
                    ("Categorical columns in exog must contain only integer values. "
                     "See skforecast docs for more info about how to include categorical "
                     "features https://skforecast.org/"
                     "latest/user_guides/categorical-features.html")
                )
    else:
        if exog.dtypes not in ['float', 'int', 'category']:
            warnings.warn(
                ("`exog` may contain only `int`, `float` or `category` dtypes. Most "
                 "machine learning models do not allow other types of values. "
                 "Fitting the forecaster may fail."), DataTypeWarning
            )
        if exog.dtypes == 'category' and exog.cat.categories.dtype not in [int, np.int32, np.int64]:
            raise TypeError(
                ("If exog is of type category, it must contain only integer values. "
                 "See skforecast docs for more info about how to include categorical "
                 "features https://skforecast.org/"
                 "latest/user_guides/categorical-features.html")
            )
         
    return


def check_interval(
    interval: list=None,
    alpha: float=None
) -> None:
    """
    Check provided confidence interval sequence is valid.

    Parameters
    ----------
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`.

    alpha : float, default `None`
        The confidence intervals used in ForecasterSarimax are (1 - alpha) %.
    
    """

    if interval is not None:
        if not isinstance(interval, list):
            raise TypeError(
                ("`interval` must be a `list`. For example, interval of 95% "
                 "should be as `interval = [2.5, 97.5]`.")
            )

        if len(interval) != 2:
            raise ValueError(
                ("`interval` must contain exactly 2 values, respectively the "
                 "lower and upper interval bounds. For example, interval of 95% "
                 "should be as `interval = [2.5, 97.5]`.")
            )

        if (interval[0] < 0.) or (interval[0] >= 100.):
            raise ValueError(
                f'Lower interval bound ({interval[0]}) must be >= 0 and < 100.'
            )

        if (interval[1] <= 0.) or (interval[1] > 100.):
            raise ValueError(
                f'Upper interval bound ({interval[1]}) must be > 0 and <= 100.'
            )

        if interval[0] >= interval[1]:
            raise ValueError(
                f'Lower interval bound ({interval[0]}) must be less than the '
                f'upper interval bound ({interval[1]}).'
            )
    
    if alpha is not None:
        if not isinstance(alpha, float):
            raise TypeError(
                ('`alpha` must be a `float`. For example, interval of 95% '
                 'should be as `alpha = 0.05`.')
            )

        if (alpha <= 0.) or (alpha >= 1):
            raise ValueError(
                f'`alpha` must have a value between 0 and 1. Got {alpha}.'
            )

    return


def check_predict_input(
    forecaster_name: str,
    steps: int,
    fitted: bool,
    included_exog: bool,
    index_type: type,
    index_freq: str,
    window_size: int,
    last_window: Optional[Union[pd.Series, pd.DataFrame]]=None,
    last_window_exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    exog_type: Optional[Union[type, None]]=None,
    exog_col_names: Optional[Union[list, None]]=None,
    interval: Optional[list]=None,
    alpha: Optional[float]=None,
    max_steps: Optional[int]=None,
    levels: Optional[Union[str, list]]=None,
    series_col_names: Optional[list]=None
) -> None:
    """
    Check all inputs of predict method. This is a helper function to validate
    that inputs used in predict method match attributes of a forecaster already
    trained.

    Parameters
    ----------
    forecaster_name : str
        Forecaster name. ForecasterAutoreg, ForecasterAutoregCustom, 
        ForecasterAutoregDirect, ForecasterAutoregMultiSeries, 
        ForecasterAutoregMultiVariate, ForecasterAutoregMultiSeriesCustom.

    steps : int
        Number of future steps predicted.

    fitted: Bool
        Tag to identify if the regressor has been fitted (trained).

    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.

    index_type : type
        Type of index of the input used in training.

    index_freq : str
        Frequency of Index of the input used in training.

    window_size: int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.

    last_window : pandas Series, pandas DataFrame, default `None`
        Values of the series used to create the predictors (lags) need in the 
        first iteration of prediction (t + 1).

    last_window_exog : pandas Series, pandas DataFrame, default `None`
        Values of the exogenous variables aligned with `last_window` in 
        ForecasterSarimax predictions.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s.

    exog_type : type, default `None`
        Type of exogenous variable/s used in training.
        
    exog_col_names : list, default `None`
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`.

    alpha : float, default `None`
        The confidence intervals used in ForecasterSarimax are (1 - alpha) %.

    max_steps: int, default `None`
        Maximum number of steps allowed (`ForecasterAutoregDirect` and 
        `ForecasterAutoregMultiVariate`).
            
    levels : str, list, default `None`
        Time series to be predicted (`ForecasterAutoregMultiSeries` and
        `ForecasterAutoregMultiSeriesCustom`).

    series_col_names : list, default `None`
        Names of the columns used during fit (`ForecasterAutoregMultiSeries`, 
        `ForecasterAutoregMultiVariate` and `ForecasterAutoregMultiSeriesCustom`).
    
    """

    if not fitted:
        raise sklearn.exceptions.NotFittedError(
            ('This Forecaster instance is not fitted yet. Call `fit` with '
             'appropriate arguments before using predict.')
        )
    
    if isinstance(steps, int) and steps < 1:
        raise ValueError(
            f'`steps` must be an integer greater than or equal to 1. Got {steps}.'
        )

    if isinstance(steps, list) and min(steps) < 1:
        raise ValueError(
           (f"The minimum value of `steps` must be equal to or greater than 1. "
            f"Got {min(steps)}.")
        )

    if max_steps is not None:
        if max(steps) > max_steps:
            raise ValueError(
                (f"The maximum value of `steps` must be less than or equal to "
                 f"the value of steps defined when initializing the forecaster. "
                 f"Got {max(steps)}, but the maximum is {max_steps}.")
            )

    if interval is not None or alpha is not None:
        check_interval(interval=interval, alpha=alpha)
    
    if forecaster_name in ['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiSeriesCustom']:
        if levels is not None and not isinstance(levels, (str, list)):
            raise TypeError(
                '`levels` must be a `list` of column names, a `str` of a column name or `None`.'
            )
        if len(set(levels) - set(series_col_names)) != 0:
            raise ValueError(
                f'`levels` must be in `series_col_names` : {series_col_names}.'
            )

    if exog is None and included_exog:
        raise ValueError(
            ('Forecaster trained with exogenous variable/s. '
             'Same variable/s must be provided when predicting.')
        )
        
    if exog is not None and not included_exog:
        raise ValueError(
            ('Forecaster trained without exogenous variable/s. '
             '`exog` must be `None` when predicting.')
        )
        
    # Checks last_window
    # Check last_window type (pd.Series or pd.DataFrame according to forecaster)
    if forecaster_name in ['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiVariate',
                           'ForecasterAutoregMultiSeriesCustom']:
        if not isinstance(last_window, pd.DataFrame):
            raise TypeError(
                f'`last_window` must be a pandas DataFrame. Got {type(last_window)}.'
            )
        
        if forecaster_name in ['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiSeriesCustom'] and \
            len(set(levels) - set(last_window.columns)) != 0:
            raise ValueError(
                (f'`last_window` must contain a column(s) named as the level(s) to be predicted.\n'
                 f'    `levels` : {levels}.\n'
                 f'    `last_window` columns : {list(last_window.columns)}.')
            )
        
        if forecaster_name == 'ForecasterAutoregMultiVariate' and \
            (series_col_names != list(last_window.columns)):
            raise ValueError(
                (f'`last_window` columns must be the same as `series` column names.\n'
                 f'    `last_window` columns : {list(last_window.columns)}.\n'
                 f'    `series` columns      : {series_col_names}.')
            )    
    else:    
        if not isinstance(last_window, pd.Series):
            raise TypeError(
                f'`last_window` must be a pandas Series. Got {type(last_window)}.'
            )
    
    # Check last_window len, nulls and index (type and freq)
    if len(last_window) < window_size:
        raise ValueError(
            (f'`last_window` must have as many values as needed to '
             f'generate the predictors. For this forecaster it is {window_size}.')
        )
    if last_window.isnull().any().all():
        raise ValueError(
            ("`last_window` has missing values.")
        )
    _, last_window_index = preprocess_last_window(
                               last_window  = last_window.iloc[:0],
                               return_values = False
                           ) 
    if not isinstance(last_window_index, index_type):
        raise TypeError(
            (f'Expected index of type {index_type} for `last_window`. '
             f'Got {type(last_window_index)}.')
        )
    if isinstance(last_window_index, pd.DatetimeIndex):
        if not last_window_index.freqstr == index_freq:
            raise TypeError(
                (f'Expected frequency of type {index_freq} for `last_window`. '
                 f'Got {last_window_index.freqstr}.')
            )

    # Checks exog
    if exog is not None:
        # Check type, nulls and expected type
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise TypeError('`exog` must be a pandas Series or DataFrame.')
        if exog.isnull().any().any():
            warnings.warn(
                ('`exog` has missing values. Most of machine learning models do not allow '
                 'missing values. `predict` method may fail.'), MissingValuesExogWarning
            )
        if not isinstance(exog, exog_type):
            raise TypeError(
                f"Expected type for `exog`: {exog_type}. Got {type(exog)}."    
            )
        
        # Check exog has many values as distance to max step predicted
        last_step = max(steps) if isinstance(steps, list) else steps
        if len(exog) < last_step:
            raise ValueError(
                (f'`exog` must have at least as many values as the distance to '
                 f'the maximum step predicted, {last_step}.')
            )

        # Check all columns are in the pandas DataFrame
        if isinstance(exog, pd.DataFrame):
            col_missing = set(exog_col_names).difference(set(exog.columns))
            if col_missing:
                raise ValueError(
                    (f"Missing columns in `exog`. Expected {exog_col_names}. "
                     f"Got {exog.columns.to_list()}.") 
                )

        # Check index dtype and freq
        _, exog_index = preprocess_exog(
                            exog          = exog.iloc[:0, ],
                            return_values = False
                        )
        if not isinstance(exog_index, index_type):
            raise TypeError(
                (f'Expected index of type {index_type} for `exog`. '
                 f'Got {type(exog_index)}.')
            )   
        if isinstance(exog_index, pd.DatetimeIndex):
            if not exog_index.freqstr == index_freq:
                raise TypeError(
                    (f'Expected frequency of type {index_freq} for `exog`. '
                     f'Got {exog_index.freqstr}.')
                )

        # Check exog starts one step ahead of last_window end.
        expected_index = expand_index(last_window.index, 1)[0]
        if expected_index != exog.index[0]:
            raise ValueError(
                (f'To make predictions `exog` must start one step ahead of `last_window`.\n'
                 f'    `last_window` ends at : {last_window.index[-1]}.\n'
                 f'    `exog` starts at      : {exog.index[0]}.\n'
                 f'     Expected index       : {expected_index}.')
            )

    # Checks ForecasterSarimax
    if forecaster_name == 'ForecasterSarimax':
        # Check last_window_exog type, len, nulls and index (type and freq)
        if last_window_exog is not None:
            if not included_exog:
                raise ValueError(
                    ('Forecaster trained without exogenous variable/s. '
                     '`last_window_exog` must be `None` when predicting.')
                )

            if not isinstance(last_window_exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    (f'`last_window_exog` must be a pandas Series or a '
                     f'pandas DataFrame. Got {type(last_window_exog)}.')
                )
            if len(last_window_exog) < window_size:
                raise ValueError(
                    (f'`last_window_exog` must have as many values as needed to '
                     f'generate the predictors. For this forecaster it is {window_size}.')
                )
            if last_window_exog.isnull().any().all():
                warnings.warn(
                ('`last_window_exog` has missing values. Most of machine learning models '
                 'do not allow missing values. `predict` method may fail.'),
                MissingValuesExogWarning
            )
            _, last_window_exog_index = preprocess_last_window(
                                            last_window   = last_window_exog.iloc[:0],
                                            return_values = False
                                        ) 
            if not isinstance(last_window_exog_index, index_type):
                raise TypeError(
                    (f'Expected index of type {index_type} for `last_window_exog`. '
                     f'Got {type(last_window_exog_index)}.')
                )
            if isinstance(last_window_exog_index, pd.DatetimeIndex):
                if not last_window_exog_index.freqstr == index_freq:
                    raise TypeError(
                        (f'Expected frequency of type {index_freq} for `last_window_exog`. '
                         f'Got {last_window_exog_index.freqstr}.')
                    )

            # Check all columns are in the pd.DataFrame, last_window_exog
            if isinstance(last_window_exog, pd.DataFrame):
                col_missing = set(exog_col_names).difference(set(last_window_exog.columns))
                if col_missing:
                    raise ValueError(
                        (f'Missing columns in `exog`. Expected {exog_col_names}. '
                         f'Got {last_window_exog.columns.to_list()}.') 
                    )

    return


def preprocess_y(
    y: pd.Series,
    return_values: bool=True
) -> Tuple[Union[None, np.ndarray], pd.Index]:
    """
    Returns values and index of series separately. Index is overwritten 
    according to the next rules:
        If index is of type DatetimeIndex and has frequency, nothing is 
        changed.
        If index is of type RangeIndex, nothing is changed.
        If index is of type DatetimeIndex but has no frequency, a 
        RangeIndex is created.
        If index is not of type DatetimeIndex, a RangeIndex is created.
    
    Parameters
    ----------        
    y : pandas Series
        Time series.

    return_values : bool, default `True`
        If `True` return the values of `y` as numpy ndarray. This option is 
        intended to avoid copying data when it is not necessary.

    Returns 
    -------
    y_values : None, numpy ndarray
        Numpy array with values of `y`.

    y_index : pandas Index
        Index of `y` modified according to the rules.
    
    """
    
    if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None:
        y_index = y.index
    elif isinstance(y.index, pd.RangeIndex):
        y_index = y.index
    elif isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
        warnings.warn(
            ("`y` has DatetimeIndex index but no frequency. "
             "Index is overwritten with a RangeIndex of step 1.")
        )
        y_index = pd.RangeIndex(
                      start = 0,
                      stop  = len(y),
                      step  = 1
                  )
    else:
        warnings.warn(
            ("`y` has no DatetimeIndex nor RangeIndex index. "
             "Index is overwritten with a RangeIndex.")
        )
        y_index = pd.RangeIndex(
                      start = 0,
                      stop  = len(y),
                      step  = 1
                  )

    y_values = y.to_numpy() if return_values else None

    return y_values, y_index


def preprocess_last_window(
    last_window: Union[pd.Series, pd.DataFrame],
    return_values: bool=True
 ) -> Tuple[np.ndarray, pd.Index]:
    """
    Returns values and index of series separately. Index is overwritten 
    according to the next rules:
        If index is of type DatetimeIndex and has frequency, nothing is 
        changed.
        If index is of type RangeIndex, nothing is changed.
        If index is of type DatetimeIndex but has no frequency, a 
        RangeIndex is created.
        If index is not of type DatetimeIndex, a RangeIndex is created.
    
    Parameters
    ----------        
    last_window : pandas Series, pandas DataFrame
        Time series values.

    return_values : bool, default `True`
        If `True` return the values of `last_window` as numpy ndarray. This option 
        is intended to avoid copying data when it is not necessary.

    Returns 
    -------
    last_window_values : numpy ndarray
        Numpy array with values of `last_window`.

    last_window_index : pandas Index
        Index of `last_window` modified according to the rules.
    
    """
    
    if isinstance(last_window.index, pd.DatetimeIndex) and last_window.index.freq is not None:
        last_window_index = last_window.index
    elif isinstance(last_window.index, pd.RangeIndex):
        last_window_index = last_window.index
    elif isinstance(last_window.index, pd.DatetimeIndex) and last_window.index.freq is None:
        warnings.warn(
            ("`last_window` has DatetimeIndex index but no frequency. "
             "Index is overwritten with a RangeIndex of step 1.")
        )
        last_window_index = pd.RangeIndex(
                                start = 0,
                                stop  = len(last_window),
                                step  = 1
                            )
    else:
        warnings.warn(
            ("`last_window` has no DatetimeIndex nor RangeIndex index. "
             "Index is overwritten with a RangeIndex.")
        )
        last_window_index = pd.RangeIndex(
                                start = 0,
                                stop  = len(last_window),
                                step  = 1
                            )

    last_window_values = last_window.to_numpy() if return_values else None

    return last_window_values, last_window_index


def preprocess_exog(
    exog: Union[pd.Series, pd.DataFrame],
    return_values: bool=True
) -> Tuple[Union[None, np.ndarray], pd.Index]:
    """
    Returns values and index of series or data frame separately. Index is
    overwritten  according to the next rules:
        If index is of type DatetimeIndex and has frequency, nothing is 
        changed.
        If index is of type RangeIndex, nothing is changed.
        If index is of type DatetimeIndex but has no frequency, a 
        RangeIndex is created.
        If index is not of type DatetimeIndex, a RangeIndex is created.

    Parameters
    ----------        
    exog : pandas Series, pandas DataFrame
        Exogenous variables.

    return_values : bool, default `True`
        If `True` return the values of `exog` as numpy ndarray. This option is 
        intended to avoid copying data when it is not necessary.

    Returns 
    -------
    exog_values : None, numpy ndarray
        Numpy array with values of `exog`.

    exog_index : pandas Index
        Index of `exog` modified according to the rules.
    
    """
    
    if isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is not None:
        exog_index = exog.index
    elif isinstance(exog.index, pd.RangeIndex):
        exog_index = exog.index
    elif isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is None:
        warnings.warn(
            ('`exog` has DatetimeIndex index but no frequency. '
             'Index is overwritten with a RangeIndex of step 1.')
        )
        exog_index = pd.RangeIndex(
                         start = 0,
                         stop  = len(exog),
                         step  = 1
                     )

    else:
        warnings.warn(
            ('`exog` has no DatetimeIndex nor RangeIndex index. '
             'Index is overwritten with a RangeIndex.')
        )
        exog_index = pd.RangeIndex(
                         start = 0,
                         stop  = len(exog),
                         step  = 1
                     )

    exog_values = exog.to_numpy() if return_values else None

    return exog_values, exog_index
    

def cast_exog_dtypes(
    exog: Union[pd.Series, pd.DataFrame],
    exog_dtypes: dict,
) -> Union[pd.Series, pd.DataFrame]: # pragma: no cover
    """
    Cast `exog` to a specified types.
    If `exog` is a pandas Series, `exog_dtypes` must be a dict with a single value.
    If `exog_dtypes` is `category` but the current type of `exog` is `float`, then
    the type is cast to `int` and then to `category`. This is done because, for
    a forecaster to accept a categorical exog, it must contain only integer values.
    Due to the internal modifications of numpy, the values may be casted to `float`,
    so they have to be re-converted to `int`.

    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variables.
    
    exog_dtypes: dict
        Dictionary with name and type of the series or data frame columns.

    Returns 
    -------
    exog

    """

    # Remove keys from exog_dtypes not in exog.columns
    exog_dtypes = {k:v for k, v in exog_dtypes.items() if k in exog.columns}
    
    if isinstance(exog, pd.Series) and exog.dtypes != list(exog_dtypes.values())[0]:
        exog = exog.astype(list(exog_dtypes.values())[0])
    elif isinstance(exog, pd.DataFrame):
        for col, initial_dtype in exog_dtypes.items():
            if exog[col].dtypes != initial_dtype:
                if initial_dtype == "category" and exog[col].dtypes==float:
                    exog[col] = exog[col].astype(int).astype("category")
                else:
                    exog[col] = exog[col].astype(initial_dtype)

    return exog


def exog_to_direct(
    exog: Union[pd.Series, pd.DataFrame],
    steps: int
)-> pd.DataFrame:
    """
    Transforms `exog` to a pandas DataFrame with the shape needed for Direct
    forecasting.
    
    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variables.

    steps : int.
        Number of steps that will be predicted using exog.

    Returns 
    -------
    exog_transformed : pandas DataFrame
        Exogenous variables transformed.
    
    """

    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError(f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}.")

    if isinstance(exog, pd.Series):
        exog = exog.to_frame()

    len_columns = len(exog)
    exog_idx = exog.index
    exog_transformed = []
    for column in exog.columns:

        exog_column_transformed = [
            (exog[column].iloc[i : len_columns - (steps - 1 - i)]).reset_index(drop=True)
            for i in range(steps)
        ]
        exog_column_transformed = pd.concat(exog_column_transformed, axis=1)
        exog_column_transformed.columns = [f"{column}_step_{i+1}" for i in range(steps)]

        exog_transformed.append(exog_column_transformed)

    if len(exog_transformed) > 1:
        exog_transformed = pd.concat(exog_transformed, axis=1)
    else:
        exog_transformed = exog_column_transformed

    exog_transformed.index = exog_idx[-len(exog_transformed):]

    return exog_transformed


def exog_to_direct_numpy(
    exog: np.ndarray,
    steps: int
)-> np.ndarray:
    """
    Transforms `exog` to `np.ndarray` with the shape needed for direct
    forecasting.
    
    Parameters
    ----------        
    exog : numpy ndarray, shape(samples,)
        Exogenous variables.

    steps : int.
        Number of steps that will be predicted using exog.

    Returns 
    -------
    exog_transformed : numpy ndarray
        Exogenous variables transformed.

    """

    exog_transformed = []
    
    if exog.ndim == 1:
        exog = np.expand_dims(exog, axis=1)

    for i in range(exog.shape[1]):
        exog_column = exog[:, i]
        exog_column_transformed = np.vstack(
            [np.roll(exog_column, j) for j in range(steps)]
        ).T[steps - 1:]
        exog_column_transformed = exog_column_transformed[:, ::-1]
        exog_transformed.append(exog_column_transformed)

    if len(exog_transformed) > 1:
        exog_transformed = np.concatenate(exog_transformed, axis=1)
    else:
        exog_transformed = exog_column_transformed

    return exog_transformed


def expand_index(
    index: Union[pd.Index, None], 
    steps: int
) -> pd.Index:
    """
    Create a new index of length `steps` starting at the end of the index.
    
    Parameters
    ----------        
    index : pd.Index, None
        Index of last window.
    
    steps : int
        Number of steps to expand.

    Returns 
    -------
    new_index : pd.Index
        New index.

    """
    
    if isinstance(index, pd.Index):
        
        if isinstance(index, pd.DatetimeIndex):
            new_index = pd.date_range(
                            index[-1] + index.freq,
                            periods = steps,
                            freq    = index.freq
                        )
        elif isinstance(index, pd.RangeIndex):
            new_index = pd.RangeIndex(
                            start = index[-1] + 1,
                            stop  = index[-1] + 1 + steps
                        )
    else: 
        new_index = pd.RangeIndex(
                        start = 0,
                        stop  = steps
                    )
    
    return new_index


def transform_series(
    series: pd.Series,
    transformer,
    fit: bool=False,
    inverse_transform: bool=False
) -> Union[pd.Series, pd.DataFrame]:
    """      
    Transform raw values of pandas Series with a scikit-learn alike transformer
    (preprocessor). The transformer used must have the following methods: fit, transform,
    fit_transform and inverse_transform. ColumnTransformers are not allowed since they
    do not have inverse_transform method.

    Parameters
    ----------
    series : pandas Series
        Series to be transformed.

    transformer : scikit-learn alike transformer (preprocessor).
        scikit-learn alike transformer (preprocessor) with methods: fit, transform,
        fit_transform and inverse_transform. ColumnTransformers are not allowed since they
        do not have inverse_transform method.

    fit : bool, default `False`
        Train the transformer before applying it.

    inverse_transform : bool, default `False`
        Transform back the data to the original representation.

    Returns
    -------
    series_transformed : pandas Series, pandas DataFrame
        Transformed Series. Depending on the transformer used, the output may be a Series
        or a DataFrame.

    """
    
    if not isinstance(series, pd.Series):
        raise TypeError(
            "`series` argument must be a pandas Series."
        )
        
    if transformer is None:
        return series

    if series.name is None:
        series.name = 'no_name'
        
    data = series.to_frame()

    if fit and hasattr(transformer, 'fit'):
        transformer.fit(data)

    # If argument feature_names_in_ exits, is overwritten to allow using the 
    # transformer on other series than those that were passed during fit.
    if hasattr(transformer, 'feature_names_in_') and transformer.feature_names_in_[0] != data.columns[0]:
        transformer = deepcopy(transformer)
        transformer.feature_names_in_ = np.array([data.columns[0]], dtype=object)

    if inverse_transform:
        values_transformed = transformer.inverse_transform(data)
    else:
        values_transformed = transformer.transform(data)   

    if hasattr(values_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense array.
        values_transformed = values_transformed.toarray()
    
    if isinstance(values_transformed, np.ndarray) and values_transformed.shape[1] == 1:
        series_transformed = pd.Series(
                                 data  = values_transformed.flatten(),
                                 index = data.index,
                                 name  = data.columns[0]
                             )
    elif isinstance(values_transformed, pd.DataFrame) and values_transformed.shape[1] == 1:
        series_transformed = values_transformed.squeeze()
    else:
        series_transformed = pd.DataFrame(
                                 data    = values_transformed,
                                 index   = data.index,
                                 columns = transformer.get_feature_names_out()
                             )

    return series_transformed


def transform_dataframe(
    df: pd.DataFrame,
    transformer,
    fit: bool=False,
    inverse_transform: bool=False
) -> pd.DataFrame:
    """      
    Transform raw values of pandas DataFrame with a scikit-learn alike
    transformer, preprocessor or ColumnTransformer. `inverse_transform` is not available
    when using ColumnTransformers.

    Parameters
    ----------
    df : pandas DataFrame
        Pandas DataFrame to be transformed.

    transformer : scikit-learn alike transformer, preprocessor or ColumnTransformer.
        scikit-learn alike transformer, preprocessor or ColumnTransformer.

    fit : bool, default `False`
        Train the transformer before applying it.

    inverse_transform : bool, default `False`
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.

    Returns
    -------
    df_transformed : pandas DataFrame
        Transformed DataFrame.
    
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "`df` argument must be a pandas DataFrame."
        )

    if transformer is None:
        return df

    if inverse_transform and isinstance(transformer, ColumnTransformer):
        raise Exception(
            '`inverse_transform` is not available when using ColumnTransformers.'
        )
 
    if not inverse_transform:
        if fit:
            values_transformed = transformer.fit_transform(df)
        else:
            values_transformed = transformer.transform(df)
    else:
        values_transformed = transformer.inverse_transform(df)

    if hasattr(values_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense
        values_transformed = values_transformed.toarray()

    if hasattr(transformer, 'get_feature_names_out'):
        feature_names_out = transformer.get_feature_names_out()
    elif hasattr(transformer, 'categories_'):   
        feature_names_out = transformer.categories_
    else:
        feature_names_out = df.columns

    df_transformed = pd.DataFrame(
                         data    = values_transformed,
                         index   = df.index,
                         columns = feature_names_out
                     )

    return df_transformed


def save_forecaster(
    forecaster, 
    file_name: str, 
    verbose: bool=True
) -> None:
    """
    Save forecaster model using joblib.

    Parameters
    ----------
    forecaster: forecaster object from skforecast library.
        Forecaster created with skforecast library.

    file_name: str
        File name given to the object.
        
    verbose: bool, default `True`
        Print summary about the forecaster saved.

    Returns 
    -------
    None

    """

    joblib.dump(forecaster, filename=file_name)

    if verbose:
        forecaster.summary()


def load_forecaster(
    file_name: str,
    verbose: bool=True
) -> object:
    """
    Load forecaster model using joblib.

    Parameters
    ----------
    file_name: str
        Object file name.

    verbose: bool, default `True`
        Print summary about the forecaster loaded.

    Returns 
    -------
    Forecaster
        Forecaster created with skforecast library.
    
    """

    forecaster = joblib.load(filename=file_name)

    if verbose:
        forecaster.summary()

    return forecaster


def _find_optional_dependency(
    package_name: str, 
    optional_dependencies: dict=optional_dependencies
) -> Tuple[str, str]:
    """
    Find if a package is an optional dependency. If true, find the version and 
    the extension it belongs to.

    Parameters
    ----------
    package_name : str
        Name of the package to check.

    optional_dependencies : dict, default optional_dependencies
        Skforecast optional dependencies.

    Return
    ------
    extra: str
        Name of the extra extension where the optional dependency is needed.

    package_version: srt
        Name and versions of the dependency.

    """

    for extra, packages in optional_dependencies.items():
        package_version = [package for package in packages if package_name in package]
        if package_version:
            return extra, package_version[0]


def check_optional_dependency(
    package_name: str
) -> None:
    """
    Check if an optional dependency is installed, if not raise an ImportError  
    with installation instructions.

    Parameters
    ----------
    package_name : str
        Name of the package to check.
    
    """

    if importlib.util.find_spec(package_name) is None:
        try:
            extra, package_version = _find_optional_dependency(package_name=package_name)
            msg = (
                f"\n'{package_name}' is an optional dependency not included in the default "
                f"skforecast installation. Please run: `pip install \"{package_version}\"` to install it."
                f"\n\nAlternately, you can install it by running `pip install skforecast[{extra}]`"
            )
        except:
            msg = f"\n'{package_name}' is needed but not installed. Please install it."
        
        raise ImportError(msg)

    
def multivariate_time_series_corr(
    time_series: pd.Series,
    other: pd.DataFrame,
    lags: Union[int, list, np.array],
    method: str='pearson'
)-> pd.DataFrame:
    """
    Compute correlation between a time_series and the lagged values of other 
    time series. 

    Parameters
    ----------
    time_series : pandas Series
        Target time series.

    other : pandas DataFrame
        Time series whose lagged values are correlated to `time_series`.

    lags : Union[int, list, numpy ndarray]
        Lags to be included in the correlation analysis.
    
    method : str, default 'pearson'
        - pearson : standard correlation coefficient.
        - kendall : Kendall Tau correlation coefficient.
        - spearman : Spearman rank correlation.
        
    Returns
    -------
    corr : pandas DataFrame
        Correlation values.

    """

    if not len(time_series) == len(other):
        raise ValueError("`time_series` and `other` must have the same length.")

    if not (time_series.index == other.index).all():
        raise ValueError("`time_series` and `other` must have the same index.")

    if isinstance(lags, int):
        lags = range(lags)

    corr = {}
    for col in other.columns:
        lag_values = {}
        for lag in lags:
            lag_values[lag] = other[col].shift(lag)

        lag_values = pd.DataFrame(lag_values)
        lag_values.insert(0, None, time_series)
        corr[col] = lag_values.corr(method=method).iloc[1:, 0]

    corr = pd.DataFrame(corr)
    corr.index = corr.index.astype('int64')
    corr.index.name = "lag"
    
    return corr