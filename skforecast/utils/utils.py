################################################################################
#                               skforecast.utils                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Any, Optional, Tuple, Callable
import warnings
import importlib
import joblib
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
import inspect
from copy import deepcopy

import skforecast
from ..exceptions import MissingValuesExogWarning
from ..exceptions import DataTypeWarning
from ..exceptions import IgnoredArgumentWarning
from ..exceptions import SkforecastVersionWarning

optional_dependencies = {
    "sarimax": [
        'pmdarima>=2.0, <2.1',
        'statsmodels>=0.12, <0.15'
    ],
    "plotting": [
        'matplotlib>=3.3, <3.9', 
        'seaborn>=0.11, <0.14', 
        'statsmodels>=0.12, <0.15'
    ]
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
        ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate.
    lags : Any
        Lags used as predictors.

    Returns
    -------
    lags : numpy ndarray
        Lags used as predictors.
    
    """

    if isinstance(lags, int) and lags < 1:
        raise ValueError("Minimum value of lags allowed is 1.")

    if isinstance(lags, (list, np.ndarray)):
        for lag in lags:
            if not isinstance(lag, (int, np.int64, np.int32)):
                raise TypeError("All values in `lags` must be int.")
        
    if isinstance(lags, (list, range, np.ndarray)) and min(lags) < 1:
        raise ValueError("Minimum value of lags allowed is 1.")

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
) -> Tuple[Union[Callable, dict], Union[str, dict], dict]:
    """
    Check weights arguments, `weight_func` and `series_weights` for the different 
    forecasters. Create `source_code_weight_func`, source code of the custom 
    function(s) used to create weights.
    
    Parameters
    ----------
    forecaster_name : str
        Forecaster name. ForecasterAutoreg, ForecasterAutoregCustom, 
        ForecasterAutoregDirect, ForecasterAutoregMultiSeries, 
        ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate.
    regressor : regressor or pipeline compatible with the scikit-learn API
        Regressor of the forecaster.
    weight_func : Callable, dict
        Argument `weight_func` of the forecaster.
    series_weights : dict
        Argument `series_weights` of the forecaster.

    Returns
    -------
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
                 f"does not accept `sample_weight` in its `fit` method."),
                 IgnoredArgumentWarning
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
                 f"does not accept `sample_weight` in its `fit` method."),
                 IgnoredArgumentWarning
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
        Dictionary with the arguments to pass to the `fit' method of the forecaster.

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
                 f"regressor's `fit` method."),
                 IgnoredArgumentWarning
            )

        if 'sample_weight' in fit_kwargs.keys():
            warnings.warn(
                ("The `sample_weight` argument is ignored. Use `weight_func` to pass "
                 "a function that defines the individual weights for each sample "
                 "based on its index."),
                 IgnoredArgumentWarning
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
    -------
    None
    
    """
    
    if not isinstance(y, pd.Series):
        raise TypeError("`y` must be a pandas Series.")
        
    if y.isnull().any():
        raise ValueError("`y` has missing values.")
    
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
    exog : Any
        Exogenous variable/s included as predictor/s.
    allow_nan : bool, default `True`
        If True, allows the presence of NaN values in `exog`. If False (default),
        issue a warning if `exog` contains NaN values.

    Returns
    -------
    None

    """
    
    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError("`exog` must be a pandas Series or DataFrame.")

    if not allow_nan:
        if exog.isnull().any().any():
            warnings.warn(
                ("`exog` has missing values. Most machine learning models do not allow "
                 "missing values. Fitting the forecaster may fail."), 
                 MissingValuesExogWarning
            )
    
    return


def get_exog_dtypes(
    exog: Union[pd.DataFrame, pd.Series]
) -> dict:
    """
    Store dtypes of `exog`.

    Parameters
    ----------
    exog : pandas DataFrame, pandas Series
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
    Issue a Warning if `exog` has columns that are not `init`, `float`, or `category`.
    
    Parameters
    ----------
    exog : pandas DataFrame, pandas Series
        Exogenous variable/s included as predictor/s.

    Returns
    -------
    None

    """

    check_exog(exog=exog, allow_nan=False)

    if isinstance(exog, pd.DataFrame):
        if not exog.select_dtypes(exclude=[np.number, 'category']).columns.empty:
            warnings.warn(
                ("`exog` may contain only `int`, `float` or `category` dtypes. Most "
                 "machine learning models do not allow other types of values. "
                 "Fitting the forecaster may fail."), DataTypeWarning
            )
        for col in exog.select_dtypes(include='category'):
            if exog[col].cat.categories.dtype not in [int, np.int32, np.int64]:
                raise TypeError(
                    ("Categorical columns in exog must contain only integer values. "
                     "See skforecast docs for more info about how to include "
                     "categorical features https://skforecast.org/"
                     "latest/user_guides/categorical-features.html")
                )
    else:
        if exog.dtype.name not in ['int', 'int8', 'int16', 'int32', 'int64', 'float', 
        'float16', 'float32', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'category']:
            warnings.warn(
                ("`exog` may contain only `int`, `float` or `category` dtypes. Most "
                 "machine learning models do not allow other types of values. "
                 "Fitting the forecaster may fail."), DataTypeWarning
            )
        if exog.dtype.name == 'category' and exog.cat.categories.dtype not in [int,
        np.int32, np.int64]:
            raise TypeError(
                ("If exog is of type category, it must contain only integer values. "
                 "See skforecast docs for more info about how to include "
                 "categorical features https://skforecast.org/"
                 "latest/user_guides/categorical-features.html")
            )
         
    return


def check_interval(
    interval: list=None,
    quantiles: float=None,
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
    quantiles : list, default `None`
        Sequence of quantiles to compute, which must be between 0 and 1 
        inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as 
        `quantiles = [0.05, 0.5, 0.95]`.
    alpha : float, default `None`
        The confidence intervals used in ForecasterSarimax are (1 - alpha) %.

    Returns
    -------
    None
    
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
                f"Lower interval bound ({interval[0]}) must be >= 0 and < 100."
            )

        if (interval[1] <= 0.) or (interval[1] > 100.):
            raise ValueError(
                f"Upper interval bound ({interval[1]}) must be > 0 and <= 100."
            )

        if interval[0] >= interval[1]:
            raise ValueError(
                f"Lower interval bound ({interval[0]}) must be less than the "
                f"upper interval bound ({interval[1]})."
            )
        
    if quantiles is not None:
        if not isinstance(quantiles, list):
            raise TypeError(
                ("`quantiles` must be a `list`. For example, quantiles 0.05, "
                 "0.5, and 0.95 should be as `quantiles = [0.05, 0.5, 0.95]`.")
            )
        
        for q in quantiles:
            if (q < 0.) or (q > 1.):
                raise ValueError(
                    ("All elements in `quantiles` must be >= 0 and <= 1.")
                )
    
    if alpha is not None:
        if not isinstance(alpha, float):
            raise TypeError(
                ("`alpha` must be a `float`. For example, interval of 95% "
                 "should be as `alpha = 0.05`.")
            )

        if (alpha <= 0.) or (alpha >= 1):
            raise ValueError(
                f"`alpha` must have a value between 0 and 1. Got {alpha}."
            )

    return


def check_predict_input(
    forecaster_name: str,
    steps: Union[int, list],
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
        ForecasterAutoregMultiSeriesCustom, ForecasterAutoregMultiVariate.
    steps : int, list
        Number of future steps predicted.
    fitted: bool
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
        `ForecasterAutoregMultiSeriesCustom` and `ForecasterAutoregMultiVariate`).

    Returns
    -------
    None

    """

    if not fitted:
        raise sklearn.exceptions.NotFittedError(
            ("This Forecaster instance is not fitted yet. Call `fit` with "
             "appropriate arguments before using predict.")
        )
    
    if isinstance(steps, (int, np.integer)) and steps < 1:
        raise ValueError(
            f"`steps` must be an integer greater than or equal to 1. Got {steps}."
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
    
    if forecaster_name in ['ForecasterAutoregMultiSeries', 
                           'ForecasterAutoregMultiSeriesCustom']:
        if levels is not None and not isinstance(levels, (str, list)):
            raise TypeError(
                ("`levels` must be a `list` of column names, a `str` of a "
                 "column name or `None`.")
            )
        if len(set(levels) - set(series_col_names)) != 0:
            raise ValueError(
                f"`levels` must be in `series_col_names` : {series_col_names}."
            )

    if exog is None and included_exog:
        raise ValueError(
            ("Forecaster trained with exogenous variable/s. "
             "Same variable/s must be provided when predicting.")
        )
        
    if exog is not None and not included_exog:
        raise ValueError(
            ("Forecaster trained without exogenous variable/s. "
             "`exog` must be `None` when predicting.")
        )
        
    # Checks last_window
    # Check last_window type (pd.Series or pd.DataFrame according to forecaster)
    if forecaster_name in ['ForecasterAutoregMultiSeries', 
                           'ForecasterAutoregMultiSeriesCustom',
                           'ForecasterAutoregMultiVariate']:
        if not isinstance(last_window, pd.DataFrame):
            raise TypeError(
                f"`last_window` must be a pandas DataFrame. Got {type(last_window)}."
            )
        
        if forecaster_name in ['ForecasterAutoregMultiSeries', 
                               'ForecasterAutoregMultiSeriesCustom'] and \
            len(set(levels) - set(last_window.columns)) != 0:
            raise ValueError(
                (f"`last_window` must contain a column(s) named as the level(s) "
                 f"to be predicted.\n"
                 f"    `levels` : {levels}.\n"
                 f"    `last_window` columns : {list(last_window.columns)}.")
            )
        
        if forecaster_name == 'ForecasterAutoregMultiVariate' and \
            (series_col_names != list(last_window.columns)):
            raise ValueError(
                (f"`last_window` columns must be the same as `series` column names.\n"
                 f"    `last_window` columns : {list(last_window.columns)}.\n"
                 f"    `series` columns      : {series_col_names}.")
            )    
    else:    
        if not isinstance(last_window, pd.Series):
            raise TypeError(
                f"`last_window` must be a pandas Series. Got {type(last_window)}."
            )
    
    # Check last_window len, nulls and index (type and freq)
    if len(last_window) < window_size:
        raise ValueError(
            (f"`last_window` must have as many values as needed to "
             f"generate the predictors. For this forecaster it is {window_size}.")
        )
    if last_window.isnull().any().all():
        raise ValueError(
            ("`last_window` has missing values.")
        )
    _, last_window_index = preprocess_last_window(
                               last_window   = last_window.iloc[:0],
                               return_values = False
                           ) 
    if not isinstance(last_window_index, index_type):
        raise TypeError(
            (f"Expected index of type {index_type} for `last_window`. "
             f"Got {type(last_window_index)}.")
        )
    if isinstance(last_window_index, pd.DatetimeIndex):
        if not last_window_index.freqstr == index_freq:
            raise TypeError(
                (f"Expected frequency of type {index_freq} for `last_window`. "
                 f"Got {last_window_index.freqstr}.")
            )

    # Checks exog
    if exog is not None:
        # Check type, nulls and expected type
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise TypeError("`exog` must be a pandas Series or DataFrame.")
        if exog.isnull().any().any():
            warnings.warn(
                ("`exog` has missing values. Most of machine learning models do "
                 "not allow missing values. `predict` method may fail."), 
                 MissingValuesExogWarning
            )
        if not isinstance(exog, exog_type):
            raise TypeError(
                f"Expected type for `exog`: {exog_type}. Got {type(exog)}."    
            )
        
        # Check exog has many values as distance to max step predicted
        last_step = max(steps) if isinstance(steps, list) else steps
        if len(exog) < last_step:
            raise ValueError(
                (f"`exog` must have at least as many values as the distance to "
                 f"the maximum step predicted, {last_step}.")
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
                (f"Expected index of type {index_type} for `exog`. "
                 f"Got {type(exog_index)}.")
            )   
        if isinstance(exog_index, pd.DatetimeIndex):
            if not exog_index.freqstr == index_freq:
                raise TypeError(
                    (f"Expected frequency of type {index_freq} for `exog`. "
                     f"Got {exog_index.freqstr}.")
                )

        # Check exog starts one step ahead of last_window end.
        expected_index = expand_index(last_window.index, 1)[0]
        if expected_index != exog.index[0]:
            raise ValueError(
                (f"To make predictions `exog` must start one step ahead of `last_window`.\n"
                 f"    `last_window` ends at : {last_window.index[-1]}.\n"
                 f"    `exog` starts at      : {exog.index[0]}.\n"
                 f"     Expected index       : {expected_index}.")
            )

    # Checks ForecasterSarimax
    if forecaster_name == 'ForecasterSarimax':
        # Check last_window_exog type, len, nulls and index (type and freq)
        if last_window_exog is not None:
            if not included_exog:
                raise ValueError(
                    ("Forecaster trained without exogenous variable/s. "
                     "`last_window_exog` must be `None` when predicting.")
                )

            if not isinstance(last_window_exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    (f"`last_window_exog` must be a pandas Series or a "
                     f"pandas DataFrame. Got {type(last_window_exog)}.")
                )
            if len(last_window_exog) < window_size:
                raise ValueError(
                    (f"`last_window_exog` must have as many values as needed to "
                     f"generate the predictors. For this forecaster it is {window_size}.")
                )
            if last_window_exog.isnull().any().all():
                warnings.warn(
                ("`last_window_exog` has missing values. Most of machine learning "
                 "models do not allow missing values. `predict` method may fail."),
                MissingValuesExogWarning
            )
            _, last_window_exog_index = preprocess_last_window(
                                            last_window   = last_window_exog.iloc[:0],
                                            return_values = False
                                        ) 
            if not isinstance(last_window_exog_index, index_type):
                raise TypeError(
                    (f"Expected index of type {index_type} for `last_window_exog`. "
                     f"Got {type(last_window_exog_index)}.")
                )
            if isinstance(last_window_exog_index, pd.DatetimeIndex):
                if not last_window_exog_index.freqstr == index_freq:
                    raise TypeError(
                        (f"Expected frequency of type {index_freq} for "
                         f"`last_window_exog`. Got {last_window_exog_index.freqstr}.")
                    )

            # Check all columns are in the pd.DataFrame, last_window_exog
            if isinstance(last_window_exog, pd.DataFrame):
                col_missing = set(exog_col_names).difference(set(last_window_exog.columns))
                if col_missing:
                    raise ValueError(
                        (f"Missing columns in `exog`. Expected {exog_col_names}. "
                         f"Got {last_window_exog.columns.to_list()}.") 
                    )

    return


def preprocess_y(
    y: Union[pd.Series, pd.DataFrame],
    return_values: bool=True
) -> Tuple[Union[None, np.ndarray], pd.Index]:
    """
    Return values and index of series separately. Index is overwritten 
    according to the next rules:
    
    - If index is of type `DatetimeIndex` and has frequency, nothing is 
    changed.
    - If index is of type `RangeIndex`, nothing is changed.
    - If index is of type `DatetimeIndex` but has no frequency, a 
    `RangeIndex` is created.
    - If index is not of type `DatetimeIndex`, a `RangeIndex` is created.
    
    Parameters
    ----------
    y : pandas Series, pandas DataFrame
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

    y_values = y.to_numpy(copy=True) if return_values else None

    return y_values, y_index


def preprocess_last_window(
    last_window: Union[pd.Series, pd.DataFrame],
    return_values: bool=True
 ) -> Tuple[np.ndarray, pd.Index]:
    """
    Return values and index of series separately. Index is overwritten 
    according to the next rules:
    
    - If index is of type `DatetimeIndex` and has frequency, nothing is 
    changed.
    - If index is of type `RangeIndex`, nothing is changed.
    - If index is of type `DatetimeIndex` but has no frequency, a 
    `RangeIndex` is created.
    - If index is not of type `DatetimeIndex`, a `RangeIndex` is created.
    
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

    last_window_values = last_window.to_numpy(copy=True) if return_values else None

    return last_window_values, last_window_index


def preprocess_exog(
    exog: Union[pd.Series, pd.DataFrame],
    return_values: bool=True
) -> Tuple[Union[None, np.ndarray], pd.Index]:
    """
    Return values and index of series or data frame separately. Index is
    overwritten  according to the next rules:
    
    - If index is of type `DatetimeIndex` and has frequency, nothing is 
    changed.
    - If index is of type `RangeIndex`, nothing is changed.
    - If index is of type `DatetimeIndex` but has no frequency, a 
    `RangeIndex` is created.
    - If index is not of type `DatetimeIndex`, a `RangeIndex` is created.

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
            ("`exog` has DatetimeIndex index but no frequency. "
             "Index is overwritten with a RangeIndex of step 1.")
        )
        exog_index = pd.RangeIndex(
                         start = 0,
                         stop  = len(exog),
                         step  = 1
                     )

    else:
        warnings.warn(
            ("`exog` has no DatetimeIndex nor RangeIndex index. "
             "Index is overwritten with a RangeIndex.")
        )
        exog_index = pd.RangeIndex(
                         start = 0,
                         stop  = len(exog),
                         step  = 1
                     )

    exog_values = exog.to_numpy(copy=True) if return_values else None

    return exog_values, exog_index
    

def cast_exog_dtypes(
    exog: Union[pd.Series, pd.DataFrame],
    exog_dtypes: dict,
) -> Union[pd.Series, pd.DataFrame]: # pragma: no cover
    """
    Cast `exog` to a specified types. This is done because, for a forecaster to 
    accept a categorical exog, it must contain only integer values. Due to the 
    internal modifications of numpy, the values may be casted to `float`, so 
    they have to be re-converted to `int`.

    - If `exog` is a pandas Series, `exog_dtypes` must be a dict with a 
    single value.
    - If `exog_dtypes` is `category` but the current type of `exog` is `float`, 
    then the type is cast to `int` and then to `category`. 

    Parameters
    ----------
    exog : pandas Series, pandas DataFrame
        Exogenous variables.
    exog_dtypes: dict
        Dictionary with name and type of the series or data frame columns.

    Returns
    -------
    exog : pandas Series, pandas DataFrame
        Exogenous variables casted to the indicated dtypes.

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

    n_rows = len(exog)
    exog_idx = exog.index
    exog_transformed = []

    for i in range(steps):
        exog_column_transformed = exog.iloc[i : n_rows - (steps - 1 - i), ]
        exog_column_transformed.index = pd.RangeIndex(len(exog_column_transformed))
        exog_column_transformed.columns = [f"{col}_step_{i+1}" 
                                           for col in exog_column_transformed.columns]
        exog_transformed.append(exog_column_transformed)

    if len(exog_transformed) > 1:
        exog_transformed = pd.concat(exog_transformed, axis=1, copy=False)
    else:
        exog_transformed = exog_column_transformed

    exog_transformed.index = exog_idx[-len(exog_transformed):]
    
    return exog_transformed


def exog_to_direct_numpy(
    exog: np.ndarray,
    steps: int
)-> np.ndarray:
    """
    Transforms `exog` to numpy ndarray with the shape needed for Direct
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

    if not isinstance(exog, np.ndarray):
        raise TypeError(f"`exog` must be a numpy ndarray. Got {type(exog)}.")

    if exog.ndim == 1:
        exog = np.expand_dims(exog, axis=1)

    n_rows = len(exog)
    exog_transformed = []

    for i in range(steps):
        exog_column_transformed = exog[i : n_rows - (steps - 1 - i)]
        exog_transformed.append(exog_column_transformed)

    if len(exog_transformed) > 1:
        exog_transformed = np.concatenate(exog_transformed, axis=1)
    else:
        exog_transformed = exog_column_transformed.copy()
    
    return exog_transformed


def expand_index(
    index: Union[pd.Index, None], 
    steps: int
) -> pd.Index:
    """
    Create a new index of length `steps` starting at the end of the index.
    
    Parameters
    ----------
    index : pandas Index, None
        Original index.
    steps : int
        Number of steps to expand.

    Returns
    -------
    new_index : pandas Index
        New index.

    """
    
    if isinstance(index, pd.Index):
        
        if isinstance(index, pd.DatetimeIndex):
            new_index = pd.date_range(
                            start   = index[-1] + index.freq,
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
    (preprocessor). The transformer used must have the following methods: fit, 
    transform, fit_transform and inverse_transform. ColumnTransformers are not 
    allowed since they do not have inverse_transform method.

    Parameters
    ----------
    series : pandas Series
        Series to be transformed.
    transformer : scikit-learn alike transformer (preprocessor).
        scikit-learn alike transformer (preprocessor) with methods: fit, transform,
        fit_transform and inverse_transform. ColumnTransformers are not allowed 
        since they do not have inverse_transform method.
    fit : bool, default `False`
        Train the transformer before applying it.
    inverse_transform : bool, default `False`
        Transform back the data to the original representation.

    Returns
    -------
    series_transformed : pandas Series, pandas DataFrame
        Transformed Series. Depending on the transformer used, the output may 
        be a Series or a DataFrame.

    """
    
    if not isinstance(series, pd.Series):
        raise TypeError(
            (f"`series` argument must be a pandas Series. Got {type(series)}.")
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
    transformer, preprocessor or ColumnTransformer. `inverse_transform` is not 
    available when using ColumnTransformers.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to be transformed.
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
            f"`df` argument must be a pandas DataFrame. Got {type(df)}"
        )

    if transformer is None:
        return df

    if inverse_transform and isinstance(transformer, ColumnTransformer):
        raise Exception(
            "`inverse_transform` is not available when using ColumnTransformers."
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
    forecaster: forecaster
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
    forecaster: forecaster
        Forecaster created with skforecast library.
    
    """

    forecaster = joblib.load(filename=file_name)

    skforecast_v = skforecast.__version__
    forecaster_v = forecaster.skforecast_version

    if forecaster_v != skforecast_v:
        warnings.warn(
            (f"The skforecast version installed in the environment differs "
             f"from the version used to create the forecaster.\n"
             f"    Installed Version  : {skforecast_v}\n"
             f"    Forecaster Version : {forecaster_v}\n"
             f"This may create incompatibilities when using the library."),
             SkforecastVersionWarning
        )

    if verbose:
        forecaster.summary()

    return forecaster


def _find_optional_dependency(
    package_name: str, 
    optional_dependencies: dict=optional_dependencies
) -> Tuple[str, str]:
    """
    Find if a package is an optional dependency. If True, find the version and 
    the extension it belongs to.

    Parameters
    ----------
    package_name : str
        Name of the package to check.
    optional_dependencies : dict, default `optional_dependencies`
        Skforecast optional dependencies.

    Returns
    -------
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

    Returns
    -------
    None
    
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
    lags : int, list, numpy ndarray
        Lags to be included in the correlation analysis.
    method : str, default 'pearson'
        - 'pearson': standard correlation coefficient.
        - 'kendall': Kendall Tau correlation coefficient.
        - 'spearman': Spearman rank correlation.

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


def check_backtesting_input(
    forecaster: object,
    steps: int,
    metric: Union[str, Callable, list],
    y: Optional[pd.Series]=None,
    series: Optional[pd.DataFrame]=None,
    initial_train_size: Optional[int]=None,
    fixed_train_size: bool=True,
    gap: int=0,
    allow_incomplete_fold: bool=True,
    refit: Optional[Union[bool, int]]=False,
    interval: Optional[list]=None,
    alpha: Optional[float]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    n_jobs: Optional[Union[int, str]]='auto',
    verbose: bool=False,
    show_progress: bool=True
) -> None:
    """
    This is a helper function to check most inputs of backtesting functions in 
    modules `model_selection`, `model_selection_multiseries` and 
    `model_selection_sarimax`.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    steps : int, list
        Number of future steps predicted.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
    y : pandas Series
        Training time series for uni-series forecasters.
    series : pandas DataFrame
        Training time series for multi-series forecasters.
    initial_train_size : int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` 
        is already trained, no initial train is done and all data is used to 
        evaluate the model.
    fixed_train_size : bool, default `True`
        If True, train size doesn't increase but moves by `steps` in each iteration.
    gap : int, default `0`
        Number of samples to be excluded after the end of each training set and 
        before the test set.
    allow_incomplete_fold : bool, default `True`
        Last fold is allowed to have a smaller number of samples than the 
        `test_size`. If `False`, the last fold is excluded.
    refit : bool, int, default `False`
        Whether to re-fit the forecaster in each iteration. If `refit` is an integer, 
        the Forecaster will be trained every that number of iterations.
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive.
    alpha : float, default `None`
        The confidence intervals used in ForecasterSarimax are (1 - alpha) %. 
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals.  If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    n_jobs : int, 'auto', default `'auto'`
            The number of jobs to run in parallel. If `-1`, then the number of jobs is 
            set to the number of cores. If 'auto', `n_jobs` is set using the fuction
            skforecast.utils.select_n_jobs_fit_forecaster.
            **New in version 0.9.0**
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress: bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    None
    
    """
    
    forecasters_uni = ['ForecasterAutoreg', 'ForecasterAutoregCustom', 
                       'ForecasterAutoregDirect', 'ForecasterSarimax',
                       'ForecasterEquivalentDate']
    forecasters_multi = ['ForecasterAutoregMultiSeries', 
                         'ForecasterAutoregMultiSeriesCustom', 
                         'ForecasterAutoregMultiVariate']
    
    forecaster_name = type(forecaster).__name__

    if forecaster_name in forecasters_uni:
        if not isinstance(y, pd.Series):
            raise TypeError("`y` must be a pandas Series.")
        data_name = 'y'
        data_length = len(y)
        
    if forecaster_name in forecasters_multi:
        if not isinstance(series, pd.DataFrame):
            raise TypeError("`series` must be a pandas DataFrame.")
        data_name = 'series'
        data_length = len(series)
        
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise TypeError(
            f"`steps` must be an integer greater than or equal to 1. Got {steps}."
        )
    if not isinstance(gap, (int, np.integer)) or gap < 0:
        raise TypeError(
            f"`gap` must be an integer greater than or equal to 0. Got {gap}."
        )
    if not isinstance(metric, (str, Callable, list)):
        raise TypeError(
            (f"`metric` must be a string, a callable function, or a list containing "
             f"multiple strings and/or callables. Got {type(metric)}.")
        )
    
    if forecaster_name == "ForecasterEquivalentDate" and isinstance(
        forecaster.offset, pd.tseries.offsets.DateOffset
    ):
        pass
    elif initial_train_size is not None:
        if not isinstance(initial_train_size, (int, np.integer)):
            raise TypeError(
                (f"If used, `initial_train_size` must be an integer greater than the "
                 f"window_size of the forecaster. Got type {type(initial_train_size)}.")
            )
        if initial_train_size >= data_length:
            raise ValueError(
                (f"If used, `initial_train_size` must be an integer smaller "
                 f"than the length of `{data_name}` ({data_length}).")
            )    
        if initial_train_size < forecaster.window_size:
            raise ValueError(
                (f"If used, `initial_train_size` must be an integer greater than "
                 f"the window_size of the forecaster ({forecaster.window_size}).")
            )
        if initial_train_size + gap >= data_length:
            raise ValueError(
                (f"The combination of initial_train_size {initial_train_size} and "
                 f"gap {gap} cannot be greater than the length of `{data_name}` "
                 f"({data_length}).")
            )
        if data_name == 'series':
            for serie in series:
                if np.isnan(series[serie].to_numpy()[:initial_train_size]).all():
                    raise ValueError(
                        (f"All values of series '{serie}' are NaN. When working "
                         f"with series of different lengths, make sure that "
                         f"`initial_train_size` has an appropriate value so that "
                         f"all series reach the first non-null value.")
                    )
    else:
        if forecaster_name == 'ForecasterSarimax':
            raise ValueError(
                (f"`initial_train_size` must be an integer smaller than the "
                 f"length of `{data_name}` ({data_length}).")
            )
        else:
            if not forecaster.fitted:
                raise NotFittedError(
                    ("`forecaster` must be already trained if no `initial_train_size` "
                     "is provided.")
                )
            if refit:
                raise ValueError(
                    "`refit` is only allowed when `initial_train_size` is not `None`."
                )
    
    if not isinstance(fixed_train_size, bool):
        raise TypeError("`fixed_train_size` must be a boolean: `True`, `False`.")
    if not isinstance(allow_incomplete_fold, bool):
        raise TypeError("`allow_incomplete_fold` must be a boolean: `True`, `False`.")
    if not isinstance(refit, (bool, int, np.integer)) or refit < 0:
        raise TypeError(f"`refit` must be a boolean or an integer greater than 0. Got {refit}.")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise TypeError(f"`n_boot` must be an integer greater than 0. Got {n_boot}.")
    if not isinstance(random_state, (int, np.integer)) or random_state < 0:
        raise TypeError(f"`random_state` must be an integer greater than 0. Got {random_state}.")
    if not isinstance(in_sample_residuals, bool):
        raise TypeError("`in_sample_residuals` must be a boolean: `True`, `False`.")
    if not isinstance(n_jobs, int) and n_jobs != 'auto':
        raise TypeError(f"`n_jobs` must be an integer or `'auto'`. Got {n_jobs}.")
    if not isinstance(verbose, bool):
        raise TypeError("`verbose` must be a boolean: `True`, `False`.")
    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean: `True`, `False`.")

    if interval is not None or alpha is not None:
        check_interval(interval=interval, alpha=alpha)

    if not allow_incomplete_fold and data_length - (initial_train_size + gap) < steps:
        raise ValueError(
            (f"There is not enough data to evaluate {steps} steps in a single "
             f"fold. Set `allow_incomplete_fold` to `True` to allow incomplete folds.\n"
             f"    Data available for test : {data_length - (initial_train_size + gap)}\n"
             f"    Steps                   : {steps}")
        )
    
    return


def select_n_jobs_backtesting(
    forecaster_name: str,
    regressor_name: str,
    refit: Union[bool, int]
) -> int:
    """
    Select the optimal number of jobs to use in the backtesting process. This
    selection is based on heuristics and is not guaranteed to be optimal.

    The number of jobs is chosen as follows:

    - If `refit` is an integer, then n_jobs=1. This is because parallelization doesn't 
    work with intermittent refit.
    - If forecaster_name is 'ForecasterAutoreg' or 'ForecasterAutoregCustom' and
    regressor_name is a linear regressor, then n_jobs=1.
    - If forecaster_name is 'ForecasterAutoreg' or 'ForecasterAutoregCustom',
    regressor_name is not a linear regressor and refit=`True`, then
    n_jobs=cpu_count().
    - If forecaster_name is 'ForecasterAutoreg' or 'ForecasterAutoregCustom',
    regressor_name is not a linear regressor and refit=`False`, then
    n_jobs=1.
    - If forecaster_name is 'ForecasterAutoregDirect' or 'ForecasterAutoregMultiVariate'
    and refit=`True`, then n_jobs=cpu_count().
    - If forecaster_name is 'ForecasterAutoregDirect' or 'ForecasterAutoregMultiVariate'
    and refit=`False`, then n_jobs=1.
    - If forecaster_name is 'ForecasterAutoregMultiseries' or 
    'ForecasterAutoregMultiseriesCustom', then n_jobs=cpu_count().
    - If forecaster_name is 'ForecasterSarimax' or 'ForecasterEquivalentDate', 
    then n_jobs=1.

    Parameters
    ----------
    forecaster_name : str
        The type of Forecaster.
    regressor_name : str
        The type of regressor.
    refit : bool, int
        If the forecaster is refitted during the backtesting process.

    Returns
    -------
    n_jobs : int
        The number of jobs to run in parallel.
    
    """

    linear_regressors = [
        regressor_name
        for regressor_name in dir(sklearn.linear_model)
        if not regressor_name.startswith('_')
    ]
    
    refit = False if refit == 0 else refit
    if not isinstance(refit, bool) and refit != 1:
        n_jobs = 1
    else:
        if forecaster_name in ['ForecasterAutoreg', 'ForecasterAutoregCustom']:
            if regressor_name in linear_regressors:
                n_jobs = 1
            else:
                n_jobs = joblib.cpu_count() if refit else 1
        elif forecaster_name in ['ForecasterAutoregDirect', 'ForecasterAutoregMultiVariate']:
            n_jobs = 1
        elif forecaster_name in ['ForecasterAutoregMultiseries', 'ForecasterAutoregMultiSeriesCustom']:
            n_jobs = joblib.cpu_count()
        elif forecaster_name in ['ForecasterSarimax', 'ForecasterEquivalentDate']:
            n_jobs = 1
        else:
            n_jobs = 1

    return n_jobs


def select_n_jobs_fit_forecaster(
    forecaster_name: str,
    regressor_name: str,
) -> int:
    """
    Select the optimal number of jobs to use in the fitting process. This
    selection is based on heuristics and is not guaranteed to be optimal. 
    
    The number of jobs is chosen as follows:
    
    - If forecaster_name is 'ForecasterAutoregDirect' or 'ForecasterAutoregMultiVariate'
    and regressor_name is a linear regressor, then n_jobs=1, otherwise n_jobs=cpu_count().
    
    Parameters
    ----------
    forecaster_name : str
        The type of Forecaster.
    regressor_name : str
        The type of regressor.

    Returns
    -------
    n_jobs : int
        The number of jobs to run in parallel.
    
    """

    linear_regressors = [
        regressor_name
        for regressor_name in dir(sklearn.linear_model)
        if not regressor_name.startswith('_')
    ]

    if forecaster_name in ['ForecasterAutoregDirect', 'ForecasterAutoregMultiVariate']:
        if regressor_name in linear_regressors:
            n_jobs = 1
        else:
            n_jobs = joblib.cpu_count()
    else:
        n_jobs = 1

    return n_jobs