################################################################################
#                                 utils                                        #
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
from sklearn.preprocessing import FunctionTransformer
import inspect

optional_dependencies = {
    "statsmodels": ['statsmodels>=0.12, <0.14'],
    "plotting": ['matplotlib>=3.3, <3.7', 'seaborn==0.11', 'statsmodels>=0.12, <0.14']
}


def initialize_lags(
    forecaster_type: str,
    lags: Any
) -> np.ndarray:
    """
    Check lags argument input and generate the corresponding numpy ndarray.
    
    Parameters
    ----------
    forecaster_type : str
        Forcaster type. ForecasterAutoreg, ForecasterAutoregCustom, 
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
        if not forecaster_type == 'ForecasterAutoregMultiVariate':
            raise TypeError(
                '`lags` argument must be an int, 1d numpy ndarray, range or list. '
                f"Got {type(lags)}."
            )
        else:
            raise TypeError(
                '`lags` argument must be a dict, int, 1d numpy ndarray, range or list. '
                f"Got {type(lags)}."
            )

    return lags


def initialize_weights(
    forecaster_type: str,
    regressor: object,
    weight_func: Union[callable, dict],
    series_weights: dict
) -> Tuple[Union[callable, dict], Union[callable, dict], dict]:
    """
    Check weights arguments, `weight_func` and `series_weights` for the different 
    forecasters. Create `source_code_weight_func`, source code of the custom 
    function(s) used to create weights.
    
    Parameters
    ----------
    forecaster_type : str
        Forcaster type. ForecasterAutoreg, ForecasterAutoregCustom, 
        ForecasterAutoregDirect, ForecasterAutoregMultiSeries, 
        ForecasterAutoregMultiVariate.

    regressor : regressor or pipeline compatible with the scikit-learn API
        Regressor of the forecaster.

    weight_func : callable, dict
        Argument `weight_func` of the forecaster.

    series_weights : dict
        Argument `series_weights` of the forecaster.
        
        
    Returns
    ----------
    weight_func : callable, dict
        Argument `weight_func` of the forecaster.

    source_code_weight_func : str, dict
        Argument `source_code_weight_func` of the forecaster.

    series_weights : dict
        Argument `series_weights` of the forecaster.
    
    """

    source_code_weight_func = None

    if weight_func is not None:
        if not isinstance(weight_func, Callable) and not forecaster_type == 'ForecasterAutoregMultiSeries':
            raise TypeError(
                f"Argument `weight_func` must be a callable. Got {type(weight_func)}."
            )
        elif not isinstance(weight_func, (Callable, dict)) and forecaster_type == 'ForecasterAutoregMultiSeries':
            raise TypeError(
                f"Argument `weight_func` must be a callable or a dict of "
                f"callables. Got {type(weight_func)}."
            )
        
        if isinstance(weight_func, dict):
            source_code_weight_func = {}
            for key in weight_func:
                source_code_weight_func[key] = inspect.getsource(weight_func[key])
        else:
            source_code_weight_func = inspect.getsource(weight_func)

        if 'sample_weight' not in inspect.getfullargspec(regressor.fit)[0]:
            warnings.warn(
                (f'Argument `weight_func` is ignored since regressor {regressor} '
                 f'does not accept `sample_weight` in its `fit` method.')
            )
            weight_func = None
            source_code_weight_func = None

    if series_weights is not None:
        if not isinstance(series_weights, dict):
            raise TypeError(
                f"Argument `series_weights` must be a dict of floats or ints."
                f"Got {type(series_weights)}."
            )
        if 'sample_weight' not in inspect.getfullargspec(regressor.fit)[0]:
            warnings.warn(
                (f'Argument `series_weights` is ignored since regressor {regressor} '
                 f'does not accept `sample_weight` in its `fit` method.')
            )
            series_weights = None

    return weight_func, source_code_weight_func, series_weights


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
    exog: Any
) -> None:
    """
    Raise Exception if `exog` is not pandas Series or pandas DataFrame, or
    if it has missing values.
    
    Parameters
    ----------        
    exog :  Any
        Exogenous variable/s included as predictor/s.

    Returns
    ----------
    None

    """
        
    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError('`exog` must be `pd.Series` or `pd.DataFrame`.')

    if exog.isnull().any().any():
        raise ValueError('`exog` has missing values.')
                
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
                ('`interval` must be a `list`. For example, interval of 95% '
                 'should be as `interval = [2.5, 97.5]`.')
            )

        if len(interval) != 2:
            raise ValueError(
                ('`interval` must contain exactly 2 values, respectively the '
                 'lower and upper interval bounds. For example, interval of 95% '
                 'should be as `interval = [2.5, 97.5]`.')
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
    forecaster_type: str,
    steps: int,
    fitted: bool,
    included_exog: bool,
    index_type: type,
    index_freq: str,
    window_size: int,
    last_window: Union[pd.Series, pd.DataFrame]=None,
    last_window_exog: Union[pd.Series, pd.DataFrame]=None,
    exog: Union[pd.Series, pd.DataFrame]=None,
    exog_type: Union[type, None]=None,
    exog_col_names: Union[list, None]=None,
    interval: list=None,
    alpha: float=None,
    max_steps: int=None,
    levels: Optional[Union[str, list]]=None,
    series_col_names: list=None
) -> None:
    """
    Check all inputs of predict method. This is a helper function to validate
    that inputs used in predict method match attributes of a forecaster already
    trained.

    Parameters
    ----------
    forecaster_type : str
        Forcaster type. ForecasterAutoreg, ForecasterAutoregCustom, 
        ForecasterAutoregDirect, ForecasterAutoregMultiSeries, 
        ForecasterAutoregMultiVariate.

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
        Time series to be predicted (`ForecasterAutoregMultiSeries`).

    series_col_names : list, default `None`
        Names of the columns used during fit (`ForecasterAutoregMultiSeries` and 
        `ForecasterAutoregMultiVariate`).
    
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
    
    if forecaster_type == 'ForecasterAutoregMultiSeries':
        if levels is not None and not isinstance(levels, (str, list)):
            raise TypeError(
                f'`levels` must be a `list` of column names, a `str` of a column name or `None`.'
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
    
    if exog is not None:
        # Check exog has many values as distance to max step predicted
        last_step = max(steps) if isinstance(steps, list) else steps
        if len(exog) < last_step:
            raise ValueError(
                f'`exog` must have at least as many values as the distance to '
                f'the maximum step predicted, {last_step}.'
            )

        # Check nulls and index type and freq
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise TypeError('`exog` must be a pandas Series or DataFrame.')
        if exog.isnull().values.any():
            raise ValueError('`exog` has missing values.')
        if not isinstance(exog, exog_type):
            raise TypeError(
                f'Expected type for `exog`: {exog_type}. Got {type(exog)}.'     
            )

        # Check all columns are in the pd.DataFrame
        if isinstance(exog, pd.DataFrame):
            col_missing = set(exog_col_names).difference(set(exog.columns))
            if col_missing:
                raise ValueError(
                    (f'Missing columns in `exog`. Expected {exog_col_names}. '
                     f'Got {exog.columns.to_list()}.') 
                )

        # Check nulls and index type and freq
        check_exog(exog = exog)
        _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])
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
        
    if last_window is not None:
        # Check last_window type (pd.Series or pd.DataFrame according to forecaster)
        if forecaster_type in ['ForecasterAutoregMultiSeries', 'ForecasterAutoregMultiVariate']:
            if not isinstance(last_window, pd.DataFrame):
                raise TypeError(
                    f'`last_window` must be a pandas DataFrame. Got {type(last_window)}.'
                )
            
            if forecaster_type == 'ForecasterAutoregMultiSeries' and \
               len(set(levels) - set(last_window.columns)) != 0:
                raise ValueError(
                    (f'`last_window` must contain a column(s) named as the level(s) to be predicted.\n'
                     f'    `levels` : {levels}.\n'
                     f'    `last_window` columns : {list(last_window.columns)}.')
                )
            
            if forecaster_type == 'ForecasterAutoregMultiVariate' and \
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
            raise ValueError('`last_window` has missing values.')
        _, last_window_index = preprocess_last_window(
                                    last_window = last_window.iloc[:0]
                                ) 
        if not isinstance(last_window_index, index_type):
            raise TypeError(
                f'Expected index of type {index_type} for `last_window`. '
                f'Got {type(last_window_index)}.'
            )
        if isinstance(last_window_index, pd.DatetimeIndex):
            if not last_window_index.freqstr == index_freq:
                raise TypeError(
                    f'Expected frequency of type {index_freq} for `last_window`. '
                    f'Got {last_window_index.freqstr}.'
                )

    # Checks ForecasterSarimax
    if forecaster_type == 'ForecasterSarimax':
        # Check if forecaster needs exog
        if last_window is not None and last_window_exog is None and included_exog:
            raise ValueError(
                ('Forecaster trained with exogenous variable/s. '
                 'Same variable/s must be provided using `last_window_exog`.')
            )   
        if last_window_exog is not None and not included_exog:
            raise ValueError(
                ('Forecaster trained without exogenous variable/s. '
                 '`last_window_exog` must be `None` when predicting.')
            )

        # If last_window_exog is provided but no last_window
        if last_window is None and last_window_exog is not None:
            raise ValueError(
                ('To make predictions unrelated to the original data, both '
                 '`last_window` and `last_window_exog` must be provided.')
            )

        # Check last_window_exog type, len, nulls and index (type and freq)
        if last_window_exog is not None:
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
                raise ValueError('`last_window_exog` has missing values.')
            _, last_window_exog_index = preprocess_last_window(
                                        last_window = last_window_exog.iloc[:0]
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
    y: pd.Series
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
    y : pandas Series
        Time series.

    Returns 
    -------
    y_values : numpy ndarray
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
            '`y` has DatetimeIndex index but no frequency. '
            'Index is overwritten with a RangeIndex of step 1.'
        )
        y_index = pd.RangeIndex(
                      start = 0,
                      stop  = len(y),
                      step  = 1
                  )
    else:
        warnings.warn(
            '`y` has no DatetimeIndex nor RangeIndex index. Index is overwritten with a RangeIndex.'
        )
        y_index = pd.RangeIndex(
                      start = 0,
                      stop  = len(y),
                      step  = 1
                  )

    y_values = y.to_numpy()

    return y_values, y_index


def preprocess_last_window(
    last_window:Union[pd.Series, pd.DataFrame]
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
            '`last_window` has DatetimeIndex index but no frequency. '
            'Index is overwritten with a RangeIndex of step 1.'
        )
        last_window_index = pd.RangeIndex(
                                start = 0,
                                stop  = len(last_window),
                                step  = 1
                                )
    else:
        warnings.warn(
            '`last_window` has no DatetimeIndex nor RangeIndex index. Index is overwritten with a RangeIndex.'
        )
        last_window_index = pd.RangeIndex(
                                start = 0,
                                stop  = len(last_window),
                                step  = 1
                                )

    last_window_values = last_window.to_numpy()

    return last_window_values, last_window_index


def preprocess_exog(
    exog: Union[pd.Series, pd.DataFrame]
) -> Tuple[np.ndarray, pd.Index]:
    """
    Returns values ​​and index of series separately. Index is overwritten 
    according to the next rules:
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

    Returns 
    -------
    exog_values : numpy ndarray
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
            '`exog` has DatetimeIndex index but no frequency. '
            'Index is overwritten with a RangeIndex of step 1.'
        )
        exog_index = pd.RangeIndex(
                        start = 0,
                        stop  = len(exog),
                        step  = 1
                        )

    else:
        warnings.warn(
            '`exog` has no DatetimeIndex nor RangeIndex index. Index is overwritten with a RangeIndex.'
        )
        exog_index = pd.RangeIndex(
                        start = 0,
                        stop  = len(exog),
                        step  = 1
                        )

    exog_values = exog.to_numpy()

    return exog_values, exog_index


def exog_to_direct(
    exog: np.ndarray,
    steps: int
)-> np.ndarray:
    """
    Transforms `exog` to `np.ndarray` with the shape needed for direct
    forecasting.
    
    Parameters
    ----------        
    exog : numpy ndarray, shape(samples,)
        Time series values.

    steps : int.
        Number of steps that will be predicted using this exog.

    Returns 
    -------
    exog_transformed : numpy ndarray

    """

    exog_transformed = []

    if exog.ndim < 2:
        exog = exog.reshape(-1, 1)

    for column in range(exog.shape[1]):

        exog_column_transformed = []

        for i in range(exog.shape[0] - (steps -1)):
            exog_column_transformed.append(exog[i:i + steps, column])

        if len(exog_column_transformed) > 1:
            exog_column_transformed = np.vstack(exog_column_transformed)

        exog_transformed.append(exog_column_transformed)

    if len(exog_transformed) > 1:
        exog_transformed = np.hstack(exog_transformed)
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

    series = series.to_frame()

    if fit and not isinstance(transformer, FunctionTransformer):
        transformer.fit(series)

    if inverse_transform:
        values_transformed = transformer.inverse_transform(series.values)
    else:
        values_transformed = transformer.transform(series.values)   

    if hasattr(values_transformed, 'toarray'):
        # If the returned values are in sparse matrix format, it is converted to dense array.
        values_transformed = values_transformed.toarray()
    
    if isinstance(values_transformed, np.ndarray) and values_transformed.shape[1] == 1:
        series_transformed = pd.Series(
                                 data  = values_transformed.flatten(),
                                 index = series.index,
                                 name  = series.columns[0]
                             )
    elif isinstance(values_transformed, pd.DataFrame) and values_transformed.shape[1] == 1:
        series_transformed = values_transformed.squeeze()
    else:
        series_transformed = pd.DataFrame(
                                 data    = values_transformed,
                                 index   = series.index,
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
    series : pandas DataFrame

    transformer : scikit-learn alike transformer, preprocessor or ColumnTransformer.
        scikit-learn alike transformer, preprocessor or ColumnTransformer.

    fit : bool, default `False`
        Train the transformer before applying it.

    inverse_transform : bool, default `False`
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.

    Returns
    -------
    series_transformed : pandas DataFrame
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
        Model created with skforecast library.

    file_name: str
        File name given to the object.
        
    verbose: bool, default `True`
        Print info about the forecaster saved

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
    Load forecaster model from disc using joblib.

    Parameters
    ----------
    forecaster: forecaster object from skforecast library.
        Forecaster created with skforecast library.

    file_name: str
        File name given to the object.

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
    corr.index = corr.index.astype(int)
    corr.index.name="lag"
    
    return corr