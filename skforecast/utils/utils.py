################################################################################
#                                 utils                                        #
#                                                                              #
# This work by Joaquin Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.    
################################################################################
# coding=utf-8

from typing import Union, List, Tuple, Any
import warnings
import numpy as np
import pandas as pd


def check_y(y: Any) -> None:
    '''
    Raise Exception if `y` is not pandas Series or if it has missing values.
    
    Parameters
    ----------        
    y : Any
        Time series values
        
    Returns
    ----------
    None
    
    '''
    
    if not isinstance(y, pd.Series):
        raise Exception('`y` must be a pandas Series.')
        
    if y.isnull().any():
        raise Exception('`y` has missing values.')
    
    return
    
    
def check_exog(exog: Any) -> None:
    '''
    Raise Exception if `exog` is not pandas Series or DataFrame, or
    if it has missing values.
    
    Parameters
    ----------        
    exog :  Any
        Exogenous variable/s included as predictor/s.

    Returns
    ----------
    None
    '''
        
    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise Exception('`exog` must be `pd.Series` or `pd.DataFrame`.')

    if exog.isnull().any().any():
        raise Exception('`exog` has missing values.')
                
    return


def check_predict_input(
    steps: int,
    fitted: bool,
    included_exog: bool,
    index_type: type,
    index_freq: str,
    window_size: int,
    last_window: pd.Series=None,
    exog: Union[pd.Series, pd.DataFrame]=None,
    exog_type: Union[type, None]=None,
    exog_col_names: Union[list, None]=None,
    max_steps: int=None
) -> None:
    '''
    Check all inputs of predict method. This is a helper function to validate
    that inputs used in predict method match attributes of a forecaster already
    trained.

    Parameters
    ----------
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

    last_window : pandas Series, default `None`
        Values of the series used to create the predictors (lags) need in the 
        first iteration of prediction (t + 1).

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s.

    exog_type : type
        Type of exogenous variable/s used in training.
        
    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    max_steps: int
        Maximum number of steps allowed.
    '''

    if not fitted:
        raise Exception(
            'This Forecaster instance is not fitted yet. Call `fit` with'
            'appropriate arguments before using predict.'
        )
    
    if steps < 1:
        raise Exception(
            f"`steps` must be integer greater than 0. Got {steps}."
        )

    if max_steps is not None:
        if steps > max_steps:
            raise Exception(
                f"`steps` must be lower or equal to the value of steps defined "
                f"when initializing the forecaster. Got {steps} but the maximum "
                f"is {max_steps}."
            )
    
    if exog is None and included_exog:
        raise Exception(
            'Forecaster trained with exogenous variable/s. '
            'Same variable/s must be provided in `predict()`.'
        )
        
    if exog is not None and not included_exog:
        raise Exception(
            'Forecaster trained without exogenous variable/s. '
            '`exog` must be `None` in `predict()`.'
        )
    
    if exog is not None:
        if len(exog) < steps:
            raise Exception(
                '`exog` must have at least as many values as `steps` predicted.'
            )
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise Exception('`exog` must be a pandas Series or DataFrame.')
        if exog.isnull().values.any():
            raise Exception('`exog` has missing values.')
        if not isinstance(exog, exog_type):
            raise Exception(
                f"Expected type for `exog`: {exog_type}. Got {type(exog)}"      
            )
        if isinstance(exog, pd.DataFrame):
            col_missing = set(exog_col_names).difference(set(exog.columns))
            if col_missing:
                raise Exception(
                    f"Missing columns in `exog`. Expected {exog_col_names}. "
                    f"Got {exog.columns.to_list()}"      
                )
        check_exog(exog = exog)
        _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])
        
        if not isinstance(exog_index, index_type):
            raise Exception(
                f"Expected index of type {index_type} for `exog`. "
                f"Got {type(exog_index)}"      
            )
        
        if isinstance(exog_index, pd.DatetimeIndex):
            if not exog_index.freqstr == index_freq:
                raise Exception(
                    f"Expected frequency of type {index_freq} for `exog`. "
                    f"Got {exog_index.freqstr}"      
                )
        
    if last_window is not None:
        if len(last_window) < window_size:
            raise Exception(
                f"`last_window` must have as many values as as needed to "
                f"calculate the predictors. For this forecaster it is {window_size}."
            )
        if not isinstance(last_window, pd.Series):
            raise Exception('`last_window` must be a pandas Series.')
        if last_window.isnull().any():
            raise Exception('`last_window` has missing values.')
        _, last_window_index = preprocess_last_window(
                                    last_window = last_window.iloc[:0]
                                ) 
        if not isinstance(last_window_index, index_type):
            raise Exception(
                f"Expected index of type {index_type} for `last_window`. "
                f"Got {type(last_window_index)}"      
            )
        if isinstance(last_window_index, pd.DatetimeIndex):
            if not last_window_index.freqstr == index_freq:
                raise Exception(
                    f"Expected frequency of type {index_type} for `last_window`. "
                    f"Got {last_window_index.freqstr}"      
                )

    return
    

def preprocess_y(y: pd.Series) -> Union[np.ndarray, pd.Index]:
    
    '''
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
        Time series

    Returns 
    -------
    y_values : numpy ndarray
        Numpy array with values of `y`.

    y_index : pandas Index
        Index of `y` modified according to the rules.
    '''
    
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


def preprocess_last_window(last_window: pd.Series) -> Union[np.ndarray, pd.Index]:
    
    '''
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
    last_window : pandas Series
        Time series values

    Returns 
    -------
    last_window_values : numpy ndarray
        Numpy array with values of `last_window`.

    last_window_index : pandas Index
        Index of `last_window` modified according to the rules.
    '''
    
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
) -> Union[np.ndarray, pd.Index]:
    
    '''
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
        Exogenous variables

    Returns 
    -------
    exog_values : numpy ndarray
        Numpy array with values of `exog`.

    exog_index : pandas Index
        Index of `exog` modified according to the rules.
    '''
    
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


def exog_to_multi_output(
    exog: np.ndarray,
    steps: int
)-> np.ndarray:
    
    '''
    Transforms `exog` to `np.ndarray` with the shape needed for multioutput
    regresors.
    
    Parameters
    ----------        
    exog : numpy ndarray, shape(samples,)
        Time series values

    steps: int.
        Number of steps that will be predicted using this exog.

    Returns 
    -------
    exog_transformed: numpy ndarray
    '''

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


def expand_index(index: Union[pd.Index, None], steps: int) -> pd.Index:
    
    '''
    Create a new index of lenght `steps` starting and the end of index.
    
    Parameters
    ----------        
    index : pd.Index, None
        Index of last window
    steps: int
        Number of steps to expand.

    Returns 
    -------
    new_index : pd.Index
    '''
    
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