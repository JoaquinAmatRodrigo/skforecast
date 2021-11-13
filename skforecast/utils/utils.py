################################################################################
#                                 utils                                        #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
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
    

def preprocess_y(y: pd.Series) -> Union[np.ndarray, pd.Index]:
    
    '''
    Returns values ​​and index of series separately. Index is overwritten
    according to the next rules:
        If index is not of type DatetimeIndex, a RangeIndex is created.
        If index is of type DatetimeIndex and but has no frequency, a
        RangeIndex is created.
        If index is of type DatetimeIndex and has frequency, nothing is
        changed.
    
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
    elif isinstance(y.index, pd.DatetimeIndex):
        warnings.warn(
            '`y` has DatetimeIndex index but no frequency. '
            'Index is overwritten with a RangeIndex.'
        )
        y_index = pd.RangeIndex(
                    start = 0,
                    stop  = len(y),
                    step  = 1
                  )
    else:
        warnings.warn(
            '`y` has no DatetimeIndex index. Index is overwritten with a RangeIndex.'
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
    Returns values ​​and index of series separately. Index is overwritten
    according to the next rules:
        If index is not of type DatetimeIndex, a RangeIndex is created.
        If index is of type DatetimeIndex and but has no frequency, a
        RangeIndex is created.
        If index is of type DatetimeIndex and has frequency, nothing is
        changed.
    
    Parameters
    ----------        
    last_window : pandas Series
        Time series values

    Returns 
    -------
    last_window_values : numpy ndarray
        Numpy array with values of `last_window`.

    last_window_index : pandas Index
        Index of of `last_window` modified according to the rules.
    '''
    
    if isinstance(last_window.index, pd.DatetimeIndex) and last_window.index.freq is not None:
        last_window_index = last_window.index
    else:
        warnings.warn(
            '`last_window` has DatetimeIndex index but no frequency. '
            'Index is overwritten with a RangeIndex.'
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
    Returns values ​​and index separately. Index is overwritten according to
    the next rules:
        If index is not of type DatetimeIndex, a RangeIndex is created.
        If index is of type DatetimeIndex and but has no frequency, a
        RangeIndex is created.
        If index is of type DatetimeIndex and has frequency, nothing is
        changed.

    Parameters
    ----------        
    exog : pd.Series, pd.DataFrame
        Exogenous variables

    Returns 
    -------
    exog_values : np.ndarray
        Numpy array with values of `exog`.
    exog_index : pd.Index
        Exog index.
    '''
    
    if isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is not None:
        exog_index = exog.index
    else:
        warnings.warn(
            ('`exog` has DatetimeIndex index but no frequency. The index is '
             'overwritten with a RangeIndex.')
        )
        exog_index = pd.RangeIndex(
                        start = 0,
                        stop  = len(exog),
                        step  = 1
                    )

    exog_values = exog.to_numpy()

    return exog_values, exog_index


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