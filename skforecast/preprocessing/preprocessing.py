################################################################################
#                           skforecast.preprocessing                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Any
from typing_extensions import Self
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


def _check_X_numpy_ndarray_1d(func):
    """
    This decorator checks if the argument X is a numpy ndarray with 1 dimension.

    Parameters
    ----------
    func : Callable
        Function to wrap.
    
    Returns
    -------
    wrapper : wrapper
        Function wrapped.

    """

    def wrapper(self, *args, **kwargs):

        if args:
            X = args[0] 
        elif 'X' in kwargs:
            X = kwargs['X']
        else:
            raise ValueError("Methods must be called with 'X' as argument.")

        if not isinstance(X, np.ndarray):
            raise TypeError(f"'X' must be a numpy ndarray. Found {type(X)}.")
        if not X.ndim == 1:
            raise ValueError(f"'X' must be a 1D array. Found {X.ndim} dimensions.")
        
        result = func(self, *args, **kwargs)
        
        return result
    
    return wrapper


class TimeSeriesDifferentiator(BaseEstimator, TransformerMixin):
    """
    Transforms a time series into a differentiated time series of order n.
    It also reverts the differentiation.

    Parameters
    ----------
    order : int
        Order of differentiation.

    Attributes
    ----------
    order : int
        Order of differentiation.
    initial_values : list
        List with the initial value of the time series after each differentiation.
        This is used to revert the differentiation.
    last_values : list
        List with the last value of the time series after each differentiation.
        This is used to revert the differentiation of a new window of data. A new
        window of data is a time series that starts right after the time series
        used to fit the transformer.

    """

    def __init__(
        self, 
        order: int=1
    ) -> None:

        if not isinstance(order, int):
            raise TypeError(f"Parameter 'order' must be an integer greater than 0. Found {type(order)}.")
        if order < 1:
            raise ValueError(f"Parameter 'order' must be an integer greater than 0. Found {order}.")

        self.order = order
        self.initial_values = []
        self.last_values = []


    @_check_X_numpy_ndarray_1d
    def fit(
        self, 
        X: np.ndarray, 
        y: Any=None
    ) -> Self:
        """
        Fits the transformer. This method only removes the values stored in
        `self.initial_values`.

        Parameters
        ----------
        X : numpy ndarray
            Time series to be differentiated.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : TimeSeriesDifferentiator

        """

        self.initial_values = []
        self.last_values = []

        for i in range(self.order):
            if i == 0:
                self.initial_values.append(X[0])
                self.last_values.append(X[-1])
                X_diff = np.diff(X, n=1)
            else:
                self.initial_values.append(X_diff[0])
                self.last_values.append(X_diff[-1])
                X_diff = np.diff(X_diff, n=1)

        return self


    @_check_X_numpy_ndarray_1d
    def transform(
        self, 
        X: np.ndarray, 
        y: Any=None
    ) -> np.ndarray:
        """
        Transforms a time series into a differentiated time series of order n and
        stores the values needed to revert the differentiation.

        Parameters
        ----------
        X : numpy ndarray
            Time series to be differentiated.
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_diff : numpy ndarray
            Differentiated time series. The length of the array is the same as
            the original time series but the first n=`order` values are nan.

        """

        X_diff = np.diff(X, n=self.order)
        X_diff = np.append((np.full(shape=self.order, fill_value=np.nan)), X_diff)

        return X_diff


    @_check_X_numpy_ndarray_1d
    def inverse_transform(
        self, 
        X: np.ndarray, 
        y: Any=None
    ) -> np.ndarray:
        """
        Reverts the differentiation. To do so, the input array is assumed to be
        a differentiated time series of order n that starts right after the
        the time series used to fit the transformer.

        Parameters
        ----------
        X : numpy ndarray
            Differentiated time series.
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_diff : numpy ndarray
            Reverted differentiated time series.
        
        """

        # Remove initial nan values if present
        X = X[np.argmax(~np.isnan(X)):]
        for i in range(self.order):
            if i == 0:
                X_undiff = np.insert(X, 0, self.initial_values[-1])
                X_undiff = np.cumsum(X_undiff, dtype=float)
            else:
                X_undiff = np.insert(X_undiff, 0, self.initial_values[-(i+1)])
                X_undiff = np.cumsum(X_undiff, dtype=float)

        return X_undiff


    @_check_X_numpy_ndarray_1d
    def inverse_transform_next_window(
        self,
        X: np.ndarray,
        y: Any=None
    ) -> np.ndarray:
        """
        Reverts the differentiation. The input array `x` is assumed to be a 
        differentiated time series of order n that starts right after the
        the time series used to fit the transformer.

        Parameters
        ----------
        X : numpy ndarray
            Differentiated time series. It is assumed o start right after
            the time series used to fit the transformer.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_undiff : numpy ndarray
            Reverted differentiated time series.
        
        """

        # Remove initial nan values if present
        X = X[np.argmax(~np.isnan(X)):]

        for i in range(self.order):
            if i == 0:
                X_undiff = np.cumsum(X, dtype=float) + self.last_values[-1]
            else:
                X_undiff = np.cumsum(X_undiff, dtype=float) + self.last_values[-(i+1)]

        return X_undiff


def series_long_to_dict(
    data: pd.DataFrame,
    series_id: str,
    index: str,
    values: str,
    freq: str,
) -> dict:
    """
    Convert long format series to dictionary.

    Parameters
    ----------
    data: pandas DataFrame
        Long format series.
    series_id: str
        Column name with the series identifier.
    index: str
        Column name with the time index.
    values: str
        Column name with the values.
    freq: str
        Frequency of the series.

    Returns
    -------
    series_dict: dict
        Dictionary with the series.

    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    for col in [series_id, index, values]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in `data`.")

    series_dict = {}
    for k, v in data.groupby(series_id):
        series_dict[k] = v.set_index(index)[values].asfreq(freq).rename(k)
        series_dict[k].index.name = None

    return series_dict


def exog_long_to_dict(
    data: pd.DataFrame,
    series_id: str,
    index: str,
    freq: str,
    dropna: bool=False,
) -> dict:
    """
    Convert long format exogenous variables to dictionary.

    Parameters
    ----------
    data: pandas DataFrame
        Long format exogenous variables.
    series_id: str
        Column name with the series identifier.
    index: str
        Column name with the time index.
    freq: str
        Frequency of the series.
    dropna: bool, default `False`
        If True, drop columns with all values as NaN. This is useful when
        there are series without some exogenous variables.
        
    Returns
    -------
    exog_dict: dict
        Dictionary with the exogenous variables.

    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    for col in [series_id, index]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in `data`.")

    exog_dict = dict(tuple(data.groupby(series_id)))
    exog_dict = {
        k: v.set_index(index).asfreq(freq).drop(columns=series_id)
        for k, v in exog_dict.items()
    }

    for k in exog_dict.keys():
        exog_dict[k].index.name = None

    if dropna:
        exog_dict = {k: v.dropna(how="all", axis=1) for k, v in exog_dict.items()}

    return exog_dict


class CalendarFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer for extracting calendar features from the DateTime index of a DataFrame.

    Parameters:
    ----------
    cyclic_encoding : bool, default=False
        If True, applies cyclic encoding to specified datetime features.
    
    features : list of str, default=[
        'year', 
        'month',
        'week',
        'day_of_week',
        'day_of_month',
        'day_of_year',
        'weekend',
        'hour',
        'minute',
        'second'
    ]
        List of calendar features to extract from the index.
    
    max_values : dict, default={
        'month': 12,
        'week': 52,
        'day_of_week': 6, # 0 is Monday
        'day_of_month': 31,
        'day_of_year': 366,
        'hour': 24,
        'minute': 60,
        'second': 60
    }
        Dictionary of maximum values for the cyclic encoding of calendar features.
    """

    def __init__(self, cyclic_encoding=False, features=None, max_values=None):
        self.cyclic_encoding = cyclic_encoding
        self.features = features if features is not None else [
            'year', 
            'month',
            'week',
            'day_of_week',
            'day_of_month',
            'day_of_year',
            'weekend',
            'hour',
            'minute',
            'second'
        ]
        self.max_values = max_values if max_values is not None else {
            'month': 12,
            'week': 52,
            'day_of_week': 6, # 0 is Monday
            'day_of_month': 31,
            'day_of_year': 366,
            'hour': 24,
            'minute': 60,
            'second': 60
        }

    def fit(self, X, y=None):
        """Fit method - does nothing and is here for compatibility with sklearn pipeline."""
        return self
    
    def transform(self, X):
        """
        Transform method to extract and encode calendar features from the index.
        
        Parameters:
        ----------
        X : DataFrame
            Input dataframe with a datetime index.
        
        Returns:
        -------
        DataFrame
            DataFrame with the extracted (and optionally encoded) calendar features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Dataframe must have a datetime index")
    
        X_new = pd.DataFrame(index=X.index)

        datetime_attrs = {
            'year': 'year',
            'month': 'month',
            'week': lambda idx: idx.isocalendar().week,
            'day_of_week': 'dayofweek',
            'day_of_year': 'dayofyear',
            'day_of_month': 'day',
            'weekend': lambda idx: idx.weekday >= 5,
            'hour': 'hour',
            'minute': 'minute',
            'second': 'second'
        }

        for feature, attr in datetime_attrs.items():
            if feature in self.features:
                try:
                    if callable(attr):
                        X_new[feature] = attr(X.index)
                    else:
                        X_new[feature] = getattr(X.index, attr)
                except AttributeError as e:
                    pass

        if self.cyclic_encoding:
            for feature, max_val in self.max_values.items():
                if feature in X_new.columns:
                    X_new[f'{feature}_sin'] = np.sin(2 * np.pi * X_new[feature] / max_val)
                    X_new[f'{feature}_cos'] = np.cos(2 * np.pi * X_new[feature] / max_val)
                    X_new = X_new.drop(feature, axis=1)

        return X_new
