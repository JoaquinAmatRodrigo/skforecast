################################################################################
#                           skforecast.preprocessing                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Any, Union, Optional
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
        order: int = 1
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
        y: Any = None
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
        y: Any = None
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
        y: Any = None
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
                X_undiff = np.insert(X_undiff, 0, self.initial_values[-(i + 1)])
                X_undiff = np.cumsum(X_undiff, dtype=float)

        return X_undiff


    @_check_X_numpy_ndarray_1d
    def inverse_transform_next_window(
        self,
        X: np.ndarray,
        y: Any = None
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
                X_undiff = np.cumsum(X_undiff, dtype=float) + self.last_values[-(i + 1)]

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
    dropna: bool = False,
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


def create_datetime_features(
    X: Union[pd.Series, pd.DataFrame],
    features: list = None,
    encoding: str = "cyclical",
    max_values: dict = None,
) -> pd.DataFrame:
    """
    Extract datetime features from the DateTime index of a pandas DataFrame or Series.

    Parameters
    ----------
    X : pandas Series, pandas DataFrame
        Input DataFrame or Series with a datetime index.
    features : list, default `[
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
    ]`
        List of calendar features (strings) to extract from the index.
    encoding : str, default `'cyclical'`
        Encoding method for the extracted features. Options are None, 'cyclical' or
        'onehot'.
    max_values : dict, default {
        'month': 12,
        'week': 52,
        'day_of_week': 7,
        'day_of_month': 31,
        'day_of_year': 365,
        'hour': 24,
        'minute': 60,
        'second': 60
    }
        Dictionary of maximum values for the cyclical encoding of calendar features.

    Returns
    -------
    X_new : pandas DataFrame
        DataFrame with the extracted (and optionally encoded) datetime features.
    
    """

    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError("Input `X` must be a pandas Series or DataFrame")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("Input `X` must have a pandas DatetimeIndex")
    if encoding not in ["cyclical", "onehot", None]:
        raise ValueError("Encoding must be one of 'cyclical', 'onehot' or None")

    default_features = [
        "year",
        "month",
        "week",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "weekend",
        "hour",
        "minute",
        "second",
    ]
    features = features or default_features

    default_max_values = {
        "month": 12,
        "week": 52,
        "day_of_week": 7,
        "day_of_month": 31,
        "day_of_year": 365,
        "hour": 24,
        "minute": 60,
        "second": 60,
    }
    max_values = max_values or default_max_values

    X_new = pd.DataFrame(index=X.index)

    datetime_attrs = {
        "year": "year",
        "month": "month",
        "week": lambda idx: idx.isocalendar().week,
        "day_of_week": "dayofweek",
        "day_of_year": "dayofyear",
        "day_of_month": "day",
        "weekend": lambda idx: (idx.weekday >= 5).astype(int),
        "hour": "hour",
        "minute": "minute",
        "second": "second",
    }

    not_supported_features = set(features) - set(datetime_attrs.keys())
    if not_supported_features:
        raise ValueError(
            f"Features {not_supported_features} are not supported. "
            f"Supported features are {list(datetime_attrs.keys())}."
        )

    for feature in features:
        attr = datetime_attrs[feature]
        X_new[feature] = (
            attr(X.index) if callable(attr) else getattr(X.index, attr).astype(int)
        )

    if encoding == "cyclical":
        cols_to_drop = []
        for feature, max_val in max_values.items():
            if feature in X_new.columns:
                X_new[f"{feature}_sin"] = np.sin(2 * np.pi * X_new[feature] / max_val)
                X_new[f"{feature}_cos"] = np.cos(2 * np.pi * X_new[feature] / max_val)
                cols_to_drop.append(feature)
        X_new = X_new.drop(columns=cols_to_drop)
    elif encoding == "onehot":
        X_new = pd.get_dummies(
            X_new, columns=features, drop_first=False, sparse=False, dtype=int
        )

    return X_new


class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for extracting datetime features from the DateTime index of a
    pandas DataFrame or Series. It can also apply encoding to the extracted features.

    Parameters
    ----------
    features : list, default `[
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
    ]`
        List of calendar features (strings) to extract from the index.
    encoding : str, default `'cyclical'`
        Encoding method for the extracted features. Options are None, 'cyclical' or
        'onehot'.
    max_values : dict, default {
        'month': 12,
        'week': 52,
        'day_of_week': 7,
        'day_of_month': 31,
        'day_of_year': 365,
        'hour': 24,
        'minute': 60,
        'second': 60
    }
        Dictionary of maximum values for the cyclical encoding of calendar features.
    
    Attributes
    ----------
    features : list
        List of calendar features to extract from the index.
    encoding : str
        Encoding method for the extracted features.
    max_values : dict
        Dictionary of maximum values for the cyclical encoding of calendar features.
    
    """

    def __init__(
        self,
        features: Optional[list] = None,
        encoding: Optional[str] = "cyclical",
        max_values: Optional[dict] = None
    ) -> None:

        if encoding not in ["cyclical", "onehot", None]:
            raise ValueError("Encoding must be one of 'cyclical', 'onehot' or None")

        self.features = (
            features
            if features is not None
            else [
                "year",
                "month",
                "week",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "weekend",
                "hour",
                "minute",
                "second",
            ]
        )
        self.encoding = encoding
        self.max_values = (
            max_values
            if max_values is not None
            else {
                "month": 12,
                "week": 52,
                "day_of_week": 7,
                "day_of_month": 31,
                "day_of_year": 365,
                "hour": 24,
                "minute": 60,
                "second": 60,
            }
        )

    def fit(self, X, y=None):
        """
        A no-op method to satisfy the scikit-learn API.
        """
        return self

    def transform(
        self,
        X: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create datetime features from the DateTime index of a pandas DataFrame or Series.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame
            Input DataFrame or Series with a datetime index.
        
        Returns
        -------
        X_new : pandas DataFrame
            DataFrame with the extracted (and optionally encoded) datetime features.

        """

        X_new = create_datetime_features(
                    X          = X,
                    encoding   = self.encoding,
                    features   = self.features,
                    max_values = self.max_values,
                )

        return X_new
