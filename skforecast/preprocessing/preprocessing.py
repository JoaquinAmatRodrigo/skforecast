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


def _check_X_numpy_ndarray_1d(ensure_1d=True):
    """
    This decorator checks if the argument X is a numpy ndarray with 1 dimension.

    Parameters
    ----------
    ensure_1d : bool, default=True
        Whether to ensure if X is a 1D numpy array.
    
    Returns
    -------
    decorator : Callable
        A decorator function.

    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):

            if args:
                X = args[0] 
            elif 'X' in kwargs:
                X = kwargs['X']
            else:
                raise ValueError("Methods must be called with 'X' as argument.")

            if not isinstance(X, np.ndarray):
                raise TypeError(f"'X' must be a numpy ndarray. Found {type(X)}.")
            if ensure_1d and not X.ndim == 1:
                raise ValueError(f"'X' must be a 1D array. Found {X.ndim} dimensions.")
            
            result = func(self, *args, **kwargs)
            
            return result
        
        return wrapper
    
    return decorator


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


    @_check_X_numpy_ndarray_1d()
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


    @_check_X_numpy_ndarray_1d()
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


    @_check_X_numpy_ndarray_1d()
    def inverse_transform(
        self, 
        X: np.ndarray, 
        y: Any = None
    ) -> np.ndarray:
        """
        Reverts the differentiation. To do so, the input array is assumed to be
        the same time series used to fit the transformer but differentiated.

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


    @_check_X_numpy_ndarray_1d(ensure_1d=False)
    def inverse_transform_next_window(
        self,
        X: np.ndarray,
        y: Any = None
    ) -> np.ndarray:
        """
        Reverts the differentiation. The input array `X` is assumed to be a 
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
        
        array_ndim = X.ndim
        if array_ndim == 1:
            X = X.reshape(-1, 1)

        # Remove initial rows with nan values if present
        X = X[~np.isnan(X).any(axis=1)]

        for i in range(self.order):
            if i == 0:
                X_undiff = np.cumsum(X, axis=0, dtype=float) + self.last_values[-1]
            else:
                X_undiff = np.cumsum(X_undiff, axis=0, dtype=float) + self.last_values[-(i + 1)]

        if array_ndim == 1:
            X_undiff = X_undiff.ravel()

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
    features: Optional[list] = None,
    encoding: str = "cyclical",
    max_values: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Extract datetime features from the DateTime index of a pandas DataFrame or Series.

    Parameters
    ----------
    X : pandas Series, pandas DataFrame
        Input DataFrame or Series with a datetime index.
    features : list, default `None`
        List of calendar features (strings) to extract from the index. When `None`,
        the following features are extracted: 'year', 'month', 'week', 'day_of_week',
        'day_of_month', 'day_of_year', 'weekend', 'hour', 'minute', 'second'.
    encoding : str, default `'cyclical'`
        Encoding method for the extracted features. Options are None, 'cyclical' or
        'onehot'.
    max_values : dict, default `None`
        Dictionary of maximum values for the cyclical encoding of calendar features.
        When `None`, the following values are used: {'month': 12, 'week': 52, 
        'day_of_week': 7, 'day_of_month': 31, 'day_of_year': 365, 'hour': 24, 
        'minute': 60, 'second': 60}.

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
    features : list, default `None`
        List of calendar features (strings) to extract from the index. When `None`,
        the following features are extracted: 'year', 'month', 'week', 'day_of_week',
        'day_of_month', 'day_of_year', 'weekend', 'hour', 'minute', 'second'.
    encoding : str, default `'cyclical'`
        Encoding method for the extracted features. Options are None, 'cyclical' or
        'onehot'.
    max_values : dict, default `None`
        Dictionary of maximum values for the cyclical encoding of calendar features.
        When `None`, the following values are used: {'month': 12, 'week': 52, 
        'day_of_week': 7, 'day_of_month': 31, 'day_of_year': 365, 'hour': 24, 
        'minute': 60, 'second': 60}.
    
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
        encoding: str = "cyclical",
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


class RollingFeatures():
    """
    This class computes rolling features.

    Parameters
    ----------
    stats : str, list
        Statistics to compute over the rolling window. Can be a `string` or a `list`,
        and can have repeats. Must be common between pandas and numpy 
        (e.g., 'mean', 'std', 'min', 'max', 'sum', 'median', etc.). 
    window_sizes : int, list
        Size of the rolling window for each statistic. If an `int`, all stats share 
        the same window size. If a `list`, it should have the same length as stats.
    features_names : list, default `None`
        Names of the output features. If `None`, default names will be used in the 
        format '_SeriesName_statistic'.
    min_periods : int, list, default `None`
        Minimum number of observations in window required to have a value. 
        Similar to pandas rolling `min_periods` argument. If `None`, defaults 
        to `window_sizes`.
    closed : str, list, default `'left'`
        - If 'left', the last point in the window is excluded from calculations.
        - If 'right', the first point in the window is excluded from calculations.
        - If 'both', the no points in the window are excluded from calculations.
        - If 'neither', the first and last points in the window are excluded from calculations.
    fill_strategy : str, list, default `None`
        Strategy to fill missing values in `transform_batch` method. Common pandas 
        methods for filling  missing values (e.g., 'ffill', 'bfill', 'mean', 
        'median', 'zero').
    fill_strategy_predict : float, list, default `None`
        Value(s) to fill missing data in the `transform` method. Can be a single 
        `float` or a `list`. Filling with a constant value is appropriate since only 
        one window is calculated in `transform`.
    
    Attributes
    ----------
    stats : list
        Statistics to compute over the rolling window.
    window_sizes : list
        Size of the rolling window for each statistic.
    features_names : list
        Names of the output features.
    min_periods : list
        Minimum number of observations in window required to have a value.
    closed : list
        Closing method for the rolling window.
    fill_strategy : list
        Strategy to fill missing values in `transform_batch` method.
    fill_strategy_predict : list
        Value(s) to fill missing data in the `transform` method.
        
    """

    def __init__(
        self, 
        stats, 
        window_sizes, 
        features_names: Union[str, list] = None, 
        min_periods: Union[int, list] = None,
        closed: Union[str, list] = 'left',
        fill_strategy: Union[str, list] = None,
        fill_strategy_predict: Union[float, list] = None
    ) -> None:
        
        # TODO: Return of method or method to set attributes?
        self._validate_preprocess_params(
            stats, 
            window_sizes, 
            features_names, 
            min_periods,
            closed,
            fill_strategy,
            fill_strategy_predict
        )

        # (
        #     stats,
        #     window_sizes,
        #     features_names,
        #     min_periods,
        #     closed,
        #     fill_strategy,
        #     fill_strategy_predict
        # ) = self._validate_preprocess_params(
        #         stats, 
        #         window_sizes, 
        #         features_names, 
        #         min_periods,
        #         closed,
        #         fill_strategy,
        #         fill_strategy_predict
        #     )
        
        # self.stats = stats
        # self.window_sizes = window_sizes
        # self.features_names = features_names
        # self.min_periods = min_periods
        # self.closed = closed
        # self.fill_strategy = fill_strategy
        # self.fill_strategy_predict = fill_strategy_predict

        window_params_list = []
        for i in range(len(self.stats)):
            window_params = (self.window_sizes[i], self.min_periods[i], self.closed[i])
            window_params_list.append(window_params)

        # Find unique window parameter combinations
        _unique_rolling_windows = {}
        for i, params in enumerate(window_params_list):
            key = f"{params[0]}_{params[1]}_{params[2]}"
            if key not in _unique_rolling_windows:
                _unique_rolling_windows[key] = {
                    'params': {'window': params[0], 'min_periods': params[1], 'closed': params[2]},
                    'stats_idx': [], 
                    'rolling_obj': None
                }
            _unique_rolling_windows[key]['stats_idx'].append(i)

        self._unique_rolling_windows = _unique_rolling_windows


    def _validate_preprocess_params(
        self, 
        stats, 
        window_sizes, 
        features_names: Union[str, list] = None, 
        min_periods: Union[int, list] = None,
        closed: Union[str, list] = 'left',
        fill_strategy: Union[str, list] = None,
        fill_strategy_predict: Union[float, list] = None
    ) -> Union[list, list, list, list, list, list, list]:
        """
        """

        # stats
        if isinstance(stats, str):
            stats = [stats]
        elif isinstance(stats, list):
            pass
        else:
            raise TypeError(
                f"`stats` must be a string or a list of strings. Got {type(stats)}."
            )
        
        allowed_stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'var']
        for stat in set(stats):
            if stat not in allowed_stats:
                raise ValueError(
                    f"Statistic '{stat}' is not allowed. Allowed stats are {allowed_stats}."
                )
        
        self.stats = stats
        n_stats = len(stats)
        
        # window_sizes
        if isinstance(window_sizes, int):
            self.window_sizes = [window_sizes] * n_stats
        elif isinstance(window_sizes, list):
            n_window_sizes = len(window_sizes)
            if n_window_sizes != n_stats:
                raise ValueError(
                    (f"Length of `window_sizes` list ({n_window_sizes}) "
                     f"must match length of `stats` list ({n_stats}).")
                )
            self.window_sizes = window_sizes
        else:
            raise TypeError(
                f"`window_sizes` must be an int or a list of ints. Got {type(window_sizes)}."
            )
        
        # features_names
        if features_names is None:
            self.features_names = None  # Will generate default names later
        elif isinstance(features_names, list):
            n_features_names = len(window_sizes)
            if n_features_names != n_stats:
                raise ValueError(
                    (f"Length of `features_names` list ({n_features_names}) "
                     f"must match length of `stats` list ({n_stats}).")
                )
            self.features_names = features_names
        else:
            raise TypeError(
                f"`features_names` must be a list of strings or None. Got {type(window_sizes)}."
            )
        
        # min_periods
        if min_periods is None:
            self.min_periods = self.window_sizes
        elif isinstance(min_periods, int):
            self.min_periods = [min_periods] * n_stats
        elif isinstance(min_periods, list):
            n_min_periods = len(min_periods)
            if n_min_periods != n_stats:
                raise ValueError(
                    (f"Length of `min_periods` list ({n_min_periods}) "
                     f"must match length of `stats` list ({n_stats}).")
                )
            self.min_periods = min_periods
        else:
            raise TypeError(
                f"`min_periods` must be an int, list of ints, or None. Got {type(window_sizes)}."
            )
        
        # closed
        allowed_closed = ['right', 'left', 'both', 'neither']
        if isinstance(closed, str):
            closed = [closed] * n_stats
        elif isinstance(closed, list):
            n_closed = len(closed)
            if n_closed != n_stats:
                raise ValueError(
                    (f"Length of `closed` list ({n_closed}) "
                     f"must match length of `stats` list ({n_stats}).")
                )
        else:
            raise TypeError(
                f"`closed` must be a string or a list of strings. Got {type(window_sizes)}."
            )
        
        allowed_closed = ['right', 'left', 'both', 'neither']
        for c in set(closed):
            if c not in allowed_closed:
                raise ValueError(
                    f"'{c}' is not allowed. Allowed `closed` values are {allowed_closed}."
                )
        
        self.closed = closed
        
        # fill_strategy
        if fill_strategy is None:
            fill_strategy = [None] * n_stats
        elif isinstance(fill_strategy, str):
            fill_strategy = [fill_strategy] * n_stats
        elif isinstance(fill_strategy, list):
            n_fill_strategy = len(fill_strategy)
            if n_fill_strategy != n_stats:
                raise ValueError(
                    (f"Length of `fill_strategy` list ({n_fill_strategy}) "
                     f"must match length of `stats` list ({n_stats})")
                )
        else:
            raise TypeError(
                f"`fill_strategy` must be a string or a list of strings. Got {type(fill_strategy)}"
            )
        
        # TODO: Complete
        # allowed_fill_strategy = ['right', 'left', 'both', 'neither']
        # for fs in set(fill_strategy):
        #     if fs not in allowed_fill_strategy:
        #         raise ValueError(
        #             (f"'{fs}' is not allowed. Allowed `fill_strategy` "
        #              f"values are {allowed_fill_strategy}.")
        #         )
        
        self.fill_strategy = fill_strategy
        
        # fill_strategy_predict        
        if fill_strategy_predict is None:
            fill_strategy_predict = [None] * n_stats
        elif isinstance(fill_strategy_predict, (int, float)):
            fill_strategy_predict = [fill_strategy_predict] * n_stats
        elif isinstance(fill_strategy_predict, list):
            n_fill_strategy_predict = len(fill_strategy_predict)
            if n_fill_strategy_predict != n_stats:
                raise ValueError(
                    (f"Length of `fill_strategy_predict` list ({n_fill_strategy_predict}) "
                     f"must match length of `stats` list ({n_stats})")
                )
        else:
            raise TypeError(
                f"`fill_strategy_predict` must be a float or a list of float. Got {type(fill_strategy_predict)}"
            )
        
        # TODO: Complete
        # allowed_fill_strategy_predict = ['right', 'left', 'both', 'neither']
        # for fs in set(fill_strategy_predict):
        #     if fs not in allowed_fill_strategy_predict:
        #         raise ValueError(
        #             (f"'{fs}' is not allowed. Allowed `fill_strategy_predict` "
        #              f"values are {allowed_fill_strategy_predict}.")
        #         )
        
        self.fill_strategy_predict = fill_strategy_predict


    def transform_batch(
        self, 
        series: pd.Series
    ) -> pd.DataFrame:
        """
        Transform an entire pandas Series using rolling windows and compute the 
        specified statistics.

        Parameters
        ----------
        series : pandas Series
            The input data series to transform.

        Returns
        -------
        results : pandas DataFrame
            A DataFrame containing the rolling features.
        
        """
        
        results = pd.DataFrame(index=series.index)
        series_name = series.name if series.name is not None else 'y'

        for k in self._unique_rolling_windows.keys():
            print(self._unique_rolling_windows[k]['params'])
            rolling_obj = series.rolling(**self._unique_rolling_windows[k]['params'])
            self._unique_rolling_windows[k]['rolling_obj'] = rolling_obj
        
        for i, stat in enumerate(self.stats):
            window_size = self.window_sizes[i]
            min_periods = self.min_periods[i]
            closed = self.closed[i]
            fill_strategy = self.fill_strategy[i]

            key = f"{window_size}_{min_periods}_{closed}"
            rolling_obj = self._unique_rolling_windows[key]['rolling_obj']
            
            # Generate feature name
            if self.features_names is not None:
                feature_name = self.features_names[i]
            else:
                feature_name = f"_{series_name}_{stat}"
            
            # Get the function corresponding to the statistic
            if stat == 'mean':
                stat_series = rolling_obj.mean()
            elif stat == 'std':
                stat_series = rolling_obj.std()
            elif stat == 'var':
                stat_series = rolling_obj.var()
            elif stat == 'min':
                stat_series = rolling_obj.min()
            elif stat == 'max':
                stat_series = rolling_obj.max()
            elif stat == 'sum':
                stat_series = rolling_obj.sum()
            elif stat == 'median':
                stat_series = rolling_obj.median()
            else:
                raise ValueError(f"Statistic '{stat}' is not implemented")
            
            # TODO: Complete
            # Fill missing values if fill_strategy is specified
            if fill_strategy:
                if fill_strategy == 'mean':
                    stat_series = stat_series.fillna(stat_series.mean())
                elif fill_strategy == 'median':
                    stat_series = stat_series.fillna(stat_series.median())
                elif fill_strategy == 'ffill':
                    stat_series = stat_series.fillna(method='ffill')
                elif fill_strategy == 'bfill':
                    stat_series = stat_series.fillna(method='bfill')
                elif fill_strategy == 'zero':
                    stat_series = stat_series.fillna(0)
                else:
                    # Try to convert fill_strategy to a number
                    try:
                        fill_value = float(fill_strategy)
                        stat_series = stat_series.fillna(fill_value)
                    except ValueError:
                        raise ValueError(f"fill_strategy '{fill_strategy}' is not recognized")
            
            results[feature_name] = stat_series
        
        return results


    def transform(
        self, 
        array: np.ndarray
    ) -> np.ndarray:
        """
        Transform a numpy array using rolling windows and compute the specified statistics.

        Parameters
        ----------
        array : np.ndarray
            The input data array to transform. Should be at least as long as the maximum window size.

        Returns
        -------
        results : numpy array
            An array containing the computed statistics.
        
        """
        
        results = []
        
        array = np.asarray(array)
        n_samples = len(array)
        max_window_size = max(self.window_sizes)
        
        if n_samples < max_window_size:
            raise ValueError(f"Input array must have at least {max_window_size} observations")
        
        for i, stat in enumerate(self.stats):
            window_size = self.window_sizes[i]
            data_window = array[-window_size:]  # Take the last window_size observations
            
            # Compute the statistic using numpy, handling NaNs
            if stat == 'mean':
                value = np.nanmean(data_window)
            elif stat == 'std':
                value = np.nanstd(data_window, ddof=1)  # ddof=1 for sample std
            elif stat == 'var':
                value = np.nanvar(data_window, ddof=1)
            elif stat == 'min':
                value = np.nanmin(data_window)
            elif stat == 'max':
                value = np.nanmax(data_window)
            elif stat == 'sum':
                value = np.nansum(data_window)
            elif stat == 'median':
                value = np.nanmedian(data_window)
            else:
                raise ValueError(f"Statistic '{stat}' is not implemented")
            
            # Handle fill_strategy_predict
            fill_value = self.fill_strategy_predict[i]
            if np.isnan(value):
                if fill_value is not None:
                    value = fill_value
                else:
                    value = np.nan  # Leave as NaN if no fill_value is specified
            
            results.append(value)
        
        return np.array(results)
