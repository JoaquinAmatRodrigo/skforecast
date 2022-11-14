################################################################################
#                        ForecasterAutoregMultiSeries                          #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional, Callable
import warnings
import logging
import sys
import inspect
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from sklearn.base import clone
from copy import copy, deepcopy

import skforecast
from ..ForecasterBase import ForecasterBase
from ..utils import initialize_lags
from ..utils import check_y
from ..utils import check_exog
from ..utils import preprocess_y
from ..utils import preprocess_last_window
from ..utils import preprocess_exog
from ..utils import expand_index
from ..utils import check_predict_input
from ..utils import transform_series
from ..utils import transform_dataframe

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)

class ForecasterAutoregMultiSeries(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster for multiple series.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        
    lags : int, list, numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list`, `numpy ndarray` or `range`: include only lags present in `lags`,
            all elements must be int.

    transformer_series : transformer (preprocessor) or dict of transformers, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        If a single transformer is passed, it is cloned and applied to all series. If a
        dict, a different transformer can be used for each series. Transformation is
        applied to each `series` before training the forecaster.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
    
    transformer_exog : transformer, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    series_weights : dict, default `None`
        Weights associated with each series {'series_column_name' : float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method.
        If `series_weights` is provided, a weight of 1 is given to all series not present
        in `series_weights`. If `None`, all levels have the same weight. See Notes section
        for more details on the use of the weights.
        **New in version 0.6.0**

    weight_func : callable, dict, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        If dict {'series_column_name' : callable} a different function can be
        used for each series.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. See Notes section for more details on the use of the weights.
        **New in version 0.6.0**
    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        
    lags : numpy ndarray
        Lags used as predictors.

    transformer_series : transformer (preprocessor) or dict of transformers, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        If a single transformer is passed, it is cloned and applied to all series. If a
        dict, a different transformer can be used for each series. Transformation is
        applied to each `series` before training the forecaster.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        
    transformer_series_ : dict
        Dictionary with the transformer for each series. It is created cloning the objects
        in `transformer_series` and is used internally to avoid overwriting.

    transformer_exog : transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    series_weights : dict, default `None`
        Weights associated with each series {'series_column_name' : float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method.
        If `series_weights` is provided, a weight of 1 is given to all series not present
        in `series_weights`. If `None`, all levels have the same weight. See Notes section
        for more details on the use of the weights.
        **New in version 0.6.0**

    series_weights_ : dict
        Weights associated with each series.It is created as a clone of `series_weights`
        and is used internally to avoid overwriting.
        **New in version 0.6.0**

    weight_func : callable, dict, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        If dict {'series_column_name' : callable} a different function can be
        used for each series.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. See Notes section for more details on the use of the weights.
        **New in version 0.6.0**

    weight_func_ : dict
        Dictionary with the `weight_func` for each series. It is created cloning the objects
        in `weight_func` and is used internally to avoid overwriting.
        **New in version 0.6.0**

    source_code_weight_func : str, dict
        Source code of the custom function(s) used to create weights.
        **New in version 0.6.0**

    max_lag : int
        Maximum value of lag included in `lags`.
        
    window_size : int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.

    last_window : pandas Series
        Last window the forecaster has seen during training. It stores the
        values needed to predict the next `step` right after the training data.
        
    index_type : type
        Type of index of the input used in training.
        
    index_freq : str
        Frequency of Index of the input used in training.

    index_values : pandas Index
        Values of Index of the input used in training.

    training_range: pandas Index
        First and last values of index of the data used during training.
        
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
        
    exog_type : type
        Type of exogenous variable/s used in training.
        
    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    series_col_names : list
        Names of the series (levels) used during training.

    X_train_col_names : list
        Names of columns of the matrix created internally for training.
        
    in_sample_residuals : dict
        Residuals of the model when predicting training data. Only stored up to
        1000 values in the form `{level: residuals}`.
        
    out_sample_residuals : dict
        Residuals of the model when predicting non-training data. Only stored
        up to 1000 values in the form `{level: residuals}`. Use 
        `set_out_sample_residuals` to set values.
        
    fitted : Bool
        Tag to identify if the regressor has been fitted (trained).

    creation_date : str
        Date of creation.

    fit_date : str
        Date of last fit.

    skforcast_version : str
        Version of skforecast library used to create the forecaster.

    python_version : str
        Version of python used to create the forecaster.


    Notes
    -----

    The weights are used to control the influence that each observation has on the
    training of the model. `ForecasterAutoregMultiseries` accepts two types of weights:

    + series_weights : controls the relative importance of each series. If a series has
    twice as much weight as the others, the observations of that series influence the
    training twice as much. The higher the weight of a series relative to the others,
    the more the model will focus on trying to learn that series.

    + weight_func : controls the relative importance of each observation according to its
    index value. For example, a function that assigns a lower weight to certain dates.

    If the two types of weights are indicated, they are multiplied to create the final
    weights. The resulting `sample_weight` cannot have negative values.
    
    """
    
    def __init__(
        self,
        regressor: object,
        lags: Union[int, np.ndarray, list],
        transformer_series: Optional[Union[object, dict]]=None,
        transformer_exog: Optional[object]=None,
        series_weights: Optional[dict]=None,
        weight_func: Optional[Union[callable, dict]]=None
    ) -> None:
        
        self.regressor               = regressor
        self.transformer_series      = transformer_series
        self.transformer_series_     = None
        self.transformer_exog        = transformer_exog
        self.series_weights          = series_weights
        self.series_weights_         = None
        self.weight_func             = weight_func
        self.weight_func_            = None
        self.source_code_weight_func = None
        self.index_type              = None
        self.index_freq              = None
        self.index_values            = None
        self.training_range          = None
        self.last_window             = None
        self.included_exog           = False
        self.exog_type               = None
        self.exog_col_names          = None
        self.series_col_names        = None
        self.X_train_col_names       = None
        self.in_sample_residuals     = None
        self.out_sample_residuals    = None
        self.fitted                  = False
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                = None
        self.skforcast_version       = skforecast.__version__
        self.python_version          = sys.version.split(" ")[0]
        
        self.lags = initialize_lags(type(self), lags)
        self.max_lag = max(self.lags)
        self.window_size = self.max_lag

        if series_weights is not None:
            if not isinstance(series_weights, dict):
                raise TypeError(
                    f"Argument `series_weights` must be a dict of floats or ints."
                    f"Got {type(series_weights)}."
                )
            if 'sample_weight' not in inspect.getfullargspec(self.regressor.fit)[0]:
                warnings.warm(
                    f"""
                    Argument `series_weights` is ignored since regressor {self.regressor}
                    does not accept `sample_weight` in its `fit` method.
                    """
                )
                self.series_weights = None

        if weight_func is not None:
            if not isinstance(weight_func, (Callable, dict)):
                raise TypeError(
                    f"Argument `weight_func` must be a callable or a dict of "
                    f"callables. Got {type(weight_func)}."
                )

            if isinstance(weight_func, dict):
                self.source_code_weight_func = {}
                for key in weight_func:
                    self.source_code_weight_func[key] = inspect.getsource(weight_func[key])
            else:
                self.source_code_weight_func = inspect.getsource(weight_func)

            if 'sample_weight' not in inspect.getfullargspec(self.regressor.fit)[0]:
                warnings.warn(
                    f"""
                    Argument `weight_func` is ignored since regressor {self.regressor}
                    does not accept `sample_weight` in its `fit` method.
                    """
                )
                self.weight_func = None
                self.weight_func_ = None
                self.source_code_weight_func = None


    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterAutoregMultiSeries object is printed.
        """

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            name_pipe_steps = tuple(name + "__" for name in self.regressor.named_steps.keys())
            params = {key : value for key, value in self.regressor.get_params().items() \
                      if key.startswith(name_pipe_steps)}
        else:
            params = self.regressor.get_params()

        info = (
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"{str(type(self)).split('.')[1]} \n"
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
            f"Transformer for series: {self.transformer_series} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Window size: {self.window_size} \n"
            f"Series levels (names): {self.series_col_names} \n"
            f"Series weights: {self.series_weights} \n"
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Exogenous included: {self.included_exog} \n"
            f"Type of exogenous variable: {self.exog_type} \n"
            f"Exogenous variables names: {self.exog_col_names} \n"
            f"Training range: {self.training_range.to_list() if self.fitted else None} \n"
            f"Training index type: {str(self.index_type).split('.')[-1][:-2] if self.fitted else None} \n"
            f"Training index frequency: {self.index_freq if self.fitted else None} \n"
            f"Regressor parameters: {params} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforcast_version} \n"
            f"Python version: {self.python_version} \n"
        )

        return info

    
    def _create_lags(
        self, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """       
        Transforms a 1d array into a 2d array (X) and a 1d array (y). Each row
        in X is associated with a value of y and it represents the lags that
        precede it.
        
        Notice that, the returned matrix X_data, contains the lag 1 in the first
        column, the lag 2 in the second column and so on.
        
        Parameters
        ----------        
        y : 1d numpy ndarray
            Training time series.

        Returns 
        -------
        X_data : 2d numpy ndarray, shape (samples - max(self.lags), len(self.lags))
            2d numpy array with the lagged values (predictors).
        
        y_data : 1d numpy ndarray, shape (samples - max(self.lags),)
            Values of the time series related to each row of `X_data`.
        
        """
          
        n_splits = len(y) - self.max_lag
        if n_splits <= 0:
            raise ValueError(
                f'The maximum lag ({self.max_lag}) must be less than the length '
                f'of the series ({len(y)}).'
            )
        
        X_data = np.full(shape=(n_splits, len(self.lags)), fill_value=np.nan, dtype=float)

        for i, lag in enumerate(self.lags):
            X_data[:, i] = y[self.max_lag - lag: -lag]

        y_data = y[self.max_lag:]
            
        return X_data, y_data


    def create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Index, pd.Index]:
        """
        Create training matrices from multiple time series and exogenous
        variables.
        
        Parameters
        ----------        
        series : pandas DataFrame
            Training time series.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `series` and their indexes must be aligned.

        Returns 
        -------
        X_train : pandas DataFrame
            Pandas DataFrame with the training values (predictors).
            
        y_train : pandas Series, shape (len(series) - self.max_lag, )
            Values (target) of the time series related to each row of `X_train`.

        y_index : pandas Index
            Index of `series`.

        y_train_index: pandas Index
            Index of `y_train`.
        
        """

        if not isinstance(series, pd.DataFrame):
            raise TypeError(f'`series` must be a pandas DataFrame. Got {type(series)}.')

        series_col_names = list(series.columns)

        self.transformer_series_ = deepcopy(self.transformer_series)
        if self.transformer_series is None:
            self.transformer_series_ = {serie: None for serie in series_col_names}
        elif not isinstance(self.transformer_series, dict):
            self.transformer_series_ = {serie: clone(self.transformer_series) 
                                        for serie in series_col_names}
        else:
            if list(self.transformer_series_.keys()) != series_col_names:
                raise ValueError(
                    (f'When `transformer_series` parameter is a `dict`, its keys '
                     f'must be the same as `series` column names.\n'
                     f'    `transformer_series` keys : {list(self.transformer_series_.keys())}.\n'
                     f'    `series` columns          : {series_col_names}.')
                )
        
        X_levels = []
        X_train_col_names = [f"lag_{lag}" for lag in self.lags]

        for i, serie in enumerate(series.columns):

            y = series[serie]
            check_y(y=y)
            y = transform_series(
                    series            = y,
                    transformer       = self.transformer_series_[serie],
                    fit               = True,
                    inverse_transform = False
                )

            y_values, y_index = preprocess_y(y=y)
            X_train_values, y_train_values = self._create_lags(y=y_values)

            if i == 0:
                X_train = X_train_values
                y_train = y_train_values
            else:
                X_train = np.vstack((X_train, X_train_values))
                y_train = np.append(y_train, y_train_values)

            X_level = [serie]*len(X_train_values)
            X_levels.extend(X_level)

        if exog is not None:
            if len(exog) != len(series):
                raise ValueError(
                    f'`exog` must have same number of samples as `series`. '
                    f'length `exog`: ({len(exog)}), length `series`: ({len(series)})'
                )
            check_exog(exog=exog)
            if isinstance(exog, pd.Series):
                exog = transform_series(
                            series            = exog,
                            transformer       = self.transformer_exog,
                            fit               = True,
                            inverse_transform = False
                       )
            else:
                exog = transform_dataframe(
                            df                = exog,
                            transformer       = self.transformer_exog,
                            fit               = True,
                            inverse_transform = False
                       )
            exog_values, exog_index = preprocess_exog(exog=exog)
            if not (exog_index[:len(y_index)] == y_index).all():
                raise ValueError(
                    ('Different index for `series` and `exog`. They must be equal '
                     'to ensure the correct alignment of values.')      
                )
            col_names_exog = exog.columns if isinstance(exog, pd.DataFrame) else [exog.name]
            X_train_col_names.extend(col_names_exog)

            # The first `self.max_lag` positions have to be removed from exog
            # since they are not in X_train. Then exog is cloned as many times
            # as series.
            if exog_values.ndim == 1:
                X_train = np.column_stack((
                              X_train,
                              np.tile(exog_values[self.max_lag:, ], series.shape[1])
                          )) 

            else:
                X_train = np.column_stack((
                              X_train,
                              np.tile(exog_values[self.max_lag:, ], [series.shape[1], 1])
                          ))

        X_levels = pd.Series(X_levels)
        X_levels = pd.get_dummies(X_levels, dtype=float)
        X_train_col_names.extend(X_levels.columns)
        X_train = np.column_stack((X_train, X_levels.values))

        X_train = pd.DataFrame(
                      data    = X_train,
                      columns = X_train_col_names
                  )

        y_train = pd.Series(
                      data  = y_train,
                      name  = 'y'
                  )

        y_train_index = pd.Index(
                            np.tile(
                                y_index[self.max_lag: ].values,
                                reps = len(series_col_names)
                            )
                        )
        
        self.X_train_col_names = X_train_col_names

        return X_train, y_train, y_index, y_train_index

    
    def create_sample_weights(
        self,
        series: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train_index: pd.Index,
    )-> np.ndarray:
        """
        Crate weights for each observation according to the forecaster's attributes
        `series_weights` and `weight_func`. The resulting weights are product of both
        types of weights.

        Parameters
        ----------
        series : pandas DataFrame
            Time series used to create `X_train` with the method `create_train_X_y`.
        X_train : pandas DataFrame
            Dataframe generated with the method `create_train_X_y`, first return.
        y_train_index : pandas Index
            Index of `y_train` generated with the method `create_train_X_y`, fourth return.

        Returns
        -------
        weights : numpy ndarray
            Weights to use in `fit` method.
        
        """

        weights = None
        weights_samples = None
        weights_series = None

        if self.series_weights is not None:
            # Series not present in series_weights have a weight of 1 in all their samples.
            # Keys in series_weights not present in series are ignored.
            series_not_in_series_weights = set(series.columns) - set(self.series_weights.keys())
            if series_not_in_series_weights:
                    logging.warning(
                        f"{series_not_in_series_weights} not present in `series_weights`."
                        f" A weight of 1 is given to all their samples."
                    )
            self.series_weights_ = dict.fromkeys(series.columns, 1.)
            self.series_weights_.update((k, v) for k, v in self.series_weights.items() if k in self.series_weights_)
            weights_series = [np.repeat(self.series_weights_[serie], sum(X_train[serie])) 
                             for serie in series.columns]
            weights_series = np.concatenate(weights_series)

        if self.weight_func is not None:
            if isinstance(self.weight_func, Callable):
                self.weight_func_ = dict.fromkeys(series.columns, self.weight_func)
            else:
                # Series not present in weight_func have a weight of 1 in all their samples
                series_not_in_weight_func = set(series.columns) - set(self.weight_func.keys())
                if series_not_in_weight_func:
                    logging.warning(
                        f"{series_not_in_weight_func} not present in `weight_func`."
                        f" A weight of 1 is given to all their samples."
                    )
                    print(series_not_in_weight_func)
                self.weight_func_ = dict.fromkeys(series.columns, lambda index: np.ones_like(index, dtype=float))
                self.weight_func_.update((k, v) for k, v in self.weight_func.items() if k in self.weight_func_)
                
            weights_samples = []
            for key in self.weight_func_.keys():
                print(key)
                index = y_train_index[X_train[X_train[key] == 1.0].index]
                weights_samples.append(self.weight_func_[key](index))
            weights_samples = np.concatenate(weights_samples)

        if weights_series is not None:
            weights = weights_series
            if weights_samples is not None:
                weights = weights * weights_samples
        else:
            if weights_samples is not None:
                weights = weights_samples

        if weights is not None:
            if np.isnan(weights).any():
                raise ValueError(
                    "The resulting `weights` cannot have NaN values."
                )
            if np.any(weights < 0):
                raise ValueError(
                    "The resulting `weights` cannot have negative values."
                )
            if np.sum(weights) == 0:
                raise ValueError(
                    ("The resulting `weights` cannot be normalized because "
                     "the sum of the weights is zero.")
                )

        return weights

        
    def fit(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        store_in_sample_residuals: bool=True
    ) -> None:
        """
        Training Forecaster.
        
        Parameters
        ----------        
        series : pandas DataFrame
            Training time series.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `series` and their indexes must be aligned so
            that series[i] is regressed on exog[i].

        store_in_sample_residuals : bool, default `True`
            if True, in_sample_residuals are stored.

        Returns 
        -------
        None
        
        """
        
        # Reset values in case the forecaster has already been fitted.
        self.index_type          = None
        self.index_freq          = None
        self.index_values        = None
        self.last_window         = None
        self.included_exog       = False
        self.exog_type           = None
        self.exog_col_names      = None
        self.series_col_names    = None
        self.X_train_col_names   = None
        self.in_sample_residuals = None
        self.fitted              = False
        self.training_range      = None
        
        self.series_col_names = list(series.columns)

        if self.series_weights is not None:
            if list(self.series_weights.keys()) != self.series_col_names:
                raise ValueError(
                    (f'`series_weights` must include all series levels (column names of series).\n'
                     f'    `series_col_names`  = {self.series_col_names}.\n'
                     f'    `series_weights` = {list(self.series_weights.keys())}.')
                )

        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else [exog.name]

            if len(set(self.exog_col_names) - set(self.series_col_names)) != len(self.exog_col_names):
                raise ValueError(
                    (f'`exog` cannot contain a column named the same as one of the series'
                     f' (column names of series).\n'
                     f'    `series` columns : {self.series_col_names}.\n'
                     f'    `exog`   columns : {self.exog_col_names}.')
                )

        X_train, y_train, y_index, y_train_index = self.create_train_X_y(series=series, exog=exog)
        sample_weight = self.create_sample_weights(
                            series        = series,
                            X_train       = X_train,
                            y_train_index = y_train_index,
                        )

        if sample_weight is not None:
            if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
                self.regressor.fit(X=X_train, y=y_train, sample_weight=sample_weight)
            else:
                self.regressor.fit(X=X_train.to_numpy(), y=y_train.to_numpy(), sample_weight=sample_weight)
        else:
            if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
                self.regressor.fit(X=X_train, y=y_train)
            else:
                self.regressor.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
            
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = y_index[[0, -1]]
        self.index_type = type(y_index)
        if isinstance(y_index, pd.DatetimeIndex):
            self.index_freq = y_index.freqstr
        else: 
            self.index_freq = y_index.step
        self.index_values = y_index

        in_sample_residuals = {}
        
        # This is done to save time during fit in functions such as backtesting()
        if store_in_sample_residuals:

            if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
                residuals = y_train - self.regressor.predict(X_train)
            else:
                residuals = y_train - self.regressor.predict(X_train.to_numpy())

            for serie in series.columns:
                in_sample_residuals[serie] = residuals.values[X_train[serie] == 1.]
                if len(in_sample_residuals[serie]) > 1000:
                    # Only up to 1000 residuals are stored
                    rng = np.random.default_rng(seed=123)
                    in_sample_residuals[serie] = rng.choice(
                                                a       = in_sample_residuals[serie], 
                                                size    = 1000, 
                                                replace = False
                                            )
        else:
            for serie in series.columns:
                in_sample_residuals[serie] = np.array([None])

        self.in_sample_residuals = in_sample_residuals

        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        self.last_window = series.iloc[-self.max_lag:, ].copy()


    def _recursive_predict(
        self,
        steps: int,
        level: str,
        last_window: np.ndarray,
        exog: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Predict n steps ahead. It is an iterative process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
        level : str
            Time series to be predicted.
        
        last_window : numpy ndarray
            Values of the series used to create the predictors (lags) need in the 
            first iteration of prediction (t + 1).
            
        exog : numpy ndarray, default `None`
            Exogenous variable/s included as predictor/s.

        Returns 
        -------
        predictions : numpy ndarray
            Predicted values.
        
        """
        
        predictions = np.full(shape=steps, fill_value=np.nan)

        for i in range(steps):
            X = last_window[-self.lags].reshape(1, -1)
            if exog is not None:
                X = np.column_stack((X, exog[i, ].reshape(1, -1)))
            
            levels_dummies = np.zeros(shape=(1, len(self.series_col_names)), dtype=float)
            levels_dummies[0][self.series_col_names.index(level)] = 1.

            X = np.column_stack((X, levels_dummies.reshape(1, -1)))

            with warnings.catch_warnings():
                # Suppress scikit-learn warning: "X does not have valid feature names,
                # but NoOpTransformer was fitted with feature names".
                warnings.simplefilter("ignore")
                prediction = self.regressor.predict(X)
                predictions[i] = prediction.ravel()[0]

            # Update `last_window` values. The first position is discarded and 
            # the new prediction is added at the end.
            last_window = np.append(last_window[1:], prediction)

        return predictions

            
    def predict(
        self,
        steps: int,
        levels: Optional[Union[str, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> pd.DataFrame:
        """
        Predict n steps ahead. It is an recursive process in which, each prediction,
        is used as a predictor for the next step.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.

        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels will be predicted.
            **New in version 0.6.0**

        last_window : pandas DataFrame, default `None`
            Values of the series used to create the predictors (lags) need in the
            first iteration of prediction (t + 1).

            If `last_window = None`, the values stored in `self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.

        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        predictions : pandas DataFrame
            Predicted values, one column for each level.

        """
        
        if levels is None:
            levels = self.series_col_names
        elif isinstance(levels, str):
            levels = [levels]
        
        check_predict_input(
            forecaster_type  = type(self).__name__,
            steps            = steps,
            fitted           = self.fitted,
            included_exog    = self.included_exog,
            index_type       = self.index_type,
            index_freq       = self.index_freq,
            window_size      = self.window_size,
            last_window      = last_window,
            exog             = exog,
            exog_type        = self.exog_type,
            exog_col_names   = self.exog_col_names,
            interval         = None,
            max_steps        = None,
            levels           = levels,
            series_col_names = self.series_col_names
        )
        
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                exog = transform_dataframe(
                           df                = exog,
                           transformer       = self.transformer_exog,
                           fit               = False,
                           inverse_transform = False
                       )
            else:
                exog = transform_series(
                           series            = exog,
                           transformer       = self.transformer_exog,
                           fit               = False,
                           inverse_transform = False
                       )
            
            exog_values, _ = preprocess_exog(
                                 exog = exog.iloc[:steps, ]
                             )
        else:
            exog_values = None

        predictions = []

        for level in levels:

            if last_window is None:
                last_window = self.last_window.copy()

            last_window_level = transform_series(
                                    series            = last_window[level],
                                    transformer       = self.transformer_series_[level],
                                    fit               = False,
                                    inverse_transform = False
                                )
            last_window_values, last_window_index = preprocess_last_window(
                                                        last_window = last_window_level
                                                    )
                
            preds_level = self._recursive_predict(
                              steps       = steps,
                              level       = level,
                              last_window = copy(last_window_values),
                              exog        = copy(exog_values)
                          )

            preds_level = pd.Series(
                              data  = preds_level,
                              index = expand_index(
                                          index = last_window_index,
                                          steps = steps
                                      ),
                              name = level
                          )

            preds_level = transform_series(
                              series            = preds_level,
                              transformer       = self.transformer_series_[level],
                              fit               = False,
                              inverse_transform = True
                          )

            predictions.append(preds_level)    

        predictions = pd.concat(predictions, axis=1)

        return predictions
    
    
    def _estimate_boot_interval(
        self,
        steps: int,
        level: str,
        last_window: Optional[np.ndarray]=None,
        exog: Optional[np.ndarray]=None,
        interval: list=[5, 95],
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
    ) -> np.ndarray:
        """
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. This method only returns prediction intervals.
        See predict_intervals() to calculate both, predictions and intervals.
        
        Parameters
        ----------   
        steps : int
            Number of future steps predicted.
            
        level : str
            Time series to be predicted.
            
        last_window : 1d numpy ndarray shape (, max_lag), default `None`
            Values of the series used to create the predictors (lags) needed in the 
            first iteration of prediction (t + 1).
    
            If `last_window = `None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : numpy ndarray, default `None`
            Exogenous variable/s included as predictor/s.
            
        n_boot : int, default `500`
            Number of bootstrapping iterations used to estimate prediction
            intervals.

        random_state : int
            Sets a seed to the random generator, so that boot intervals are always 
            deterministic.
            
        interval : list, default `[5, 95]`
            Confidence of the prediction interval estimated. Sequence of 
            percentiles to compute, which must be between 0 and 100 inclusive. 
            For example, interval of 95% should be as `interval = [2.5, 97.5]`.
            
        in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create prediction intervals. If `False`, out of
            sample residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
            
        Returns 
        -------
        prediction_interval : numpy ndarray, shape (steps, 2)
            Interval estimated for each prediction by bootstrapping:
                lower_bound: lower bound of the interval.
                upper_bound: upper bound interval of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
        
        """
        
        if last_window is None:
            last_window = self.last_window[level]
            last_window = last_window.values

        boot_predictions = np.full(
                                shape      = (steps, n_boot),
                                fill_value = np.nan,
                                dtype      = float
                           )
        rng = np.random.default_rng(seed=random_state)
        seeds = rng.integers(low=0, high=10000, size=n_boot)

        for i in range(n_boot):
            # In each bootstraping iteration the initial last_window and exog 
            # need to be restored.
            last_window_boot = last_window.copy()
            if exog is not None:
                exog_boot = exog.copy()
            else:
                exog_boot = None

            if in_sample_residuals:
                residuals = self.in_sample_residuals[level]
            else:
                residuals = self.out_sample_residuals[level]

            rng = np.random.default_rng(seed=seeds[i])
            sample_residuals = rng.choice(
                                    a       = residuals,
                                    size    = steps,
                                    replace = True
                               )

            for step in range(steps):
                prediction = self._recursive_predict(
                                steps       = 1,
                                level       = level,
                                last_window = last_window_boot,
                                exog        = exog_boot 
                             )
                
                prediction_with_residual  = prediction + sample_residuals[step]
                boot_predictions[step, i] = prediction_with_residual

                last_window_boot = np.append(
                                       last_window_boot[1:],
                                       prediction_with_residual
                                   )
                
                if exog is not None:
                    exog_boot = exog_boot[1:]
                            
        prediction_interval = np.percentile(boot_predictions, q=interval, axis=1)
        prediction_interval = prediction_interval.transpose()
        
        return prediction_interval
    
        
    def predict_interval(
        self,
        steps: int,
        levels: Optional[Union[str, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        interval: list=[5, 95],
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
    ) -> pd.DataFrame:
        """
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. Both, predictions and intervals, are returned.
        
        Parameters
        ---------- 
        steps : int
            Number of future steps predicted.

        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels will be predicted.  
            **New in version 0.6.0**  
            
        last_window : pandas DataFrame, default `None`
            Values of the series used to create the predictors (lags) needed in the 
            first iteration of prediction (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
            
        interval : list, default `[5, 95]`
            Confidence of the prediction interval estimated. Sequence of 
            percentiles to compute, which must be between 0 and 100 inclusive. 
            For example, interval of 95% should be as `interval = [2.5, 97.5]`.
            
        n_boot : int, default `500`
            Number of bootstrapping iterations used to estimate prediction
            intervals.

        random_state : int, default 123
            Sets a seed to the random generator, so that boot intervals are always 
            deterministic.
            
        in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create prediction intervals. If `False`, out of
            sample residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).

        Returns 
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.
                level: predictions.
                level_lower_bound: lower bound of the interval.
                level_upper_bound: upper bound interval of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.

        """
        
        if levels is None:
            levels = self.series_col_names
        elif isinstance(levels, str):
            levels = [levels]

        for level in levels:
            if in_sample_residuals and (self.in_sample_residuals[level] == None).any():
                raise ValueError(
                    (f"`forecaster.in_sample_residuals['{level}']` contains `None` values. "
                      "Try using `fit` method with `in_sample_residuals=True` or set in "
                      "`predict_interval` method `in_sample_residuals=False` and use "
                      "`out_sample_residuals` (see `set_out_sample_residuals()`).")
                )
        
        if not in_sample_residuals and self.out_sample_residuals is None:
            raise ValueError(
                ('`forecaster.out_sample_residuals` is `None`. Use '
                 '`in_sample_residuals=True` or method `set_out_sample_residuals()` '
                 'before `predict_interval()`.')
            )

        if not in_sample_residuals and len(set(levels) - set(self.out_sample_residuals.keys())) != 0:
            raise ValueError(
                ('Not `forecaster.out_sample_residuals` for levels: {set(levels) - set(self.out_sample_residuals.keys())}. '
                 'Use method `set_out_sample_residuals()`.')
            )
        
        check_predict_input(
            forecaster_type    = type(self).__name__,
            steps              = steps,
            fitted             = self.fitted,
            included_exog      = self.included_exog,
            index_type         = self.index_type,
            index_freq         = self.index_freq,
            window_size        = self.window_size,
            last_window        = last_window,
            exog               = exog,
            exog_type          = self.exog_type,
            exog_col_names     = self.exog_col_names,
            interval           = interval,
            max_steps          = None,
            levels             = levels,
            series_col_names   = self.series_col_names
        ) 
        
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                exog = transform_dataframe(
                            df                = exog,
                            transformer       = self.transformer_exog,
                            fit               = False,
                            inverse_transform = False
                       )
            else:
                exog = transform_series(
                            series            = exog,
                            transformer       = self.transformer_exog,
                            fit               = False,
                            inverse_transform = False
                       )
            
            exog_values, _ = preprocess_exog(
                                exog = exog.iloc[:steps, ]
                                )
        else:
            exog_values = None
        
        predictions = []

        for level in levels:

            if last_window is None:
                last_window = self.last_window.copy()
            
            last_window_level = transform_series(
                                    series            = last_window[level],
                                    transformer       = self.transformer_series_[level],
                                    fit               = False,
                                    inverse_transform = False
                                )
            last_window_values, last_window_index = preprocess_last_window(
                                                        last_window = last_window_level
                                                    )
            
            # Since during predict() `last_window_values` and `exog_values` are modified,
            # the originals are stored to be used later.
            last_window_values_original = last_window_values.copy()
            if exog is not None:
                exog_values_original = exog_values.copy()
            else:
                exog_values_original = None
                
            preds_level = self._recursive_predict(
                              steps       = steps,
                              level       = level,
                              last_window = last_window_values,
                              exog        = exog_values
                          )

            preds_level_interval = self._estimate_boot_interval(
                                       steps       = steps,
                                       level       = level,
                                       last_window = copy(last_window_values_original),
                                       exog        = copy(exog_values_original),
                                       interval    = interval,
                                       n_boot      = n_boot,
                                       random_state = random_state,
                                       in_sample_residuals = in_sample_residuals
                                   )
            
            preds_level = np.column_stack((preds_level, preds_level_interval))

            preds_level = pd.DataFrame(
                              data = preds_level,
                              index = expand_index(
                                          index = last_window_index,
                                          steps = steps
                                      ),
                              columns = [level, f'{level}_lower_bound', f'{level}_upper_bound']
                          )

            if self.transformer_series_[level]:
                for col in preds_level.columns:
                    preds_level[col] = self.transformer_series_[level].inverse_transform(preds_level[[col]])

            predictions.append(preds_level) 
        
        predictions = pd.concat(predictions, axis=1)

        return predictions

    
    def set_params(
        self, 
        **params: dict
    ) -> None:
        """
        Set new values to the parameters of the scikit learn model stored in the
        forecaster.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns 
        -------
        self
        
        """

        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)
        
        
    def set_lags(
        self, 
        lags: Union[int, list, np.ndarray, range]
    ) -> None:
        """      
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, 1D np.array, range
            Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                `int`: include lags from 1 to `lags`.
                `list` or `np.array`: include only lags present in `lags`.

        Returns 
        -------
        None
        
        """
        
        self.lags = initialize_lags(type(self), lags)            
        self.max_lag  = max(self.lags)
        self.window_size = max(self.lags)
        
        
    def set_out_sample_residuals(
        self, 
        residuals: pd.DataFrame,
        append: bool=True,
        transform: bool=True,
        random_state: int=123
    )-> None:
        """
        Set new values to the attribute `out_sample_residuals`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process.
        
        Parameters
        ----------
        residuals : pandas DataFrame
            Values of residuals. If len(residuals) > 1000, only a random sample
            of 1000 values are stored. Columns must be the same as `levels`.
            
        append : bool, default `True`
            If `True`, new residuals are added to the once already stored in the
            attribute `out_sample_residuals`. Once the limit of 1000 values is
            reached, no more values are appended. If False, `out_sample_residuals`
            is overwritten with the new residuals.

        transform : bool, default `True`
            If `True`, new residuals are transformed using self.transformer_series.

        random_state : int, default `123`
            Sets a seed to the random sampling for reproducible output.
        
        Returns 
        -------
        self

        """

        if not isinstance(residuals, pd.DataFrame):
            raise TypeError(
                f"`residuals` argument must be a pandas DataFrame. Got {type(residuals)}."
            )

        if not self.fitted:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `set_out_sample_residuals()`.")
            )
        
        out_sample_residuals = {}

        for level in residuals.columns:

            if not level in self.series_col_names:
                continue
            else:      

                residuals_level = residuals[level]

                if not transform and self.transformer_series_[level] is not None:
                    warnings.warn(
                        ('Argument `transform` is set to `False` but forecaster was trained '
                         f'using a transformer {self.transformer_series_[level]} for level {level}. '
                         'Ensure that the new residuals are already transformed or set `transform=True`.')
                    )

                if transform and self.transformer_series_ and self.transformer_series_[level]:
                    warnings.warn(
                        ('Residuals will be transformed using the same transformer used '
                         f'when training the forecaster for level {level} : ({self.transformer_series_[level]}). '
                         'Ensure that the new residuals are on the same scale as the '
                         'original time series. ')
                    )

                    residuals_level = transform_series(
                                          series            = residuals_level,
                                          transformer       = self.transformer_series_[level],
                                          fit               = False,
                                          inverse_transform = False
                                      )

                if len(residuals_level) > 1000:
                    rng = np.random.default_rng(seed=random_state)
                    residuals_level = rng.choice(a=residuals_level, size=1000, replace=False)
        
                if append and self.out_sample_residuals is not None:

                    if not level in self.out_sample_residuals.keys():
                        raise ValueError(
                            f'{level} does not exists in `forecaster.out_sample_residuals` keys: {list(self.out_sample_residuals.keys())}'
                        )

                    free_space = max(0, 1000 - len(self.out_sample_residuals[level]))
                    if len(residuals_level) < free_space:
                        residuals_level = np.hstack((
                                              self.out_sample_residuals[level],
                                              residuals_level
                                          ))
                    else:
                        residuals_level = np.hstack((
                                              self.out_sample_residuals[level],
                                              residuals_level[:free_space]
                                          ))

                out_sample_residuals[level] = np.array(residuals_level)

        self.out_sample_residuals = out_sample_residuals

    
    def get_feature_importance(
        self
    ) -> pd.DataFrame:
        """      
        Return feature importance of the regressor stored in the
        forecaster. Only valid when regressor stores internally the feature
        importance in the attribute `feature_importances_` or `coef_`.

        Parameters
        ----------
        self

        Returns
        -------
        feature_importance : pandas DataFrame
            Feature importance associated with each predictor.
        
        """

        if self.fitted == False:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importance()`.")
            )

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            estimator = self.regressor[-1]
        else:
            estimator = self.regressor

        try:
            feature_importance = pd.DataFrame({
                                    'feature': self.X_train_col_names,
                                    'importance' : estimator.feature_importances_
                                })
        except:   
            try:
                feature_importance = pd.DataFrame({
                                        'feature': self.X_train_col_names,
                                        'importance' : estimator.coef_
                                    })
            except:
                warnings.warn(
                    f"Impossible to access feature importance for regressor of type {type(estimator)}. "
                    f"This method is only valid when the regressor stores internally "
                    f"the feature importance in the attribute `feature_importances_` "
                    f"or `coef_`."
                )

                feature_importance = None

        return feature_importance