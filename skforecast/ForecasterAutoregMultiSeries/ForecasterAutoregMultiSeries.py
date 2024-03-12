################################################################################
#                        ForecasterAutoregMultiSeries                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable
import warnings
import logging
import sys
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from copy import copy
import inspect

import skforecast
from ..ForecasterBase import ForecasterBase
from ..exceptions import MissingValuesWarning
from ..exceptions import IgnoredArgumentWarning
from ..utils import initialize_lags
from ..utils import initialize_weights
from ..utils import initialize_transformer_series
from ..utils import check_select_fit_kwargs
from ..utils import check_preprocess_series
from ..utils import check_preprocess_exog_multiseries
from ..utils import align_series_and_exog_multiseries
from ..utils import get_exog_dtypes
from ..utils import check_exog_dtypes
from ..utils import check_interval
from ..utils import check_predict_input
from ..utils import preprocess_last_window
from ..utils import expand_index
from ..utils import transform_series
from ..utils import transform_dataframe
from ..preprocessing import TimeSeriesDifferentiator

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
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
    encoding : str, default `'ordinal_category'`
        Encoding used to identify the different series. 
        
        - If `'ordinal'`, a single column is created with integer values from 0 
        to n_series - 1. 
        - If `'ordinal_category'`, a single column is created with integer 
        values from 0 to n_series - 1 and the column is transformed into 
        pandas.category dtype so that it can be used as a categorical variable. 
        - If `'onehot'`, a binary column is created for each series.
        **New in version 0.12.0**
    transformer_series : transformer (preprocessor), dict, default `sklearn.preprocessing.StandardScaler`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. Transformation is applied to each `series` before training 
        the forecaster. ColumnTransformers are not allowed since they do not have 
        inverse_transform method.

        - If single transformer: it is cloned and applied to all series. 
        - If `dict` of transformers: a different transformer can be used for each series.
    transformer_exog : transformer, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, dict, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates. 
        Ignored if `regressor` does not have the argument `sample_weight` in its 
        `fit` method. See Notes section for more details on the use of the weights.

        - If single function: it is applied to all series. 
        - If `dict` {'series_column_name' : Callable}: a different function can be
        used for each series, a weight of 1 is given to all series not present in
        `weight_func`.
    series_weights : dict, default `None`
        Weights associated with each series {'series_column_name' : float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method. 
        See Notes section for more details on the use of the weights.

        - If a `dict` is provided, a weight of 1 is given to all series not present
        in `series_weights`.
        - If `None`, all levels have the same weight.
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Differentiation is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
        **New in version 0.12.0**
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.

    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    lags : numpy ndarray
        Lags used as predictors.
    encoding : str
        Encoding used to identify the different series. 
        
        - If `'ordinal'`, a single column is created with integer values from 0 
        to n_series - 1. 
        - If `'ordinal_category'`, a single column is created with integer 
        values from 0 to n_series - 1 and the column is transformed into 
        pandas.category dtype so that it can be used as a categorical variable. 
        - If `'onehot'`, a binary column is created for each series.
        **New in version 0.12.0**
    transformer_series : transformer (preprocessor), dict
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. Transformation is applied to each `series` before training 
        the forecaster. ColumnTransformers are not allowed since they do not have 
        inverse_transform method.

        - If single transformer: it is cloned and applied to all series. 
        - If `dict` of transformers: a different transformer can be used for each series.
    transformer_series_ : dict
        Dictionary with the transformer for each series. It is created cloning the 
        objects in `transformer_series` and is used internally to avoid overwriting.
    transformer_exog : transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, dict
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates. 
        Ignored if `regressor` does not have the argument `sample_weight` in its 
        `fit` method. See Notes section for more details on the use of the weights.

        - If single function: it is applied to all series. 
        - If `dict` {'series_column_name' : Callable}: a different function can be
        used for each series, a weight of 1 is given to all series not present 
        in `weight_func`.
    weight_func_ : dict
        Dictionary with the `weight_func` for each series. It is created cloning the 
        objects in `weight_func` and is used internally to avoid overwriting.
    source_code_weight_func : str, dict
        Source code of the custom function(s) used to create weights.
    series_weights : dict, default `None`
        Weights associated with each series {'series_column_name' : float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method. 
        See Notes section for more details on the use of the weights.

        - If a `dict` is provided, a weight of 1 is given to all series not present
        in `series_weights`.
        - If `None`, all levels have the same weight.
    differentiation : int
        Order of differencing applied to the time series before training the 
        forecaster.
    differentiator : TimeSeriesDifferentiator
        Skforecast object used to differentiate the time series.
    differentiator_ : dict
        Dictionary with the `differentiator` for each series. It is created cloning the
        objects in `differentiator` and is used internally to avoid overwriting.
    series_weights_ : dict
        Weights associated with each series.It is created as a clone of `series_weights`
        and is used internally to avoid overwriting.
    encoder : sklearn.preprocessing
        Scikit-learn preprocessing encoder used to encode the series.
        **New in version 0.12.0**
    encoding_mapping : dict
        Mapping of the encoding used to identify the different series.
    max_lag : int
        Maximum value of lag included in `lags`.
    window_size : int
        Size of the window needed to create the predictors. It is equal to `max_lag`.
    last_window : dict
        Last window of training data for each series. It stores the values 
        needed to predict the next `step` immediately after the training data.
    index_type : type
        Type of index of the input used in training.
    index_freq : str
        Frequency of Index of the input used in training.
    training_range: dict
        First and last values of index of the data used during training for each 
        series.
    series_col_names : list
        Names of the series (levels) used during training.
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_type : type
        Type of exogenous variable/s used in training.
    exog_dtypes : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated before the transformation.
    exog_col_names : list
        Names of the exogenous variables used during training.
    X_train_col_names : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals : dict
        Residuals of the model when predicting training data. Only stored up to
        1000 values in the form `{level: residuals}`. If `transformer_series` 
        is not `None`, residuals are stored in the transformed scale.
    out_sample_residuals : dict
        Residuals of the models when predicting non training data. Only stored
        up to 1000 values in the form `{level: residuals}`. If `transformer_series` 
        is not `None`, residuals are assumed to be in the transformed scale. Use 
        `set_out_sample_residuals()` method to set values.
    fitted : bool
        Tag to identify if the regressor has been fitted (trained).
    creation_date : str
        Date of creation.
    fit_date : str
        Date of last fit.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.

    Notes
    -----
    The weights are used to control the influence that each observation has on the
    training of the model. `ForecasterAutoregMultiseries` accepts two types of weights. 
    If the two types of weights are indicated, they are multiplied to create the final
    weights. The resulting `sample_weight` cannot have negative values.

    - `series_weights` : controls the relative importance of each series. If a 
    series has twice as much weight as the others, the observations of that series 
    influence the training twice as much. The higher the weight of a series 
    relative to the others, the more the model will focus on trying to learn 
    that series.
    - `weight_func` : controls the relative importance of each observation 
    according to its index value. For example, a function that assigns a lower 
    weight to certain dates.
    
    """
    
    def __init__(
        self,
        regressor: object,
        lags: Union[int, np.ndarray, list],
        encoding : str='onehot',
        transformer_series: Optional[Union[object, dict]]=StandardScaler(),
        transformer_exog: Optional[object]=None,
        weight_func: Optional[Union[Callable, dict]]=None,
        series_weights: Optional[dict]=None,
        differentiation: Optional[int]=None,
        fit_kwargs: Optional[dict]=None,
        forecaster_id: Optional[Union[str, int]]=None
    ) -> None:
        
        self.regressor               = regressor
        self.transformer_series      = transformer_series
        self.transformer_series_     = None
        self.transformer_exog        = transformer_exog
        self.encoding                = encoding
        self.encoder                 = None
        self.encoding_mapping        = {}
        self.weight_func             = weight_func
        self.weight_func_            = None
        self.source_code_weight_func = None
        self.series_weights          = series_weights
        self.series_weights_         = None
        self.differentiation         = differentiation
        self.differentiator          = None
        self.differentiator_         = None
        self.last_window             = None
        self.index_type              = None
        self.index_freq              = None
        self.training_range          = None
        self.series_col_names        = None
        self.included_exog           = False
        self.exog_type               = None
        self.exog_dtypes             = None
        self.exog_col_names          = None
        self.X_train_col_names       = None
        self.in_sample_residuals     = None
        self.out_sample_residuals    = None
        self.fitted                  = False
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                = None
        self.skforecast_version      = skforecast.__version__
        self.python_version          = sys.version.split(" ")[0]
        self.forecaster_id           = forecaster_id
        
        self.lags = initialize_lags(type(self).__name__, lags)
        self.max_lag = max(self.lags)
        self.window_size = self.max_lag

        self.weight_func, self.source_code_weight_func, self.series_weights = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = series_weights
        )

        if self.differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    (f"Argument `differentiation` must be an integer equal to or "
                     f"greater than 1. Got {differentiation}.")
                )
            self.window_size += self.differentiation
            self.differentiator = TimeSeriesDifferentiator(order=self.differentiation)

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )

        if self.encoding not in ['ordinal', 'ordinal_category', 'onehot']:
            raise ValueError(
                (f"Argument `encoding` must be one of the following values: 'ordinal', "
                 f"'ordinal_category', 'onehot'. Got {self.encoding}.")
            )

        if self.encoding == 'onehot':
            self.encoder = OneHotEncoder(
                               categories    = 'auto',
                               sparse_output = False,
                               drop          = None,
                               dtype         = int
                           ).set_output(transform='pandas')
        else:
            self.encoder = OrdinalEncoder(
                               categories = 'auto',
                               dtype      = int
                           ).set_output(transform='pandas')


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
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
            f"Transformer for series: {self.transformer_series} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Series encoding: {self.encoding} \n"
            f"Window size: {self.window_size} \n"
            f"Series levels (names): {self.series_col_names} \n"
            f"Series weights: {self.series_weights} \n"
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Differentiation order: {self.differentiation} \n"
            f"Exogenous included: {self.included_exog} \n"
            f"Type of exogenous variable: {self.exog_type} \n"
            f"Exogenous variables names: {self.exog_col_names} \n"
            f"Training range: {self.training_range.to_list() if self.fitted else None} \n"
            f"Training index type: {str(self.index_type).split('.')[-1][:-2] if self.fitted else None} \n"
            f"Training index frequency: {self.index_freq if self.fitted else None} \n"
            f"Regressor parameters: {params} \n"
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    
    def _create_lags(
        self, 
        y: np.ndarray, 
        series_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a 1d array into a 2d array (X) and a 1d array (y). Each row
        in X is associated with a value of y and it represents the lags that
        precede it.
        
        Notice that, the returned matrix X_data, contains the lag 1 in the first
        column, the lag 2 in the second column and so on.
        
        Parameters
        ----------
        y : numpy ndarray
            1d numpy ndarray Training time series.
        series_name : str
            Name of the series.

        Returns
        -------
        X_data : numpy ndarray
            2d numpy ndarray with the lagged values (predictors). 
            Shape: (samples - max(self.lags), len(self.lags))
        y_data : numpy ndarray
            1d numpy ndarray with the values of the time series related to each 
            row of `X_data`. 
            Shape: (samples - max(self.lags), )
        
        """
          
        n_splits = len(y) - self.max_lag
        if n_splits <= 0:
            raise ValueError(
                (f"The maximum lag ({self.max_lag}) must be less than the length "
                 f"of the series '{series_name}', ({len(y)}).")
            )
        
        X_data = np.full(shape=(n_splits, len(self.lags)), fill_value=np.nan, dtype=float)

        for i, lag in enumerate(self.lags):
            X_data[:, i] = y[self.max_lag - lag: -lag]

        y_data = y[self.max_lag:]
            
        return X_data, y_data


    def _create_train_X_y_single_series(
        self,
        y: pd.Series,
        ignore_exog: bool,
        exog: Optional[pd.DataFrame]=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series and exogenous
        variables. This method does not transform the exog variables.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        ignore_exog : bool
            If `True`, `exog` is ignored.
        exog : pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        X_train_lags : pandas DataFrame
            Training values of lags
            Shape: (len(y) - self.max_lag, len(self.lags))
        X_train_exog : pandas DataFrame
            Training values of exogenous variables.
            Shape: (len(y) - self.max_lag, len(exog.columns))
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.
            Shape: (len(y) - self.max_lag, )
        
        """

        series_name = y.name
        fit_transformer = False if self.fitted else True
        y = transform_series(
                series            = y,
                transformer       = self.transformer_series_[series_name],
                fit               = fit_transformer,
                inverse_transform = False
            )
        
        y_values = y.to_numpy()
        y_index = y.index

        if self.differentiation is not None:
            if not self.fitted:
                y_values = self.differentiator_[series_name].fit_transform(y_values)
            else:
                differentiator = clone(self.differentiator_[series_name])
                y_values = differentiator.fit_transform(y_values)
        
        X_train, y_train = self._create_lags(y=y_values, series_name=series_name)

        X_train_lags = pd.DataFrame(
                           data    = X_train,
                           columns = [f"lag_{i}" for i in self.lags],
                           index   = y_index[self.max_lag: ]
                       )
        X_train_lags['_level_skforecast'] = series_name

        if ignore_exog:
            X_train_exog = None
        else:
            if exog is not None:
                # The first `self.max_lag` positions have to be removed from exog
                # since they are not in X_train.
                X_train_exog = exog.iloc[self.max_lag:, ]
            else:
                X_train_exog = pd.DataFrame(
                                   data    = np.nan,
                                   columns = ['_dummy_exog_col_to_keep_shape'],
                                   index   = y_index[self.max_lag: ]
                               )

        y_train = pd.Series(
                      data  = y_train,
                      index = y_index[self.max_lag: ],
                      name  = 'y'
                  )

        if self.differentiation is not None:
            X_train_lags = X_train_lags.iloc[self.differentiation: ]
            y_train = y_train.iloc[self.differentiation: ]
            if X_train_exog is not None:
                X_train_exog = X_train_exog.iloc[self.differentiation: ]
        
        return X_train_lags, X_train_exog, y_train


    def _create_train_X_y(
        self,
        series: Union[pd.DataFrame, dict],
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]]=None,
        drop_nan: bool=False,
        store_last_window: Union[bool, list]=True,
    ) -> Tuple[pd.DataFrame, pd.Series, dict, list, list, dict, dict]:
        """
        Create training matrices from multiple time series and exogenous
        variables. See Notes section for more details depending on the type of
        `series` and `exog`.
        
        Parameters
        ----------
        series : pandas DataFrame, dict
            Training time series.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        drop_nan : bool, default `False`
            NaNs detected in `y_train` will be dropped since the target variable 
            cannot have NaN values. Same rows are dropped from `X_train` to 
            maintain alignment. Regarding `X_train`:

            - If `True`, drop NaNs in X_train and same rows in y_train.
            - If `False`, leave NaNs in X_train and warn the user.
        store_last_window : bool, list, default `True`
            Whether or not to store the last window of training data.

            - If `True`, last_window is stored for all series. 
            - If `list`, last_window is stored for the series present in the list.
            - If `False`, last_window is not stored.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors).
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.
        series_indexes : dict
            Dictionary with the index of each series.
        series_col_names : list
            Names of the series (levels) used during training.
        exog_col_names : list
            Names of the exogenous variables used during training.
        exog_dtypes : dict
            Type of each exogenous variable/s used in training. If `transformer_exog` 
            is used, the dtypes are calculated before the transformation.
        last_window : dict
            Last window of training data for each series. It stores the values 
            needed to predict the next `step` immediately after the training data.

        Notes
        -----
        - If `series` is a pandas DataFrame and `exog` is a pandas Series or 
        DataFrame, each exog is duplicated for each series. Exog must have the
        same index as `series` (type, length and frequency).
        - If `series` is a pandas DataFrame and `exog` is a dict of pandas Series 
        or DataFrames. Each key in `exog` must be a column in `series` and the 
        values are the exog for each series. Exog must have the same index as 
        `series` (type, length and frequency).
        - If `series` is a dict of pandas Series, `exog`must be a dict of pandas
        Series or DataFrames. The keys in `series` and `exog` must be the same.
        All series and exog must have a pandas DatetimeIndex with the same 
        frequency.
        
        """

        series_dict, series_indexes = check_preprocess_series(series=series)
        input_series_is_dict = isinstance(series, dict)
        series_col_names = list(series_dict.keys())

        if self.fitted and not (series_col_names == self.series_col_names):
            raise ValueError(
                (f"Once the Forecaster has been trained, `series` must have the "
                 f"same columns as the series used during training:\n" 
                 f" Got      : {series_col_names}\n"
                 f" Expected : {self.series_col_names}")
            )
        
        exog_dict = {serie: None for serie in series_col_names}
        exog_col_names = None
        if exog is not None:
            exog_dict, exog_col_names = check_preprocess_exog_multiseries(
                                            input_series_is_dict = input_series_is_dict,
                                            series_indexes       = series_indexes,
                                            series_col_names     = series_col_names,
                                            exog                 = exog,
                                            exog_dict            = exog_dict
                                        )
            
        if self.fitted and not (exog_col_names == self.exog_col_names):
            if self.exog_col_names is None:
                raise ValueError(
                    ("Once the Forecaster has been trained, `exog` must be `None` "
                     "because no exogenous variables were added during training.")
                )
            else:
                raise ValueError(
                    (f"Once the Forecaster has been trained, `exog` must have the "
                     f"same columns as the series used during training:\n" 
                     f" Got      : {exog_col_names}\n"
                     f" Expected : {self.exog_col_names}")
                )

        if not self.fitted:
            self.transformer_series_ = initialize_transformer_series(
                                           series_col_names = series_col_names,
                                           transformer_series = self.transformer_series
                                       )
        
        if self.differentiation is None:
            self.differentiator_ = {serie: None for serie in series_col_names}
        else:
            if not self.fitted:
                self.differentiator_ = {serie: clone(self.differentiator) 
                                        for serie in series_col_names}

        series_dict, exog_dict = align_series_and_exog_multiseries(
                                     series_dict          = series_dict,
                                     input_series_is_dict = input_series_is_dict,
                                     exog_dict            = exog_dict
                                 )

        # TODO: parallelize
        # ======================================================================
        ignore_exog = True if exog is None else False
        input_matrices = [
            [series_dict[k], exog_dict[k], ignore_exog]
             for k in series_dict.keys()
        ]

        X_train_lags_buffer = []
        X_train_exog_buffer = []
        y_train_buffer = []
        for matrices in input_matrices:

            X_train_lags, X_train_exog, y_train = (
                self._create_train_X_y_single_series(
                    y           = matrices[0],
                    exog        = matrices[1],
                    ignore_exog = matrices[2],
                )
            )
            
            X_train_lags_buffer.append(X_train_lags)
            X_train_exog_buffer.append(X_train_exog)
            y_train_buffer.append(y_train)
        # ======================================================================

        X_train = pd.concat(X_train_lags_buffer, axis=0)
        y_train = pd.concat(y_train_buffer, axis=0)

        if self.fitted:
            encoded_values = self.encoder.transform(X_train[['_level_skforecast']])
        else:
            encoded_values = self.encoder.fit_transform(X_train[['_level_skforecast']])
            for i, code in enumerate(self.encoder.categories_[0]):
                self.encoding_mapping[code] = i

        X_train = pd.concat([
                      X_train.drop(columns='_level_skforecast'),
                      encoded_values
                  ], axis=1)

        if self.encoding == 'onehot':
            X_train.columns = X_train.columns.str.replace('_level_skforecast_', '')
        elif self.encoding == 'ordinal_category':
            X_train['_level_skforecast'] = X_train['_level_skforecast'].astype('category')
        
        del encoded_values

        exog_dtypes = None
        if exog is not None:

            X_train_exog = pd.concat(X_train_exog_buffer, axis=0)
            if '_dummy_exog_col_to_keep_shape' in X_train_exog.columns:
                X_train_exog = X_train_exog.drop(columns=['_dummy_exog_col_to_keep_shape'])
            
            exog_dtypes = get_exog_dtypes(exog=X_train_exog)

            fit_transformer = False if self.fitted else True
            X_train_exog = transform_dataframe(
                               df                = X_train_exog,
                               transformer       = self.transformer_exog,
                               fit               = fit_transformer,
                               inverse_transform = False
                           )
            
            check_exog_dtypes(X_train_exog, call_check_exog=False)
            if not (X_train_exog.index == X_train.index).all():
                raise ValueError(
                    ("Different index for `series` and `exog` after transformation. "
                     "They must be equal to ensure the correct alignment of values.")
                )
            
            X_train = pd.concat([X_train, X_train_exog], axis=1)

        if y_train.isnull().any():
            mask = y_train.notna().to_numpy()
            y_train = y_train.iloc[mask]
            X_train = X_train.iloc[mask,]
            warnings.warn(
                ("NaNs detected in `y_train`. They have been dropped since the "
                 "target variable cannot have NaN values. Same rows have been "
                 "dropped from `X_train` to maintain alignment."),
                 MissingValuesWarning
            )

        if drop_nan:
            # TODO: review when we have a full exog as NaN
            if X_train.isnull().any().any():
                mask = X_train.notna().all(axis=1).to_numpy()
                X_train = X_train.iloc[mask, ]
                y_train = y_train.iloc[mask]
                warnings.warn(
                    ("NaNs detected in `X_train`. They have been dropped. If "
                     "you want to keep them, set `drop_nan = False`. Same rows"
                     "have been removed from `y_train` to maintain alignment."),
                     MissingValuesWarning
                )
        else:
            if X_train.isnull().any().any():
                warnings.warn(
                    ("NaNs detected in `X_train`. Some regressor do not allow "
                     "NaN values during training. If you want to drop them, "
                     "set `drop_nan = True`."),
                     MissingValuesWarning
                )

        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        if store_last_window:

            store_series = (
                series_col_names if store_last_window is True else store_last_window
            )
        
            series_not_in_series_dict = set(store_series) - set(series_col_names)
            if series_not_in_series_dict:
                warnings.warn(
                    (f"{series_not_in_series_dict} not present in `series`. No "
                     f"last window is stored for them."),
                    IgnoredArgumentWarning
                )
            
            last_window = {
                k: v.iloc[-self.max_lag:].copy()
                for k, v in series_dict.items()
                if k in store_series
            }

        return (
            X_train,
            y_train,
            series_indexes,
            series_col_names,
            exog_col_names,
            exog_dtypes,
            last_window,
        )


    def create_train_X_y(
        self,
        series: Union[pd.DataFrame, dict],
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]]=None,
        drop_nan: bool=False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrices from multiple time series and exogenous
        variables. See Notes section for more details depending on the type of
        `series` and `exog`.
        
        Parameters
        ----------
        series : pandas DataFrame, dict
            Training time series.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        drop_nan : bool, default `False`
            NaNs detected in `y_train` will be dropped since the target variable 
            cannot have NaN values. Same rows are dropped from `X_train` to 
            maintain alignment. Regarding `X_train`:

            - If `True`, drop NaNs in X_train and same rows in y_train.
            - If `False`, leave NaNs in X_train and warn the user.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors).
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.

        Notes
        -----
        - If `series` is a pandas DataFrame and `exog` is a pandas Series or 
        DataFrame, each exog is duplicated for each series. Exog must have the
        same index as `series` (type, length and frequency).
        - If `series` is a pandas DataFrame and `exog` is a dict of pandas Series 
        or DataFrames. Each key in `exog` must be a column in `series` and the 
        values are the exog for each series. Exog must have the same index as 
        `series` (type, length and frequency).
        - If `series` is a dict of pandas Series, `exog`must be a dict of pandas
        Series or DataFrames. The keys in `series` and `exog` must be the same.
        All series and exog must have a pandas DatetimeIndex with the same 
        frequency.
        
        """

        X_train, y_train = self._create_train_X_y(
                               series            = series, 
                               exog              = exog, 
                               drop_nan          = drop_nan, 
                               store_last_window = False
                           )[0, 1]

        return X_train, y_train


    def create_sample_weights(
        self,
        series_col_names: list,
        X_train: pd.DataFrame
    )-> np.ndarray:
        """
        Crate weights for each observation according to the forecaster's attributes
        `series_weights` and `weight_func`. The resulting weights are product of both
        types of weights.

        Parameters
        ----------
        series_col_names : list
            Names of the series (levels) used during training.
        X_train : pandas DataFrame
            Dataframe created with the `create_train_X_y` method, first return.

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
            series_not_in_series_weights = set(series_col_names) - set(self.series_weights.keys())
            if series_not_in_series_weights:
                warnings.warn(
                    (f"{series_not_in_series_weights} not present in `series_weights`. "
                     f"A weight of 1 is given to all their samples."),
                     IgnoredArgumentWarning
                )
            self.series_weights_ = {col: 1. for col in series_col_names}
            self.series_weights_.update(
                (k, v) 
                for k, v in self.series_weights.items() 
                if k in self.series_weights_
            )

            if self.encoding == "onehot":
                weights_series = [
                    np.repeat(self.series_weights_[serie], sum(X_train[serie]))
                    for serie in series_col_names
                ]
            else:
                weights_series = [
                    np.repeat(
                        self.series_weights_[serie],
                        sum(X_train["_level_skforecast"] == self.encoding_mapping[serie]),
                    )
                    for serie in series_col_names
                ]
            
            weights_series = np.concatenate(weights_series)

        if self.weight_func is not None:
            if isinstance(self.weight_func, Callable):
                self.weight_func_ = {col: copy(self.weight_func) 
                                     for col in series_col_names}
            else:
                # Series not present in weight_func have a weight of 1 in all their samples
                series_not_in_weight_func = set(series_col_names) - set(self.weight_func.keys())
                if series_not_in_weight_func:
                    warnings.warn(
                        (f"{series_not_in_weight_func} not present in `weight_func`. "
                         f"A weight of 1 is given to all their samples."),
                         IgnoredArgumentWarning
                    )
                self.weight_func_ = {col: lambda x: np.ones_like(x, dtype=float) 
                                     for col in series_col_names}
                self.weight_func_.update(
                    (k, v) 
                    for k, v in self.weight_func.items() 
                    if k in self.weight_func_
                )
               
            weights_samples = []
            for key in self.weight_func_.keys():
                if self.encoding == "onehot":
                    idx = X_train.index[X_train[key] == 1.0]
                else:
                    idx = X_train.index[X_train["_level_skforecast"] == self.encoding_mapping[key]]
                weights_samples.append(self.weight_func_[key](idx))
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
        series: Union[pd.DataFrame, dict],
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]]=None,
        drop_nan: bool=False,
        store_last_window: Union[bool, list]=True,
        store_in_sample_residuals: bool=True
    ) -> None:
        """
        Training Forecaster. See Notes section for more details depending on 
        the type of `series` and `exog`.

        Additional arguments to be passed to the `fit` method of the regressor 
        can be added with the `fit_kwargs` argument when initializing the forecaster.
        
        Parameters
        ----------
        series : pandas DataFrame, dict
            Training time series.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        drop_nan : bool, default `False`
            NaNs detected in `y_train` will be dropped since the target variable 
            cannot have NaN values. Same rows are dropped from `X_train` to 
            maintain alignment. Regarding `X_train`:

            - If `True`, drop NaNs in X_train and same rows in y_train.
            - If `False`, leave NaNs in X_train and warn the user.
        store_last_window : bool, list, default `True`
            Whether or not to store the last window of training data.

            - If `True`, last_window is stored for all series. 
            - If `list`, last_window is stored for the series present in the list.
            - If `False`, last_window is not stored.
        store_in_sample_residuals : bool, default `True`
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting.

        Returns
        -------
        None

        Notes
        -----
        - If `series` is a pandas DataFrame and `exog` is a pandas Series or 
        DataFrame, each exog is duplicated for each series. Exog must have the
        same index as `series` (type, length and frequency).
        - If `series` is a pandas DataFrame and `exog` is a dict of pandas Series 
        or DataFrames. Each key in `exog` must be a column in `series` and the 
        values are the exog for each series. Exog must have the same index as 
        `series` (type, length and frequency).
        - If `series` is a dict of pandas Series, `exog`must be a dict of pandas
        Series or DataFrames. The keys in `series` and `exog` must be the same.
        All series and exog must have a pandas DatetimeIndex with the same 
        frequency.
        
        """
        
        # Reset values in case the forecaster has already been fitted.
        self.series_col_names    = None
        self.index_type          = None
        self.index_freq          = None
        self.last_window         = None
        self.included_exog       = False
        self.exog_type           = None
        self.exog_dtypes         = None
        self.exog_col_names      = None
        self.series_col_names    = None
        self.X_train_col_names   = None
        self.in_sample_residuals = None
        self.fitted              = False
        self.training_range      = None

        (
            X_train,
            y_train,
            series_indexes,
            series_col_names,
            exog_col_names,
            exog_dtypes,
            last_window
        ) = self._create_train_X_y(
                series=series, exog=exog, drop_nan=drop_nan, store_last_window=store_last_window
        )

        sample_weight = self.create_sample_weights(
                            series_col_names = series_col_names,
                            X_train          = X_train
                        )

        if sample_weight is not None:
            self.regressor.fit(
                X             = X_train,
                y             = y_train,
                sample_weight = sample_weight,
                **self.fit_kwargs
            )
        else:
            self.regressor.fit(X=X_train, y=y_train, **self.fit_kwargs)

        self.series_col_names = series_col_names
        self.X_train_col_names = X_train.columns.to_list()
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')

        self.training_range = {k: v[[0, -1]] for k, v in series_indexes.items()}
        unique_index = series_indexes[series_col_names[0]]
        self.index_type = type(unique_index)
        if isinstance(unique_index, pd.DatetimeIndex):
            self.index_freq = unique_index.freqstr
        else: 
            self.index_freq = unique_index.step

        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = exog_col_names
            self.exog_dtypes = exog_dtypes

        in_sample_residuals = {}
        if store_in_sample_residuals:

            residuals = y_train - self.regressor.predict(X_train)

            for col in series_col_names:
                if self.encoding == 'onehot':
                    in_sample_residuals[col] = residuals.loc[X_train[col] == 1.].to_numpy()
                else:
                    encoded_value = self.encoding_mapping[col]
                    in_sample_residuals[col] = (
                        residuals.loc[X_train['_level_skforecast'] == encoded_value].to_numpy()
                    )
                if len(in_sample_residuals[col]) > 1000:
                    # Only up to 1000 residuals are stored
                    rng = np.random.default_rng(seed=123)
                    in_sample_residuals[col] = rng.choice(
                                                   a       = in_sample_residuals[col], 
                                                   size    = 1000, 
                                                   replace = False
                                               )
        else:
            for col in series_col_names:
                in_sample_residuals[col] = None

        self.in_sample_residuals = in_sample_residuals

        if store_last_window:
            self.last_window = last_window


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
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
        exog : numpy ndarray, default `None`
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        predictions : numpy ndarray
            Predicted values.
        
        """
        
        predictions = np.full(shape=steps, fill_value=np.nan)
        level_encoded = np.array([self.encoding_mapping[level]], dtype='float64')

        for i in range(steps):

            X = last_window[-self.lags].reshape(1, -1)        

            if self.encoding == 'onehot':
                levels_dummies = np.zeros(shape=(1, len(self.series_col_names)), dtype=float)
                levels_dummies[0][self.series_col_names.index(level)] = 1.
                X = np.column_stack((X, levels_dummies.reshape(1, -1)))
            else:
                X = np.column_stack((X, level_encoded))

            if exog is not None:
                X = np.column_stack((X, exog[i, ].reshape(1, -1)))
    
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
        is used as a predictor for the next step. Only levels whose last window
        ends at the same datetime index can be predicted together.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
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
        
        input_levels_is_list = False 
        if levels is None:
            levels = self.series_col_names
        elif isinstance(levels, str):
            levels = [levels]
        else:
            input_levels_is_list = True

        if last_window is None and self.fitted:
            not_available_last_window = set(levels) - set(self.last_window.keys())
            if not_available_last_window:
                warnings.warn(
                    (f"{not_available_last_window} are excluded from prediction "
                     f"since they were not stored in `last_window` attribute "
                     f"during training. If you don't want to retrain the "
                     f"Forecaster, provide `last_window` as argument."),
                    IgnoredArgumentWarning
                )
                levels = [level for level in levels 
                          if level not in not_available_last_window]

                if not levels:
                    raise ValueError(
                        ("No series to predict. None of the series are present in "
                         "`last_window` attribute. Provide `last_window` as argument "
                         "in predict method.")
                    )

            training_range_levels = [
                v[-1] for k, v in self.training_range.items()
                if k in levels
            ]
            if len(set(training_range_levels)) > 1:
                max_training_range = max(training_range_levels)
                selected_levels = [
                    k 
                    for k, v in self.last_window.items()
                    if k in levels and v.index[-1] == max_training_range
                ]

                series_excluded_from_last_window = set(levels) - set(selected_levels)
                levels = selected_levels
                
                if input_levels_is_list and series_excluded_from_last_window:
                    warnings.warn(
                        (f"Found series with different ends of training range. "
                         f"Only series whose last window ends at the same index "
                         f"can be predicted together. Series that not reach the "
                         f"maximum index, {max_training_range}, are excluded "
                         f"from prediction: {series_excluded_from_last_window}."),
                        IgnoredArgumentWarning
                    )

            last_window = pd.DataFrame(
                {k: v for 
                 k, v in self.last_window.items() 
                 if k in levels}
            )

        check_predict_input(
            forecaster_name  = type(self).__name__,
            steps            = steps,
            fitted           = self.fitted,
            included_exog    = self.included_exog,
            index_type       = self.index_type,
            index_freq       = self.index_freq,
            window_size      = self.window_size,
            last_window      = last_window,
            last_window_exog = None,
            exog             = exog,
            exog_type        = self.exog_type,
            exog_col_names   = self.exog_col_names,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = levels,
            series_col_names = self.series_col_names
        )

        last_window = last_window.iloc[-self.window_size:, ].copy()
        
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
            check_exog_dtypes(exog=exog)
            exog_values = exog.to_numpy()[:steps]
        else:
            exog_values = None

        predictions = []

        for level in levels:

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
                              last_window = last_window_values,
                              exog        = exog_values
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


    def predict_bootstrapping(
        self,
        steps: int,
        levels: Optional[Union[str, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
    ) -> dict:
        """
        Generate multiple forecasting predictions using a bootstrapping process. 
        By sampling from a collection of past observed errors (the residuals),
        each iteration of bootstrapping generates a different set of predictions. 
        Only levels whose last window ends at the same datetime index can be 
        predicted together. See the Notes section for more information. 
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. 
        n_boot : int, default `500`
            Number of bootstrapping iterations used to estimate predictions.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot predictions are always 
            deterministic.      
        in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).

        Returns
        -------
        boot_predictions : dict
            Predictions generated by bootstrapping for each level.
            {level: pandas DataFrame, shape (steps, n_boot)}

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
        Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

        """
        
        if self.fitted:

            input_levels_is_list = False 
            if levels is None:
                levels = self.series_col_names
            elif isinstance(levels, str):
                levels = [levels]
            else:
                input_levels_is_list = True
            
            if last_window is None:
                not_available_last_window = set(levels) - set(self.last_window.keys())
                if not_available_last_window:
                    warnings.warn(
                        (f"{not_available_last_window} are excluded from prediction "
                         f"since they were not stored in `last_window` attribute "
                         f"during training. If you don't want to retrain the "
                         f"Forecaster, provide `last_window` as argument."),
                         IgnoredArgumentWarning
                    )
                    levels = [level for level in levels 
                              if level not in not_available_last_window]

                    if not levels:
                        raise ValueError(
                            ("No series to predict. None of the series are present in "
                             "`last_window` attribute. Provide `last_window` as argument "
                             "in predict method.")
                        )

                training_range_levels = [
                    v[-1] for k, v in self.training_range.items()
                    if k in levels
                ]
                if len(set(training_range_levels)) > 1:
                    max_training_range = max(training_range_levels)
                    selected_levels = [
                        k 
                        for k, v in self.last_window.items()
                        if k in levels and v.index[-1] == max_training_range
                    ]

                    series_excluded_from_last_window = set(levels) - set(selected_levels)
                    levels = selected_levels
                    
                    if input_levels_is_list and series_excluded_from_last_window:
                        warnings.warn(
                            (f"Found series with different ends of training range. "
                             f"Only series whose last window ends at the same index "
                             f"can be predicted together. Series that not reach the "
                             f"maximum index, {max_training_range}, are excluded "
                             f"from prediction: {series_excluded_from_last_window}."),
                             IgnoredArgumentWarning
                        )

                last_window = pd.DataFrame(
                    {k: v for 
                     k, v in self.last_window.items() 
                     if k in levels}
                )

            if in_sample_residuals:
                if not set(levels).issubset(set(self.in_sample_residuals.keys())):
                    raise ValueError(
                        (f"Not `forecaster.in_sample_residuals` for levels: "
                         f"{set(levels) - set(self.in_sample_residuals.keys())}.")
                    )
                residuals_levels = self.in_sample_residuals
            else:
                if self.out_sample_residuals is None:
                    raise ValueError(
                        ("`forecaster.out_sample_residuals` is `None`. Use "
                         "`in_sample_residuals=True` or method "
                         "`set_out_sample_residuals()` before `predict_interval()`, "
                         "`predict_bootstrapping()`,`predict_quantiles()` or "
                         "`predict_dist()`.")
                    )
                else:
                    if not set(levels).issubset(set(self.out_sample_residuals.keys())):
                        raise ValueError(
                            (f"Not `forecaster.out_sample_residuals` for levels: "
                             f"{set(levels) - set(self.out_sample_residuals.keys())}. "
                             f"Use method `set_out_sample_residuals()`.")
                        )
                residuals_levels = self.out_sample_residuals
                    
            check_residuals = (
                "forecaster.in_sample_residuals" if in_sample_residuals
                else "forecaster.out_sample_residuals"
            )
            for level in levels:
                if residuals_levels[level] is None:
                    raise ValueError(
                        (f"forecaster residuals for level '{level}' are `None`. "
                         f"Check `{check_residuals}`.")
                    )
                elif (any(element is None for element in residuals_levels[level]) or
                      np.any(np.isnan(residuals_levels[level]))):
                    raise ValueError(
                        (f"forecaster residuals for level '{level}' contains `None` "
                         f"or `NaNs` values. Check `{check_residuals}`.")
                    )

        check_predict_input(
            forecaster_name  = type(self).__name__,
            steps            = steps,
            fitted           = self.fitted,
            included_exog    = self.included_exog,
            index_type       = self.index_type,
            index_freq       = self.index_freq,
            window_size      = self.window_size,
            last_window      = last_window,
            last_window_exog = None,
            exog             = exog,
            exog_type        = self.exog_type,
            exog_col_names   = self.exog_col_names,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = levels,
            series_col_names = self.series_col_names
        )

        last_window = last_window.iloc[-self.window_size:, ].copy()

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
            exog_values = exog.to_numpy()[:steps]
        else:
            exog_values = None
        
        boot_predictions = {}

        for level in levels:
        
            last_window_level = transform_series(
                                    series            = last_window[level],
                                    transformer       = self.transformer_series_[level],
                                    fit               = False,
                                    inverse_transform = False
                                )
            last_window_values, last_window_index = preprocess_last_window(
                                                        last_window = last_window_level
                                                    )

            level_boot_predictions = np.full(
                                         shape      = (steps, n_boot),
                                         fill_value = np.nan,
                                         dtype      = float
                                     )
            rng = np.random.default_rng(seed=random_state)
            seeds = rng.integers(low=0, high=10000, size=n_boot)

            residuals = residuals_levels[level]

            for i in range(n_boot):
                # In each bootstraping iteration the initial last_window and exog 
                # need to be restored.
                last_window_boot = last_window_values.copy()
                exog_boot = exog_values.copy() if exog is not None else None

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
                    
                    prediction_with_residual = prediction + sample_residuals[step]
                    level_boot_predictions[step, i] = prediction_with_residual[0]

                    last_window_boot = np.append(
                                           last_window_boot[1:],
                                           prediction_with_residual
                                       )
                    if exog is not None:
                        exog_boot = exog_boot[1:]

            level_boot_predictions = pd.DataFrame(
                                         data    = level_boot_predictions,
                                         index   = expand_index(last_window_index, steps=steps),
                                         columns = [f"pred_boot_{i}" for i in range(n_boot)]
                                     )

            if self.transformer_series_[level]:
                for col in level_boot_predictions.columns:
                    level_boot_predictions[col] = transform_series(
                                                      series            = level_boot_predictions[col],
                                                      transformer       = self.transformer_series_[level],
                                                      fit               = False,
                                                      inverse_transform = True
                                                  )
            
            boot_predictions[level] = level_boot_predictions
        
        return boot_predictions


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
        intervals. Both predictions and intervals are returned.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels will be predicted.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
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
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot predictions are always 
            deterministic.
        in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

            - level: predictions.
            - level_lower_bound: lower bound of the interval.
            - level_upper_bound: upper bound of the interval.

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
        
        check_interval(interval=interval)

        preds = self.predict(
                    steps       = steps,
                    levels      = levels,
                    last_window = last_window,
                    exog        = exog
                )

        boot_predictions = self.predict_bootstrapping(
                               steps               = steps,
                               levels              = levels,
                               last_window         = last_window,
                               exog                = exog,
                               n_boot              = n_boot,
                               random_state        = random_state,
                               in_sample_residuals = in_sample_residuals
                           )

        interval = np.array(interval)/100
        predictions = []

        for level in levels:
            preds_interval = boot_predictions[level].quantile(q=interval, axis=1).transpose()
            preds_interval.columns = [f'{level}_lower_bound', f'{level}_upper_bound']
            predictions.append(preds[level])
            predictions.append(preds_interval)
        
        predictions = pd.concat(predictions, axis=1)

        return predictions


    def predict_quantiles(
        self,
        steps: int,
        levels: Optional[Union[str, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        quantiles: list=[0.05, 0.5, 0.95],
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
    ) -> pd.DataFrame:
        """
        Calculate the specified quantiles for each step. After generating 
        multiple forecasting predictions through a bootstrapping process, each 
        quantile is calculated for each step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels will be predicted.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. 
        quantiles : list, default `[0.05, 0.5, 0.95]`
            Sequence of quantiles to compute, which must be between 0 and 1 
            inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as 
            `quantiles = [0.05, 0.5, 0.95]`.
        n_boot : int, default `500`
            Number of bootstrapping iterations used to estimate quantiles.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot quantiles are always 
            deterministic.
        in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create quantiles. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).

        Returns
        -------
        predictions : pandas DataFrame
            Quantiles predicted by the forecaster.

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
        
        check_interval(quantiles=quantiles)

        boot_predictions = self.predict_bootstrapping(
                               steps               = steps,
                               levels              = levels,
                               last_window         = last_window,
                               exog                = exog,
                               n_boot              = n_boot,
                               random_state        = random_state,
                               in_sample_residuals = in_sample_residuals
                           )

        predictions = []

        for level in levels:
            preds_quantiles = boot_predictions[level].quantile(q=quantiles, axis=1).transpose()
            preds_quantiles.columns = [f'{level}_q_{q}' for q in quantiles]
            predictions.append(preds_quantiles)
        
        predictions = pd.concat(predictions, axis=1)

        return predictions


    def predict_dist(
        self,
        steps: int,
        distribution: object,
        levels: Optional[Union[str, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
    ) -> pd.DataFrame:
        """
        Fit a given probability distribution for each step. After generating 
        multiple forecasting predictions through a bootstrapping process, each 
        step is fitted to the given distribution.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        distribution : Object
            A distribution object from scipy.stats.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels will be predicted.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        n_boot : int, default `500`
            Number of bootstrapping iterations used to estimate predictions.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot predictions are always 
            deterministic.
        in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).

        Returns
        -------
        predictions : pandas DataFrame
            Distribution parameters estimated for each step and level.

        """
        
        if levels is None:
            levels = self.series_col_names
        elif isinstance(levels, str):
            levels = [levels]

        boot_samples = self.predict_bootstrapping(
                           steps               = steps,
                           levels              = levels,
                           last_window         = last_window,
                           exog                = exog,
                           n_boot              = n_boot,
                           random_state        = random_state,
                           in_sample_residuals = in_sample_residuals
                       )

        param_names = [p for p in inspect.signature(distribution._pdf).parameters if not p=='x'] + ["loc","scale"]
        predictions = []

        for level in levels:
            param_values = np.apply_along_axis(lambda x: distribution.fit(x), axis=1, arr=boot_samples[level])
            level_param_names = [f'{level}_{p}' for p in param_names]

            pred_level = pd.DataFrame(
                             data    = param_values,
                             columns = level_param_names,
                             index   = boot_samples[level].index
                         )
            
            predictions.append(pred_level)
        
        predictions = pd.concat(predictions, axis=1)

        return predictions

    
    def set_params(
        self, 
        params: dict
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
        None
        
        """

        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)


    def set_fit_kwargs(
        self, 
        fit_kwargs: dict
    ) -> None:
        """
        Set new values for the additional keyword arguments passed to the `fit` 
        method of the regressor.
        
        Parameters
        ----------
        fit_kwargs : dict
            Dict of the form {"argument": new_value}.

        Returns
        -------
        None
        
        """

        self.fit_kwargs = check_select_fit_kwargs(self.regressor, fit_kwargs=fit_kwargs)
        
        
    def set_lags(
        self, 
        lags: Union[int, list, np.ndarray, range]
    ) -> None:
        """      
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, numpy ndarray, range
            Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.

            - `int`: include lags from 1 to `lags` (included).
            - `list`, `1d numpy ndarray` or `range`: include only lags present in 
            `lags`, all elements must be int.

        Returns
        -------
        None
        
        """
        
        self.lags = initialize_lags(type(self).__name__, lags)            
        self.max_lag  = max(self.lags)
        self.window_size = max(self.lags)
        
        
    def set_out_sample_residuals(
        self, 
        residuals: dict,
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
        residuals : dict
            Dictionary of numpy ndarrays with the residuals of each level in the
            form {level: residuals}. If len(residuals) > 1000, only a random 
            sample of 1000 values are stored. Keys must be the same as `levels`.
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
        None

        """

        if not isinstance(residuals, dict) or not all(isinstance(x, np.ndarray) for x in residuals.values()):
            raise TypeError(
                (f"`residuals` argument must be a dict of numpy ndarrays in the form "
                 "`{level: residuals}`. " 
                 f"Got {type(residuals)}.")
            )

        if not self.fitted:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `set_out_sample_residuals()`.")
            )
        
        if self.out_sample_residuals is None:
            self.out_sample_residuals = {level: None for level in self.series_col_names}

        if not set(self.out_sample_residuals.keys()).issubset(set(residuals.keys())):
            warnings.warn(
                (f"""
                Only residuals of levels 
                {set(self.out_sample_residuals.keys()).intersection(set(residuals.keys()))} 
                are updated.
                """), IgnoredArgumentWarning
            )

        residuals = {key: value 
                     for key, value in residuals.items() 
                     if key in self.out_sample_residuals.keys()}

        for level, value in residuals.items():

            residuals_level = value

            if not transform and self.transformer_series_[level] is not None:
                warnings.warn(
                    ("Argument `transform` is set to `False` but forecaster was "
                    f"trained using a transformer {self.transformer_series_[level]} "
                    f"for level {level}. Ensure that the new residuals are "
                     "already transformed or set `transform=True`.")
                )

            if transform and self.transformer_series_ and self.transformer_series_[level]:
                warnings.warn(
                    ("Residuals will be transformed using the same transformer used "
                    f"when training the forecaster for level {level} : "
                    f"({self.transformer_series_[level]}). Ensure that the new "
                     "residuals are on the same scale as the original time series.")
                )
                residuals_level = transform_series(
                                      series            = pd.Series(residuals_level, name='residuals'),
                                      transformer       = self.transformer_series_[level],
                                      fit               = False,
                                      inverse_transform = False
                                  ).to_numpy()

            if len(residuals_level) > 1000:
                rng = np.random.default_rng(seed=random_state)
                residuals_level = rng.choice(a=residuals_level, size=1000, replace=False)
    
            if append and self.out_sample_residuals[level] is not None:
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

            self.out_sample_residuals[level] = residuals_level

    
    def get_feature_importances(
        self,
        sort_importance: bool=True
    ) -> pd.DataFrame:
        """
        Return feature importances of the regressor stored in the
        forecaster. Only valid when regressor stores internally the feature
        importances in the attribute `feature_importances_` or `coef_`.

        Parameters
        ----------
        sort_importance: bool, default `True`
            If `True`, sorts the feature importances in descending order.

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.
        
        """

        if not self.fitted:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
            )

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            estimator = self.regressor[-1]
        else:
            estimator = self.regressor

        if hasattr(estimator, 'feature_importances_'):
            feature_importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            feature_importances = estimator.coef_
        else:
            warnings.warn(
                (f"Impossible to access feature importances for regressor of type "
                 f"{type(estimator)}. This method is only valid when the "
                 f"regressor stores internally the feature importances in the "
                 f"attribute `feature_importances_` or `coef_`.")
            )
            feature_importances = None

        if feature_importances is not None:
            feature_importances = pd.DataFrame({
                                      'feature': self.X_train_col_names,
                                      'importance': feature_importances
                                  })
            if sort_importance:
                feature_importances = feature_importances.sort_values(
                                          by='importance', ascending=False
                                      )

        return feature_importances