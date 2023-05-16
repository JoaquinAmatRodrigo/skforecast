################################################################################
#                   ForecasterAutoregMultiSeriesCustom                         #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional, Callable
import warnings
import logging
import sys
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from sklearn.base import clone
from copy import copy, deepcopy
import inspect

import skforecast
from ..ForecasterBase import ForecasterBase
from ..exceptions import IgnoredArgumentWarning
from ..utils import initialize_weights
from ..utils import check_select_fit_kwargs
from ..utils import check_y
from ..utils import check_exog
from ..utils import get_exog_dtypes
from ..utils import check_exog_dtypes
from ..utils import check_interval
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

class ForecasterAutoregMultiSeriesCustom(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster for multiple series with a custom
    function to create predictors.
    **New in version 0.7.0**
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        
    fun_predictors : Callable
        Function that receives a time series as input (numpy ndarray) and returns
        another numpy ndarray with the predictors. The same function is applied 
        to all series.
        
    window_size : int
        Size of the window needed by `fun_predictors` to create the predictors.

    name_predictors : list, default `None`
        Name of the predictors returned by `fun_predictors`. If `None`, predictors are
        named using the prefix 'custom_predictor_<i>' where `i` is the index of the position
        the predictor has in the returned array of `fun_predictors`.

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

    weight_func : Callable, dict, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        If dict {'series_column_name' : Callable} a different function can be
        used for each series, a weight of 1 is given to all series not present
        in `weight_func`. Ignored if `regressor` does not have the argument 
        `sample_weight` in its `fit` method. See Notes section for more details 
        on the use of the weights. 

    series_weights : dict, default `None`
        Weights associated with each series {'series_column_name' : float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method.
        If `series_weights` is provided, a weight of 1 is given to all series not present
        in `series_weights`. If `None`, all levels have the same weight. See Notes section
        for more details on the use of the weights.

    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
        **New in version 0.8.0**

    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
        **New in version 0.7.0**
    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        
    fun_predictors : Callable
        Function that receives a time series as input (numpy ndarray) and returns
        another numpy ndarray with the predictors. The same function is applied 
        to all series.

    source_code_fun_predictors : str
        Source code of the custom function used to create the predictors.
        
    window_size : int
        Size of the window needed by `fun_predictors` to create the predictors.

    name_predictors : list, default `None`
        Name of the predictors returned by `fun_predictors`. If `None`, predictors are
        named using the prefix 'custom_predictor_<i>' where `i` is the index of the position
        the predictor has in the returned array of `fun_predictors`.

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

    weight_func : Callable, dict, default `None`
        Function that defines the individual weights of each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        If dict {'series_column_name': Callable} a different function can be
        used for each series, a weight of 1 is given to all series not present
        in `weight_func`. Ignored if `regressor` does not have the argument 
        `sample_weight` in its `fit` method. See Notes section for more details 
        on the use of the weights.

    weight_func_ : dict
        Dictionary with the `weight_func` for each series. It is created cloning the objects
        in `weight_func` and is used internally to avoid overwriting.

    source_code_weight_func : str, dict
        Source code of the custom function(s) used to create weights.

    series_weights : dict, default `None`
        Weights associated with each series {'series_column_name': float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method.
        If `series_weights` is provided, a weight of 1 is given to all series not present
        in `series_weights`. If `None`, all levels have the same weight. See Notes section
        for more details on the use of the weights.

    series_weights_ : dict
        Weights associated with each series.It is created as a clone of `series_weights`
        and is used internally to avoid overwriting.
        
    window_size : int
        Size of the window needed by `fun_predictors` to create the predictors.

    last_window : pandas Series
        Last window seen by the forecaster during training. It stores the values 
        needed to predict the next `step` immediately after the training data.
        
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

    exog_dtypes : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated after the transformation.
        
    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    series_col_names : list
        Names of the series (levels) used during training.

    X_train_col_names : list
        Names of columns of the matrix created internally for training.

    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
        **New in version 0.8.0**
        
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

    forecaster_id : str, int default `None`
        Name used as an identifier of the forecaster.
        **New in version 0.7.0**


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
        fun_predictors: Callable, 
        window_size: int,
        name_predictors: Optional[list]=None,
        transformer_series: Optional[Union[object, dict]]=None,
        transformer_exog: Optional[object]=None,
        weight_func: Optional[Union[Callable, dict]]=None,
        series_weights: Optional[dict]=None,
        fit_kwargs: Optional[dict]=None,
        forecaster_id: Optional[Union[str, int]]=None
    ) -> None:
        
        self.regressor                  = regressor
        self.fun_predictors             = fun_predictors
        self.source_code_fun_predictors = None
        self.window_size                = window_size
        self.name_predictors            = name_predictors
        self.transformer_series         = transformer_series
        self.transformer_series_        = None
        self.transformer_exog           = transformer_exog
        self.weight_func                = weight_func
        self.weight_func_               = None
        self.source_code_weight_func    = None
        self.series_weights             = series_weights
        self.series_weights_            = None
        self.index_type                 = None
        self.index_freq                 = None
        self.index_values               = None
        self.training_range             = None
        self.last_window                = None
        self.included_exog              = False
        self.exog_type                  = None
        self.exog_dtypes                = None
        self.exog_col_names             = None
        self.series_col_names           = None
        self.X_train_col_names          = None
        self.in_sample_residuals        = None
        self.out_sample_residuals       = None
        self.fitted                     = False
        self.creation_date              = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                   = None
        self.skforcast_version          = skforecast.__version__
        self.python_version             = sys.version.split(" ")[0]
        self.forecaster_id              = forecaster_id

        if not isinstance(window_size, int):
            raise TypeError(
                f"Argument `window_size` must be an int. Got {type(window_size)}."
            )

        if not isinstance(fun_predictors, Callable):
            raise TypeError(
                f"Argument `fun_predictors` must be a Callable. Got {type(fun_predictors)}."
            )
        
        self.source_code_fun_predictors = inspect.getsource(fun_predictors)

        self.weight_func, self.source_code_weight_func, self.series_weights = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = series_weights
        )

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )


    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterAutoregMultiSeriesCustom object is printed.
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
            f"Predictors created with function: {self.fun_predictors.__name__} \n"
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
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforcast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info


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
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}.")

        if len(series) < self.window_size + 1:
            raise ValueError(
                (f"`series` must have as many values as the windows_size needed by "
                 f"{self.fun_predictors.__name__}. For this Forecaster the "
                 f"minimum length is {self.window_size + 1}")
            )

        series_col_names = list(series.columns)

        if self.transformer_series is None:
            self.transformer_series_ = {serie: None for serie in series_col_names}
        elif not isinstance(self.transformer_series, dict):
            self.transformer_series_ = {serie: clone(self.transformer_series) 
                                        for serie in series_col_names}
        else:
            self.transformer_series_ = {serie: None for serie in series_col_names}
            # Only elements already present in transformer_series_ are updated
            self.transformer_series_.update(
                (k, v) for k, v in deepcopy(self.transformer_series).items() if k in self.transformer_series_
            )
            series_not_in_transformer_series = set(series.columns) - set(self.transformer_series.keys())
            if series_not_in_transformer_series:
                    warnings.warn(
                        (f"{series_not_in_transformer_series} not present in `transformer_series`."
                         f" No transformation is applied to these series."),
                         IgnoredArgumentWarning
                    )
        
        if exog is not None:
            if len(exog) != len(series):
                raise ValueError(
                    (f"`exog` must have same number of samples as `series`. "
                     f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
                )
            check_exog(exog=exog, allow_nan=True)
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
            
            check_exog(exog=exog, allow_nan=False)
            check_exog_dtypes(exog)
            self.exog_dtypes = get_exog_dtypes(exog=exog)

            _, exog_index = preprocess_exog(exog=exog, return_values=False)
            if not (exog_index[:len(series.index)] == series.index).all():
                raise ValueError(
                    ("Different index for `series` and `exog`. They must be equal "
                     "to ensure the correct alignment of values.")
                )

        X_levels = []
        
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

            temp_X_train  = []
            temp_y_train  = []

            for j in range(len(y) - self.window_size):

                train_index = np.arange(j, self.window_size + j)
                test_index  = self.window_size + j

                temp_X_train.append(self.fun_predictors(y=y_values[train_index]))
                temp_y_train.append(y_values[test_index])

            X_train_values = np.vstack(temp_X_train)
            y_train_values = np.array(temp_y_train)

            if np.isnan(X_train_values).any():
                raise ValueError(
                    f"`fun_predictors()` is returning `NaN` values for series {serie}."
                )

            if i == 0:
                X_train = X_train_values
                y_train = y_train_values
            else:
                X_train = np.vstack((X_train, X_train_values))
                y_train = np.append(y_train, y_train_values)

            X_level = [serie]*len(X_train_values)
            X_levels.extend(X_level)

        if self.name_predictors is None:
            X_train_col_names = [f"custom_predictor_{i}" for i in range(X_train.shape[1])]
        else:
            if len(self.name_predictors) != X_train.shape[1]:
                raise ValueError(
                    (f"The length of provided predictors names (`name_predictors`) do not "
                     f"match the number of columns created by `fun_predictors()`.")
                )
            X_train_col_names = self.name_predictors.copy()

        # y_values correspond only to the last series of `series`. Since the columns
        # of X_train are the same for all series, the check is the same.
        expected = self.fun_predictors(y_values[:-1])
        observed = X_train[-1, :]

        if expected.shape != observed.shape or not (expected == observed).all():
            raise ValueError(
                (f"The `window_size` argument ({self.window_size}), declared when "
                 f"initializing the forecaster, does not correspond to the window "
                 f"used by `fun_predictors()`.")
            )
        
        X_levels = pd.Series(X_levels)
        X_levels = pd.get_dummies(X_levels, dtype=float)

        X_train = pd.DataFrame(
                      data    = X_train,
                      columns = X_train_col_names
                  )

        if exog is not None:
            # The first `self.window_size` positions have to be removed from exog
            # since they are not in X_train. Then exog is cloned as many times
            # as series.
            exog_to_train = exog.iloc[self.window_size:, ]
            exog_to_train = pd.concat([exog_to_train]*len(series_col_names)).reset_index(drop=True)
        else:
            exog_to_train = None

        X_train = pd.concat([X_train, exog_to_train, X_levels], axis=1)
        self.X_train_col_names = X_train.columns.to_list()

        y_train = pd.Series(
                      data = y_train,
                      name = 'y'
                  )

        y_train_index = pd.Index(
                            np.tile(
                                y_index[self.window_size: ].values,
                                reps = len(series_col_names)
                            )
                        )

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
                warnings.warn(
                    (f"{series_not_in_series_weights} not present in `series_weights`."
                     f" A weight of 1 is given to all their samples."),
                    IgnoredArgumentWarning
                )
            self.series_weights_ = {col: 1. for col in series.columns}
            self.series_weights_.update((k, v) for k, v in self.series_weights.items() if k in self.series_weights_)
            weights_series = [np.repeat(self.series_weights_[serie], sum(X_train[serie])) 
                              for serie in series.columns]
            weights_series = np.concatenate(weights_series)

        if self.weight_func is not None:
            if isinstance(self.weight_func, Callable):
                self.weight_func_ = {col: copy(self.weight_func) for col in series.columns}
            else:
                # Series not present in weight_func have a weight of 1 in all their samples
                series_not_in_weight_func = set(series.columns) - set(self.weight_func.keys())
                if series_not_in_weight_func:
                    warnings.warn(
                        (f"{series_not_in_weight_func} not present in `weight_func`."
                         f" A weight of 1 is given to all their samples."),
                        IgnoredArgumentWarning
                    )
                self.weight_func_ = {col: lambda x: np.ones_like(x, dtype=float) for col in series.columns}
                self.weight_func_.update((k, v) for k, v in self.weight_func.items() if k in self.weight_func_)
                
            weights_samples = []
            for key in self.weight_func_.keys():
                idx = y_train_index[X_train[X_train[key] == 1.0].index]
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
        self.exog_dtypes         = None
        self.exog_col_names      = None
        self.series_col_names    = None
        self.X_train_col_names   = None
        self.in_sample_residuals = None
        self.fitted              = False
        self.training_range      = None
        
        self.series_col_names = list(series.columns)

        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else [exog.name]

            if len(set(self.exog_col_names) - set(self.series_col_names)) != len(self.exog_col_names):
                raise ValueError(
                    (f"`exog` cannot contain a column named the same as one of the series"
                     f" (column names of series).\n"
                     f"    `series` columns : {self.series_col_names}.\n"
                     f"    `exog`   columns : {self.exog_col_names}.")
                )

        X_train, y_train, y_index, y_train_index = self.create_train_X_y(series=series, exog=exog)
        sample_weight = self.create_sample_weights(
                            series        = series,
                            X_train       = X_train,
                            y_train_index = y_train_index,
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

            residuals = y_train - self.regressor.predict(X_train)

            for serie in series.columns:
                in_sample_residuals[serie] = residuals.loc[X_train[serie] == 1.].to_numpy()
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

        # The last time window of training data is stored so that predictors in
        # the first iteration of `predict()` can be calculated.
        self.last_window = series.iloc[-self.window_size:].copy()


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

        for i in range(steps):
            X = self.fun_predictors(y=last_window).reshape(1, -1)
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
        
        if levels is None:
            levels = self.series_col_names
        elif isinstance(levels, str):
            levels = [levels]

        if last_window is None:
            last_window = deepcopy(self.last_window)

        last_window = last_window.iloc[-self.window_size:, ]
        
        check_predict_input(
            forecaster_name  = type(self).__name__,
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
            check_exog_dtypes(exog=exog)
            exog_values = exog.iloc[:steps, ].to_numpy()
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
        See the Notes section for more information. 
        
        Parameters
        ----------   
        steps : int
            Number of future steps predicted.

        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels will be predicted.
            
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
    
            If `last_window = None`, the values stored in `self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
            
        n_boot : int, default `500`
            Number of bootstrapping iterations used to estimate prediction
            intervals.

        random_state : int, default `123`
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
        boot_predictions : dict
            Predictions generated by bootstrapping for each level. 
            {level: pandas DataFrame, shape (steps, n_boot)}

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
        Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

        """
        
        if levels is None:
            levels = self.series_col_names
        elif isinstance(levels, str):
            levels = [levels]

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
                     "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
                     "before `predict_interval()`, `predict_bootstrapping()` or "
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
                
        check_residuals = 'forecaster.in_sample_residuals' if in_sample_residuals else 'forecaster.out_sample_residuals'
        for level in levels:
            if residuals_levels[level] is None:
                raise ValueError(
                    (f"forecaster residuals for level '{level}' are `None`. Check `{check_residuals}`.")
                )
            elif (residuals_levels[level] == None).any():
                raise ValueError(
                    (f"forecaster residuals for level '{level}' contains `None` values. Check `{check_residuals}`.")
                )

        if last_window is None:
            last_window = deepcopy(self.last_window)

        last_window = last_window.iloc[-self.window_size:, ]

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
            
            exog_values = exog.iloc[:steps, ].to_numpy()
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
                    level_boot_predictions[step, i] = prediction_with_residual

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
            A distribution object from scipy.stats. For example scipy.stats.norm.

        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels will be predicted.  
            
        last_window : pandas DataFrame, default `None`
            Values of the series used to create the predictors needed in the first
            re of prediction (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
            
        n_boot : int, default `500`
            Number of bootstrapping iterations used to estimate prediction
            intervals.

        random_state : int, default `123`
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
        self
        
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
        self

        """

        if not isinstance(residuals, dict) or not all(isinstance(x, np.ndarray) for x in residuals.values()):
            raise TypeError(
                f"`residuals` argument must be a dict of numpy ndarrays in the form "
                "`{level: residuals}`. " 
                f"Got {type(residuals)}."
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

        residuals = {key: value for key, value in residuals.items() if key in self.out_sample_residuals.keys()}

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
        self
    ) -> pd.DataFrame:
        """
        Return feature importances of the regressor stored in the
        forecaster. Only valid when regressor stores internally the feature
        importances in the attribute `feature_importances_` or `coef_`.

        Parameters
        ----------
        self

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

        return feature_importances
    

    def get_feature_importance(
        self
    ) -> pd.DataFrame:
        """
        This method has been replaced by `get_feature_importances()`.

        Return feature importances of the regressor stored in the
        forecaster. Only valid when regressor stores internally the feature
        importances in the attribute `feature_importances_` or `coef_`.

        Parameters
        ----------
        self

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.
        
        """

        warnings.warn(
            ("get_feature_importance() method has been renamed to get_feature_importances()."
             "This method will be removed in skforecast 0.9.0.")
        )

        return self.get_feature_importances()