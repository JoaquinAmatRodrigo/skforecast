################################################################################
#                        ForecasterAutoregMultiVariate                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
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
import inspect
from copy import copy, deepcopy
from itertools import chain
from joblib import Parallel, delayed, cpu_count

import skforecast
from ..ForecasterBase import ForecasterBase
from ..exceptions import IgnoredArgumentWarning
from ..utils import initialize_lags
from ..utils import initialize_weights
from ..utils import check_select_fit_kwargs
from ..utils import check_y
from ..utils import check_exog
from ..utils import get_exog_dtypes
from ..utils import check_exog_dtypes
from ..utils import check_predict_input
from ..utils import check_interval
from ..utils import preprocess_y
from ..utils import preprocess_last_window
from ..utils import preprocess_exog
from ..utils import exog_to_direct
from ..utils import exog_to_direct_numpy
from ..utils import expand_index
from ..utils import transform_series
from ..utils import transform_dataframe
from ..utils import select_n_jobs_fit_forecaster

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)

class ForecasterAutoregMultiVariate(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    autoregressive multivariate direct multi-step forecaster. A separate model 
    is created for each forecast time step. See documentation for more details.

    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    level : str
        Name of the time series to be predicted.
    steps : int
        Maximum number of future steps the forecaster will predict when using
        method `predict()`. Since a different model is created for each step,
        this value must be defined before training.
    lags : int, list, numpy ndarray, range, dict
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.

            - `int`: include lags from 1 to `lags` (included).
            - `list`, `1d numpy ndarray` or `range`: include only lags present in 
            `lags`, all elements must be int.
            - `dict`: create different lags for each series. 
            {'series_column_name': lags}.
    transformer_series : transformer (preprocessor), dict, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. Transformation is applied to each `series` before training 
        the forecaster. ColumnTransformers are not allowed since they do not have 
        inverse_transform method.

            - If single transformer: it is cloned and applied to all series. 
            - If `dict` of transformers: a different transformer can be used for each 
            series.
    transformer_exog : transformer, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_fit_forecaster.
        **New in version 0.9.0**
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.

    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        An instance of this regressor is trained for each step. All of them 
        are stored in `self.regressors_`.
    regressors_ : dict
        Dictionary with regressors trained for each step. They are initialized 
        as a copy of `regressor`.
    steps : int
        Number of future steps the forecaster will predict when using method
        `predict()`. Since a different model is created for each step, this value
        should be defined before training.
    lags : numpy ndarray, dict
        Lags used as predictors.
    lags_ : dict
        Dictionary containing the lags of each series. Created from `lags` and 
        used internally.
    transformer_series : transformer (preprocessor), dict, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. Transformation is applied to each `series` before training 
        the forecaster. ColumnTransformers are not allowed since they do not have 
        inverse_transform method.

            - If single transformer: it is cloned and applied to all series. 
            - If `dict` of transformers: a different transformer can be used for each 
            series.
    transformer_series_ : dict
        Dictionary with the transformer for each series. It is created cloning the 
        objects in `transformer_series` and is used internally to avoid overwriting.
    transformer_exog : transformer
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its
        `fit` method. The resulting `sample_weight` cannot have negative values.
    source_code_weight_func : str
        Source code of the custom function used to create weights.
    max_lag : int
        Maximum value of lag included in `lags`.
    window_size : int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.
    last_window : pandas Series
        Last window seen by the forecaster during training. It stores the values 
        needed to predict the next `step` immediately after the training data.   
    index_type : type
        Type of index of the input used in training.
    index_freq : str
        Frequency of Index of the input used in training.
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
        Names of the series used during training.
    X_train_col_names : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals : dict
        Residuals of the models when predicting training data. Only stored up to
        1000 values per model in the form `{step: residuals}`. If `transformer_series` 
        is not `None`, residuals are stored in the transformed scale.
    out_sample_residuals : dict
        Residuals of the models when predicting non training data. Only stored
        up to 1000 values per model in the form `{step: residuals}`. If `transformer_series` 
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
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the fuction
        skforecast.utils.select_n_jobs_fit_forecaster.
        **New in version 0.9.0**
    forecaster_id : str, int
        Name used as an identifier of the forecaster.

    Notes
    -----
    A separate model is created for each forecasting time step. It is important to
    note that all models share the same parameter and hyperparameter configuration.
    
    """
    
    def __init__(
        self,
        regressor: object,
        level: str,
        steps: int,
        lags: Union[int, np.ndarray, list, dict],
        transformer_series: Optional[Union[object, dict]]=None,
        transformer_exog: Optional[object]=None,
        weight_func: Optional[Callable]=None,
        fit_kwargs: Optional[dict]=None,
        n_jobs: Optional[Union[int, str]]='auto',
        forecaster_id: Optional[Union[str, int]]=None
    ) -> None:
        
        self.regressor               = regressor
        self.level                   = level
        self.steps                   = steps
        self.transformer_series      = transformer_series
        self.transformer_series_     = None
        self.transformer_exog        = transformer_exog
        self.weight_func             = weight_func
        self.source_code_weight_func = None
        self.max_lag                 = None
        self.window_size             = None
        self.last_window             = None
        self.index_type              = None
        self.index_freq              = None
        self.training_range          = None
        self.included_exog           = False
        self.exog_type               = None
        self.exog_dtypes             = None
        self.exog_col_names          = None
        self.series_col_names        = None
        self.X_train_col_names       = None
        self.fitted                  = False
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                = None
        self.skforecast_version      = skforecast.__version__
        self.python_version          = sys.version.split(" ")[0]
        self.forecaster_id           = forecaster_id

        if not isinstance(level, str):
            raise TypeError(
                f"`level` argument must be a str. Got {type(level)}."
            )

        if not isinstance(steps, int):
            raise TypeError(
                (f"`steps` argument must be an int greater than or equal to 1. "
                 f"Got {type(steps)}.")
            )

        if steps < 1:
            raise ValueError(
                f"`steps` argument must be greater than or equal to 1. Got {steps}."
            )
        
        if not isinstance(n_jobs, int) and n_jobs != 'auto':
            raise TypeError(
                f"`n_jobs` must be an integer or `'auto'`. Got {type(n_jobs)}."
            )
        
        self.regressors_ = {step: clone(self.regressor) for step in range(1, steps + 1)}

        if isinstance(lags, dict):
            self.lags = {}
            for key in lags:
                self.lags[key] = initialize_lags(
                                     forecaster_name = type(self).__name__,
                                     lags            = lags[key]
                                 )
        else:
            self.lags = initialize_lags(
                            forecaster_name = type(self).__name__, 
                            lags            = lags
                        )

        self.lags_ = self.lags
        self.max_lag = (
            max(list(chain(*self.lags.values())))
            if isinstance(self.lags, dict)
            else max(self.lags)
        )
        self.window_size = self.max_lag
            
        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = None
        )

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )

        self.in_sample_residuals = {step: None for step in range(1, steps + 1)}
        self.out_sample_residuals = None

        if n_jobs == 'auto':
            self.n_jobs = select_n_jobs_fit_forecaster(
                              forecaster_name = type(self).__name__,
                              regressor_name  = type(self.regressor).__name__,
                          )
        else:
            self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
    

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterAutoregMultiVariate object is printed.
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
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Window size: {self.window_size} \n"
            f"Target series, level: {self.level} \n"
            f"Multivariate series (names): {self.series_col_names} \n"
            f"Maximum steps predicted: {self.steps} \n"
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
        lags: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a 1d array into a 2d array (X) and a 2d array (y). Each row
        in X is associated with a value of y and it represents the lags that
        precede it.
        
        Notice that, the returned matrix X_data, contains the lag 1 in the first
        column, the lag 2 in the second column and so on.
        
        Parameters
        ----------
        y : numpy ndarray
            1d numpy ndarray Training time series.
        lags : numpy ndarray
            lags to create.

        Returns
        -------
        X_data : numpy ndarray
            2d numpy ndarray with the lagged values (predictors). 
            Shape: (samples - max(self.lags), len(self.lags))
        y_data : numpy ndarray
            2d numpy ndarray with the values of the time series related to each 
            row of `X_data` for each step. 
            Shape: (len(self.steps), samples - max(self.lags))
        
        """
          
        n_splits = len(y) - self.max_lag - (self.steps - 1) # rows of y_data
        if n_splits <= 0:
            raise ValueError(
                (f"The maximum lag ({self.max_lag}) must be less than the length "
                 f"of the series minus the number of steps ({len(y)-(self.steps-1)}).")
            )
        
        X_data = np.full(shape=(n_splits, len(lags)), fill_value=np.nan, dtype=float)
        for i, lag in enumerate(lags):
            X_data[:, i] = y[self.max_lag - lag : -(lag + self.steps - 1)] 

        y_data = np.full(shape=(self.steps, n_splits), fill_value=np.nan, dtype=float)
        for step in range(self.steps):
            y_data[step, ] = y[self.max_lag + step : self.max_lag + step + n_splits]
            
        return X_data, y_data


    def create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Create training matrices from multiple time series and exogenous
        variables. The resulting matrices contain the target variable and predictors
        needed to train all the regressors (one per step).
        
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
            Training values (predictors) for each step. Note that the index 
            corresponds to that of the last step. It is updated for the corresponding 
            step in the filter_train_X_y_for_step method.
            Shape: (len(series) - self.max_lag, len(self.lags)*len(series.columns) + exog.shape[1]*steps)
        y_train : dict
            Values (target) of the time series related to each row of `X_train` 
            for each step of the form {step: y_step_[i]}.
            Shape of each series: (len(y) - self.max_lag, )
        
        """

        if not isinstance(series, pd.DataFrame):
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}.")
        
        series_col_names = list(series.columns)

        if self.level not in series_col_names:
            raise ValueError(
                (f"One of the `series` columns must be named as the `level` of the forecaster.\n"
                 f"    forecaster `level` : {self.level}.\n"
                 f"    `series` columns   : {series_col_names}.")
            )

        self.lags_ = self.lags
        if isinstance(self.lags_, dict):
            if list(self.lags_.keys()) != series_col_names:
                raise ValueError(
                    (f"When `lags` parameter is a `dict`, its keys must be the "
                     f"same as `series` column names.\n"
                     f"    Lags keys        : {list(self.lags_.keys())}.\n"
                     f"    `series` columns : {series_col_names}.")
                )
        else:
            self.lags_ = {serie: self.lags_ for serie in series_col_names}

        if len(series) < self.max_lag + self.steps:
            raise ValueError(
                (f"Minimum length of `series` for training this forecaster is "
                 f"{self.max_lag + self.steps}. Got {len(series)}. Reduce the "
                 f"number of predicted steps, {self.steps}, or the maximum "
                 f"lag, {self.max_lag}, if no more data is available.")
            )

        if self.transformer_series is None:
            self.transformer_series_ = {serie: None for serie in series_col_names}
        elif not isinstance(self.transformer_series, dict):
            self.transformer_series_ = {serie: clone(self.transformer_series) 
                                        for serie in series_col_names}
        else:
            self.transformer_series_ = {serie: None for serie in series_col_names}
            # Only elements already present in transformer_series_ are updated
            self.transformer_series_.update(
                (k, v) for k, v in deepcopy(self.transformer_series).items()
                if k in self.transformer_series_
            )
            series_not_in_transformer_series = (
                set(series.columns) - set(self.transformer_series.keys())
            )
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
            # Need here for filter_train_X_y_for_step to work without fitting
            self.included_exog = True 
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
            if not (exog.index[:len(series)] == series.index).all():
                raise ValueError(
                    ("Different index for `series` and `exog`. They must be equal "
                     "to ensure the correct alignment of values.") 
                )

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
            X_train_values, y_train_values = self._create_lags(
                                                 y    = y_values,
                                                 lags = self.lags_[serie]
                                             )
            if i == 0:
                X_train = X_train_values
            else:
                X_train = np.hstack((X_train, X_train_values))

            if serie == self.level:
                y_train = y_train_values

        X_train_col_names = [f"{key}_lag_{lag}" for key in self.lags_
                             for lag in self.lags_[key]]
        X_train = pd.DataFrame(
                      data    = X_train,
                      columns = X_train_col_names,
                      index   = y_index[self.max_lag + (self.steps -1): ]
                  )

        if exog is not None:
            # Transform exog to match direct format
            # The first `self.max_lag` positions have to be removed from X_exog
            # since they are not in X_lags.
            exog_to_train = exog_to_direct(
                                exog  = exog,
                                steps = self.steps
                            ).iloc[-X_train.shape[0]:, :]
            exog_to_train.index = exog_index[-X_train.shape[0]:]
            X_train = pd.concat((X_train, exog_to_train), axis=1)
        
        self.X_train_col_names = X_train.columns.to_list()

        y_train = {step: pd.Series(
                             data  = y_train[step-1], 
                             index = y_index[self.max_lag + step-1:][:len(y_train[0])],
                             name  = f"{self.level}_step_{step}"
                         )
                   for step in range(1, self.steps + 1)}
                        
        return X_train, y_train

    
    def filter_train_X_y_for_step(
        self,
        step: int,
        X_train: pd.DataFrame,
        y_train: dict,
        remove_suffix: bool=False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select the columns needed to train a forecaster for a specific step.  
        The input matrices should be created using `create_train_X_y` method. 
        This method updates the index of `X_train` to the corresponding one 
        according to `y_train`. If `remove_suffix=True` the suffix "_step_i" 
        will be removed from the column names. 

        Parameters
        ----------
        step : int
            step for which columns must be selected selected. Starts at 1.
        X_train : pandas DataFrame
            Dataframe created with the `create_train_X_y` method, first return.
        y_train : dict
            Dict created with the `create_train_X_y` method, second return.
        remove_suffix : bool, default `False`
            If True, suffix "_step_i" is removed from the column names.

        Returns
        -------
        X_train_step : pandas DataFrame
            Training values (predictors) for the selected step.
        y_train_step : pandas Series
            Values (target) of the time series related to each row of `X_train`.
            Shape: (len(y) - self.max_lag)

        """

        if (step < 1) or (step > self.steps):
            raise ValueError(
                (f"Invalid value `step`. For this forecaster, minimum value is 1 "
                 f"and the maximum step is {self.steps}.")
            )

        y_train_step = y_train[step]

        # Matrix X_train starts at index 0.
        if not self.included_exog:
            X_train_step = X_train
        else:
            len_columns_lags = len(list(chain(*self.lags_.values())))
            idx_columns_lags = np.arange(len_columns_lags)
            n_exog = (len(self.X_train_col_names) - len_columns_lags) / self.steps
            idx_columns_exog = (
                np.arange((step-1)*n_exog, (step)*n_exog) + idx_columns_lags[-1] + 1 
            )
            idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
            X_train_step = X_train.iloc[:, idx_columns]

        X_train_step.index = y_train_step.index

        if remove_suffix:
            X_train_step.columns = [col_name.replace(f"_step_{step}", "")
                                    for col_name in X_train_step.columns]
            y_train_step.name = y_train_step.name.replace(f"_step_{step}", "")

        return  X_train_step, y_train_step

    
    def create_sample_weights(
        self,
        X_train: pd.DataFrame
    )-> np.ndarray:
        """
        Crate weights for each observation according to the forecaster's attribute
        `weight_func`. 

        Parameters
        ----------
        X_train : pandas DataFrame
            Dataframe created with `create_train_X_y` and filter_train_X_y_for_step`
            methods, first return.

        Returns
        -------
        sample_weight : numpy ndarray
            Weights to use in `fit` method.
        
        """

        sample_weight = None

        if self.weight_func is not None:
            sample_weight = self.weight_func(X_train.index)

        if sample_weight is not None:
            if np.isnan(sample_weight).any():
                raise ValueError(
                    "The resulting `sample_weight` cannot have NaN values."
                )
            if np.any(sample_weight < 0):
                raise ValueError(
                    "The resulting `sample_weight` cannot have negative values."
                )
            if np.sum(sample_weight) == 0:
                raise ValueError(
                    ("The resulting `sample_weight` cannot be normalized because "
                     "the sum of the weights is zero.")
                )

        return sample_weight

        
    def fit(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        store_in_sample_residuals: bool=True
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the regressor 
        can be added with the `fit_kwargs` argument when initializing the forecaster.

        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `series` and their indexes must be aligned so
            that series[i] is regressed on exog[i].
        store_in_sample_residuals : bool, default `True`
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting.

        Returns
        -------
        None
        
        """
        
        # Reset values in case the forecaster has already been fitted.
        self.index_type          = None
        self.index_freq          = None
        self.last_window         = None
        self.included_exog       = False
        self.exog_type           = None
        self.exog_dtypes         = None
        self.exog_col_names      = None
        self.series_col_names    = None
        self.X_train_col_names   = None
        self.in_sample_residuals = {step: None for step in range(1, self.steps + 1)}
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
                    (f"`exog` cannot contain a column named the same as one of the "
                     f"series (column names of series).\n"
                     f"    `series` columns : {self.series_col_names}.\n"
                     f"    `exog`   columns : {self.exog_col_names}.")
                )

        X_train, y_train = self.create_train_X_y(series=series, exog=exog)

        def fit_forecaster(regressor, X_train, y_train, step, store_in_sample_residuals):
            """
            Auxiliary function to fit each of the forecaster's regressors in parallel.

            Parameters
            ----------
            regressor : object
                Regressor to be fitted.
            X_train : pandas DataFrame
                Dataframe created with the `create_train_X_y` method, first return.
            y_train : dict
                Dict created with the `create_train_X_y` method, second return.
            step : int
                Step of the forecaster to be fitted.
            store_in_sample_residuals : bool
                If `True`, in-sample residuals will be stored in the forecaster object
                after fitting.
            
            Returns
            -------
            Tuple with the step, fitted regressor and in-sample residuals.

            """

            X_train_step, y_train_step = self.filter_train_X_y_for_step(
                                             step          = step,
                                             X_train       = X_train,
                                             y_train       = y_train,
                                             remove_suffix = True
                                         )
            sample_weight = self.create_sample_weights(X_train=X_train_step)
            if sample_weight is not None:
                regressor.fit(
                    X             = X_train_step,
                    y             = y_train_step,
                    sample_weight = sample_weight,
                    **self.fit_kwargs
                )
            else:
                regressor.fit(
                    X = X_train_step,
                    y = y_train_step,
                    **self.fit_kwargs
                )

            # This is done to save time during fit in functions such as backtesting()
            if store_in_sample_residuals:
                residuals = (
                    (y_train_step - regressor.predict(X_train_step))
                ).to_numpy()

                if len(residuals) > 1000:
                    # Only up to 1000 residuals are stored
                        rng = np.random.default_rng(seed=123)
                        residuals = rng.choice(
                                        a       = residuals, 
                                        size    = 1000, 
                                        replace = False
                                    )
            else:
                residuals = None

            return step, regressor, residuals

        results_fit = (
            Parallel(n_jobs=self.n_jobs)
            (delayed(fit_forecaster)
            (
                regressor                 = copy(self.regressor),
                X_train                   = X_train,
                y_train                   = y_train,
                step                      = step,
                store_in_sample_residuals = store_in_sample_residuals
            )
            for step in range(1, self.steps + 1))
        )

        self.regressors_ = {step: regressor 
                            for step, regressor, _ in results_fit}

        if store_in_sample_residuals:
            self.in_sample_residuals = {step: residuals 
                                        for step, _, residuals in results_fit}
        
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = preprocess_y(
                                y = series[self.level],
                                return_values = False
                              )[1][[0, -1]]
        self.index_type = type(X_train.index)
        if isinstance(X_train.index, pd.DatetimeIndex):
            self.index_freq = X_train.index.freqstr
        else: 
            self.index_freq = X_train.index.step

        self.last_window = series.iloc[-self.max_lag:].copy()

            
    def predict(
        self,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        levels: Any=None
    ) -> pd.DataFrame:
        """
        Predict n steps ahead

        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
                - If `int`: Only steps within the range of 1 to int are predicted.
                - If `list`: List of ints. Only the steps contained in the list 
                are predicted.
                - If `None`: As many steps are predicted as were defined at 
                initialization.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        levels : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Predicted values.

        """

        if isinstance(steps, int):
            steps = list(np.arange(steps) + 1)
        elif steps is None:
            steps = list(np.arange(self.steps) + 1)
        elif isinstance(steps, list):
            steps = list(np.array(steps))
        
        for step in steps:
            if not isinstance(step, (int, np.int64, np.int32)):
                raise TypeError(
                    (f"`steps` argument must be an int, a list of ints or `None`. "
                     f"Got {type(steps)}.")
                )

        if last_window is None:
            last_window = self.last_window
        
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
            max_steps        = self.steps,
            levels           = None,
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
            exog_values = exog_to_direct_numpy(
                              exog  = exog.to_numpy()[:max(steps)],
                              steps = max(steps)
                          )[0]
        else:
            exog_values = None

        X_lags = np.array([[]], dtype=float)
        
        for serie in self.series_col_names:
            
            last_window_serie = transform_series(
                                    series            = last_window[serie],
                                    transformer       = self.transformer_series_[serie],
                                    fit               = False,
                                    inverse_transform = False
                                )
        
            last_window_values, last_window_index = preprocess_last_window(
                                                        last_window = last_window_serie
                                                    )

            X_lags = np.hstack(
                        [X_lags, last_window_values[-self.lags_[serie]].reshape(1, -1)]
                     )

        predictions = np.full(shape=len(steps), fill_value=np.nan)

        if exog is None:
            Xs = [X_lags] * len(steps)
        else:
            n_exog = exog.shape[1] if isinstance(exog, pd.DataFrame) else 1
            Xs = [
                np.hstack(
                    [X_lags, exog_values[(step-1)*n_exog:(step)*n_exog].reshape(1, -1)]
                )
                for step in steps
            ]

        regressors = [self.regressors_[step] for step in steps]
        with warnings.catch_warnings():
            # Suppress scikit-learn warning: "X does not have valid feature names,
            # but NoOpTransformer was fitted with feature names".
            warnings.simplefilter("ignore")
            predictions = [
                regressor.predict(X)[0] for regressor, X in zip(regressors, Xs)
            ]

        idx = expand_index(index=last_window_index, steps=max(steps))
        predictions = pd.DataFrame(
                          data    = predictions,
                          columns = [self.level],
                          index   = idx[np.array(steps)-1]
                      )

        predictions = transform_dataframe(
                          df                = predictions,
                          transformer       = self.transformer_series_[self.level],
                          fit               = False,
                          inverse_transform = True
                      )

        return predictions


    def predict_bootstrapping(
        self,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True,
        levels: Any=None
    ) -> pd.DataFrame:
        """
        Generate multiple forecasting predictions using a bootstrapping process. 
        By sampling from a collection of past observed errors (the residuals),
        each iteration of bootstrapping generates a different set of predictions. 
        See the Notes section for more information. 
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
                - If `int`: Only steps within the range of 1 to int are predicted.
                - If `list`: List of ints. Only the steps contained in the list 
                are predicted.
                - If `None`: As many steps are predicted as were defined at 
                initialization.
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
        levels : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        boot_predictions : pandas DataFrame
            Predictions generated by bootstrapping.
            Shape: (steps, n_boot)

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
        Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

        """

        if self.fitted:
            if isinstance(steps, int):
                steps = list(np.arange(steps) + 1)
            elif steps is None:
                steps = list(np.arange(self.steps) + 1)
            elif isinstance(steps, list):
                steps = list(np.array(steps))

            if in_sample_residuals:
                if not set(steps).issubset(set(self.in_sample_residuals.keys())):
                    raise ValueError(
                        (f"Not `forecaster.in_sample_residuals` for steps: "
                         f"{set(steps) - set(self.in_sample_residuals.keys())}.")
                    )
                residuals = self.in_sample_residuals
            else:
                if self.out_sample_residuals is None:
                    raise ValueError(
                        ("`forecaster.out_sample_residuals` is `None`. Use "
                         "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
                         "before `predict_interval()`, `predict_bootstrapping()`, "
                         "`predict_quantiles()` or `predict_dist()`.")
                    )
                else:
                    if not set(steps).issubset(set(self.out_sample_residuals.keys())):
                        raise ValueError(
                            (f"Not `forecaster.out_sample_residuals` for steps: "
                             f"{set(steps) - set(self.out_sample_residuals.keys())}. "
                             f"Use method `set_out_sample_residuals()`.")
                        )
                residuals = self.out_sample_residuals
            
            check_residuals = (
                'forecaster.in_sample_residuals' if in_sample_residuals
                else 'forecaster.out_sample_residuals'
            )
            for step in steps:
                if residuals[step] is None:
                    raise ValueError(
                        (f"forecaster residuals for step {step} are `None`. "
                         f"Check {check_residuals}.")
                    )
                elif (residuals[step] == None).any():
                    raise ValueError(
                        (f"forecaster residuals for step {step} contains `None` values. "
                         f"Check {check_residuals}.")
                    )

        predictions = self.predict(
                          steps       = steps,
                          last_window = last_window,
                          exog        = exog 
                      )

        # Predictions must be in the transformed scale before adding residuals
        predictions = transform_dataframe(
                          df                = predictions,
                          transformer       = self.transformer_series_[self.level],
                          fit               = False,
                          inverse_transform = False
                      )
        boot_predictions = pd.concat([predictions] * n_boot, axis=1)
        boot_predictions.columns= [f"pred_boot_{i}" for i in range(n_boot)]

        for i, step in enumerate(steps):
            rng = np.random.default_rng(seed=random_state)
            sample_residuals = rng.choice(
                                   a       = residuals[step],
                                   size    = n_boot,
                                   replace = True
                               )
            boot_predictions.iloc[i, :] = boot_predictions.iloc[i, :] + sample_residuals

        if self.transformer_series_[self.level]:
            for col in boot_predictions.columns:
                boot_predictions[col] = transform_series(
                                            series            = boot_predictions[col],
                                            transformer       = self.transformer_series_[self.level],
                                            fit               = False,
                                            inverse_transform = True
                                        )
        
        return boot_predictions

    
    def predict_interval(
        self,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        interval: list=[5, 95],
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True,
        levels: Any=None
    ) -> pd.DataFrame:
        """
        Bootstrapping based predicted intervals.
        Both predictions and intervals are returned.
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
                - If `int`: Only steps within the range of 1 to int are predicted.
                - If `list`: List of ints. Only the steps contained in the list 
                are predicted.
                - If `None`: As many steps are predicted as were defined at 
                initialization.
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
        levels : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

                - pred: predictions.
                - lower_bound: lower bound of the interval.
                - upper_bound: upper bound of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
        
        """

        check_interval(interval=interval)

        predictions = self.predict(
                          steps       = steps,
                          last_window = last_window,
                          exog        = exog
                      )

        boot_predictions = self.predict_bootstrapping(
                               steps               = steps,
                               last_window         = last_window,
                               exog                = exog,
                               n_boot              = n_boot,
                               random_state        = random_state,
                               in_sample_residuals = in_sample_residuals
                           )

        interval = np.array(interval)/100
        predictions_interval = boot_predictions.quantile(q=interval, axis=1).transpose()
        predictions_interval.columns = ['lower_bound', 'upper_bound']
        predictions = pd.concat((predictions, predictions_interval), axis=1)

        return predictions

    
    def predict_quantiles(
        self,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        quantiles: list=[0.05, 0.5, 0.95],
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True,
        levels: Any=None
    ) -> pd.DataFrame:
        """
        Bootstrapping based predicted quantiles.
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
                - If `int`: Only steps within the range of 1 to int are predicted.
                - If `list`: List of ints. Only the steps contained in the list 
                are predicted.
                - If `None`: As many steps are predicted as were defined at 
                initialization.
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
        levels : Ignored
            Not used, present here for API consistency by convention.

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

        check_interval(quantiles=quantiles)

        boot_predictions = self.predict_bootstrapping(
                               steps               = steps,
                               last_window         = last_window,
                               exog                = exog,
                               n_boot              = n_boot,
                               random_state        = random_state,
                               in_sample_residuals = in_sample_residuals
                           )

        predictions = boot_predictions.quantile(q=quantiles, axis=1).transpose()
        predictions.columns = [f'q_{q}' for q in quantiles]

        return predictions
    

    def predict_dist(
        self,
        distribution: object,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.DataFrame]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True,
        levels: Any=None
    ) -> pd.DataFrame:
        """
        Fit a given probability distribution for each step. After generating 
        multiple forecasting predictions through a bootstrapping process, each 
        step is fitted to the given distribution.
        
        Parameters
        ----------
        distribution : Object
            A distribution object from scipy.stats.
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
                - If `int`: Only steps within the range of 1 to int are predicted.
                - If `list`: List of ints. Only the steps contained in the list 
                are predicted.
                - If `None`: As many steps are predicted as were defined at 
                initialization.
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
        levels : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Distribution parameters estimated for each step.

        """
        
        boot_samples = self.predict_bootstrapping(
                           steps               = steps,
                           last_window         = last_window,
                           exog                = exog,
                           n_boot              = n_boot,
                           random_state        = random_state,
                           in_sample_residuals = in_sample_residuals
                       )       

        param_names = [p for p in inspect.signature(distribution._pdf).parameters 
                       if not p=='x'] + ["loc","scale"]
        param_values = np.apply_along_axis(
                           lambda x: distribution.fit(x),
                           axis = 1,
                           arr  = boot_samples
                       )
        predictions = pd.DataFrame(
                          data    = param_values,
                          columns = param_names,
                          index   = boot_samples.index
                      )

        return predictions
    

    def set_params(
        self, 
        params: dict
    ) -> None:
        """
        Set new values to the parameters of the scikit learn model stored in the
        forecaster. It is important to note that all models share the same 
        configuration of parameters and hyperparameters.
        
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
        self.regressors_ = {step: clone(self.regressor)
                            for step in range(1, self.steps + 1)}


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
        lags: Union[int, np.ndarray, list, dict]
    ) -> None:
        """      
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, numpy ndarray, range, dict
            Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.

                - `int`: include lags from 1 to `lags` (included).
                - `list`, `1d numpy ndarray` or `range`: include only lags present in 
                `lags`, all elements must be int.
                - `dict`: create different lags for each series. 
                {'series_column_name': lags}.

        Returns
        -------
        None
        
        """

        if isinstance(lags, dict):
            self.lags = {}
            for key in lags:
                self.lags[key] = initialize_lags(
                                     forecaster_name = type(self).__name__,
                                     lags            = lags[key]
                                 )
        else:
            self.lags = initialize_lags(
                            forecaster_name = type(self).__name__, 
                            lags            = lags
                        )
        
        self.lags_ = self.lags
        self.max_lag = (
            max(list(chain(*self.lags.values()))) if isinstance(self.lags, dict)
            else max(self.lags)
        )
        self.window_size = self.max_lag


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
            Dictionary of numpy ndarrays with the residuals of each model in the
            form {step: residuals}. If len(residuals) > 1000, only a random 
            sample of 1000 values are stored.
        append : bool, default `True`
            If `True`, new residuals are added to the once already stored in the
            attribute `out_sample_residuals`. Once the limit of 1000 values is
            reached, no more values are appended. If False, `out_sample_residuals`
            is overwritten with the new residuals.
        transform : bool, default `True`
            If `True`, new residuals are transformed using self.transformer_y.
        random_state : int, default `123`
            Sets a seed to the random sampling for reproducible output.

        Returns
        -------
        None

        """

        if not isinstance(residuals, dict) or not all(isinstance(x, np.ndarray) for x in residuals.values()):
            raise TypeError(
                (f"`residuals` argument must be a dict of numpy ndarrays in the form "
                 "`{step: residuals}`. " 
                 f"Got {type(residuals)}.")
            )

        if not self.fitted:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `set_out_sample_residuals()`.")
            )
        
        if self.out_sample_residuals is None:
            self.out_sample_residuals = {step: None 
                                         for step in range(1, self.steps + 1)}
        
        if not set(self.out_sample_residuals.keys()).issubset(set(residuals.keys())):
            warnings.warn(
                (f"""
                Only residuals of models (steps) 
                {set(self.out_sample_residuals.keys()).intersection(set(residuals.keys()))} 
                are updated.
                """), IgnoredArgumentWarning
            )

        residuals = {key: value for key, value in residuals.items()
                     if key in self.out_sample_residuals.keys()}

        if not transform and self.transformer_series_[self.level] is not None:
            warnings.warn(
                (f"Argument `transform` is set to `False` but forecaster was trained "
                 f"using a transformer {self.transformer_series_[self.level]}. Ensure "
                 f"that the new residuals are already transformed or set `transform=True`.")
            )

        if transform and self.transformer_series_[self.level] is not None:
            warnings.warn(
                (f"Residuals will be transformed using the same transformer used when "
                 f"training the forecaster ({self.transformer_series_[self.level]}). Ensure "
                 f"the new residuals are on the same scale as the original time series.")
            )
            for key, value in residuals.items():
                residuals[key] = transform_series(
                                     series            = pd.Series(value, name='residuals'),
                                     transformer       = self.transformer_series_[self.level],
                                     fit               = False,
                                     inverse_transform = False
                                 ).to_numpy()
    
        for key, value in residuals.items():
            if len(value) > 1000:
                rng = np.random.default_rng(seed=random_state)
                value = rng.choice(a=value, size=1000, replace=False)

            if append and self.out_sample_residuals[key] is not None:
                free_space = max(0, 1000 - len(self.out_sample_residuals[key]))
                if len(value) < free_space:
                    value = np.hstack((
                                self.out_sample_residuals[key],
                                value
                            ))
                else:
                    value = np.hstack((
                                self.out_sample_residuals[key],
                                value[:free_space]
                            ))
            
            self.out_sample_residuals[key] = value

    
    def get_feature_importances(
        self,
        step: int
    ) -> pd.DataFrame:
        """
        Return feature importance of the model stored in the forecaster for a
        specific step. Since a separate model is created for each forecast time
        step, it is necessary to select the model from which retrieve information.
        Only valid when regressor stores internally the feature importances in
        the attribute `feature_importances_` or `coef_`. Otherwise, it returns  
        `None`.

        Parameters
        ----------
        step : int
            Model from which retrieve information (a separate model is created 
            for each forecast time step). First step is 1.

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.
        
        """

        if not isinstance(step, int):
            raise TypeError(
                f'`step` must be an integer. Got {type(step)}.'
            )

        if not self.fitted:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
            )

        if (step < 1) or (step > self.steps):
            raise ValueError(
                (f"The step must have a value from 1 to the maximum number of steps "
                 f"({self.steps}). Got {step}.")
            )

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            estimator = self.regressors_[step][-1]
        else:
            estimator = self.regressors_[step]
        
        len_columns_lags = len(list(chain(*self.lags_.values())))
        idx_columns_lags = np.arange(len_columns_lags)
        if self.included_exog:
            idx_columns_exog = np.flatnonzero(
                                [name.endswith(f"step_{step}")
                                 for name in self.X_train_col_names]
                               )
        else:
            idx_columns_exog = np.array([], dtype=int)
        
        idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
        feature_names = [self.X_train_col_names[i].replace(f"_step_{step}", "") 
                         for i in idx_columns]

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
                                      'feature': feature_names,
                                      'importance': feature_importances
                                  })

        return feature_importances