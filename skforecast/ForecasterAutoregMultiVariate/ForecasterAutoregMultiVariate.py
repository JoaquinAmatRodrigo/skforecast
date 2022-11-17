################################################################################
#                        ForecasterAutoregMultiVariate                         #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional
import warnings
import logging
import sys
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from sklearn.base import clone
from copy import deepcopy
from itertools import chain

import skforecast
from ..ForecasterBase import ForecasterBase
from ..utils import initialize_lags
from ..utils import initialize_weights
from ..utils import check_y
from ..utils import check_exog
from ..utils import preprocess_y
from ..utils import preprocess_last_window
from ..utils import preprocess_exog
from ..utils import exog_to_direct
from ..utils import expand_index
from ..utils import check_predict_input
from ..utils import transform_series
from ..utils import transform_dataframe

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)

class ForecasterAutoregMultiVariate(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    autoregressive multivariate direct multi-step forecaster. A separate model 
    is created for each forecast time step. See documentation for more details.
    **New in version 0.6.0**

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
            `int`: include lags from 1 to `lags` (included).
            `list`, `numpy ndarray` or `range`: include only lags present in `lags`,
                all elements must be int.
            `dict`: create different lags for each series. {'series_column_name': lags}.

    transformer_series : transformer or dict of transformers, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        If a single transformer is passed, it is cloned and applied to all series.
        If dict, a different transformer can be used for each series {'series_column_name':
        transformer}. Transformation is applied to each series before training the forecaster.
        ColumnTransformers are not allowed since they do not have inverse_transform method.

    transformer_exog : transformer, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    weight_func : callable, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its
        `fit` method. The resulting `sample_weight` cannot have negative values.

    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        One instance of this regressor is trained for each step. All
        them are stored in `self.regressors_`.

    regressors_ : dict
        Dictionary with regressors trained for each step. They are initialized as a copy
        of `regressor`.
        
    steps : int
        Number of future steps the forecaster will predict when using method
        `predict()`. Since a different model is created for each step, this value
        should be defined before training.
        
    lags : int, list, numpy ndarray, range, dict
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list`, `numpy ndarray` or `range`: include only lags present in `lags`,
                all elements must be int.
            `dict`: create different lags for each series. {'series_column_name': lags}.
        
    lags_ : dict
        Dictionary with the of the lags for each series. Created from `lags` and used
        internally.

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

    transformer_exog : transformer, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    weight_func : callable, default `None`
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
        Last window the forecaster has seen during training. It stores the
        values needed to predict the next `step` right after the training data.
        
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
        
    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    series_col_names : list
        Names of the series used during training.

    X_train_col_names : list
        Names of columns of the matrix created internally for training.
        
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
    
    """
    
    def __init__(
        self,
        regressor: object,
        level: str,
        steps: int,
        lags: Union[int, np.ndarray, list, dict],
        transformer_series: Optional[Union[object, dict]]=None,
        transformer_exog: Optional[object]=None,
        weight_func: Optional[callable]=None
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
        self.exog_col_names          = None
        self.series_col_names        = None
        self.X_train_col_names       = None
        self.fitted                  = False
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                = None
        self.skforcast_version       = skforecast.__version__
        self.python_version          = sys.version.split(" ")[0]

        if not isinstance(level, str):
            raise TypeError(
                f"`level` argument must be a str. Got {type(level)}."
            )

        if not isinstance(steps, int):
            raise TypeError(
                f"`steps` argument must be an int greater than or equal to 1. "
                f"Got {type(steps)}."
            )

        if steps < 1:
            raise ValueError(
                f"`steps` argument must be greater than or equal to 1. Got {steps}."
            )
        
        self.regressors_ = {step: clone(self.regressor) for step in range(steps)}

        if isinstance(lags, dict):
            self.lags = {}
            for key in lags:
                self.lags[key] = initialize_lags(forecaster_type=type(self), lags=lags[key])
        else:
            self.lags = initialize_lags(forecaster_type=type(self), lags=lags)

        self.lags_ = self.lags
        self.max_lag = max(list(chain(*self.lags.values()))) if isinstance(self.lags, dict) else max(self.lags)
        self.window_size = self.max_lag
            
        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_type = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = None
        )
    

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
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"{str(type(self)).split('.')[1]} \n"
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
            f"Transformer for series: {self.transformer_series} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Window size: {self.window_size} \n"
            f"Multivariate series (names): {self.series_col_names} \n"
            f"Maximum steps predicted: {self.steps} \n"
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
        y: np.ndarray,
        lags: np.ndarray,
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

        lags : 1d numpy ndarray
            lags to create.

        Returns 
        -------
        X_data : 2d numpy ndarray, shape (samples - max(self.lags), len(self.lags))
            2d numpy array with the lagged values (predictors).
        
        y_data : 1d numpy ndarray, shape (samples - max(self.lags),)
            Values of the time series related to each row of `X_data`.
        
        """
          
        n_splits = len(y) - self.max_lag - (self.steps - 1) # rows of y_data
        if n_splits <= 0:
            raise ValueError(
                f'The maximum lag ({self.max_lag}) must be less than the length '
                f'of the series minus the number of steps ({len(y)-(self.steps-1)}).'
            )
        
        X_data = np.full(shape=(n_splits, len(lags)), fill_value=np.nan, dtype=float)
        for i, lag in enumerate(lags):
            X_data[:, i] = y[self.max_lag - lag : -(lag + self.steps - 1)] 

        y_data = np.full(shape=(n_splits, self.steps), fill_value=np.nan, dtype=float)
        for step in range(self.steps):
            y_data[:, step] = y[self.max_lag + step : self.max_lag + step + n_splits]
            
        return X_data, y_data


    def create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        X_train : pandas DataFrame, shape (len(series) - self.max_lag, len(self.lags)*len(series.columns) + exog.shape[1]*steps)
            Pandas DataFrame with the training values (predictors) for each step.
            
        y_train : pandas DataFrame, shape (len(series) - self.max_lag, )
            Values (target) of the time series related to each row of `X_train` 
            for each step.
        
        """

        if not isinstance(series, pd.DataFrame):
            raise TypeError(f'`series` must be a pandas DataFrame. Got {type(series)}.')
        
        series_col_names = list(series.columns)

        if self.level not in series_col_names:
            raise ValueError(
                (f'One of the `series` columns must be named as the `level` of the forecaster.\n'
                 f'    forecaster `level` : {self.level}.\n'
                 f'    `series` columns   : {series_col_names}.')
            )

        self.lags_ = self.lags
        if isinstance(self.lags_, dict):
            if list(self.lags_.keys()) != series_col_names:
                raise ValueError(
                    (f'When `lags` parameter is a `dict`, its keys must be the '
                     f'same as `series` column names.\n'
                     f'    Lags keys        : {list(self.lags_.keys())}.\n'
                     f'    `series` columns : {series_col_names}.')
                )
        else:
            self.lags_ = {serie: self.lags_ for serie in series_col_names}

        if len(series) < self.max_lag + self.steps:
            raise ValueError(
                f'Minimum length of `series` for training this forecaster is '
                f'{self.max_lag + self.steps}. Got {len(series)}.'
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
                (k, v) for k, v in deepcopy(self.transformer_series).items() if k in self.transformer_series_
            )
            series_not_in_transformer_series = set(series.columns) - set(self.transformer_series.keys())
            if series_not_in_transformer_series:
                    warnings.warn(
                        f"{series_not_in_transformer_series} not present in `transformer_series`."
                        f" No transformation is applied to these series."
                    )
        
        y_train_col_names = [f"{self.level}_step_{i+1}" for i in range(self.steps)]
        X_train_col_names = [f"{key}_lag_{lag}" for key in self.lags_ for lag in self.lags_[key]]

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
            X_train_values, y_train_values = self._create_lags(y=y_values, lags=self.lags_[serie])

            if i == 0:
                X_train = X_train_values
            else:
                X_train = np.hstack((X_train, X_train_values))

            if serie == self.level:
                y_train = y_train_values

        if exog is not None:
            if len(exog) != len(series):
                raise ValueError(
                    f'`exog` must have same number of samples as `series`. '
                    f'length `exog`: ({len(exog)}), length `series`: ({len(series)})'
                )
            check_exog(exog=exog)
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
            exog_values, exog_index = preprocess_exog(exog=exog)
            if not (exog_index[:len(y_index)] == y_index).all():
                raise ValueError(
                    ('Different index for `series` and `exog`. They must be equal '
                     'to ensure the correct alignment of values.')      
                )
            col_names_exog = exog.columns if isinstance(exog, pd.DataFrame) else [exog.name]

            # Transform exog to match direct format
            X_exog = exog_to_direct(exog=exog_values, steps=self.steps)
            col_names_exog = [f"{col_name}_step_{i+1}" for col_name in col_names_exog for i in range(self.steps)]
            X_train_col_names.extend(col_names_exog)

            # The first `self.max_lag` positions have to be removed from X_exog
            # since they are not in X_lags.
            X_exog = X_exog[-X_train.shape[0]:, ]
            X_train = np.column_stack((X_train, X_exog))

        X_train = pd.DataFrame(
                      data    = X_train,
                      columns = X_train_col_names,
                      index   = y_index[self.max_lag + (self.steps -1): ]
                  )
        self.X_train_col_names = X_train_col_names
        y_train = pd.DataFrame(
                      data    = y_train,
                      index   = y_index[self.max_lag + (self.steps -1): ],
                      columns = y_train_col_names,
                  )
                        
        return X_train, y_train

    
    def filter_train_X_y_for_step(
        self,
        step: int,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select columns needed to train a forecaster for a specific step. The input
        matrices should be created with created with `create_train_X_y()`.         

        Parameters
        ----------
        step : int
            step for which columns must be selected selected. Starts at 1.

        X_train : pandas DataFrame
            Pandas DataFrame with the training values (predictors).
            
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.


        Returns 
        -------
        X_train_step : pandas DataFrame
            Pandas DataFrame with the training values (predictors) for step.
            
        y_train_step : pandas Series, shape (len(y) - self.max_lag)
            Values (target) of the time series related to each row of `X_train`.

        """

        if (step < 1) or (step > self.steps):
            raise ValueError(
                f"Invalid value `step`. For this forecaster, minimum value is 1 "
                f"and the maximum step is {self.steps}."
            )

        step = step - 1 # To start at 0

        y_train_step = y_train.iloc[:, step]

        if not self.included_exog:
            X_train_step = X_train
        else:
            len_columns_lags = len(list(chain(*self.lags_.values())))
            idx_columns_lags = np.arange(len_columns_lags)
            idx_columns_exog = np.arange(X_train.shape[1])[len_columns_lags + step::self.steps]
            idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
            X_train_step = X_train.iloc[:, idx_columns]

        return  X_train_step, y_train_step

    
    def create_sample_weights(
        self,
        X_train: pd.DataFrame,
    )-> np.ndarray:
        """
        Crate weights for each observation according to the forecaster's attribute
        `weight_func`. 

        Parameters
        ----------
        X_train : pandas DataFrame
            Dataframe generated with the methods `create_train_X_y` and 
            `filter_train_X_y_for_step`, first return.

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
        store_in_sample_residuals: Any=None
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

        store_in_sample_residuals : Ignored
            Not used, present here for API consistency by convention.

        Returns 
        -------
        None
        
        """
        
        # Reset values in case the forecaster has already been fitted.
        self.index_type        = None
        self.index_freq        = None
        self.last_window       = None
        self.included_exog     = False
        self.exog_type         = None
        self.exog_col_names    = None
        self.series_col_names  = None
        self.X_train_col_names = None
        self.fitted            = False
        self.training_range    = None
        
        self.series_col_names = list(series.columns)

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

        X_train, y_train = self.create_train_X_y(series=series, exog=exog)
       
        # Train one regressor for each step
        for step in range(self.steps):

            X_train_step, y_train_step = self.filter_train_X_y_for_step(
                                             step    = step + 1,
                                             X_train = X_train,
                                             y_train = y_train
                                         )
            sample_weight = self.create_sample_weights(X_train=X_train_step)
            if sample_weight is not None:
                if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
                    self.regressors_[step].fit(
                                            X = X_train_step,
                                            y = y_train_step,
                                            sample_weight = sample_weight
                                          )
                else:
                    self.regressors_[step].fit(
                                            X = X_train_step.to_numpy(),
                                            y = y_train_step.to_numpy(),
                                            sample_weight = sample_weight
                                          )
            else:
                if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
                    self.regressors_[step].fit(X=X_train_step, y=y_train_step)
                else:
                    self.regressors_[step].fit(X=X_train_step.to_numpy(), y=y_train_step.to_numpy())
        
            
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = preprocess_y(y=series[self.level])[1][[0, -1]]
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
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> pd.DataFrame:
        """
        Predict n steps ahead

        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            If int:
                Only steps within the range of 1 to int are predicted.
        
            If list:
                List of ints. Only the steps contained in the list are predicted.

            If `None`:
                As many steps are predicted as were defined at initialization.

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
            Predicted values.

        """

        if isinstance(steps, int):
            steps = list(range(steps))
        elif steps is None:
            steps = list(range(self.steps))
        elif isinstance(steps, list):
            steps = list(np.array(steps) - 1) # To start at 0 for indexing

        for step in steps:
            if not isinstance(step, (int, np.int64, np.int32)):
                raise TypeError(
                    f"`steps` argument must be an int, a list of ints or `None`. "
                    f"Got {type(steps)}."
                )
        
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
            max_steps        = self.steps,
            levels           = None,
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
                                 exog = exog.iloc[:max(steps)+1, ]
                             )
            exog_values = exog_to_direct(exog=exog_values, steps=max(steps)+1)
        else:
            exog_values = None

        X_lags = np.array([[]], dtype=float)

        if last_window is None:
            last_window = self.last_window.copy()
        
        for serie in self.series_col_names:
            
            last_window[serie] = transform_series(
                                     series            = last_window[serie],
                                     transformer       = self.transformer_series_[serie],
                                     fit               = False,
                                     inverse_transform = False
                                 )
        
            last_window_values, last_window_index = preprocess_last_window(
                                                        last_window = last_window[serie]
                                                    )

            X_lags = np.hstack([X_lags, last_window_values[-self.lags_[serie]].reshape(1, -1)])

        predictions = np.full(shape=len(steps), fill_value=np.nan)

        for i, step in enumerate(steps):
            regressor = self.regressors_[step]
            if exog is None:
                X = X_lags
            else:
                # Only columns from exog related with the current step are selected.
                X = np.hstack([X_lags, exog_values[0][step::max(steps)+1].reshape(1, -1)])
            with warnings.catch_warnings():
                # Suppress scikit-learn warning: "X does not have valid feature names,
                # but NoOpTransformer was fitted with feature names".
                warnings.simplefilter("ignore")
                predictions[i] = regressor.predict(X)

        idx = expand_index(index=last_window_index, steps=max(steps)+1)

        predictions = pd.DataFrame(
                          data    = predictions,
                          columns = [self.level],
                          index   = idx[steps]
                      )

        predictions = transform_dataframe(
                          df                = predictions,
                          transformer       = self.transformer_series_[self.level],
                          fit               = False,
                          inverse_transform = True
                      )

        return predictions
    

    def set_params(
        self, 
        **params: dict
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
        self
        
        """

        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)
        self.regressors_ = {step: clone(self.regressor) for step in range(self.steps)}
        
        
    def set_lags(
        self, 
        lags: Union[int, np.ndarray, list, dict]
    ) -> None:
        """      
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, 1d numpy ndarray, range, dict
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list`, `numpy ndarray` or `range`: include only lags present in `lags`,
                all elements must be int.
            `dict`: generate different lags for each series used to fit the 
                regressors. {'series_column_name': lags}.

        Returns 
        -------
        None
        
        """
        
        if isinstance(lags, dict):
            self.lags = {}
            for key in lags:
                self.lags[key] = initialize_lags(forecaster_type=type(self), lags=lags[key])
        else:
            self.lags = initialize_lags(forecaster_type=type(self), lags=lags)
        
        self.lags_ = self.lags
        self.max_lag = max(list(chain(*self.lags.values()))) if isinstance(self.lags, dict) else max(self.lags)
        self.window_size = self.max_lag

    
    def get_feature_importance(
        self,
        step: int
    ) -> pd.DataFrame:
        """      
        Return impurity-based feature importance of the model stored in
        the forecaster for a specific step. Since a separate model is created for
        each forecast time step, it is necessary to select the model from which
        retrieve information. Only valid when regressor stores internally the 
        feature importance in the attribute `feature_importances_` or `coef_`.

        Parameters
        ----------
        step : int
            Model from which retrieve information (a separate model is created 
            for each forecast time step). First step is 1.

        Returns
        -------
        feature_importance : pandas DataFrame
            Feature importance associated with each predictor.
        
        """

        if not isinstance(step, int):
            raise TypeError(
                f'`step` must be an integer. Got {type(step)}.'
            )

        if self.fitted == False:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importance()`.")
            )

        if (step < 1) or (step > self.steps):
            raise ValueError(
                f"The step must have a value from 1 to the maximum number of steps "
                f"({self.steps}). Got {step}."
            )

        # Stored regressors start at index 0
        step = step - 1

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            estimator = self.regressors_[step][-1]
        else:
            estimator = self.regressors_[step]
        
        len_columns_lags = len(list(chain(*self.lags_.values())))
        idx_columns_lags = np.arange(len_columns_lags)
        idx_columns_exog = np.array([], dtype=int)
        if self.included_exog:
            idx_columns_exog = np.arange(len(self.X_train_col_names))[len_columns_lags + step::self.steps]
        idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
        feature_names = [self.X_train_col_names[i] for i in idx_columns]
        feature_names = [name.replace(f"_step_{step+1}", "") for name in feature_names]

        try:
            feature_importance = pd.DataFrame({
                                    'feature': feature_names,
                                    'importance': estimator.feature_importances_
                                 })
        except:   
            try:
                feature_importance = pd.DataFrame({
                                        'feature': feature_names,
                                        'importance': estimator.coef_
                                     })
            except:
                warnings.warn(
                    f"Impossible to access feature importance for regressor of type "
                    f"{type(estimator)}. This method is only valid when the "
                    f"regressor stores internally the feature importance in the "
                    f"attribute `feature_importances_` or `coef_`."
                )

                feature_importance = None

        return feature_importance