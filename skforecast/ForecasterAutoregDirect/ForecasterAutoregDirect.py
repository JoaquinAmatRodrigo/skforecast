################################################################################
#                         ForecasterAutoregDirect                              #
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
import inspect
from copy import copy

import skforecast
from ..ForecasterBase import ForecasterBase
from ..utils import initialize_lags
from ..utils import initialize_weights
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
from ..utils import expand_index
from ..utils import transform_series
from ..utils import transform_dataframe

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class ForecasterAutoregDirect(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    autoregressive direct multi-step forecaster. A separate model is created for
    each forecast time step. See documentation for more details.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        
    lags : int, list, 1d numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list`, `numpy ndarray` or range: include only lags present in `lags`.
            
    steps : int
        Maximum number of future steps the forecaster will predict when using
        method `predict()`. Since a different model is created for each step,
        this value should be defined before training.

    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.

    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    weight_func : Callable, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
        **New in version 0.6.0**

    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
    
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
        
    lags : numpy ndarray
        Lags used as predictors.
        
    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.

    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
        
    weight_func : Callable
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method.
        **New in version 0.6.0**

    source_code_weight_func : str
        Source code of the custom function used to create weights.
        **New in version 0.6.0**
        
    max_lag : int
        Maximum value of lag included in `lags`.
        
    window_size : int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.

    last_window : pandas Series
        Last window the forecaster has seen during trained. It stores the
        values needed to predict the next `step` right after the training data.
        
    index_type : type
        Type of index of the input used in training.
        
    index_freq : str
        Frequency of Index of the input used in training.
        
    training_range : pandas Index
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

    X_train_col_names : list
        Names of columns of the matrix created internally for training.

    in_sample_residuals : dict
        Residuals of the models when predicting training data. Only stored up to
        1000 values per model in the form `{step: residuals}`. If `transformer_y` 
        is not `None`, residuals are stored in the transformed scale.
        
    out_sample_residuals : dict
        Residuals of the models when predicting non training data. Only stored
        up to 1000 values per model in the form `{step: residuals}`. If `transformer_y` 
        is not `None`, residuals are assumed to be in the transformed scale. Use 
        `set_out_sample_residuals()` method to set values.
        
    fitted : bool
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
        
    Notes
    -----
    A separate model is created for each forecasting time step. It is important to
    note that all models share the same parameter and hyperparameter configuration.
     
    """
    
    def __init__(
        self, 
        regressor: object,
        steps: int,
        lags: Union[int, np.ndarray, list],
        transformer_y: Optional[object]=None,
        transformer_exog: Optional[object]=None,
        weight_func: Optional[Callable]=None,
        forecaster_id: Optional[Union[str, int]]=None
    ) -> None:
        
        self.regressor               = regressor
        self.steps                   = steps
        self.transformer_y           = transformer_y
        self.transformer_exog        = transformer_exog
        self.weight_func             = weight_func
        self.source_code_weight_func = None
        self.last_window             = None
        self.index_type              = None
        self.index_freq              = None
        self.training_range          = None
        self.included_exog           = False
        self.exog_type               = None
        self.exog_dtypes             = None
        self.exog_col_names          = None
        self.X_train_col_names       = None
        self.fitted                  = False
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                = None
        self.skforcast_version       = skforecast.__version__
        self.python_version          = sys.version.split(" ")[0]
        self.forecaster_id           = forecaster_id

        if not isinstance(steps, int):
            raise TypeError(
                (f"`steps` argument must be an int greater than or equal to 1. "
                 f"Got {type(steps)}.")
            )

        if steps < 1:
            raise ValueError(
                f"`steps` argument must be greater than or equal to 1. Got {steps}."
            )
        
        self.regressors_ = {step: clone(self.regressor) for step in range(1, steps + 1)}
        self.lags = initialize_lags(type(self).__name__, lags)
        self.max_lag = max(self.lags)
        self.window_size = self.max_lag

        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = None
        )

        self.in_sample_residuals = {step: None for step in range(1, steps + 1)}
        self.out_sample_residuals = None
    

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterAutoregDirect object is printed.
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
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Window size: {self.window_size} \n"
            f"Maximum steps predicted: {self.steps} \n"
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
            f"Forecaster id: {self.forecaster_id} \n"
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

        n_splits = len(y) - self.max_lag - (self.steps - 1) # rows of y_data
        if n_splits <= 0:
            raise ValueError(
                (f"The maximum lag ({self.max_lag}) must be less than the length "
                 f"of the series minus the number of steps ({len(y)-(self.steps-1)}).")
            )
        
        X_data = np.full(shape=(n_splits, len(self.lags)), fill_value=np.nan, dtype=float)
        for i, lag in enumerate(self.lags):
            X_data[:, i] = y[self.max_lag - lag : -(lag + self.steps - 1)] 

        y_data = np.full(shape=(n_splits, self.steps), fill_value=np.nan, dtype=float)
        for step in range(self.steps):
            y_data[:, step] = y[self.max_lag + step : self.max_lag + step + n_splits]
            
        return X_data, y_data


    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training matrices from univariate time series and exogenous
        variables. The resulting matrices contain the target variable and predictors
        needed to train all the regressors (one per step).
        
        Parameters
        ----------        
        y : pandas Series
            Training time series.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.

        Returns 
        -------
        X_train : pandas DataFrame, shape (len(y) - self.max_lag, len(self.lags) + exog.shape[1]*steps)
            Pandas DataFrame with the training values (predictors) for each step.
            
        y_train : pandas DataFrame, shape (len(y) - self.max_lag, )
            Values (target) of the time series related to each row of `X_train` 
            for each step.
        
        """

        if len(y) < self.max_lag + self.steps:
            raise ValueError(
                (f"Minimum length of `y` for training this forecaster is "
                 f"{self.max_lag + self.steps}. Got {len(y)}. Reduce the "
                 f"number of predicted steps, {self.steps}, or the maximum "
                 f"lag, {self.max_lag}, if no more data is available.")
            )

        check_y(y=y)
        y = transform_series(
                series            = y,
                transformer       = self.transformer_y,
                fit               = True,
                inverse_transform = False
            )
        y_values, y_index = preprocess_y(y=y)

        if exog is not None:
            if len(exog) != len(y):
                raise ValueError(
                    (f"`exog` must have same number of samples as `y`. "
                     f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
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
            if not (exog_index[:len(y_index)] == y_index).all():
                raise ValueError(
                    ("Different index for `y` and `exog`. They must be equal "
                     "to ensure the correct alignment of values.")      
                )

        X_train, y_train = self._create_lags(y=y_values)
        X_train_col_names = [f"lag_{i}" for i in self.lags]
        X_train = pd.DataFrame(
                      data    = X_train,
                      columns = X_train_col_names,
                      index   = y_index[self.max_lag + (self.steps -1): ]
                  )

        if exog is not None:
            # Transform exog to match direct format
            # The first `self.max_lag` positions have to be removed from X_exog
            # since they are not in X_lags.
            exog_to_train = exog_to_direct(exog=exog, steps=self.steps).iloc[-X_train.shape[0]:, :]
            X_train = pd.concat((X_train, exog_to_train), axis=1)

        self.X_train_col_names = X_train.columns.to_list()

        y_train_col_names = [f"y_step_{i+1}" for i in range(self.steps)]
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
            Step for which columns must be selected selected. Starts at 1.

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
                (f"Invalid value `step`. For this forecaster, minimum value is 1 "
                 f"and the maximum step is {self.steps}.")
            )

        step = step - 1 # Matrices X_train and y_train start at index 0.
        y_train_step = y_train.iloc[:, step]

        if not self.included_exog:
            X_train_step = X_train
        else:
            idx_columns_lags = np.arange(len(self.lags))
            idx_columns_exog = np.arange(X_train.shape[1])[len(self.lags) + step::self.steps]
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
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> None:
        """
        Training Forecaster.
        
        Parameters
        ----------        
        y : pandas Series
            Training time series.
        
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned so
            that y[i] is regressed on exog[i].

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
        self.X_train_col_names   = None
        self.in_sample_residuals = {step: None for step in range(1, self.steps + 1)}
        self.fitted              = False
        self.training_range      = None

        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else exog.name

        X_train, y_train = self.create_train_X_y(y=y, exog=exog)
        
        # Train one regressor for each step 
        for step in range(1, self.steps + 1): 
            # self.regressors_ and self.filter_train_X_y_for_step expect
            # first step to start at value 1
            X_train_step, y_train_step = self.filter_train_X_y_for_step(
                                             step    = step,
                                             X_train = X_train,
                                             y_train = y_train
                                         )
            sample_weight = self.create_sample_weights(X_train=X_train_step)
            if sample_weight is not None:
                self.regressors_[step].fit(
                    X             = X_train_step,
                    y             = y_train_step,
                    sample_weight = sample_weight
                )
            else:
                self.regressors_[step].fit(X=X_train_step, y=y_train_step)
                
            residuals = (y_train_step - self.regressors_[step].predict(X_train_step)).to_numpy()

            if len(residuals) > 1000:
                # Only up to 1000 residuals are stored
                    rng = np.random.default_rng(seed=123)
                    residuals = rng.choice(
                                    a       = residuals, 
                                    size    = 1000, 
                                    replace = False
                                )

            self.in_sample_residuals[step] = residuals

        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = preprocess_y(y=y, return_values=False)[1][[0, -1]]
        self.index_type = type(X_train.index)
        if isinstance(X_train.index, pd.DatetimeIndex):
            self.index_freq = X_train.index.freqstr
        else: 
            self.index_freq = X_train.index.step

        self.last_window = y.iloc[-self.max_lag:].copy()


    def predict(
        self,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.Series]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> pd.Series:
        """
        Predict n steps ahead.

        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            If `int`:
                Only steps within the range of 1 to int are predicted.
        
            If `list`:
                List of ints. Only the steps contained in the list are predicted.

            If `None`:
                As many steps are predicted as were defined at initialization.

        last_window : pandas Series, default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of prediction (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns 
        -------
        predictions : pandas Series
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
            last_window = copy(self.last_window)

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
            series_col_names = None
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
            exog_values = exog_to_direct(exog=exog.iloc[:max(steps), ], steps=max(steps)).to_numpy()
        else:
            exog_values = None

        last_window = transform_series(
                          series            = last_window,
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = False
                      )
        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )

        X_lags = last_window_values[-self.lags].reshape(1, -1)
        
        predictions = np.full(shape=len(steps), fill_value=np.nan)

        for i, step in enumerate(steps):
            regressor = self.regressors_[step]
            if exog is None:
                X = X_lags
            else:
                # Only columns from exog related with the current step are selected.
                X = np.hstack([X_lags, exog_values[0][step-1::max(steps)].reshape(1, -1)])
            with warnings.catch_warnings():
                # Suppress scikit-learn warning: "X does not have valid feature names,
                # but NoOpTransformer was fitted with feature names".
                warnings.simplefilter("ignore")
                predictions[i] = regressor.predict(X)

        idx = expand_index(index=last_window_index, steps=max(steps))

        predictions = pd.Series(
                          data  = predictions.reshape(-1),
                          index = idx[np.array(steps)-1],
                          name  = 'pred'
                      )

        predictions = transform_series(
                          series            = predictions,
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = True
                      )

        return predictions


    def predict_bootstrapping(
        self,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.Series]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
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
        
            If `int`:
                Only steps within the range of 1 to int are predicted.
        
            If `list`:
                List of ints. Only the steps contained in the list are predicted.

            If `None`:
                As many steps are predicted as were defined at initialization.

        last_window : pandas Series, default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of prediction (t + 1).
    
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
        boot_predictions : pandas DataFrame, shape (steps, n_boot)
            Predictions generated by bootstrapping.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
        Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

        """
        
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
                     "before `predict_interval()`, `predict_bootstrapping()` or "
                     "`predict_dist()`.")
                )
            else:
                if not set(steps).issubset(set(self.out_sample_residuals.keys())):
                    raise ValueError(
                        (f"Not `forecaster.out_sample_residuals` for steps: "
                         f"{set(steps) - set(self.out_sample_residuals.keys())}. "
                         f"Use method `set_out_sample_residuals()`.")
                    )
            residuals = self.out_sample_residuals
        
        check_residuals = "forecaster.in_sample_residuals" if in_sample_residuals else "forecaster.out_sample_residuals"
        for step in steps:
            if residuals[step] is None:
                raise ValueError(
                    (f"forecaster residuals for step {step} are `None`. Check {check_residuals}.")
                )
            elif (residuals[step] == None).any():
                raise ValueError(
                    (f"forecaster residuals for step {step} contains `None` values. Check {check_residuals}.")
                )

        predictions = self.predict(
                          steps       = steps,
                          last_window = last_window,
                          exog        = exog 
                      )

        # Predictions must be in the transformed scale before adding residuals
        predictions = transform_series(
                          series            = predictions,
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = False
                      )
        boot_predictions = pd.concat([predictions] * n_boot, axis=1)
        boot_predictions.columns= [f"pred_boot_{i}" for i in range(n_boot)]

        rng = np.random.default_rng(seed=random_state)
        for i, step in enumerate(steps):
            sample_residuals = rng.choice(
                                   a       = residuals[step],
                                   size    = n_boot,
                                   replace = True
                               )
            boot_predictions.iloc[i, :] = boot_predictions.iloc[i, :] + sample_residuals

        if self.transformer_y:
            for col in boot_predictions.columns:
                boot_predictions[col] = transform_series(
                                            series            = boot_predictions[col],
                                            transformer       = self.transformer_y,
                                            fit               = False,
                                            inverse_transform = True
                                        )
        
        return boot_predictions

    
    def predict_interval(
        self,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.Series]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        interval: list=[5, 95],
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
    ) -> pd.DataFrame:
        """
        Bootstrapping based prediction intervals.
        Both predictions and intervals are returned.
        
        Parameters
        ---------- 
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            If `int`:
                Only steps within the range of 1 to int are predicted.
        
            If `list`:
                List of ints. Only the steps contained in the list are predicted.

            If `None`:
                As many steps are predicted as were defined at initialization.

        last_window : pandas Series, default `None`
            Values of the series used to create the predictors (lags) need in the 
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
            Values predicted by the forecaster and their estimated interval:

            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound interval of the interval.

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
    

    def predict_dist(
        self,
        distribution: object,
        steps: Optional[Union[int, list]]=None,
        last_window: Optional[pd.Series]=None,
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
        distribution : Object
            A distribution object from scipy.stats.
        
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            If `int`:
                Only steps within the range of 1 to int are predicted.
        
            If `list`:
                List of ints. Only the steps contained in the list are predicted.

            If `None`:
                As many steps are predicted as were defined at initialization.
            
        last_window : pandas Series, default `None`
            Values of the series used to create the predictors (lags) needed in the 
            first iteration of prediction (t + 1).
    
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

        param_names = [p for p in inspect.signature(distribution._pdf).parameters if not p=='x'] + ["loc","scale"]
        param_values = np.apply_along_axis(lambda x: distribution.fit(x), axis=1, arr=boot_samples)
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
        self
        
        """
        
        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)
        self.regressors_ = {step: clone(self.regressor) for step in range(1, self.steps + 1)}


    def set_lags(
        self, 
        lags: Union[int, list, np.ndarray, range]
    ) -> None:
        """      
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, 1D np.ndarray, range
            Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                `int`: include lags from 1 to `lags`.
                `list` or `np.ndarray`: include only lags present in `lags`.

        Returns 
        -------
        None
        
        """
        
        self.lags = initialize_lags(type(self).__name__, lags)
        self.max_lag = max(self.lags)
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
        self

        """

        if not isinstance(residuals, dict) or not all(isinstance(x, np.ndarray) for x in residuals.values()):
            raise TypeError(
                f"`residuals` argument must be a dict of numpy ndarrays in the form "
                "`{step: residuals}`. " 
                f"Got {type(residuals)}."
            )

        if not self.fitted:
            raise sklearn.exceptions.NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `set_out_sample_residuals()`.")
            )
        
        if self.out_sample_residuals is None:
            self.out_sample_residuals = {step: None for step in range(1, self.steps + 1)}
        
        if not set(self.out_sample_residuals.keys()).issubset(set(residuals.keys())):
            warnings.warn(
                f"""
                Only residuals of models (steps) 
                {set(self.out_sample_residuals.keys()).intersection(set(residuals.keys()))} 
                are updated.
                """
            )

        residuals = {key: value for key, value in residuals.items() if key in self.out_sample_residuals.keys()}

        if not transform and self.transformer_y is not None:
            warnings.warn(
                (f"Argument `transform` is set to `False` but forecaster was trained "
                 f"using a transformer {self.transformer_y}. Ensure that the new residuals "
                 f"are already transformed or set `transform=True`.")
            )

        if transform and self.transformer_y is not None:
            warnings.warn(
                (f"Residuals will be transformed using the same transformer used "
                 f"when training the forecaster ({self.transformer_y}). Ensure the "
                 f"new residuals are on the same scale as the original time series.")
            )
            for key, value in residuals.items():
                residuals[key] = transform_series(
                                     series            = pd.Series(value, name='residuals'),
                                     transformer       = self.transformer_y,
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

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            estimator = self.regressors_[step][-1]
        else:
            estimator = self.regressors_[step]

        idx_columns_lags = np.arange(len(self.lags))
        idx_columns_exog = np.array([], dtype=int)
        if self.included_exog:
            idx_columns_exog = np.arange(len(self.X_train_col_names))[len(self.lags) + step-1::self.steps]
        idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
        feature_names = [self.X_train_col_names[i] for i in idx_columns]
        feature_names = [name.replace(f"_step_{step}", "") for name in feature_names]
        try:
            feature_importance = pd.DataFrame({
                                     'feature': feature_names,
                                     'importance' : estimator.feature_importances_
                                 })
        except:   
            try:
                feature_importance = pd.DataFrame({
                                         'feature': feature_names,
                                         'importance' : estimator.coef_
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