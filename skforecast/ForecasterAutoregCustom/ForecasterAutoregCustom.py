################################################################################
#                        ForecasterAutoregCustom                               #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional
import warnings
import logging
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from sklearn.base import clone
from copy import copy
from inspect import getsource

import skforecast
from ..ForecasterBase import ForecasterBase
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


class ForecasterAutoregCustom(ForecasterBase):
    '''
    This class turns any regressor compatible with the scikit-learn API into a
    recursive (multi-step) forecaster with a custom function to create predictors.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        
    fun_predictors: Callable
        Function that takes a numpy ndarray as a window of values as input and  
        returns a numpy ndarray with the predictors associated with that window.
        
    window_size: int
        Size of the window needed by `fun_predictors` to create the predictors.

    transformer_y : transformer (preprocessor) compatible with the scikit-learn
                    preprocessing API, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.

    transformer_exog : transformer (preprocessor) compatible with the scikit-learn
                       preprocessing API, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    
    
    Attributes
    ----------
    regressor : regressor compatible with the scikit-learn API
        An instance of a regressor compatible with the scikit-learn API.
        
    create_predictors: Callable
        Function that takes a numpy ndarray as a window of values as input and  
        returns a numpy ndarray with the predictors associated with that window.

    source_code_create_predictors: str
        Source code of the custom function used to create the predictors.
        
    window_size: int
        Size of the window needed by `fun_predictors` to create the predictors.

    transformer_y : transformer (preprocessor) compatible with the scikit-learn
                    preprocessing API, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.

    transformer_exog : transformer (preprocessor) compatible with the scikit-learn
                    preprocessing API, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
        
    last_window : pandas Series
        Last window the forecaster has seen during trained. It stores the
        values needed to predict the next `step` right after the training data.
        
    window_size: int
        Size of the window needed by `fun_predictors` to create the predictors.
        
    fitted: Bool
        Tag to identify if the regressor has been fitted (trained).
        
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

    X_train_col_names : list
        Names of columns of the matrix created internally for training.
        
    in_sample_residuals: pandas Series
        Residuals of the model when predicting training data. Only stored up to
        1000 values.
        
    out_sample_residuals: pandas Series
        Residuals of the model when predicting non training data. Only stored
        up to 1000 values. Use `set_out_sample_residuals` to set values.

    creation_date: str
        Date of creation.

    fit_date: str
        Date of last fit.

    skforcast_version: str
        Version of skforecast library used to create the forecaster.
     
    '''
    
    def __init__(
        self, 
        regressor, 
        fun_predictors: callable, 
        window_size: int,
        transformer_y = None,
        transformer_exog = None,
    ) -> None:
        
        self.regressor                     = regressor
        self.create_predictors             = fun_predictors
        self.source_code_create_predictors = None
        self.window_size                   = window_size
        self.transformer_y                 = transformer_y
        self.transformer_exog              = transformer_exog
        self.index_type                    = None
        self.index_freq                    = None
        self.training_range                = None
        self.last_window                   = None
        self.included_exog                 = False
        self.exog_type                     = None
        self.exog_col_names                = None
        self.X_train_col_names             = None
        self.in_sample_residuals           = None
        self.out_sample_residuals          = None
        self.fitted                        = False
        self.creation_date                 = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                      = None
        self.skforcast_version             = skforecast.__version__
        
        if not isinstance(window_size, int):
            raise Exception(
                f'`window_size` must be int, got {type(window_size)}'
            )

        if not callable(fun_predictors):
            raise Exception(
                f'`fun_predictors` must be callable, got {type(fun_predictors)}.'
            )
    
        self.source_code_create_predictors = getsource(fun_predictors)
                
        
    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterAutoregCustom object is printed.
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
            f"Predictors created with function: {self.create_predictors.__name__} \n"
            f"Window size: {self.window_size} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Included exogenous: {self.included_exog} \n"
            f"Type of exogenous variable: {self.exog_type} \n"
            f"Exogenous variables names: {self.exog_col_names} \n"
            f"Training range: {self.training_range.to_list() if self.fitted else None} \n"
            f"Training index type: {str(self.index_type).split('.')[-1][:-2] if self.fitted else None} \n"
            f"Training index frequency: {self.index_freq if self.fitted else None} \n"
            f"Regressor parameters: {params} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforcast_version} \n"
        )

        return info

    
    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series.
        
        Parameters
        ----------        
        y : pandas Series
            Training time series.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.


        Returns 
        -------
        X_train : pandas DataFrame
            Pandas DataFrame with the training values (predictors).
            
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.
        
        """
        
        if len(y) < self.window_size + 1:
            raise Exception(
                f'`y` must have as many values as the windows_size needed by '
                f'{self.create_predictors.__name__}. For this Forecaster the '
                f'minimum length is {self.window_size + 1}'
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
                raise Exception(
                    "`exog` must have same number of samples as `y`."
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
                raise Exception(
                    ('Different index for `y` and `exog`. They must be equal '
                    'to ensure the correct alignment of values.')      
                )
       
        X_train  = []
        y_train  = []

        for i in range(len(y) - self.window_size):

            train_index = np.arange(i, self.window_size + i)
            test_index  = self.window_size + i

            X_train.append(self.create_predictors(y=y_values[train_index]))
            y_train.append(y_values[test_index])
        
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        X_train_col_names = [f"custom_predictor_{i}" for i in range(X_train.shape[1])]

        if np.isnan(X_train).any():
            raise Exception(
                f"`create_predictors()` is returning `NaN` values."
            )
        
        if exog is not None:
            col_names_exog = exog.columns if isinstance(exog, pd.DataFrame) else [exog.name]
            X_train_col_names.extend(col_names_exog)
            # The first `self.window_size` positions have to be removed from exog
            # since they are not in X_train.
            X_train = np.column_stack((X_train, exog_values[self.window_size:, ]))

        X_train = pd.DataFrame(
                    data    = X_train,
                    columns = X_train_col_names,
                    index   = y_index[self.window_size: ]
                  )
        self.X_train_col_names = X_train_col_names
        y_train = pd.Series(
                    data  = y_train,
                    index = y_index[self.window_size: ],
                    name  = 'y'
                 )
                        
        return X_train, y_train

        
    def fit(
        self,
        y: pd.Series,
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None
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
        self.index_type           = None
        self.index_freq           = None
        self.last_window          = None
        self.included_exog        = False
        self.exog_type            = None
        self.exog_col_names       = None
        self.in_sample_residuals  = None
        self.X_train_col_names    = None
        self.fitted               = False
        self.training_range       = None
        
        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else exog.name
        
        X_train, y_train = self.create_train_X_y(y=y, exog=exog)
        
        if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
            self.regressor.fit(X=X_train, y=y_train)
        else:
            self.regressor.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
        
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = preprocess_y(y=y)[1][[0, -1]]
        self.index_type = type(X_train.index)
        if isinstance(X_train.index, pd.DatetimeIndex):
            self.index_freq = X_train.index.freqstr
        else: 
            self.index_freq = X_train.index.step

        if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
            residuals = y_train - self.regressor.predict(X_train)
        else:
            residuals = y_train - self.regressor.predict(X_train.to_numpy())
            
        residuals = pd.Series(
                        data  = residuals,
                        index = y_train.index,
                        name  = 'in_sample_residuals'
                    )

        if len(residuals) > 1000:
            # Only up to 1000 residuals are stored
            residuals = residuals.sample(n=1000, random_state=123, replace=False)
                                                  
        self.in_sample_residuals = residuals
        
        # The last time window of training data is stored so that predictors in
        # the first iteration of `predict()` can be calculated.
        self.last_window = y.iloc[-self.window_size:].copy()


    def _recursive_predict(
        self,
        steps: int,
        last_window: np.ndarray,
        exog: np.ndarray
    ) -> np.array:
        """
        Predict n steps ahead. It is an iterative process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
        last_window : numpy ndarray
            Values of the series used to create the predictors (lags) need in the 
            first iteration of prediction (t + 1).
            
        exog : numpy ndarray, pandas DataFrame
            Exogenous variable/s included as predictor/s.

        Returns 
        -------
        predictions : numpy ndarray
            Predicted values.
            
        """

        predictions = np.full(shape=steps, fill_value=np.nan)

        for i in range(steps):
            X = self.create_predictors(y=last_window).reshape(1, -1)
            if np.isnan(X).any():
                raise Exception(
                    f"`create_predictors()` is returning `NaN` values."
                )
            if exog is not None:
                X = np.column_stack((X, exog[i, ].reshape(1, -1)))

            with warnings.catch_warnings():
                # Suppress scikitlearn warning: "X does not have valid feature names,
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
        last_window: Optional[pd.Series]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> pd.Series:
        """
        Predict n steps ahead. It is an recursive process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
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

        check_predict_input(
            forecaster_type = type(self),
            steps           = steps,
            fitted          = self.fitted,
            included_exog   = self.included_exog,
            index_type      = self.index_type,
            index_freq      = self.index_freq,
            window_size     = self.window_size,
            last_window     = last_window,
            exog            = exog,
            exog_type       = self.exog_type,
            exog_col_names  = self.exog_col_names,
            interval        = None,
            max_steps       = None,
            level           = None,
            series_levels   = None
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
        
        if last_window is None:
            last_window = self.last_window.copy()
        
        last_window = transform_series(
                            series            = last_window,
                            transformer       = self.transformer_y,
                            fit               = False,
                            inverse_transform = False
                      )
        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )
            
        predictions = self._recursive_predict(
                        steps       = steps,
                        last_window = copy(last_window_values),
                        exog        = copy(exog_values)
                      )

        predictions = pd.Series(
                        data  = predictions,
                        index = expand_index(
                                    index = last_window_index,
                                    steps = steps
                                ),
                        name = 'pred'
                      )

        predictions = transform_series(
                        series            = predictions,
                        transformer       = self.transformer_y,
                        fit               = False,
                        inverse_transform = True
                      )

        return predictions

    
    
    def _estimate_boot_interval(
        self,
        steps: int,
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
            
        last_window : 1d numpy ndarray shape (, max_lag), default `None`
            Values of the series used to create the predictors (lags) needed in the 
            first iteration of prediction (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
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
                first column = lower bound of the interval.
                second column= upper bound interval of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
            
        """

        if last_window is None:
            last_window = self.last_window.values
        
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
                residuals = self.in_sample_residuals
            else:
                residuals = self.out_sample_residuals

            rng = np.random.default_rng(seed=seeds[i])
            sample_residuals = rng.choice(
                                    a       = residuals,
                                    size    = steps,
                                    replace = True
                               )

            for step in range(steps):
                prediction = self._recursive_predict(
                                steps       = 1,
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
        last_window: Optional[pd.Series]=None,
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
            
        last_window : pandas Series, default `None`
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
            Values predicted by the forecaster and their estimated interval:
                column pred = predictions.
                column lower_bound = lower bound of the interval.
                column upper_bound = upper bound interval of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
            
        """
        
        check_predict_input(
            forecaster_type = type(self),
            steps           = steps,
            fitted          = self.fitted,
            included_exog   = self.included_exog,
            index_type      = self.index_type,
            index_freq      = self.index_freq,
            window_size     = self.window_size,
            last_window     = last_window,
            exog            = exog,
            exog_type       = self.exog_type,
            exog_col_names  = self.exog_col_names,
            interval        = interval,
            max_steps       = None,
            level           = None,
            series_levels   = None
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
            
        if last_window is None:
            last_window = self.last_window.copy()
        
        last_window = transform_series(
                            series            = last_window,
                            transformer       = self.transformer_y,
                            fit               = False,
                            inverse_transform = False
                      )
        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )
        
        # Since during predict() `last_window` and `exog` are modified, the
        # originals are stored to be used later
        last_window_values_original = last_window_values.copy()
        if exog is not None:
            exog_values_original = exog_values.copy()
        else:
            exog_values_original = None
        
        predictions = self._recursive_predict(
                            steps       = steps,
                            last_window = last_window_values,
                            exog        = exog_values
                      )

        predictions_interval = self._estimate_boot_interval(
                                    steps       = steps,
                                    last_window = copy(last_window_values_original),
                                    exog        = copy(exog_values_original),
                                    interval    = interval,
                                    n_boot      = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                               )
        
        predictions = np.column_stack((predictions, predictions_interval))

        predictions = pd.DataFrame(
                        data = predictions,
                        index = expand_index(
                                    index = last_window_index,
                                    steps = steps
                                ),
                        columns = ['pred', 'lower_bound', 'upper_bound']
                      )
                      
        if self.transformer_y:
            for col in predictions.columns:
                predictions[col] = self.transformer_y.inverse_transform(predictions[[col]])

        return predictions

    
    def set_params(
        self, 
        **params: dict
    ) -> None:
        """
        Set new values to the parameters of the scikit learn model stored in the
        ForecasterAutoregCustom.
        
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
        
    
    def set_out_sample_residuals(
        self, 
        residuals: pd.Series, 
        append: bool=True,
        transform: bool=True
    )-> None:
        """
        Set new values to the attribute `out_sample_residuals`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process.
        
        Parameters
        ----------
        residuals : pd.Series
            Values of residuals. If len(residuals) > 1000, only a random sample
            of 1000 values are stored.
            
        append : bool, default `True`
            If `True`, new residuals are added to the once already stored in the
            attribute `out_sample_residuals`. Once the limit of 1000 values is
            reached, no more values are appended. If False, `out_sample_residuals`
            is overwritten with the new residuals.

        transform : bool, default `True`
            If `True`, new residuals are transformed using self.transformer_y.
            
        Returns 
        -------
        self

        """

        if not isinstance(residuals, pd.Series):
            raise Exception(
                f"`residuals` argument must be `pd.Series`. Got {type(residuals)}"
            )

        if transform and self.transformer_y is not None:
            residuals = transform_series(
                            series            = residuals,
                            transformer       = self.transformer_y,
                            fit               = False,
                            inverse_transform = False
                        ) 
            
        if len(residuals) > 1000:
            rng = np.random.default_rng(seed=123)
            residuals = rng.choice(a=residuals, size=1000, replace=False)
            residuals = pd.Series(residuals)   
    
        if append and self.out_sample_residuals is not None:
            free_space = max(0, 1000 - len(self.out_sample_residuals))
            if len(residuals) < free_space:
                residuals = np.hstack((
                                self.out_sample_residuals,
                                residuals
                            ))
            else:
                residuals = np.hstack((
                                self.out_sample_residuals,
                                residuals[:free_space]
                            ))

        self.out_sample_residuals = pd.Series(residuals)

    
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