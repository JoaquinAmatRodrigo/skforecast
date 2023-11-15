################################################################################
#                        ForecasterAutoregCustom                               #
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
import inspect

import skforecast
from ..ForecasterBase import ForecasterBase
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
from ..preprocessing import TimeSeriesDifferentiator

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class ForecasterAutoregCustom(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive (multi-step) forecaster with a custom function to create predictors.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    fun_predictors : Callable
        Function that receives a time series as input (numpy ndarray) and returns 
        another numpy ndarray with the predictors.
    window_size : int
        Size of the window needed by `fun_predictors` to create the predictors.
    name_predictors : list, default `None`
        Name of the predictors returned by `fun_predictors`. If `None`, predictors are
        named using the prefix 'custom_predictor_<i>' where `i` is the index of the position
        the predictor has in the returned array of `fun_predictors`.
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
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Diferentiarion is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
        **New in version 0.10.0**
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    regressor : regressor compatible with the scikit-learn API
        An instance of a regressor compatible with the scikit-learn API.
    fun_predictors : Callable
        Function that receives a time series as input (numpy ndarray) and returns
        another numpy ndarray with the predictors.
        **New in version 0.7.0**
    source_code_fun_predictors : str
        Source code of the custom function used to create the predictors.
        **New in version 0.7.0**
    window_size : int
        Size of the window needed by `fun_predictors` to create the predictors.
        If `differentiation` is not `None`, `window_size` is increased by the
        order of differentiation.
    name_predictors : list
        Name of the predictors returned by `fun_predictors`. If `None`, predictors are
        named using the prefix 'custom_predictor_<i>' where `i` is the index of the position
        the predictor has in the returned array of `fun_predictors`.
    transformer_y : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.
    transformer_exog : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Differentiation is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
        **New in version 0.10.0**
    source_code_weight_func : str
        Source code of the custom function used to create weights.
    last_window : pandas Series
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the values needed to predict the
        next step immediately after the training data. These values are stored
        in the original scale of the time series before undergoing any transformations
        or differentiation. When `differentiation` parameter is specified, the
        dimensions of the `last_window` are expanded as many values as the order
        of differentiation. For example, if `fun_predictors()` requires 7 lagged
        values and `differentiation=1`, `last_window` will contain 8 values.
    index_type : type
        Type of index of the input used in training.
    index_freq : str
        Frequency of Index of the input used in training.
    training_range : pandas Index
        First and last values of index of the data used during training.
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_type : type
        Type of exogenous data (pandas Series or DataFrame) used in training.
    exog_dtypes : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated after the transformation.
    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.
    X_train_col_names : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals : numpy ndarray
        Residuals of the model when predicting training data. Only stored up to
        1000 values. If `transformer_y` is not `None`, residuals are stored in the
        transformed scale.
    out_sample_residuals : numpy ndarray
        Residuals of the model when predicting non training data. Only stored
        up to 1000 values. If `transformer_y` is not `None`, residuals
        are assumed to be in the transformed scale. Use `set_out_sample_residuals` 
        method to set values.
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

    """
    
    def __init__(
        self, 
        regressor: object, 
        fun_predictors: Callable, 
        window_size: int,
        name_predictors: Optional[list]=None,
        transformer_y: Optional[object]=None,
        transformer_exog: Optional[object]=None,
        weight_func: Optional[Callable]=None,
        differentiation: Optional[int]=None,
        fit_kwargs: Optional[dict]=None,
        forecaster_id: Optional[Union[str, int]]=None
    ) -> None:
        
        self.regressor                  = regressor
        self.fun_predictors             = fun_predictors
        self.source_code_fun_predictors = None
        self.window_size                = window_size
        self.name_predictors            = name_predictors
        self.transformer_y              = transformer_y
        self.transformer_exog           = transformer_exog
        self.weight_func                = weight_func
        self.differentiation            = differentiation
        self.differentiator             = None
        self.source_code_weight_func    = None
        self.last_window                = None
        self.index_type                 = None
        self.index_freq                 = None
        self.training_range             = None
        self.included_exog              = False
        self.exog_type                  = None
        self.exog_dtypes                = None
        self.exog_col_names             = None
        self.X_train_col_names          = None
        self.in_sample_residuals        = None
        self.out_sample_residuals       = None
        self.fitted                     = False
        self.creation_date              = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                   = None
        self.skforecast_version         = skforecast.__version__
        self.python_version             = sys.version.split(" ")[0]
        self.forecaster_id              = forecaster_id
        
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(
                (f"Argument `window_size` must be an integer equal to or "
                 f"greater than 1. Got {window_size}.")
            )
        
        if self.differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    (f"Argument `differentiation` must be an integer equal to or "
                     f"greater than 1. Got {differentiation}.")
                )
            self.window_size += self.differentiation
            self.differentiator = TimeSeriesDifferentiator(order=self.differentiation)

        if not isinstance(fun_predictors, Callable):
            raise TypeError(
                f"Argument `fun_predictors` must be a Callable. Got {type(fun_predictors)}."
            )
            
        self.source_code_fun_predictors = inspect.getsource(fun_predictors)

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
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Regressor: {self.regressor} \n"
            f"Predictors created with function: {self.fun_predictors.__name__} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Window size: {self.window_size} \n"
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

    
    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series and exogenous
        variables.
        
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
            raise ValueError(
                (f"`y` must have as many values as the windows_size needed by "
                 f"{self.fun_predictors.__name__}. For this Forecaster the "
                 f"minimum length is {self.window_size + 1}")
            )

        check_y(y=y)
        y = transform_series(
                series            = y,
                transformer       = self.transformer_y,
                fit               = True,
                inverse_transform = False
            )
        y_values, y_index = preprocess_y(y=y)

        if self.differentiation is not None:
            y_values = self.differentiator.fit_transform(y_values)
        
        if exog is not None:
            if len(exog) != len(y):
                raise ValueError(
                    f'`exog` must have same number of samples as `y`. '
                    f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
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
            if not (exog_index[:len(y_index)] == y_index).all():
                raise ValueError(
                    ("Different index for `y` and `exog`. They must be equal "
                     "to ensure the correct alignment of values.")
                )
       
        X_train  = []
        y_train  = []

        for i in range(len(y) - self.window_size):

            train_index = np.arange(i, self.window_size + i)
            test_index  = self.window_size + i

            X_train.append(self.fun_predictors(y=y_values[train_index]))
            y_train.append(y_values[test_index])
        
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)

        if self.name_predictors is None:
            X_train_col_names = [f"custom_predictor_{i}" for i in range(X_train.shape[1])]
        else:
            if len(self.name_predictors) != X_train.shape[1]:
                raise ValueError(
                    ("The length of provided predictors names (`name_predictors`) do not "
                     "match the number of columns created by `fun_predictors()`.")
                )
            X_train_col_names = self.name_predictors.copy()

        if np.isnan(X_train).any():
            raise ValueError(
                "`fun_predictors()` is returning `NaN` values."
            )

        expected = self.fun_predictors(y_values[:-1])
        observed = X_train[-1, :]

        if expected.shape != observed.shape or not (expected == observed).all():
            raise ValueError(
                (f"The `window_size` argument ({self.window_size}), declared when "
                 f"initializing the forecaster, does not correspond to the window "
                 f"used by `fun_predictors()`.")
            )
        
        X_train = pd.DataFrame(
                      data    = X_train,
                      columns = X_train_col_names,
                      index   = y_index[self.window_size: ]
                  )
        
        if exog is not None:
            # The first `self.window_size` positions have to be removed from exog
            # since they are not in X_train.
            exog_to_train = exog.iloc[self.window_size:, ]
            exog_to_train.index = exog_index[self.window_size:]
            check_exog_dtypes(exog_to_train)
            X_train = pd.concat((X_train, exog_to_train), axis=1)
        
        self.X_train_col_names = X_train.columns.to_list()
        y_train = pd.Series(
                      data  = y_train,
                      index = y_index[self.window_size: ],
                      name  = 'y'
                  )

        return X_train, y_train


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
            Dataframe created with the `create_train_X_y` method, first return.

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
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        store_in_sample_residuals: bool=True
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the regressor 
        can be added with the `fit_kwargs` argument when initializing the forecaster.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned so
            that y[i] is regressed on exog[i].
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
        self.X_train_col_names   = None
        self.in_sample_residuals = None
        self.fitted              = False
        self.training_range      = None
        
        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else exog.name

        X_train, y_train = self.create_train_X_y(y=y, exog=exog)
        sample_weight = self.create_sample_weights(X_train=X_train)

        if sample_weight is not None:
            self.regressor.fit(X=X_train, y=y_train, sample_weight=sample_weight,
                               **self.fit_kwargs)
        else:
            self.regressor.fit(X=X_train, y=y_train, **self.fit_kwargs)

        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = preprocess_y(y=y, return_values=False)[1][[0, -1]]
        self.index_type = type(X_train.index)
        if isinstance(X_train.index, pd.DatetimeIndex):
            self.index_freq = X_train.index.freqstr
        else: 
            self.index_freq = X_train.index.step
        
        # This is done to save time during fit in functions such as backtesting()
        if store_in_sample_residuals:

            residuals = (y_train - self.regressor.predict(X_train)).to_numpy()

            if len(residuals) > 1000:
                # Only up to 1000 residuals are stored
                rng = np.random.default_rng(seed=123)
                residuals = rng.choice(
                                a       = residuals, 
                                size    = 1000, 
                                replace = False
                            )
                                                    
            self.in_sample_residuals = residuals
        
        # The last time window of training data is stored so that predictors in
        # the first iteration of `predict()` can be calculated. It also includes
        # the values need to calculate the diferenctiation.
        self.last_window = y.iloc[-self.window_size:].copy()


    def _recursive_predict(
        self,
        steps: int,
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
            if np.isnan(X).any():
                raise ValueError(
                    "`fun_predictors()` is returning `NaN` values."
                )
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
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        
        """

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
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )

        last_window = last_window.iloc[-self.window_size:].copy()

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
        
        last_window = transform_series(
                          series            = last_window,
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = False
                      )
        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )
        if self.differentiation is not None:
            last_window_values = self.differentiator.fit_transform(last_window_values)
            
        predictions = self._recursive_predict(
                          steps       = steps,
                          last_window = last_window_values,
                          exog        = exog_values
                      )
        
        if self.differentiation is not None:
            predictions = self.differentiator.inverse_transform_next_window(predictions)

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

    
    def predict_bootstrapping(
        self,
        steps: int,
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
        steps : int
            Number of future steps predicted.
        last_window : pandas Series, default `None`
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
        boot_predictions : pandas DataFrame
            Predictions generated by bootstrapping.
            Shape: (steps, n_boot)

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
        Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

        """
        
        if not in_sample_residuals and self.out_sample_residuals is None:
            raise ValueError(
                ("`forecaster.out_sample_residuals` is `None`. Use "
                 "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
                 "before `predict_interval()`, `predict_bootstrapping()`, "
                 "`predict_quantiles()` or `predict_dist()`.")
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
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )

        last_window = last_window.iloc[-self.window_size:].copy()

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
        
        last_window = transform_series(
                          series            = last_window,
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = False
                      )
        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )
        if self.differentiation is not None:
            last_window_values = self.differentiator.fit_transform(last_window_values)

        boot_predictions = np.full(
                               shape      = (steps, n_boot),
                               fill_value = np.nan,
                               dtype      = float
                           )
        rng = np.random.default_rng(seed=random_state)
        seeds = rng.integers(low=0, high=10000, size=n_boot)

        if in_sample_residuals:
            residuals = self.in_sample_residuals
        else:
            residuals = self.out_sample_residuals

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
                                 last_window = last_window_boot,
                                 exog        = exog_boot 
                             )

                prediction_with_residual  = prediction + sample_residuals[step]
                boot_predictions[step, i] = prediction_with_residual[0]

                last_window_boot = np.append(
                                       last_window_boot[1:],
                                       prediction_with_residual
                                   )
                
                if exog is not None:
                    exog_boot = exog_boot[1:]

            if self.differentiation is not None:
                boot_predictions[:, i] = (
                    self.differentiator.inverse_transform_next_window(boot_predictions[:, i])
                )

        boot_predictions = pd.DataFrame(
                               data    = boot_predictions,
                               index   = expand_index(last_window_index, steps=steps),
                               columns = [f"pred_boot_{i}" for i in range(n_boot)]
                           )

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
        steps: int,
        last_window: Optional[pd.Series]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        interval: list=[5, 95],
        n_boot: int=500,
        random_state: int=123,
        in_sample_residuals: bool=True
    ) -> pd.DataFrame:
        """
        Iterative process in which each prediction is used as a predictor
        for the next step, and bootstrapping is used to estimate prediction
        intervals. Both predictions and intervals are returned.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        last_window : pandas Series, default `None`
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
        steps: int,
        last_window: Optional[pd.Series]=None,
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
        last_window : pandas Series, default `None`
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
            prediction error to create prediction quantiles. If `False`, out of
            sample residuals are used. In the latter case, the user should have
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
        steps: int,
        distribution: object,
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
        steps : int
            Number of future steps predicted.
        distribution : Object
            A distribution object from scipy.stats.
        last_window : pandas Series, default `None`
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
        
    
    def set_out_sample_residuals(
        self, 
        residuals: np.ndarray, 
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
        residuals : numpy ndarray
            Values of residuals. If len(residuals) > 1000, only a random sample
            of 1000 values are stored.
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

        if not isinstance(residuals, np.ndarray):
            raise TypeError(
                f"`residuals` argument must be `numpy ndarray`. Got {type(residuals)}."
            )

        if not transform and self.transformer_y is not None:
            warnings.warn(
                (f"Argument `transform` is set to `False` but forecaster was trained "
                 f"using a transformer {self.transformer_y}. Ensure that the new residuals "
                 f"are already transformed or set `transform=True`.")
            )

        if transform and self.transformer_y is not None:
            warnings.warn(
                (f"Residuals will be transformed using the same transformer used "
                 f"when training the forecaster ({self.transformer_y}). Ensure that the "
                 f"new residuals are on the same scale as the original time series.")
            )

            residuals = transform_series(
                            series            = pd.Series(residuals, name='residuals'),
                            transformer       = self.transformer_y,
                            fit               = False,
                            inverse_transform = False
                        ).to_numpy()
            
        if len(residuals) > 1000:
            rng = np.random.default_rng(seed=random_state)
            residuals = rng.choice(a=residuals, size=1000, replace=False)
    
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

        self.out_sample_residuals = residuals

    
    def get_feature_importances(
        self
    ) -> pd.DataFrame:
        """
        Return feature importances of the regressor stored in the forecaster.
        Only valid when regressor stores internally the feature importances in the
        attribute `feature_importances_` or `coef_`. Otherwise, returns `None`.

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