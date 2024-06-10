################################################################################
#                            ForecasterAutoreg                                 #
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
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import clone
import inspect

import skforecast
from ..ForecasterBase import ForecasterBase
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
from ..utils import expand_index
from ..utils import transform_series
from ..utils import transform_dataframe
from ..preprocessing import TimeSeriesDifferentiator

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class ForecasterAutoreg(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API
    lags : int, list, numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1. 
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
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
        Differentiation is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
        **New in version 0.10.0**
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    binner_kwargs : dict, default `None`
        Additional arguments to pass to the `KBinsDiscretizer` used to discretize 
        the residuals into k bins according to the predicted values associated 
        with each residual. The `encode' argument is always set to 'ordinal' 
        and `dtype' to np.float64.
        **New in version 0.12.0**
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    lags : numpy ndarray
        Lags used as predictors.
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
    binner : sklearn.preprocessing.KBinsDiscretizer
        `KBinsDiscretizer` used to discretize residuals into k bins according 
        to the predicted values associated with each residual.
        **New in version 0.12.0**
    binner_kwargs : dict
        Additional arguments to pass to the `KBinsDiscretizer` used to discretize 
        the residuals into k bins according to the predicted values associated 
        with each residual. The `encode' argument is always set to 'ordinal' 
        and `dtype' to np.float64.
        **New in version 0.12.0**
    source_code_weight_func : str
        Source code of the custom function used to create weights.
    differentiation : int
        Order of differencing applied to the time series before training the 
        forecaster.
    differentiator : TimeSeriesDifferentiator
        Skforecast object used to differentiate the time series.
    max_lag : int
        Maximum value of lag included in `lags`.
    window_size : int
        Size of the window needed to create the predictors. It is equal to `max_lag`.
    window_size_diff : int
        Size of the window extended by the order of differentiation. When using
        differentiation, the `window_size` is increased by the order of differentiation
        so that the predictors can be created correctly.
    last_window : pandas Series
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the values needed to predict the
        next step immediately after the training data. These values are stored
        in the original scale of the time series before undergoing any transformations
        or differentiation. When `differentiation` parameter is specified, the
        dimensions of the `last_window` are expanded as many values as the order
        of differentiation. For example, if `lags` = 7 and `differentiation` = 1,
        `last_window` will have 8 values.
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
        Names of the exogenous variables used during training.
    X_train_col_names : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals : numpy ndarray
        Residuals of the model when predicting training data. Only stored up to
        1000 values. If `transformer_y` is not `None`, residuals are stored in the
        transformed scale.
    in_sample_residuals_by_bin : dict
        In sample residuals binned according to the predicted value each residual
        is associated with. Only stored up to 200 values per bin. If `transformer_y`
        is not `None`, residuals are stored in the transformed scale.
        **New in version 0.12.0**
    out_sample_residuals : numpy ndarray
        Residuals of the model when predicting non training data. Only stored
        up to 1000 values. If `transformer_y` is not `None`, residuals
        are assumed to be in the transformed scale. Use `set_out_sample_residuals` 
        method to set values.
    out_sample_residuals_by_bin : dict
        Out of sample residuals binned according to the predicted value each residual
        is associated with. Only stored up to 200 values per bin. If `transformer_y`
        is not `None`, residuals are assumed to be in the transformed scale.
        **New in version 0.12.0**
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
        lags: Union[int, np.ndarray, list],
        transformer_y: Optional[object]=None,
        transformer_exog: Optional[object]=None,
        weight_func: Optional[Callable]=None,
        differentiation: Optional[int]=None,
        fit_kwargs: Optional[dict]=None,
        binner_kwargs: Optional[dict]=None,
        forecaster_id: Optional[Union[str, int]]=None
    ) -> None:
        
        self.regressor                   = regressor
        self.transformer_y               = transformer_y
        self.transformer_exog            = transformer_exog
        self.weight_func                 = weight_func
        self.source_code_weight_func     = None
        self.differentiation             = differentiation
        self.differentiator              = None
        self.last_window                 = None
        self.index_type                  = None
        self.index_freq                  = None
        self.training_range              = None
        self.included_exog               = False
        self.exog_type                   = None
        self.exog_dtypes                 = None
        self.exog_col_names              = None
        self.X_train_col_names           = None
        self.in_sample_residuals         = None
        self.out_sample_residuals        = None
        self.in_sample_residuals_by_bin  = None
        self.out_sample_residuals_by_bin = None
        self.fitted                      = False
        self.creation_date               = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                    = None
        self.skforecast_version          = skforecast.__version__
        self.python_version              = sys.version.split(" ")[0]
        self.forecaster_id               = forecaster_id
       
        self.lags = initialize_lags(type(self).__name__, lags)
        self.max_lag = max(self.lags)
        self.window_size = self.max_lag
        self.window_size_diff = self.max_lag
        
        self.binner_kwargs = binner_kwargs
        if binner_kwargs is None:
            self.binner_kwargs = {
                'n_bins': 10, 'encode': 'ordinal', 'strategy': 'quantile',
                'subsample': 10000, 'random_state': 789654, 'dtype': np.float64
            }
        else:
            self.binner_kwargs = binner_kwargs
            self.binner_kwargs['encode'] = 'ordinal'
            self.binner_kwargs['dtype'] = np.float64
        self.binner = KBinsDiscretizer(**self.binner_kwargs)
        self.binner_intervals = None

        if self.differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    (f"Argument `differentiation` must be an integer equal to or "
                     f"greater than 1. Got {differentiation}.")
                )
            self.window_size_diff += self.differentiation
            self.differentiator = TimeSeriesDifferentiator(order=self.differentiation)
            
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
        Information displayed when a ForecasterAutoreg object is printed.
        """

        if isinstance(self.regressor, Pipeline):
            name_pipe_steps = tuple(name + "__" for name in self.regressor.named_steps.keys())
            params = {key : value for key, value in self.regressor.get_params().items() \
                      if key.startswith(name_pipe_steps)}
        else:
            params = self.regressor.get_params(deep=True)

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
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
        y : numpy ndarray
            1d numpy ndarray Training time series.

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
                 f"of the series ({len(y)}).")
            )
        
        X_data = np.full(shape=(n_splits, len(self.lags)), fill_value=np.nan, dtype=float)

        for i, lag in enumerate(self.lags):
            X_data[:, i] = y[self.max_lag - lag: -lag]

        y_data = y[self.max_lag:]
        
        return X_data, y_data


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
            Training values (predictors).
            Shape: (len(y) - self.max_lag, len(self.lags))
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.
            Shape: (len(y) - self.max_lag, )
        
        """
        
        check_y(y=y)
        fit_transformer = False if self.fitted else True
        y = transform_series(
                series            = y,
                transformer       = self.transformer_y,
                fit               = fit_transformer,
                inverse_transform = False
            )
        y_values, y_index = preprocess_y(y=y)

        if self.differentiation is not None:
            if not self.fitted:
                y_values = self.differentiator.fit_transform(y_values)
            else:
                differentiator = clone(self.differentiator)
                y_values = differentiator.fit_transform(y_values)
        
        if exog is not None:
            
            check_exog(exog=exog, allow_nan=True)
            if len(exog) != len(y):
                raise ValueError(
                    (f"`exog` must have same number of samples as `y`. "
                     f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
                )
            
            if isinstance(exog, pd.Series):
                exog = transform_series(
                           series            = exog,
                           transformer       = self.transformer_exog,
                           fit               = fit_transformer,
                           inverse_transform = False
                       )
            else:
                exog = transform_dataframe(
                           df                = exog,
                           transformer       = self.transformer_exog,
                           fit               = fit_transformer,
                           inverse_transform = False
                       )
            
            check_exog_dtypes(exog, call_check_exog=True)

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
                      index   = y_index[self.max_lag: ]
                  )
        
        if exog is not None:
            # The first `self.max_lag` positions have to be removed from exog
            # since they are not in X_train.
            exog_to_train = exog.iloc[self.max_lag:, ]
            exog_to_train.index = exog_index[self.max_lag:]
            X_train = pd.concat((X_train, exog_to_train), axis=1)
        
        # TODO: move self to fit method and make X_train_col_names a return
        if not self.fitted:
            self.X_train_col_names = X_train.columns.to_list()
        
        y_train = pd.Series(
                      data  = y_train,
                      index = y_index[self.max_lag: ],
                      name  = 'y'
                  )

        if self.differentiation is not None:
            X_train = X_train.iloc[self.differentiation: ]
            y_train = y_train.iloc[self.differentiation: ]
                        
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
        store_last_window: bool=True,
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
        store_last_window : bool, default `True`
            Whether or not to store the last window of training data.
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

        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_dtypes = get_exog_dtypes(exog=exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else exog.name
        
        # This is done to save time during fit in functions such as backtesting()
        if store_in_sample_residuals:
            in_sample_predictions = pd.Series(
                                        data  = self.regressor.predict(X_train),
                                        index = X_train.index
                                    )
            self._binning_in_sample_residuals(
                y_true = y_train,
                y_pred = in_sample_predictions
            )
        
        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated. It
        # also includes the values need to calculate the diferenctiation.
        if store_last_window:
            self.last_window = y.iloc[-self.window_size_diff:].copy()


    def _binning_in_sample_residuals(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> None:
        """
        Binning residuals according to the predicted value each residual is
        associated with. First a sklearn.preprocessing.KBinsDiscretizer
        is fitted to the predicted values. Then, residuals are binned according
        to the predicted value each residual is associated with. Residuals are
        stored in the forecaster object as `in_sample_residuals` and
        `in_sample_residuals_by_bin`. Only up to 200 residuals are stored per bin.

        Parameters
        ----------
        y_true : pandas Series
            True values of the time series.
        y_pred : pandas Series
            Predicted values of the time series.     

        Returns
        -------
        None
        
        """

        y_pred = y_pred.rename('prediction')
        residuals = (y_true - y_pred).rename('residual')
        data = pd.merge(
                   residuals,
                   y_pred,
                   left_index  = True,
                   right_index = True
               )
        self.binner.fit(data[['prediction']].to_numpy())
        data['bin'] = self.binner.transform(data[['prediction']].to_numpy()).astype(int)
        self.in_sample_residuals_by_bin = (
            data.groupby('bin')['residual'].apply(np.array).to_dict()
        )

        # Only up to 200 residuals are stored per bin
        for k, v in self.in_sample_residuals_by_bin.items():
            # TODO: Include `random_state` in fit method to allow the user 
            # change the residual sample stored.
            rng = np.random.default_rng(seed=95123)
            if len(v) > 200:
                sample = rng.choice(a=v, size=200, replace=False)
                self.in_sample_residuals_by_bin[k] = sample

        self.in_sample_residuals = np.concatenate(list(
            self.in_sample_residuals_by_bin.values()
        ))

        self.binner_intervals = {
            i: (
                self.binner.bin_edges_[0][i],
                (
                    self.binner.bin_edges_[0][i + 1]
                    if i + 1 < len(self.binner.bin_edges_[0])
                    else None
                ),
            )
            for i in range(len(self.binner.bin_edges_[0]) - 1)
        }


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
        last_window = np.concatenate((last_window, predictions))

        for i in range(steps):

            X = last_window[-self.lags - (steps - i)].reshape(1, -1)
            
            if exog is not None:
                X = np.column_stack((X, exog[i, ].reshape(1, -1)))
            with warnings.catch_warnings():
                # Suppress scikit-learn warning: "X does not have valid feature names,
                # but NoOpTransformer was fitted with feature names".
                warnings.simplefilter("ignore")
                prediction = self.regressor.predict(X).ravel()[0]
                predictions[i] = prediction

            # Update `last_window` values. The first position is discarded and 
            # the new prediction is added at the end.
            last_window[-(steps - i)] = prediction

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
            window_size      = self.window_size_diff,
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

        last_window = last_window.iloc[-self.window_size_diff:].copy()

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
        n_boot: int=250,
        random_state: int=123,
        in_sample_residuals: bool=True,
        binned_residuals: bool=False,
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
        binned_residuals : bool, default `False`
            If `True`, residuals used in each bootstrapping iteration are selected
            conditioning on the predicted values. If `False`, residuals are selected
            randomly without conditioning on the predicted values.
            **WARNING: This argument is newly introduced and requires special attention.
            It is still experimental and may undergo changes.**
            **New in version 0.12.0**

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

        # TODO: Move to check_predict_input(), validate why it was not there.
        if not in_sample_residuals:
            if not binned_residuals and self.out_sample_residuals is None:
                raise ValueError(
                    ("`forecaster.out_sample_residuals` is `None`. Use "
                     "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
                     "before `predict_interval()`, `predict_bootstrapping()`, "
                     "`predict_quantiles()` or `predict_dist()`.")
                )
            if binned_residuals and self.out_sample_residuals_by_bin is None:
                raise ValueError(
                    ("`forecaster.out_sample_residuals_by_bin` is `None`. Use "
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
            window_size      = self.window_size_diff,
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

        last_window = last_window.iloc[-self.window_size_diff:]

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
            residuals_by_bin = self.in_sample_residuals_by_bin
        else:
            residuals = self.out_sample_residuals
            residuals_by_bin = self.out_sample_residuals_by_bin

        for i in range(n_boot):
            # In each bootstraping iteration the initial last_window and exog 
            # need to be restored.
            last_window_boot = last_window_values.copy()
            exog_boot = exog_values.copy() if exog is not None else None

            rng = np.random.default_rng(seed=seeds[i])
            if not binned_residuals:
                sampled_residuals = rng.choice(
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
                if binned_residuals:
                    predicted_bin = (
                        self.binner.transform(prediction.reshape(1, -1)).astype(int)[0][0]
                    )
                    sampled_residual = rng.choice(a=residuals_by_bin[predicted_bin], size=1)
                else:
                    sampled_residual = sampled_residuals[step]

                prediction_with_residual  = prediction + sampled_residual
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
        n_boot: int=250,
        random_state: int=123,
        in_sample_residuals: bool=True,
        binned_residuals: bool=False
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
        binned_residuals : bool, default `False`
            If `True`, residuals used in each bootstrapping iteration are selected
            conditioning on the predicted values. If `False`, residuals are selected
            randomly without conditioning on the predicted values.
            **WARNING: This argument is newly introduced and requires special attention.
            It is still experimental and may undergo changes.**
            **New in version 0.12.0**

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
                               in_sample_residuals = in_sample_residuals,
                               binned_residuals    = binned_residuals
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
        n_boot: int=250,
        random_state: int=123,
        in_sample_residuals: bool=True,
        binned_residuals: bool=False
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
        binned_residuals : bool, default `False`
            If `True`, residuals used in each bootstrapping iteration are selected
            conditioning on the predicted values. If `False`, residuals are selected
            randomly without conditioning on the predicted values.
            **WARNING: This argument is newly introduced and requires special attention.
            It is still experimental and may undergo changes.**
            **New in version 0.12.0**

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
                               in_sample_residuals = in_sample_residuals,
                               binned_residuals    = binned_residuals
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
        n_boot: int=250,
        random_state: int=123,
        in_sample_residuals: bool=True,
        binned_residuals: bool=False
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
        binned_residuals : bool, default `False`
            If `True`, residuals used in each bootstrapping iteration are selected
            conditioning on the predicted values. If `False`, residuals are selected
            randomly without conditioning on the predicted values.
            **WARNING: This argument is newly introduced and requires special attention.
            It is still experimental and may undergo changes.**
            **New in version 0.12.0**

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
                           in_sample_residuals = in_sample_residuals,
                           binned_residuals    = binned_residuals
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

        
    def set_lags(
        self, 
        lags: Union[int, list, np.ndarray, range]
    ) -> None:
        """
        Set new value to the attribute `lags`. Attributes `max_lag`, 
        `window_size` and  `window_size_diff` are also updated.
        
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
        self.max_lag = max(self.lags)
        self.window_size = max(self.lags)
        self.window_size_diff = max(self.lags)
        if self.differentiation is not None:
            self.window_size_diff += self.differentiation        


    def set_out_sample_residuals(
        self, 
        residuals: Union[pd.Series, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]]=None,
        append: bool=True,
        transform: bool=True,
        random_state: int=123
    )-> None:
        """
        Set new values to the attribute `out_sample_residuals`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process. If `y_pred` is provided, residuals
        are binned according to the predicted value they are associated with. If
        `y_pred` is `None`, residuals are stored without binning. Only up to 200
        residuals are stored per bin.
        
        Parameters
        ----------
        residuals : pandas Series, numpy ndarray
            Values of residuals. If `y_pred` is `None`, at most 1000 values are
            stored. If `y_pred` is not `None`, at most 200 * n_bins values are
            stored, where `n_bins` is the number of bins used in `self.binner`.
        y_pred : pandas Series, numpy ndarray, default `None`
            Predicted values of the time series from which the residuals have been
            calculated. This argument is used to bin residuals according to the
            predicted values. `y_pred` and `residuals` must be of the same class
            (both pandas Series or both numpy ndarray) must have the same length
            and, if they are pandas Series, the same index. 
            
            - If `y_pred` is `None`, residuals are not binned.
            - If affter binning, a bin has more than 200 residuals, only a random
                sample of 200 residuals is stored.
            - If affter binning, a bin binning is empty, it is filled with a
            random sample of residuals from other bins. This is done to ensure
            that all bins have at least one residual and can be used in the
            prediction process.
            **New in version 0.12.0**
        append : bool, default `True`
            If `True`, new residuals are added to the once already stored in the
            forecaster. Once the limit of 200 values per bin is reached, no more values
            are appended. If False, stored residuals are overwritten with the new
            residuals.
        transform : bool, default `True`
            If `True`, new residuals are transformed using self.transformer_y.
        random_state : int, default `123`
            Sets a seed to the random sampling for reproducible output.

        Returns
        -------
        None

        """

        if not isinstance(residuals, (np.ndarray, pd.Series)):
            raise TypeError(
                (f"`residuals` argument must be `numpy ndarray` or `pandas Series`, "
                 f"but found {type(residuals)}.")
            )

        if not isinstance(y_pred, (np.ndarray, pd.Series, type(None))):
            raise TypeError(
                (f"`y_pred` argument must be `numpy ndarray`, `pandas Series` or `None`, "
                 f"but found {type(y_pred)}.")
            )

        if y_pred is not None and len(residuals) != len(y_pred):
            raise ValueError(
                (f"`residuals` and `y_pred` must have the same length, but found "
                 f"{len(residuals)} and {len(y_pred)}.")
            )

        if isinstance(residuals, pd.Series) and isinstance(y_pred, pd.Series):
            if not residuals.index.equals(y_pred.index):
                raise ValueError(
                    (f"`residuals` and `y_pred` must have the same index, but found "
                     f"{residuals.index} and {y_pred.index}.")
                )

        if y_pred is not None and not self.fitted:
            raise NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `set_out_sample_residuals()`.")
            )

        if isinstance(residuals, np.ndarray):
            residuals = pd.Series(residuals, name='residuals')
        else:
            residuals = residuals.rename('residuals').reset_index(drop=True)

        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, name='prediction')
        elif isinstance(y_pred, pd.Series):
            y_pred = y_pred.rename('prediction').reset_index(drop=True)

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
                            series            = residuals,
                            transformer       = self.transformer_y,
                            fit               = False,
                            inverse_transform = False
                        ).to_numpy()
        
        if y_pred is None:
            # Residuals are not binned.
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
        else:
            # Residuals are binned according to the predicted values.
            data = pd.merge(
                       residuals,
                       y_pred,
                       left_index  = True,
                       right_index = True
                   )
            data['bin'] = self.binner.transform(data[['prediction']].to_numpy()).astype(int)
            residuals_by_bin = data.groupby('bin')['residuals'].apply(np.array).to_dict()

            if append and self.out_sample_residuals_by_bin is not None:
                for k, v in residuals_by_bin.items():
                    if k in self.out_sample_residuals_by_bin:
                        free_space = max(0, 200 - len(self.out_sample_residuals_by_bin[k]))
                        if len(v) < free_space:
                            self.out_sample_residuals_by_bin[k] = np.hstack((
                                self.out_sample_residuals_by_bin[k],
                                v
                            ))
                        else:
                            self.out_sample_residuals_by_bin[k] = np.hstack((
                                self.out_sample_residuals_by_bin[k],
                                v[:free_space]
                            ))
                    else:
                        self.out_sample_residuals_by_bin[k] = v
            else:
                self.out_sample_residuals_by_bin = residuals_by_bin

            for k, v in self.out_sample_residuals_by_bin.items():
                rng = np.random.default_rng(seed=123)
                if len(v) > 200:
                    # Only up to 200 residuals are stored per bin
                    sample = rng.choice(a=v, size=200, replace=False)
                    self.out_sample_residuals_by_bin[k] = sample
                
            self.out_sample_residuals = np.concatenate(list(
                                            self.out_sample_residuals_by_bin.values()
                                        ))

            for k in self.in_sample_residuals_by_bin.keys():
                if k not in self.out_sample_residuals_by_bin:
                    self.out_sample_residuals_by_bin[k] = np.array([])
            
            empty_bins = [k for k, v in self.out_sample_residuals_by_bin.items() if len(v) == 0]
            if empty_bins:
                warnings.warn(
                    (f"The following bins have no out of sample residuals: {empty_bins}. "
                     f"No predicted values fall in the interval "
                     f"{[self.binner_intervals[bin] for bin in empty_bins]}. "
                     f"Empty bins will be filled with a random sample of residuals from "
                     f"the other bins.")
                )
                for k in empty_bins:
                    rng = np.random.default_rng(seed=123)
                    self.out_sample_residuals_by_bin[k] = rng.choice(
                                                              a       = self.out_sample_residuals,
                                                              size    = 200,
                                                              replace = True
                                                          )


    def get_feature_importances(
        self,
        sort_importance: bool=True
    ) -> pd.DataFrame:
        """
        Return feature importances of the regressor stored in the forecaster.
        Only valid when regressor stores internally the feature importances in the
        attribute `feature_importances_` or `coef_`. Otherwise, returns `None`.

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
            raise NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
            )

        if isinstance(self.regressor, Pipeline):
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