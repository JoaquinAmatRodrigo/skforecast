################################################################################
#                        ForecasterRnn                                         #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under the BSD 3-Clause License.                                              #
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
from copy import deepcopy

import skforecast
from ..ForecasterBase import ForecasterBase
from ..exceptions import IgnoredArgumentWarning
from ..utils import initialize_weights
from ..utils import check_select_fit_kwargs
from ..utils import check_y
from ..utils import check_predict_input
from ..utils import check_interval
from ..utils import preprocess_y
from ..utils import preprocess_last_window
from ..utils import expand_index
from ..utils import transform_series
from ..utils import transform_dataframe

logging.basicConfig(
    format="%(name)-10s %(levelname)-5s %(message)s",
    level=logging.INFO,
)


class ForecasterRnn(ForecasterBase):
    """
    This class turns any regressor compatible with the TensorFlow API into a
    TensorFlow RNN multi-serie multi-step forecaster. A unique model is created
    to forecast all time steps and series. See documentation for more details.

    Parameters
    ----------
    regressor : regressor or pipeline compatible with the TensorFlow API
        An instance of a regressor or pipeline compatible with the TensorFlow API.
    levels : str, list, default `None`
        Name of one or more time series to be predicted. This determine the series
        the forecaster will be handling. If `None`, all series used during training
        will be available for prediction.
    steps : Ignored
        Not used, present here for API consistency by convention.
    lags  : Ignored
        Not used, present here for API consistency by convention. 
    transformer_series : object, dict, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and
        inverse_transform. Transformation is applied to each `series` before training
        the forecaster. ColumnTransformers are not allowed since they do not have
        inverse_transform method.
        
            - If single transformer: it is cloned and applied to all series.
            - If `dict` of transformers: a different transformer can be used for each
            series.
    transformer_exog : Ignored
        Not used, present here for API consistency by convention. 
    weight_func : Ignored
        Not used, present here for API consistency by convention. 
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    n_jobs : Ignored
        Not used, present here for API consistency by convention.
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.

    Attributes
    ----------
    regressor : regressor or pipeline compatible with the TensorFlow API
        An instance of a regressor or pipeline compatible with the TensorFlow API.
        An instance of this regressor is trained for each step. All of them
        are stored in `self.regressors_`.
    levels : str, list, default `None`
        Name of one or more time series to be predicted. This determine the series
        the forecaster will be handling. If `None`, all series used during training
        will be available for prediction.
    steps : int
        Number of future steps the forecaster will predict when using method
        `predict()`. Since a different model is created for each step, this value
        should be defined before training.
    lags : numpy ndarray
        Lags used as predictors.
    transformer_series : object, dict, default `None`
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
    transformer_exog : Ignored
        Not used, present here for API consistency by convention.
    max_lag : int
        Maximum value of lag included in `lags`.
    window_size : int
        Size of the window needed to create the predictors.
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
    X_train_dim_names : dict
        Labels for the multi-dimensional arrays created internally for training.
    y_train_dim_names : dict
        Labels for the multi-dimensional arrays created internally for training.
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
    skforcast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    history : dict
        Dictionary with the history of the training of each step. It is created
        internally to avoid overwriting.

    Notes
    -----


    """

    def __init__(
        self,
        regressor: object,
        levels: Union[str, list],
        transformer_series: Optional[Union[object, dict]] = None,
        weight_func: Optional[Callable] = None,
        fit_kwargs: Optional[dict] = None,
        forecaster_id: Optional[Union[str, int]] = None,
        n_jobs=None,
        step=None,
        lags=None,
        transformer_exog=None,
    ) -> None:
        self.levels = levels
        self.transformer_series = transformer_series
        self.transformer_series_ = None
        self.transformer_exog = None
        self.weight_func = weight_func
        self.source_code_weight_func = None
        self.max_lag = None
        self.window_size = None
        self.last_window = None
        self.index_type = None
        self.index_freq = None
        self.training_range = None
        self.included_exog = False
        self.exog_type = None
        self.exog_dtypes = None
        self.exog_col_names = None
        self.series_col_names = None
        self.X_train_dim_names = None
        self.y_train_dim_names = None
        self.fitted = False
        self.creation_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        self.fit_date = None
        self.skforcast_version = skforecast.__version__
        self.python_version = sys.version.split(" ")[0]
        self.forecaster_id = forecaster_id
        self.history = None

        # Infer parameters from the model
        self.regressor = regressor
        layer_init = self.regressor.layers[0]
        self.lags = np.arange(layer_init.input_shape[0][1]) + 1
        self.max_lag = np.max(self.lags)
        self.window_size = self.max_lag
        self.series = layer_init.input_shape[0][2]
        layer_end = self.regressor.layers[-1]
        self.steps = layer_end.output_shape[1]
        self.outputs = layer_end.output_shape[-1]

        if not isinstance(levels, (list, str, type(None))):
            raise TypeError(
                f"`levels` argument must be a string, list or. Got {type(levels)}."
            )

        if isinstance(levels, str):
            self.levels = [levels]

        self.series_val = None
        if "series_val" in fit_kwargs:
            self.series_val = fit_kwargs["series_val"]
            fit_kwargs.pop("series_val")

        self.fit_kwargs = check_select_fit_kwargs(
            regressor=self.regressor, fit_kwargs=fit_kwargs
        )


    def __repr__(self) -> str:
        """
        Information displayed when a ForecasterRnn object is printed.
        """

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            name_pipe_steps = tuple(
                name + "__" for name in self.regressor.named_steps.keys()
            )
            params = {
                key: value
                for key, value in self.regressor.get_params().items()
                if key.startswith(name_pipe_steps)
            }
        else:
            params = self.regressor.get_params()

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
            f"Transformer for series: {self.transformer_series} \n"
            f"Window size: {self.window_size} \n"
            f"Target series, levels: {self.levels} \n"
            f"Multivariate series (names): {self.series_col_names} \n"
            f"Maximum steps predicted: {self.steps} \n"
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

    def _create_lags(
        self,
        y: np.ndarray,
        lags: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a 1d array into a 3d array (X) and a 3d array (y). Each row
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
            3d numpy ndarray with the lagged values (predictors).
            Shape: (samples - max(self.lags), len(self.lags))
        y_data : numpy ndarray
            3d numpy ndarray with the values of the time series related to each
            row of `X_data` for each step.
            Shape: (len(self.steps), samples - max(self.lags))

        """

        n_splits = len(y) - self.max_lag - (self.steps - 1)  # rows of y_data
        if n_splits <= 0:
            raise ValueError(
                (
                    f"The maximum lag ({self.max_lag}) must be less than the length "
                    f"of the series minus the number of steps ({len(y)-(self.steps-1)})."
                )
            )

        X_data = np.full(shape=(n_splits, len(lags)), fill_value=np.nan, dtype=float)
        for i, lag in enumerate(lags):
            X_data[:, i] = y[self.max_lag - lag : -(lag + self.steps - 1)]

        y_data = np.full(shape=(n_splits, self.steps), fill_value=np.nan, dtype=float)
        for step in range(self.steps):
            y_data[:, step] = y[self.max_lag + step : self.max_lag + step + n_splits]

        return X_data, y_data

    def create_train_X_y(
        self, series: pd.DataFrame, exog=None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Create training matrices. The resulting multi-dimensional matrices contain
        the target variable and predictors needed to train the model.

        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        exog : Ignored
            Not used, present here for API consistency by convention. This type of
            forecaster does not allow exogenous variables.
            
        Returns
        -------
        X_train : np.ndarray
            Training values (predictors) for each step. The resulting array has
            3 dimensions: (time_points, n_lags, n_series)
        y_train : np.ndarray
            Values (target) of the time series related to each row of `X_train`.
            The resulting array has 3 dimensions: (time_points, n_steps, n_levels)
        dimension_names : dict
            Labels for the multi-dimensional arrays created internally for training.

        """
        if not isinstance(series, pd.DataFrame):
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}.")

        series_col_names = list(series.columns)

        if not set(self.levels).issubset(set(series.columns)):
            raise ValueError(
                (
                    f"`levels` defined when initializing the forecaster must be included "
                    f"in `series` used for trainng. {set(self.levels) - set(series.columns)} "
                    f"not found."
                )
            )

        if len(series) < self.max_lag + self.steps:
            raise ValueError(
                (
                    f"Minimum length of `series` for training this forecaster is "
                    f"{self.max_lag + self.steps}. Got {len(series)}. Reduce the "
                    f"number of predicted steps, {self.steps}, or the maximum "
                    f"lag, {self.max_lag}, if no more data is available."
                )
            )

        if self.transformer_series is None:
            self.transformer_series_ = {serie: None for serie in series_col_names}
        elif not isinstance(self.transformer_series, dict):
            self.transformer_series_ = {
                serie: clone(self.transformer_series) for serie in series_col_names
            }
        else:
            self.transformer_series_ = {serie: None for serie in series_col_names}
            # Only elements already present in transformer_series_ are updated
            self.transformer_series_.update(
                (k, v)
                for k, v in deepcopy(self.transformer_series).items()
                if k in self.transformer_series_
            )
            series_not_in_transformer_series = set(series.columns) - set(
                self.transformer_series.keys()
            )
            if series_not_in_transformer_series:
                warnings.warn(
                    (
                        f"{series_not_in_transformer_series} not present in `transformer_series`."
                        f" No transformation is applied to these series."
                    ),
                    IgnoredArgumentWarning,
                )

        # Step 1: Create lags for all columns
        X_train = []
        y_train = []

        for i, serie in enumerate(series.columns):
            x = series[serie]
            check_y(y=x)
            x = transform_series(
                series=x,
                transformer=self.transformer_series_[serie],
                fit=True,
                inverse_transform=False,
            )
            X, _ = self._create_lags(x, self.lags)
            X_train.append(X)

        for i, serie in enumerate(self.levels):
            y = series[serie]
            check_y(y=y)
            y = transform_series(
                series=y,
                transformer=self.transformer_series_[serie],
                fit=True,
                inverse_transform=False,
            )

            _, y = self._create_lags(y, self.lags)
            y_train.append(y)

        X_train = np.stack(X_train, axis=2)
        y_train = np.stack(y_train, axis=2)

        train_index = series.index.to_list()[self.max_lag:(len(series.index.to_list()) - self.steps+1)]
        dimension_names = {
            "X_train": {
                0:train_index, 
                1:["lag_" + str(l) for l in self.lags], 
                2:series.columns.to_list()
            }, 
            "y_train": {
                0:train_index, 
                1:["step_" + str(l) for l in np.array(range(self.steps))+1], 
                2:self.levels
            }
        }

        return X_train, y_train, dimension_names

    def fit(
        self, series: pd.DataFrame, exog=None, store_in_sample_residuals: bool = True
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the regressor
        can be added with the `fit_kwargs` argument when initializing the forecaster.

        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        exog : Ignored
            Not used, present here for API consistency by convention. 
        store_in_sample_residuals : bool, default `True`
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting.

        Returns
        -------
        None

        """

        # Reset values in case the forecaster has already been fitted.
        self.index_type = None
        self.index_freq = None
        self.last_window = None
        self.included_exog = None
        self.exog_type = None
        self.exog_dtypes = None
        self.exog_col_names = None
        self.series_col_names = None
        self.X_train_dim_names = None
        self.y_train_dim_names = None
        self.in_sample_residuals = None
        self.fitted = False
        self.training_range = None
        
        self.series_col_names = list(series.columns)
        X_train, y_train, X_train_dim_names = self.create_train_X_y(series=series)
        self.X_train_dim_names = X_train_dim_names["X_train"]
        self.y_train_dim_names = X_train_dim_names["y_train"]
        if self.series_val is not None:
            X_val, y_val, _ = self.create_train_X_y(series=self.series_val)
            history = self.regressor.fit(
                x=X_train, y=y_train, validation_data=(X_val, y_val), **self.fit_kwargs
            )
        else:
            history = self.regressor.fit(x=X_train, y=y_train, **self.fit_kwargs)

        self.history = history.history
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        _, y_index = preprocess_y(y=series[self.levels], return_values=False)
        self.training_range = y_index[[0, -1]]
        self.index_type = type(y_index)
        if isinstance(y_index, pd.DatetimeIndex):
            self.index_freq = y_index.freqstr
        else:
            self.index_freq = y_index.step

        self.last_window = series.iloc[-self.max_lag :].copy()

    def predict(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog=None,
        levels: Any = None,
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
        exog : Ignored
            Not used, present here for API consistency by convention.
        levels : str, list, default `None`
            Name of one or more time series to be predicted. It must be included
            in `levels` defined when initializing the forecaster. If `None`, all
            all series used during training will be available for prediction.
        Returns
        -------
        predictions : pandas DataFrame
            Predicted values.

        """

        if levels is None:
            levels = self.levels
        elif isinstance(levels, str):
            levels = [levels]
        if isinstance(steps, int):
            steps = list(np.arange(steps) + 1)
        elif steps is None:
            steps = list(np.arange(self.steps) + 1)
        elif isinstance(steps, list):
            steps = list(np.array(steps))

        for step in steps:
            if not isinstance(step, (int, np.int64, np.int32)):
                raise TypeError(
                    (
                        f"`steps` argument must be an int, a list of ints or `None`. "
                        f"Got {type(steps)}."
                    )
                )

        if last_window is None:
            last_window = self.last_window.copy()
        else:
            last_window = last_window.iloc[-self.window_size:].copy()
            

        check_predict_input(
            forecaster_name=type(self).__name__,
            steps=steps,
            fitted=self.fitted,
            included_exog=None,
            index_type=self.index_type,
            index_freq=self.index_freq,
            window_size=self.window_size,
            last_window=last_window,
            last_window_exog=None,
            exog=None,
            exog_type=None,
            exog_col_names=None,
            interval=None,
            alpha=None,
            max_steps=self.steps,
            levels=levels,
            series_col_names=self.series_col_names,
        )

        for serie_name in self.series_col_names:
            last_window_serie = transform_series(
                series=last_window[serie_name],
                transformer=self.transformer_series_[serie_name],
                fit=False,
                inverse_transform=False,
            )
            last_window_values, last_window_index = preprocess_last_window(
                last_window=last_window_serie
            )
            last_window.loc[:, serie_name] = last_window_values

        X = np.reshape(last_window.to_numpy(), (1, self.max_lag, last_window.shape[1]))
        predictions = self.regressor.predict(X, verbose=0)
        predictions_reshaped = np.reshape(
            predictions, (predictions.shape[1], predictions.shape[2])
        )
        idx = expand_index(index=last_window_index, steps=max(steps))
        predictions = pd.DataFrame(
            data=predictions_reshaped[np.array(steps) - 1],
            columns=[self.levels],
            index=idx[np.array(steps) - 1],
        )
        predictions = predictions[levels]

        for serie in levels:
            x = predictions[serie].squeeze()  # TODO check
            check_y(y=x)
            x = transform_series(
                series=x,
                transformer=self.transformer_series_[serie],
                fit=False,
                inverse_transform=True,
            )
            predictions.loc[:, serie] = x

        return predictions

    # def predict_bootstrapping(
    #     self,
    #     steps: Optional[Union[int, list]] = None,
    #     last_window: Optional[pd.DataFrame] = None,
    #     exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    #     n_boot: int = 500,
    #     random_state: int = 123,
    #     in_sample_residuals: bool = True,
    #     levels: Any = None,
    # ) -> pd.DataFrame:
    #     """
    #     Generate multiple forecasting predictions using a bootstrapping process.
    #     By sampling from a collection of past observed errors (the residuals),
    #     each iteration of bootstrapping generates a different set of predictions.
    #     See the Notes section for more information.

    #     Parameters
    #     ----------
    #     steps : int, list, None, default `None`
    #         Predict n steps. The value of `steps` must be less than or equal to the
    #         value of steps defined when initializing the forecaster. Starts at 1.

    #             - If `int`: Only steps within the range of 1 to int are predicted.
    #             - If `list`: List of ints. Only the steps contained in the list
    #             are predicted.
    #             - If `None`: As many steps are predicted as were defined at
    #             initialization.
    #     last_window : pandas DataFrame, default `None`
    #         Series values used to create the predictors (lags) needed in the
    #         first iteration of the prediction (t + 1).
    #         If `last_window = None`, the values stored in` self.last_window` are
    #         used to calculate the initial predictors, and the predictions start
    #         right after training data.
    #     exog : pandas Series, pandas DataFrame, default `None`
    #         Exogenous variable/s included as predictor/s.
    #     n_boot : int, default `500`
    #         Number of bootstrapping iterations used to estimate prediction
    #         intervals.
    #     random_state : int, default `123`
    #         Sets a seed to the random generator, so that boot intervals are always
    #         deterministic.
    #     in_sample_residuals : bool, default `True`
    #         If `True`, residuals from the training data are used as proxy of
    #         prediction error to create prediction intervals. If `False`, out of
    #         sample residuals are used. In the latter case, the user should have
    #         calculated and stored the residuals within the forecaster (see
    #         `set_out_sample_residuals()`).
    #     levelss : Ignored
    #         Not used, present here for API consistency by convention.

    #     Returns
    #     -------
    #     boot_predictions : pandas DataFrame
    #         Predictions generated by bootstrapping.
    #         Shape: (steps, n_boot)

    #     Notes
    #     -----
    #     More information about prediction intervals in forecasting:
    #     https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
    #     Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

    #     """

    #     if isinstance(steps, int):
    #         steps = list(np.arange(steps) + 1)
    #     elif steps is None:
    #         steps = list(np.arange(self.steps) + 1)
    #     elif isinstance(steps, list):
    #         steps = list(np.array(steps))

    #     if in_sample_residuals:
    #         if not set(steps).issubset(set(self.in_sample_residuals.keys())):
    #             raise ValueError(
    #                 (
    #                     f"Not `forecaster.in_sample_residuals` for steps: "
    #                     f"{set(steps) - set(self.in_sample_residuals.keys())}."
    #                 )
    #             )
    #         residuals = self.in_sample_residuals
    #     else:
    #         if self.out_sample_residuals is None:
    #             raise ValueError(
    #                 (
    #                     "`forecaster.out_sample_residuals` is `None`. Use "
    #                     "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
    #                     "before `predict_interval()`, `predict_bootstrapping()` or "
    #                     "`predict_dist()`."
    #                 )
    #             )
    #         else:
    #             if not set(steps).issubset(set(self.out_sample_residuals.keys())):
    #                 raise ValueError(
    #                     (
    #                         f"Not `forecaster.out_sample_residuals` for steps: "
    #                         f"{set(steps) - set(self.out_sample_residuals.keys())}. "
    #                         f"Use method `set_out_sample_residuals()`."
    #                     )
    #                 )
    #         residuals = self.out_sample_residuals

    #     check_residuals = (
    #         "forecaster.in_sample_residuals"
    #         if in_sample_residuals
    #         else "forecaster.out_sample_residuals"
    #     )
    #     for step in steps:
    #         if residuals[step] is None:
    #             raise ValueError(
    #                 (
    #                     f"forecaster residuals for step {step} are `None`. "
    #                     f"Check {check_residuals}."
    #                 )
    #             )
    #         elif (residuals[step] == None).any():
    #             raise ValueError(
    #                 (
    #                     f"forecaster residuals for step {step} contains `None` values. "
    #                     f"Check {check_residuals}."
    #                 )
    #             )

    #     predictions = self.predict(steps=steps, last_window=last_window, exog=exog)

    #     # Predictions must be in the transformed scale before adding residuals
    #     predictions = transform_dataframe(
    #         df=predictions,
    #         transformer=self.transformer_series_[self.levels],
    #         fit=False,
    #         inverse_transform=False,
    #     )
    #     boot_predictions = pd.concat([predictions] * n_boot, axis=1)
    #     boot_predictions.columns = [f"pred_boot_{i}" for i in range(n_boot)]

    #     for i, step in enumerate(steps):
    #         rng = np.random.default_rng(seed=random_state)
    #         sample_residuals = rng.choice(a=residuals[step], size=n_boot, replace=True)
    #         boot_predictions.iloc[i, :] = boot_predictions.iloc[i, :] + sample_residuals

    #     if self.transformer_series_[self.levels]:
    #         for col in boot_predictions.columns:
    #             boot_predictions[col] = transform_series(
    #                 series=boot_predictions[col],
    #                 transformer=self.transformer_series_[self.levels],
    #                 fit=False,
    #                 inverse_transform=True,
    #             )

    #     return boot_predictions

    # def predict_interval(
    #     self,
    #     steps: Optional[Union[int, list]] = None,
    #     last_window: Optional[pd.DataFrame] = None,
    #     exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    #     interval: list = [5, 95],
    #     n_boot: int = 500,
    #     random_state: int = 123,
    #     in_sample_residuals: bool = True,
    #     levelss: Any = None,
    # ) -> pd.DataFrame:
    #     """
    #     Bootstrapping based prediction intervals.
    #     Both predictions and intervals are returned.

    #     Parameters
    #     ----------
    #     steps : int, list, None, default `None`
    #         Predict n steps. The value of `steps` must be less than or equal to the
    #         value of steps defined when initializing the forecaster. Starts at 1.

    #             - If `int`: Only steps within the range of 1 to int are predicted.
    #             - If `list`: List of ints. Only the steps contained in the list
    #             are predicted.
    #             - If `None`: As many steps are predicted as were defined at
    #             initialization.
    #     last_window : pandas DataFrame, default `None`
    #         Series values used to create the predictors (lags) needed in the
    #         first iteration of the prediction (t + 1).
    #         If `last_window = None`, the values stored in` self.last_window` are
    #         used to calculate the initial predictors, and the predictions start
    #         right after training data.
    #     exog : pandas Series, pandas DataFrame, default `None`
    #         Exogenous variable/s included as predictor/s.
    #     interval : list, default `[5, 95]`
    #         Confidence of the prediction interval estimated. Sequence of
    #         percentiles to compute, which must be between 0 and 100 inclusive.
    #         For example, interval of 95% should be as `interval = [2.5, 97.5]`.
    #     n_boot : int, default `500`
    #         Number of bootstrapping iterations used to estimate prediction
    #         intervals.
    #     random_state : int, default `123`
    #         Sets a seed to the random generator, so that boot intervals are always
    #         deterministic.
    #     in_sample_residuals : bool, default `True`
    #         If `True`, residuals from the training data are used as proxy of
    #         prediction error to create prediction intervals. If `False`, out of
    #         sample residuals are used. In the latter case, the user should have
    #         calculated and stored the residuals within the forecaster (see
    #         `set_out_sample_residuals()`).
    #     levelss : Ignored
    #         Not used, present here for API consistency by convention.

    #     Returns
    #     -------
    #     predictions : pandas DataFrame
    #         Values predicted by the forecaster and their estimated interval.

    #             - pred: predictions.
    #             - lower_bound: lower bound of the interval.
    #             - upper_bound: upper bound of the interval.

    #     Notes
    #     -----
    #     More information about prediction intervals in forecasting:
    #     https://otexts.com/fpp2/prediction-intervals.html
    #     Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
    #     George Athanasopoulos.

    #     """

    #     check_interval(interval=interval)

    #     predictions = self.predict(steps=steps, last_window=last_window, exog=exog)

    #     boot_predictions = self.predict_bootstrapping(
    #         steps=steps,
    #         last_window=last_window,
    #         exog=exog,
    #         n_boot=n_boot,
    #         random_state=random_state,
    #         in_sample_residuals=in_sample_residuals,
    #     )

    #     interval = np.array(interval) / 100
    #     predictions_interval = boot_predictions.quantile(q=interval, axis=1).transpose()
    #     predictions_interval.columns = ["lower_bound", "upper_bound"]
    #     predictions = pd.concat((predictions, predictions_interval), axis=1)

    #     return predictions

    # def predict_dist(
    #     self,
    #     distribution: object,
    #     steps: Optional[Union[int, list]] = None,
    #     last_window: Optional[pd.DataFrame] = None,
    #     exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    #     n_boot: int = 500,
    #     random_state: int = 123,
    #     in_sample_residuals: bool = True,
    #     levelss: Any = None,
    # ) -> pd.DataFrame:
    #     """
    #     Fit a given probability distribution for each step. After generating
    #     multiple forecasting predictions through a bootstrapping process, each
    #     step is fitted to the given distribution.

    #     Parameters
    #     ----------
    #     distribution : Object
    #         A distribution object from scipy.stats.
    #     steps : int, list, None, default `None`
    #         Predict n steps. The value of `steps` must be less than or equal to the
    #         value of steps defined when initializing the forecaster. Starts at 1.

    #             - If `int`: Only steps within the range of 1 to int are predicted.
    #             - If `list`: List of ints. Only the steps contained in the list
    #             are predicted.
    #             - If `None`: As many steps are predicted as were defined at
    #             initialization.
    #     last_window : pandas DataFrame, default `None`
    #         Series values used to create the predictors (lags) needed in the
    #         first iteration of the prediction (t + 1).
    #         If `last_window = None`, the values stored in` self.last_window` are
    #         used to calculate the initial predictors, and the predictions start
    #         right after training data.
    #     exog : pandas Series, pandas DataFrame, default `None`
    #         Exogenous variable/s included as predictor/s.
    #     n_boot : int, default `500`
    #         Number of bootstrapping iterations used to estimate prediction
    #         intervals.
    #     random_state : int, default `123`
    #         Sets a seed to the random generator, so that boot intervals are always
    #         deterministic.
    #     in_sample_residuals : bool, default `True`
    #         If `True`, residuals from the training data are used as proxy of
    #         prediction error to create prediction intervals. If `False`, out of
    #         sample residuals are used. In the latter case, the user should have
    #         calculated and stored the residuals within the forecaster (see
    #         `set_out_sample_residuals()`).
    #     levelss : Ignored
    #         Not used, present here for API consistency by convention.

    #     Returns
    #     -------
    #     predictions : pandas DataFrame
    #         Distribution parameters estimated for each step.

    #     """

    #     boot_samples = self.predict_bootstrapping(
    #         steps=steps,
    #         last_window=last_window,
    #         exog=exog,
    #         n_boot=n_boot,
    #         random_state=random_state,
    #         in_sample_residuals=in_sample_residuals,
    #     )

    #     param_names = [
    #         p for p in inspect.signature(distribution._pdf).parameters if not p == "x"
    #     ] + ["loc", "scale"]
    #     param_values = np.apply_along_axis(
    #         lambda x: distribution.fit(x), axis=1, arr=boot_samples
    #     )
    #     predictions = pd.DataFrame(
    #         data=param_values, columns=param_names, index=boot_samples.index
    #     )

    #     return predictions

    def set_params(
        self, params: dict
    ) -> None:  # TODO testear
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
        self.regressor.reset_states()
        self.regressor.compile(**params)

    def set_fit_kwargs(self, fit_kwargs: dict) -> None:
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

    def set_lags(self, lags: Union[int, np.ndarray, list, dict]) -> None:
        """
        Not used, present here for API consistency by convention.

        Returns
        -------
        None

        """

        pass
    
    
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

    