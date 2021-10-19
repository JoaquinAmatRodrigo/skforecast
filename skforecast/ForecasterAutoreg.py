################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

import typing
from typing import Union, Dict, List, Tuple, Any
import warnings
import logging
import numpy as np
import pandas as pd
import sklearn
import tqdm


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


################################################################################
#                             ForecasterAutoreg                                #
################################################################################

class ForecasterAutoreg():
    '''
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster.
    
    Parameters
    ----------
    regressor : regressor compatible with the scikit-learn API
        An instance of a regressor compatible with the scikit-learn API.
        
    lags : int, list, 1D np.array, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list`, `np.array` or range: include only lags present in `lags`.

    
    Attributes
    ----------
    regressor : regressor compatible with the scikit-learn API
        An instance of a regressor compatible with the scikit-learn API.
        
    lags : 1D np.array
        Lags used as predictors.
        
    max_lag : int
        Maximum value of lag included in `lags`.
        
    window_size: int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.
        
    fitted: Bool
        Tag to identify if the regressor has been fitted (trained).
        
    y_index_type : type
        Index type of the input time series used in training.
        
    y_index_freq : str
        Index frequency of the input time series used in training.
        
    training_range: pd.Index
        First and last index of samples used during training.
        
    last_window : pd.Series
        Last window the forecaster has seen during trained. It stores the
        values needed to calculate the lags used to predict the next `step`
        after the training data.
        
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
        
    exog_type : type
        Type of exogenous variable/s used in training.
        
    exog_col_names : tuple
        Column names of exog if exog used in training is a pandas DataFrame.
        
    in_sample_residuals: np.ndarray
        Residuals of the model when predicting training data. Only stored up to
        1000 values.
        
    out_sample_residuals: np.ndarray
        Residuals of the model when predicting non training data. Only stored
        up to 1000 values.
     
    '''
    
    def __init__(self, regressor, lags: Union[int, np.ndarray, list]) -> None:
        
        self.regressor            = regressor
        self.y_index_type         = None
        self.y_index_freq         = None
        self.training_range       = None
        self.last_window          = None
        self.included_exog        = False
        self.exog_type            = None
        self.exog_col_names       = None
        self.in_sample_residuals  = None
        self.out_sample_residuals = None
        self.fitted               = False
        
        if isinstance(lags, int) and lags < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, (list, range, np.ndarray)) and min(lags) < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, int):
            self.lags = np.arange(lags) + 1
        elif isinstance(lags, (list, range)):
            self.lags = np.array(lags)
        elif isinstance(lags, np.ndarray):
            self.lags = lags
        else:
            raise Exception(
                '`lags` argument must be `int`, `1D np.ndarray`, `range` or `list`. '
                f"Got {type(lags)}"
            )
            
        self.max_lag  = max(self.lags)
        self.window_size = self.max_lag


    def __repr__(self) -> str:
        '''
        Information displayed when a ForecasterAutoreg object is printed.
        '''

        info = (
            f"{'=' * len(str(type(self)))} \n"
            f"{type(self)} \n"
            f"{'=' * len(str(type(self)))} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
            f"Window size: {self.window_size} \n"
            f"Included exogenous: {self.included_exog}, {self.exog_type if self.included_exog else ''} \n"
            f"Type of exogenous variable: {self.exog_type if self.included_exog else ''} \n"
            f"Exogenous variables names: {self.exog_col_names if self.included_exog else ''} \n"
            f"Training range: {self.training_range.to_list()} \n"
            f"Training index type: {str(self.y_index_type)} \n"
            f"Training index frequancy: {self.y_index_freq} \n"
            f"Parameters: {self.regressor.get_params()} \n"
        )

        return info

    
    
    def __create_lags(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''       
        Transforms a 1d array into a 2d array (X) and a 1d array (y).
        Each value of y is associated with a row in X that represents the lags
        that precede it.
        
        Notice that, the returned matrix X_data, contains the lag 1 in the first
        column, the lag 2 in the second column and so on.
        
        Parameters
        ----------        
        y : 1d np.ndarray
            Training time series.

        Returns 
        -------
        X_data : 2d np.ndarray, shape (samples, len(self.lags))
            2d array with the lag values (predictors).
        
        y_data : 1d np.ndarray, shape (nº samples - max(seld.lags),)
            Values of the time series related to each row of `X_data`.
            
        '''
          
        n_splits = len(y) - self.max_lag
        X_data  = np.full(shape=(n_splits, self.max_lag), fill_value=np.nan, dtype=float)
        y_data  = np.full(shape=(n_splits, 1), fill_value=np.nan, dtype= float)

        for i in range(n_splits):
            X_index = np.arange(i, self.max_lag + i)
            y_index = [self.max_lag + i]
            X_data[i, :] = y[X_index]
            y_data[i]    = y[y_index]
            
        X_data = X_data[:, -self.lags]
        y_data = y_data.ravel()
            
        return X_data, y_data


    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        '''
        Create training matrices from univariante time series.
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.
            
        exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and should be aligned so that y[i] is
            regressed on exog[i].


        Returns 
        -------
        X_train : 2D np.ndarray, shape (len(y) - self.max_lag, len(self.lags))
            2D array with the training values (predictors).
            
        y_train : 1D np.ndarray, shape (len(y) - self.max_lag,)
            Values (target) of the time series related to each row of `X_train`.
        
        '''
        
        self.__check_y(y=y)
        y_values, y_index = self.__preproces_y(y=y)
        
        if exog is not None:
            if len(exog) != len(y):
                raise Exception(
                    "`exog` must have same number of samples as `y`."
                )
            self.__check_exog(exog=exog)
            exog_values, exog_index = self.__preproces_exog(exog=exog)
            self.__check_index_aligment_y_exog(y_index=y_index, exog_index=exog_index)
            
        
        
        X_train, y_train = self.__create_lags(y=y_values)
        col_names_X_train = [f"lag_{i}" for i in self.lags]
        if exog is not None:
            col_names_exog = exog.columns if isinstance(exog, pd.DataFrame) else exog.name
            col_names_X_train.extend(col_names_exog)
            # The first `self.max_lag` positions have to be removed from exog
            # since they are not in X_train.
            X_train = np.column_stack((X_train, exog_values[self.max_lag:, ]))

        X_train = pd.DataFrame(
                    data    = X_train,
                    columns = col_names_X_train,
                    index   = y_index[self.max_lag: ]
                  )

        y_train = pd.Series(
                    data  = y_train,
                    index = y_index[self.max_lag: ],
                    name  = 'y'
                 )
                        
        return X_train, y_train

        
    def fit(
        self,
        y: pd.Series,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> None:
        '''
        Training Forecaster.
        
        Parameters
        ----------        
        y : pd.Series
            Training time series.
            
        exog : pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and should be aligned so that y[i] is
            regressed on exog[i].


        Returns 
        -------
        None
        
        '''
        
        # Reset values in case the forecaster has already been fitted.
        self.y_index_type         = None
        self.y_index_freq         = None
        self.last_window          = None
        self.included_exog        = False
        self.exog_type            = None
        self.exog_col_names       = None
        self.in_sample_residuals  = None
        self.out_sample_residuals = None
        self.fitted               = False
        self.training_range       = None
        
        self.__check_y(y=y)
        self.y_index_type = type(y.index)
        self.training_range = y.index[[0, -1]]
        if isinstance(y.index, pd.DatetimeIndex):
            self.y_index_freq = y.index.freqstr
        else: 
            self.y_index_freq = y.index.step
        
        if exog is not None:
            if len(exog) != len(y):
                raise Exception(
                    "`exog` must have same number of samples as `y`."
                )
            self.__check_exog(exog=exog)
            self.exog_type = type(exog)
            self.included_exog = True
            if isinstance(exog, pd.DataFrame):
                self.exog_col_names = exog.columns.to_list()
                self.__check_index_aligment_y_exog(y_index=y.index, exog_index=exog.index)
 
        X_train, y_train = self.create_train_X_y(y=y, exog=exog)
        
        self.regressor.fit(X=X_train, y=y_train)
        self.fitted = True            
        residuals = y_train - self.regressor.predict(X_train)
            
        if len(residuals) > 1000:
            # Only up to 1000 residuals are stored
            residuals = np.random.choice(a=residuals, size=1000, replace=False)                                              
        self.in_sample_residuals = residuals
        
        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        self.last_window = y_train.iloc[-self.max_lag:].copy()
    

    def __recursive_predict(
        self,
        steps: int,
        last_window: np.array,
        exog: np.array
    ) -> pd.Series:
        '''
        Predict n steps ahead. It is an iterative process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
        last_window : np.array
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
            
        exog : np.array, pd.DataFrame
            Exogenous variable/s included as predictor/s.

        Returns 
        -------
        predictions : np.array
            Predicted values.
            
        '''


        predictions = np.full(shape=steps, fill_value=np.nan)

        for i in range(steps):
            X = last_window[-self.lags].reshape(1, -1)
            if exog is not None:
                X = np.column_stack((X, exog[i,].reshape(1, -1)))

            prediction = self.regressor.predict(X)
            predictions[i] = prediction.ravel()[0]

            # Update `last_window` values. The first position is discarded and 
            # the new prediction is added at the end.
            last_window = np.append(last_window[1:], prediction)

        return predictions
            
            
    def predict(
        self,
        steps: int,
        last_window: pd.Series=None,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> pd.Series:
        '''
        Predict n steps ahead. It is an iterative process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
        last_window : pd.Series, default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns 
        -------
        predictions : pd.Series
            Predicted values.
            
        '''

        if not self.fitted:
            raise Exception(
                'This Forecaster instance is not fitted yet. Call `fit` with'
                'appropriate arguments before using this it.'
            )
        
        if steps < 1:
            raise Exception(
                f"`steps` must be integer greater than 0. Got {steps}."
            )
        
        if exog is None and self.included_exog:
            raise Exception(
                'Forecaster trained with exogenous variable/s. '
                'Same variable/s must be provided in `predict()`.'
            )
            
        if exog is not None and not self.included_exog:
            raise Exception(
                'Forecaster trained without exogenous variable/s. '
                '`exog` must be `None` in `predict()`.'
            )
        
        if exog is not None:
            if len(exog) < steps:
                raise Exception(
                    '`exog` must have at least as many values as `steps` predicted.'
                )
            self.__check_exog(
                exog = exog,
                ref_type = self.exog_type,
                ref_col_names = self.exog_col_names
            )
            exog_values, exog_index = self.__preproces_exog(exog=exog)
        else:
            exog_values = None
            exog_index = None
            
        if last_window is not None:
            if len(last_window) < self.max_lag:
                raise Exception(
                    f"`last_window` must have as many values as as needed to "
                    f"calculate the maximum lag ({self.max_lag})."
                )
            self.__check_last_window(last_window=last_window)
            last_window_values, last_window_index = self.__preproces_y(y=last_window)
            
        else:
            last_window_values, last_window_index = self.__preproces_y(y=self.last_window)
            
        predictions = self.__recursive_predict(
                        steps       = steps,
                        last_window = last_window_values,
                        exog        = exog_values 
                      )

        predictions = pd.Series(
                        data = predictions,
                        index = self.__expand_index(
                                    index = last_window_index,
                                    steps = steps
                                ),
                        name = 'pred'
                      )

        return predictions
    
    
    def __estimate_boot_interval(
        self,
        steps: int,
        last_window: np.ndarray,
        exog: np.ndarray,
        interval: list=[5, 95],
        n_boot: int=500,
        in_sample_residuals: bool=True
    ) -> np.ndarray:
        '''
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. This method only returns prediction intervals.
        See predict_intervals() to calculate both, predictions and intervals.
        
        Parameters
        ----------   
        steps : int
            Number of future steps predicted.
            
        last_window : 1d np.ndarray shape (, max_lag)
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : np.ndarray
            Exogenous variable/s included as predictor/s.
            
        n_boot: int, default `500`
            Number of bootstrapping iterations used to estimate prediction
            intervals.
            
        interval: list, default `[5, 95]`
            Confidence of the prediction interval estimated. Sequence of percentiles
            to compute, which must be between 0 and 100 inclusive.
            
        in_sample_residuals: bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create prediction intervals. If `False`, out of
            sample residuals are used. In the latter case, the user shoud have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
            

        Returns 
        -------
        predicction_interval : np.array, shape (steps, 2)
            Interval estimated for each prediction by bootstrapping:
                first column = lower bound of the interval.
                second column= upper bound interval of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
            
        '''
        

        boot_predictions = np.full(
                                shape      = (steps, n_boot),
                                fill_value = np.nan,
                                dtype      = float
                           )

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

            sample_residuals = np.random.choice(
                                    a       = residuals,
                                    size    = steps,
                                    replace = True
                               )

            for step in range(steps):
                prediction = self.__recursive_predict(
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
        last_window: pd.Series=None,
        exog: Union[pd.Series, pd.DataFrame]=None,
        interval: list=[5, 95],
        n_boot: int=500,
        in_sample_residuals: bool=True
    ) -> pd.DataFrame:
        '''
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. Both, predictions and intervals, are returned.
        
        Parameters
        ---------- 
        steps : int
            Number of future steps predicted.
            
        last_window : pd.Series, default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
            
        interval: list, default `[5, 100]`
            Confidence of the prediction interval estimated. Sequence of percentiles
            to compute, which must be between 0 and 100 inclusive.
            
        n_boot: int, default `500`
            Number of bootstrapping iterations used to estimate prediction
            intervals.
            
        in_sample_residuals: bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create prediction intervals. If `False`, out of
            sample residuals are used. In the latter case, the user shoud have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).

        Returns 
        -------
        predictions : pd.DataFrame
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
            
        '''
        
        if steps < 1:
            raise Exception(
                f"`steps` must be integer greater than 0. Got {steps}."
            )
            
        if not in_sample_residuals and self.out_sample_residuals is None:
            raise Exception(
                ('out_sample_residuals is empty. In order to estimate prediction '
                'intervals using out of sample residuals, the user shoud have '
                'calculated and stored the residuals within the forecaster (see'
                '`set_out_sample_residuals()`.')
            )
        
        if exog is None and self.included_exog:
            raise Exception(
                'Forecaster trained with exogenous variable/s. '
                'Same variable/s must be provided in `predict()`.'
            )
            
        if exog is not None and not self.included_exog:
            raise Exception(
                'Forecaster trained without exogenous variable/s. '
                '`exog` must be `None` in `predict()`.'
            )
        
        if exog is not None:
            if len(exog) < steps:
                raise Exception(
                    '`exog` must have at least as many values as `steps` predicted.'
                )
            self.__check_exog(
                exog = exog,
                ref_type = self.exog_type,
                ref_col_names = self.exog_col_names
            )
            exog_values, exog_index = self.__preproces_exog(exog=exog)
        else:
            exog_values = None
            exog_index = None
     
        if last_window is not None:
            if len(last_window) < self.max_lag:
                raise Exception(
                    f"`last_window` must have as many values as as needed to "
                    f"calculate the maximum lag ({self.max_lag})."
                )
            self.__check_last_window(last_window=last_window)
            last_window_values, last_window_index = self.__preproces_y(y=last_window)
            
        else:
            last_window_values, last_window_index = self.__preproces_y(y=self.last_window)
        
        # Since during predict() `last_window_values` and `exog_values` are modified,
        # the originals are stored to be used later.
        last_window_values_original = last_window_values.copy()
        if exog is not None:
            exog_values_original = exog_values.copy()
        else:
            exog_values_original = None
        
        predictions = self.__recursive_predict(
                            steps       = steps,
                            last_window = last_window_values,
                            exog        = exog_values
                      )

        predictions_interval = self.__estimate_boot_interval(
                                    steps       = steps,
                                    last_window = last_window_values_original,
                                    exog        = exog_values_original,
                                    interval    = interval,
                                    n_boot      = n_boot,
                                    in_sample_residuals = in_sample_residuals
                                )
        
        predictions = np.column_stack((predictions, predictions_interval))

        predictions = pd.DataFrame(
                        data = predictions,
                        index = self.__expand_index(
                                    index = last_window_index,
                                    steps = steps
                                ),
                        columns = ['pred', 'lower_bound', 'upper_bound']
                      )

        return predictions

    
    @staticmethod
    def __check_y(y: Any) -> None:
        '''
        Raise Exception if `y` is not a pandas Series or if it has missing values.
        If `y` index is of type DatetimeIndex without a frequancy, a warning is
        displayed.
        
        Parameters
        ----------        
        y : Any
            Time series values
            
        Returns
        ----------
        None
        
        '''
        
        if not isinstance(y, pd.Series):
            raise Exception('`y` must be a pandas Series.')
            
        if y.isnull().any():
            raise Exception('`y` has missing values.')
        
        if isinstance(y.index, pd.DatetimeIndex) and  y.index.freq is None:
            warnings.warn(
                ('`y` has DatetimeIndex index but no frequency. The index will be '
                 'overwrite with a RangeIndex.')
            )
            
        return
    
    
    @staticmethod
    def __check_last_window(last_window: Any) -> None:
        '''
        Raise Exception if `last_window` is not a pandas Series or if it has
        missing values. If `y` index is of type DatetimeIndex without a frequancy,
        a warning is displayed.
        
        Parameters
        ----------        
        last_window : Any
            Time series values
            
        Returns
        ----------
        None
        
        '''
        
        if not isinstance(last_window, pd.Series):
            raise Exception('`last_window` must be a pandas Series.')
                
        if last_window.isnull().any():
            raise Exception('`last_window` has missing values.')
        
        if isinstance(last_window.index, pd.DatetimeIndex) and last_window.index.freq is None:
            warnings.warn(
                ('`last_window` has DatetimeIndex index but no frequency. The '
                 'index will be overwrite with a RangeIndex.')
            )
            
        return
        
        
    @staticmethod
    def __check_exog(
        exog: Any,
        ref_type: type=None,
        ref_col_names: list=None
    ) -> None:
        '''
        Raise Exception if `exog` is not pd.Series or pd.DataFrame, if `exog`
        type does not match `ref_type` or if `exog` does not have the columns in
        `ref_col_names`.
        
        Parameters
        ----------        
        exog :  pd.Series, pd.DataFrame
            Exogenous variable/s included as predictor/s.

        ref_type : type, default `None`
            Type of reference for exog.
            
        ref_col_names : list, default `None`
            Shape of reference for exog.
        '''
            
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise Exception('`exog` must be `pd.Series` or `pd.DataFrame`.')
                    
        if ref_type is not None and type(exog) != ref_type:
            raise Exception(
                f"Expected type of `exog` is {ref_type}. Got {type(exog)}."
            )
            
        if ref_col_names is not None and isinstance(exog, pd.DataFrame):
            missing_cols = set(ref_col_names ).difference(set(exog.columns))
            if missing_cols:
                raise Exception(
                    f"Forecaster was trained with exogenous variables: {ref_col_names}. \n"
                    f"The following columns are missing `exog` : {missing_cols}."
                )
        return
    

    @staticmethod
    def __check_index_aligment_y_exog(y_index: pd.Index, exog_index: pd.Index) -> None:
        '''
        Raise Exception if y_index and exog_index do not align.

        Parameters
        ---------- 
        y_index : pd.Index
            Index for `y`

        exog_index : pd.Index
            Index for exogenous variable/s included as predictor/s.
        ''' 

        if not (exog_index[:len(y_index)] == y_index).all():
            raise Exception(
               ('`y` and `exog` must have same index. This is important '
               'to ensure the correct aligment of values.')      
            )

    @staticmethod
    def __check_index_compatibility(
        index: pd.Index,
        ref_index_type: type,
        ref_index_freq: str
    ) -> None:
        '''
        Raise Warning if `index` is not of type `ref_index_type` or has not
        frequancy `ref_index_freq`.

        Parameters
        ---------- 
        index : pd.Index
            Index to be check

        ref_index_type: type
            Type of reference
        
        ref_index_freq: str
            Frequancy of reference
        ''' 

        if not type(index) == ref_index_type:
            warnings.warn(
               f"Expected index of type {ref_index_type}. Got {type(index)}"      
            )

        if not index.freqstr == ref_index_freq:
            warnings.warn(
               f"Expected index of type {ref_index_type}. Got {type(index)}"      
            )
        
    @staticmethod
    def __preproces_y(y: pd.Series) -> Union[np.ndarray, pd.Index]:
        
        '''
        Returns values ​​and index of series separately. Index is overwritten
        according to the next rules:
            If index is not of type DatetimeIndex, a RangeIndex is created.
            If index is of type DatetimeIndex and but has no frequency, a
            RangeIndex is created.
            If index is of type DatetimeIndex and has frequency, nothing is
            changed.

        
        
        Parameters
        ----------        
        y : pd.Series
            Time series values

        Returns 
        -------
        y_values : np.ndarray
            Numpy array with values of `y`.
        y_index :pd:Index
            Series index.
        '''
        
        if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None:
            y_index = y.index
        else:
            y_index = pd.RangeIndex(
                        start = 0,
                        stop  = len(y),
                        step  = 1
                       )

        y_values = y.to_numpy()

        return y_values, y_index
            
        
        
    @staticmethod
    def __preproces_exog(
        exog: Union[pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Index]:
        
        '''
        Returns values ​​and index separately. Index is overwritten according to
        the next rules:
            If index is not of type DatetimeIndex, a RangeIndex is created.
            If index is of type DatetimeIndex and but has no frequency, a
            RangeIndex is created.
            If index is of type DatetimeIndex and has frequency, nothing is
            changed.

        Parameters
        ----------        
        exog : pd.Series, pd.DataFrame
            Exogenous variables

        Returns 
        -------
        exog_values : np.ndarray
            Numpy array with values of `exog`.
        exog_index : pd.Index
            Exog index.
        '''
        
        if isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is not None:
            exog_index = exog.index
        else:
            exog_index = pd.RangeIndex(
                            start = 0,
                            stop  = len(exog),
                            step  = 1
                          )

        exog_values = exog.to_numpy()

        return exog_values, exog_index

    @staticmethod
    def __expand_index(index: Union[pd.Index, None], steps: int) -> pd.Index:
        
        '''
        Create a new index of lenght `steps` starting and the end of index.
        
        Parameters
        ----------        
        index : pd.Index, None
            Index of last window
        steps: int
            Number of steps to expand.

        Returns 
        -------
        new_index : pd.Index
        '''
        
        if isinstance(index, pd.Index):
            
            if isinstance(index, pd.DatetimeIndex):
                new_index = pd.date_range(
                                index[-1] + index.freq,
                                periods = steps,
                                freq    = index.freq
                            )
            elif isinstance(index, pd.RangeIndex):
                new_index = pd.RangeIndex(
                                start = index[-1] + 1,
                                stop  = index[-1] + 1 + steps
                             )
        else: 
            new_index = pd.RangeIndex(
                            start = 0,
                            stop  = steps
                         )
        return new_index
    
    
    def set_params(self, **params: dict) -> None:
        '''
        Set new values to the parameters of the scikit learn model stored in the
        ForecasterAutoreg.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns 
        -------
        self
        
        '''
        
        self.regressor.set_params(**params)
        
        
    def set_lags(self, lags: int) -> None:
        '''      
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
        self
        
        '''
        
        if isinstance(lags, int) and lags < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, (list, range, np.ndarray)) and min(lags) < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, int):
            self.lags = np.arange(lags) + 1
        elif isinstance(lags, (list, range)):
            self.lags = np.array(lags)
        elif isinstance(lags, np.ndarray):
            self.lags = lags
        else:
            raise Exception(
                f"`lags` argument must be `int`, `1D np.ndarray`, `range` or `list`. "
                f"Got {type(lags)}"
            )
            
        self.max_lag  = max(self.lags)
        self.window_size = max(self.lags)
        
        
    def set_out_sample_residuals(self, residuals: np.ndarray, append: bool=True)-> None:
        '''
        Set new values to the attribute `out_sample_residuals`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process.
        
        Parameters
        ----------
        params : 1D np.ndarray
            Values of residuals. If len(residuals) > 1000, only a random sample
            of 1000 values are stored.
            
        append : bool, default `True`
            If `True`, new residuals are added to the once already stored in the attribute
            `out_sample_residuals`. Once the limit of 1000 values is reached, no more values
            are appended. If False, `out_sample_residuals` is overwrited with the new residuals.
            

        Returns 
        -------
        self
        
        '''
        if not isinstance(residuals, np.ndarray):
            raise Exception(
                f"`residuals` argument must be `1D np.ndarray`. Got {type(residuals)}"
            )
            
        if len(residuals) > 1000:
            residuals = np.random.choice(a=residuals, size=1000, replace=False)
                                 
        if not append or self.out_sample_residuals is None:
            self.out_sample_residuals = residuals
        else:
            free_space = max(0, 1000 - len(self.out_sample_residuals))
            if len(residuals) < free_space:
                self.out_sample_residuals = np.hstack((self.out_sample_residuals, residuals))
            else:
                self.out_sample_residuals = np.hstack((self.out_sample_residuals, residuals[:free_space]))
        

    def get_coef(self) -> np.ndarray:
        '''      
        Return estimated coefficients for the linear regression model stored in
        the forecaster. Only valid when the forecaster has been trained using
        as `regressor: `LinearRegression()`, `Lasso()` or `Ridge()`.
        
        Parameters
        ----------
        self

        Returns 
        -------
        coef : 1D np.ndarray
            Value of the coefficients associated with each predictor (lag).
            Coefficients are aligned so that `coef[i]` is the value associated
            with `self.lags[i]`.
        
        '''
        
        valid_instances = (sklearn.linear_model._base.LinearRegression,
                          sklearn.linear_model._coordinate_descent.Lasso,
                          sklearn.linear_model._ridge.Ridge
                          )
        
        if not isinstance(self.regressor, valid_instances):
            warnings.warn(
                ('Only forecasters with `regressor` `LinearRegression()`, ' +
                 ' `Lasso()` or `Ridge()` have coef.')
            )
            return
        else:
            coef = self.regressor.coef_
            
        return coef

    
    def get_feature_importances(self) -> np.ndarray:
        '''      
        Return impurity-based feature importances of the model stored in the
        forecaster. Only valid when the forecaster has been trained using
        `regressor=GradientBoostingRegressor()` or `regressor=RandomForestRegressor`.

        Parameters
        ----------
        self

        Returns 
        -------
        feature_importances : 1D np.ndarray
        Impurity-based feature importances associated with each predictor (lag).
        Values are aligned so that `feature_importances[i]` is the value
        associated with `self.lags[i]`.
        '''

        if not isinstance(self.regressor,
                        (sklearn.ensemble._forest.RandomForestRegressor,
                        sklearn.ensemble._gb.GradientBoostingRegressor)):
            warnings.warn(
                ('Only forecasters with `regressor=GradientBoostingRegressor()` '
                 'or `regressor=RandomForestRegressor`.')
            )
            return
        else:
            feature_importances = self.regressor.feature_importances_

        return feature_importances
