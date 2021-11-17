################################################################################
#                                ForecasterBase                                #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.    
################################################################################
# coding=utf-8

from abc import ABC, abstractmethod
from typing import Union, Dict, List, Tuple, Any
import logging
import pandas as pd

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class ForecasterBase(ABC):
    '''
    Base class for all forecasters in skforecast. All forecasters should specify
    all the parameters that can be set at the class level in their ``__init__``.     
    '''

    @abstractmethod
    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        '''
        Create training matrices from univariante time series and exogenous
        variables.
        
        Parameters
        ----------        
        y : pandas Series
            Time series.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.

        Returns 
        -------
        X_train : pandas DataFrame, shape (len(y) - self.max_lag, len(self.lags))
            Pandas DataFrame with the training values (predictors).
            
        y_train : pandas Series, shape (len(y) - self.max_lag, )
            Values (target) of the time series related to each row of `X_train`.
        
        '''
        
        pass


    @abstractmethod
    def fit(
        self,
        y: pd.Series,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> None:
        '''
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
        
        '''
        
        pass


    @abstractmethod        
    def predict(
        self,
        steps: int,
        last_window: pd.Series=None,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> pd.Series:
        '''
        Predict n steps ahead.
        
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
            
        '''

        pass
        

    @abstractmethod
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
        
        pass
        

    @abstractmethod    
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
        
        pass


    def _check_predict_input(
        self,
        steps: int,
        last_window: pd.Series=None,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> None:
        '''
        Check all inputs of predict method
        '''

        if not self.fitted:
            raise Exception(
                'This Forecaster instance is not fitted yet. Call `fit` with'
                'appropriate arguments before using predict.'
            )
        
        if steps < 1:
            raise Exception(
                f"`steps` must be integer greater than 0. Got {steps}."
            )

        if hasattr(self, 'steps'):
            if steps > self.steps:
                raise Exception(
                    f"`steps` must be lower or equal to the value of steps defined "
                    f"when initializing the forecaster. Got {steps} but the maximum "
                    f"is {self.steps}."
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
            if not isinstance(exog, self.exog_type):
                raise Exception(
                    f"Expected type for `exog`: {self.exog_type}. Got {type(exog)}"      
                )
            if isinstance(exog, pd.DataFrame):
                col_missing = set(self.exog_col_names).difference(set(exog.columns))
                if col_missing:
                    raise Exception(
                        f"Missing columns in `exog`. Expected {self.exog_col_names}. "
                        f"Got {exog.columns.to_list()}"      
                    )
            check_exog(exog = exog)
            _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])
            
            if not isinstance(exog_index, self.index_type):
                raise Exception(
                    f"Expected index of type {self.index_type} for `exog`. "
                    f"Got {type(exog_index)}"      
                )
            if not exog_index.freqstr == self.index_freq:
                raise Exception(
                    f"Expected frequency of type {self.index_type} for `exog`. "
                    f"Got {exog_index.freqstr}"      
                )
            
        if last_window is not None:
            if len(last_window) < self.max_lag:
                raise Exception(
                    f"`last_window` must have as many values as as needed to "
                    f"calculate the maximum lag ({self.max_lag})."
                )
            if not isinstance(last_window, pd.Series):
                raise Exception('`last_window` must be a pandas Series.')
            if last_window.isnull().any():
                raise Exception('`last_window` has missing values.')
            _, last_window_index = preprocess_last_window(
                                        last_window = last_window.iloc[:0]
                                   ) 
            if not isinstance(last_window_index, self.index_type):
                raise Exception(
                    f"Expected index of type {self.index_type} for `last_window`. "
                    f"Got {type(last_window_index)}"      
                )
            if not last_window_index.freqstr == self.index_freq:
                raise Exception(
                    f"Expected frequency of type {self.index_type} for `last_window`. "
                    f"Got {last_window_index.freqstr}"      
                )

        return
        