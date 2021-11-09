################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any
import warnings
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


################################################################################
#                                ForecasterBase                                #
################################################################################

class ForecasterBase():
    '''
    Base class for all forecasters in skforecast. All forecasters should specify
    all the parameters that can be set at the class level in their ``__init__``.     
    '''


    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        '''
        Create training matrices from univariante time series.
        
        Parameters
        ----------        
        y : pandas Series
            Training time series.
            
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

            
    def predict(
        self,
        steps: int,
        last_window: pd.Series=None,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> pd.Series:
        '''
        Predict n steps ahead. It is an recursive process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
        last_window : pandas Series, default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
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

    
    @staticmethod
    def _check_y(y: Any) -> None:
        '''
        Raise Exception if `y` is not pandas Series or if it has missing values.
        
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
        
        return
        
        
    @staticmethod
    def _check_exog(exog: Any) -> None:
        '''
        Raise Exception if `exog` is not pandas Series or DataFrame, or
        if it has missing values.
        
        Parameters
        ----------        
        exog :  Any
            Exogenous variable/s included as predictor/s.

        Returns
        ----------
        None
        '''
            
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise Exception('`exog` must be `pd.Series` or `pd.DataFrame`.')

        if exog.isnull().any().any():
            raise Exception('`exog` has missing values.')
                    
        return


    def _check_predict_input(
        self,
        steps: int,
        last_window: pd.Series=None,
        exog: Union[pd.Series, pd.DataFrame]=None
    ) -> None:
        '''
        Check all inputs of predict method
        '''

        pass    
        

    @staticmethod
    def _preproces_y(y: pd.Series) -> Union[np.ndarray, pd.Index]:
        
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
        y : pandas Series
            Time series values

        Returns 
        -------
        y_values : numpy ndarray
            Numpy array with values of `y`.

        y_index : pandas Index
            Index of of `y` modified according to the rules.
        '''
        
        if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None:
            y_index = y.index
        elif isinstance(y.index, pd.DatetimeIndex):
            warnings.warn(
                '`y` has DatetimeIndex index but no frequency. ',
                'Index is overwritten with a RangeIndex.'
            )
            y_index = pd.RangeIndex(
                        start = 0,
                        stop  = len(y),
                        step  = 1
                       )
        else:
            warnings.warn(
                '`y` has no DatetimeIndex index. Index is overwritten with a RangeIndex.'
            )
            y_index = pd.RangeIndex(
                        start = 0,
                        stop  = len(y),
                        step  = 1
                       )

        y_values = y.to_numpy()

        return y_values, y_index
            

    @staticmethod
    def _preproces_last_window(last_window: pd.Series) -> Union[np.ndarray, pd.Index]:
        
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
        last_window : pandas Series
            Time series values

        Returns 
        -------
        last_window_values : numpy ndarray
            Numpy array with values of `last_window`.

        last_window_index : pandas Index
            Index of of `last_window` modified according to the rules.
        '''
        
        if isinstance(last_window.index, pd.DatetimeIndex) and last_window.index.freq is not None:
            last_window_index = last_window.index
        else:
            warnings.warn(
                '`last_window` has DatetimeIndex index but no frequency. '
                'Index is overwritten with a RangeIndex.'
            )
            last_window_index = pd.RangeIndex(
                        start = 0,
                        stop  = len(last_window),
                        step  = 1
                       )

        last_window_values = last_window.to_numpy()

        return last_window_values, last_window_index
        

    @staticmethod
    def _preproces_exog(
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
            warnings.warn(
                ('`exog` has DatetimeIndex index but no frequency. The index is '
                 'overwritten with a RangeIndex.')
            )
            exog_index = pd.RangeIndex(
                            start = 0,
                            stop  = len(exog),
                            step  = 1
                          )

        exog_values = exog.to_numpy()

        return exog_values, exog_index

    @staticmethod
    def _expand_index(index: Union[pd.Index, None], steps: int) -> pd.Index:
        
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
        
        pass
        
        
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
        
