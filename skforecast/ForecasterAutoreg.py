################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

import typing
from typing import Union, Dict
import numpy as np
import pandas as pd
import logging
import tqdm


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


################################################################################
#                             ForecasterAutoreg                                #
################################################################################

class ForecasterAutoreg():
    '''
    This class turns a scikit-learn regressor into a recursive autoregressive
    (multi-step) forecaster.
    
    Parameters
    ----------
    regressor : scikit-learn regressor
        An instance of a scikit-learn regressor.
        
    lags : int, list, 1D np.array, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags`.
            `list` or `np.array`: include only lags present in `lags`.

    
    Attributes
    ----------
    regressor : scikit-learn regressor
        An instance of a scikit-learn regressor.
        
    lags : 1D np.array
        Lags used as predictors.
        
    max_lag : int
        Maximum value of lag included in lags.
        
    last_window : 1D np.ndarray, shape (1, max(self.lags))
        Last time window the Forecaster has seen when being trained. It contains
        the values needed to calculate the lags used to predict the next `step`
        after the training data.
        
    included_exog: bool, default `False`.
        If the Forecaster has been trained using exogenous variable.
     
    '''
    
    def __init__(self, regressor, lags: Union[int, np.ndarray, list]) -> None:
        
        self.regressor     = regressor
        self.last_window   = None
        self.included_exog = False
        
        if isinstance(lags, int):
            self.lags = np.arange(lags)
        elif isinstance(lags, (list, range)):
            self.lags = np.array(lags) -1
        elif isinstance(lags, np.ndarray):
            self.lags = lags
        else:
            raise Exception(
                f"`lags` must be `int`, `1D np.ndarray`, `range` or `list`. "
                f"Got {type(lags)}"
            )
            
        self.max_lag  = max(self.lags) + 1
                
        
    def __repr__(self) -> str:
            """
            Information displayed when a ForecasterAutoreg object is printed.
    
            """
    
            info =    "=======================" \
                    + "ForecasterAutoreg" \
                    + "=======================" \
                    + "\n" \
                    + "Regressor: " + str(self.regressor) \
                    + "\n" \
                    + "Lags: " + str(self.lags) \
                    + "\n" \
                    + "Exogenous variable: " + str(self.included_exog) \
                    + "\n" \
                    + "Parameters: " + str(self.regressor.get_params())
     
            return info

    
    
    def create_lags(self, y: Union[np.ndarray, pd.Series]) -> Dict[np.ndarray, np.ndarray]:
        '''       
        Transforms a time series into a 2D array and a 1D array where each value
        of `y` is associated with the lags that precede it.
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.

        Returns 
        -------
        X_data : 2D np.ndarray, shape (samples, len(lags))
            2D array with the lag values (predictors).
        
        y_data : 1D np.ndarray, shape (nº observaciones - max(lags),)
            Response variable Value of the time series fore each row of `X_data`.
            
        '''
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)        
        
        n_splits = len(y) - self.max_lag

        X_data  = np.full(shape=(n_splits, self.max_lag), fill_value=np.nan, dtype= float)
        y_data  = np.full(shape=(n_splits, 1), fill_value=np.nan, dtype= float)

        for i in range(n_splits):

            train_index = np.arange(i, self.max_lag + i)
            test_index  = [self.max_lag + i]

            X_data[i, :] = y[train_index]
            y_data[i]    = y[test_index]
            
        X_data = X_data[:, self.lags]
        y_data = y_data.ravel()
            
        return X_data, y_data

        
    def fit(self, y: Union[np.ndarray, pd.Series], exog: Union[np.ndarray, pd.Series]=None) -> None:
        '''
        Training ForecasterAutoreg
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.
            
        exog : 1D np.ndarray, pd.Series, default `None`
            Exogenous variable that is included as predictor. Must have the same
            number of observations as `y` and should be aligned so that y[i] is
            regressed on exog[i].


        Returns 
        -------
        self : ForecasterAutoreg
               Trained ForecasterAutoreg
        
        '''
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)
        
        if exog is not None:
            self._check_exog(exog=exog)
            exog = self._preproces_exog(exog=exog)
        
        X_train, y_train = self.create_lags(y=y)
        
        if exog is not None:
            self.included_exog = True
            self.regressor.fit(
                # The first `self.max_lag` positions have to be removed from exog
                # since they are not in X_train.
                X = np.column_stack((X_train, exog[self.max_lag:])),
                y = y_train
            )
            
        else:
            self.regressor.fit(X=X_train, y=y_train)
        
        # The last time window of training data is saved so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        self.last_window = y_train[-self.max_lag:]
        
            
    def predict(self, steps: int, X: np.ndarray=None, exog: np.ndarray=None) -> np.ndarray:
        '''
        Iterative process in which, each prediction, is used as a predictor
        for the next step.
        
        Parameters
        ----------
               
        steps : int
            Number of future steps predicted.
            
        X : 2D np.ndarray, shape (1, len(lags)), default `None`
            Value of the predictors to start the prediction process. That is,
            the value of the lags at t + 1.
    
            If `X = None`, the values stored in` self.last_window` are used to
            calculate the initial predictors, and the predictions start after
            training data.
            
        exog : 1D np.ndarray, pd.Series, default `None`
            Exogenous variable that is included as predictor.

        Returns 
        -------
        predicciones : 1D np.array, shape (steps,)
            Values predicted by the forecaster.
            
        '''
        
        if exog is None and self.included_exog:
            raise Exception(
                "Forecaster trained with exogenous variable. This same " \
                + "variable must be provided in the `predict()`."
            )
                
        if X is None:
            X = self.last_window[self.lags].reshape(1, -1)
        
        if exog is not None:
            self._check_exog(exog=exog)
            exog = self._preproces_exog(exog=exog)
            
        predictions = np.full(shape=steps, fill_value=np.nan)

        for i in range(steps):
            if exog is None:
                prediction = self.regressor.predict(X=X)
            else:
                prediction = self.regressor.predict(X=np.column_stack((X, exog[i])))
                
            predictions[i] = prediction.ravel()[0]

            # Update `X` values. The first position is discarded and the new
            # prediction is added at the end.
            X = X.flatten()
            X = np.append(X[1:], prediction)
            X = X.reshape(1, -1)


        return predictions
    
    
    def _check_y(self, y: Union[np.ndarray, pd.Series]) -> None:
        '''
        Raise Exception if `y` is not 1D `np.ndarray` or `pd.Series`.
        
        Parameters
        ----------        
        y : np.ndarray, pd.Series
            Time series values

        '''
        
        if not isinstance(y, (np.ndarray, pd.Series)):
            
            raise Exception('`y` must be `1D np.ndarray` or `pd.Series`.')
            
        elif isinstance(y, np.ndarray) and y.ndim != 1:
            
            raise Exception(
                f"`y` must be `1D np.ndarray` o `pd.Series`, "
                f"got `np.ndarray` with {y.ndim} dimensions."
            )
            
        else:
            return
        
        
    def _check_exog(self, exog: Union[np.ndarray, pd.Series]) -> None:
        '''
        Raise Exception if `exog` is not 1D `np.ndarray` or `pd.Series`.
        
        Parameters
        ----------        
        exog : np.ndarray, pd.Series
            Time series values

        '''
            
        if not isinstance(exog, (np.ndarray, pd.Series)):
            
            raise Exception('`exog` must be `1D np.ndarray` or `pd.Series`.')
            
        elif isinstance(exog, np.ndarray) and exog.ndim != 1:
            
            raise Exception(
                f"`exog` must be `1D np.ndarray` or `pd.Series`, "
                f"got `np.ndarray` with {y.ndim} dimensions."
            )
        else:
            return
        
    def _preproces_y(self, y) -> np.ndarray:
        
        '''
        Transforms `y` to 1D `np.ndarray` it is `pd.Series`.
        
        Parameters
        ----------        
        y :1D np.ndarray, pd.Series
            Time series values

        Returns 
        -------
        y: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(y, pd.Series):
            return y.to_numpy()
        else:
            return y
        
    def _preproces_exog(self, exog) -> np.ndarray:
        
        '''
        Transforms `exog` to 1D `np.ndarray` it is `pd.Series`.
        
        Parameters
        ----------        
        exog : 1D np.ndarray, pd.Series
            Time series values

        Returns 
        -------
        exog: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(exog, pd.Series):
            return exog.to_numpy()
        else:
            return exog
    
    
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
        Attribute `max_lag` is also updated.
        
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
        
        if isinstance(lags, int):
            self.lags = np.arange(lags)
        elif isinstance(lags, (list, range)):
            self.lags = np.array(lags) -1
        elif isinstance(lags, np.ndarray):
            self.lags = lags
        else:
            raise Exception(
                f"`lags` must be `int`, `1D np.ndarray`, `range` or `list`. "
                f"Got {type(lags)}"
            )
            
        self.max_lag  = max(self.lags) + 1
