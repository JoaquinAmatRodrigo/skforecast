################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

import typing
from typing import Union, Dict
import warnings
import logging
import numpy as np
import pandas as pd
import sklearn
import tqdm

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


################################################################################
#                         ForecasterAutoregMultiOutput                         #
################################################################################

class ForecasterAutoregMultiOutput():
    '''
    This class turns a scikit-learn regressor into a autoregressive multi-output
    forecaster. A separate model is created for each forecast time step. See Notes
    for more details.
    
    Parameters
    ----------
    regressor : scikit-learn regressor
        An instance of a scikit-learn regressor.
        
    lags : int, list, 1D np.array, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list` or `np.array`: include only lags present in `lags`.
            
    steps : int
        Number of future steps the forecaster will predict when using method
        `predict()`. Since a diferent model is created for each step, this value
        should be defined before training.
    
    Attributes
    ----------
    regressor : scikit-learn regressor
        An instance of a scikit-learn regressor.
        
    steps : int
        Number of future steps the forecaster will predict when using method
        `predict()`. Since a diferent model is created for each step, this value
        should be defined before training.
        
    lags : 1D np.array
        Lags used as predictors.
        
    max_lag : int
        Maximum value of lag included in lags.
        
    last_window : 1D np.ndarray
        Last time window the forecaster has seen when trained. It stores the
        values needed to calculate the lags used to predict the next `step`
        after the training data.
        
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
        
    exog_type : type
        Type used for the exogenous variable/s.
            
    exog_shape : tuple
        Shape of exog used in training.
        
    in_sample_residuals: np.ndarray
        Residuals of the model when predicting training data.
        
    out_sample_residuals: np.ndarray
        Residuals of the model when predicting non training data.
        
        
    Notes
    -----
    A separate model is created for each forecast time step using
    `sklearn.multioutput.MultiOutputRegressor`. This scikit learn class
    is a wrapper that automize fitting one regressor per target. It is important
    to note that all models share the same configuration of parameters and 
    hiperparameters.
     
    '''
    
    def __init__(self, regressor, steps: int,
                 lags: Union[int, np.ndarray, list]) -> None:
        
        self.regressor     = MultiOutputRegressor(regressor)
        self.steps         = steps
        self.last_window   = None
        self.included_exog = False
        self.exog_type     = False
        self.exog_shape    = None
        self.in_sample_residuals  = None
        self.out_sample_residuals = None
        
        if not isinstance(steps, int) or steps < 1:
            raise Exception(
                f"`steps` must be integer greater than 0. Got {steps}."
            )
        
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
                
        
    def __repr__(self) -> str:
        '''
        Information displayed when a ForecasterAutoreg object is printed.
        '''

        info =    "============================" \
                + "ForecasterAutoregMultiOutput" \
                + "============================" \
                + "\n" \
                + "Regressor: " + str(self.regressor) \
                + "\n" \
                + "Steps: " + str(self.steps) \
                + "\n" \
                + "Lags: " + str(self.lags) \
                + "\n" \
                + "Exogenous variable: " + str(self.included_exog) \
                + "\n" \
                + "Parameters: " + str(self.regressor.get_params())

        return info

    
    def create_lags(self, y: Union[np.ndarray, pd.Series]) -> Dict[np.ndarray, np.ndarray]:
        '''
        Transforms a time series into two 2D arrays of pairs predictor-response.
        
        Notice that the returned matrix X_data, contains the lag 1 in the
        first column, the lag 2 in the second column and so on.
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.

        Returns 
        -------
        X_data : 2D np.ndarray
            2D array with the lag values (predictors).
        
        y_data : 2D np.ndarray
            Values of the time series related to each row of `X_data`.
            
        '''
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)        
        
        if self.max_lag > len(y):
            raise Exception(
                f"Maximum lag can't be higer than `y` length. "
                f"Got maximum lag={self.max_lag} and `y` length={len(y)}."
            )
            
        n_splits = len(y) - self.max_lag - (self.steps -1)
        X_data  = np.full(shape=(n_splits, self.max_lag), fill_value=np.nan, dtype=float)
        y_data  = np.full(shape=(n_splits, self.steps), fill_value=np.nan, dtype= float)

        for i in range(n_splits):
            train_index = np.arange(i, self.max_lag + i)
            test_index  = np.arange(self.max_lag + i, self.max_lag + i + self.steps)

            X_data[i, :] = y[train_index]
            y_data[i, :] = y[test_index]
            
        X_data = X_data[:, -self.lags]
                    
        return X_data, y_data

        
    def fit(self, y: Union[np.ndarray, pd.Series],
            exog: Union[np.ndarray, pd.Series]=None) -> None:
        '''
        Training ForecasterAutoregMultiOutput
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.
            
        exog : np.ndarray, pd.Series, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and should be aligned so that y[i] is
            regressed on exog[i].


        Returns 
        -------
        self : ForecasterAutoregMultiOutput
            Trained ForecasterAutoregMultiOutput
        
        '''
        
        # Reset values in case the forecaster has already been fitted before.
        self.included_exog = False
        self.exog_type     = None
        self.exog_shape    = None
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)
        
        if exog is not None:
            self._check_exog(exog=exog)
            self.exog_type = type(exog)
            exog = self._preproces_exog(exog=exog)
            self.included_exog = True
            self.exog_shape = exog.shape
            
            if exog.shape[0] != len(y):
                raise Exception(
                    f"`exog` must have same number of samples as `y`"
                )
                
            # Trasform exog to match multi output format
            exog = self._exog_to_multi_output(exog=exog)               
        
        X_train, y_train = self.create_lags(y=y)
        
        if exog is not None:
            self.regressor.fit(
                # The first `self.max_lag` positions have to be removed from exog
                # since they are not in X_train.
                X = np.column_stack((X_train, exog[self.max_lag:,])),
                y = y_train
            )
            self.in_sample_residuals = \
                y_train - self.regressor.predict(
                                np.column_stack((X_train, exog[self.max_lag:,]))
                          )
        else:
            self.regressor.fit(X=X_train, y=y_train)
            self.in_sample_residuals = y_train - self.regressor.predict(X_train)
        
        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        self.last_window = np.concatenate(
                                (y_train.ravel()[::self.steps][-self.max_lag:],
                                 y_train[-1][1:])
                            )
        
            
    def predict(self, last_window: Union[np.ndarray, pd.Series]=None,
                exog: np.ndarray=None, steps=None):
        '''
        Multi-step prediction with a MultiOutputRegressor. The number of future
        steps predicted is defined when ininitializing the forecaster.
        
        Parameters
        ----------
            
        last_window : 1D np.ndarray, pd.Series, shape (, max_lag), default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : np.ndarray, pd.Series, default `None`
            Exogenous variable/s included as predictor/s.
            
        steps : Ignored
            Not used, present here for API consistency by convention.

        Returns 
        -------
        predictions : 1D np.array, shape (steps,)
            Values predicted.
            
        '''
        
        if exog is None and self.included_exog:
            raise Exception(
                f"Forecaster trained with exogenous variable/s. "
                f"Same variable/s must be provided in `predict()`."
            )
            
        if exog is not None and not self.included_exog:
            raise Exception(
                f"Forecaster trained without exogenous variable/s. "
                f"`exog` must be `None` in `predict()`."
            )
        
        if exog is not None:
            self._check_exog(
                exog=exog, ref_type = self.exog_type, ref_shape=self.exog_shape
            )
            exog = self._preproces_exog(exog=exog)
            if exog.shape[0] < self.steps:
                raise Exception(
                    f"`exog` must have at least as many values as `steps` predicted."
                )
            exog = self._exog_to_multi_output(exog=exog)
     
        if last_window is not None:
            self._check_last_window(last_window=last_window)
            last_window = self._preproces_last_window(last_window=last_window)
            if last_window.shape[0] < self.max_lag:
                raise Exception(
                    f"`last_window` must have as many values as as needed to "
                    f"calculate the maximum lag ({self.max_lag})."
                )
        else:
            last_window = self.last_window.copy()
            
        X = last_window[-self.lags].reshape(1, -1)
        if exog is None:
            predictions = self.regressor.predict(X=X)
        else:
            X = np.hstack([X, exog[:self.steps]])
            predictions = self.regressor.predict(
                            X = X
                          )
            
        predictions = predictions.reshape(-1)
            
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
            
        return
    
    
    def _check_last_window(self, last_window: Union[np.ndarray, pd.Series]) -> None:
        '''
        Raise Exception if `last_window` is not 1D `np.ndarray` or `pd.Series`.
        
        Parameters
        ----------        
        last_window : np.ndarray, pd.Series
            Time series values

        '''
        
        if not isinstance(last_window, (np.ndarray, pd.Series)):
            raise Exception('`last_window` must be `1D np.ndarray` or `pd.Series`.')
        elif isinstance(last_window, np.ndarray) and last_window.ndim != 1:
            raise Exception(
                f"`last_window` must be `1D np.ndarray` o `pd.Series`, "
                f"got `np.ndarray` with {last_window.ndim} dimensions."
            )
            
        return
        
        
    def _check_exog(self, exog: Union[np.ndarray, pd.Series], 
                    ref_type: type=None, ref_shape: tuple=None) -> None:
        '''
        Raise Exception if `exog` is not `np.ndarray` or `pd.Series`.
        If `ref_shape` is provided, raise Exception if `ref_shape[1]` do not match
        `exog.shape[1]` (number of columns).
        
        Parameters
        ----------        
        exog : np.ndarray, pd.Series
            Time series values

        '''
            
        if not isinstance(exog, (np.ndarray, pd.Series)):
            raise Exception('`exog` must be `np.ndarray` or `pd.Series`.')
            
        if isinstance(exog, np.ndarray) and exog.ndim > 2:
            raise Exception(
                    f" If `exog` is `np.ndarray`, maximum allowed dim=2. "
                    f"Got {exog.ndim}."
                )
            
        if ref_type is not None:
            
            if ref_type == pd.Series:
                if isinstance(exog, pd.Series):
                    return
                elif isinstance(exog, np.ndarray) and exog.ndim == 1:
                    return
                elif isinstance(exog, np.ndarray) and exog.shape[1] == 1:
                    return
                else:
                    raise Exception(
                        f"`exog` must be: `pd.Series`, `np.ndarray` with 1 dimension"
                        f"or `np.ndarray` with 1 column in the second dimension. "
                        f"Got `np.ndarray` with {exog.shape[1]} columns."
                    )
                    
            if ref_type == np.ndarray:
                if exog.ndim == 1 and ref_shape[1] == 1:
                    return
                elif exog.ndim == 1 and ref_shape[1] > 1:
                    raise Exception(
                        f"`exog` must have {ref_shape[1]} columns. "
                        f"Got `np.ndarray` with 1 dimension or `pd.Series`."
                    )
                elif ref_shape[1] != exog.shape[1]:
                    raise Exception(
                        f"`exog` must have {ref_shape[1]} columns. "
                        f"Got `np.ndarray` with {exog.shape[1]} columns."
                    )     
        return
    
        
    def _preproces_y(self, y) -> np.ndarray:
        
        '''
        Transforms `y` to 1D `np.ndarray` if it is `pd.Series`.
        
        Parameters
        ----------        
        y :1D np.ndarray, pd.Series
            Time series values

        Returns 
        -------
        y: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(y, pd.Series):
            return y.to_numpy().copy()
        else:
            return y.copy()
        
    def _preproces_last_window(self, last_window) -> np.ndarray:
        
        '''
        Transforms `last_window` to 1D `np.ndarray` if it is `pd.Series`.
        
        Parameters
        ----------        
        last_window :1D np.ndarray, pd.Series
            Time series values

        Returns 
        -------
        last_window: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(last_window, pd.Series):
            return last_window.to_numpy().copy()
        else:
            return last_window.copy()
        
        
    def _preproces_exog(self, exog) -> np.ndarray:
        
        '''
        Transforms `exog` to `np.ndarray` if it is `pd.Series`.
        If 1D `np.ndarray` reshape it to (n_samples, 1)
        
        Parameters
        ----------        
        exog : np.ndarray, pd.Series
            Time series values

        Returns 
        -------
        exog: np.ndarray, shape(samples,)
        '''
        
        if isinstance(exog, pd.Series):
            exog_prep = exog.to_numpy().reshape(-1, 1).copy()
        elif isinstance(exog, np.ndarray) and exog.ndim == 1:
            exog_prep = exog.reshape(-1, 1).copy()
        else:
            exog_prep = exog.copy()
            
        return exog_prep
    
    def _exog_to_multi_output(self, exog):
        
        '''
        Transforms `exog` to `np.ndarray` with the shape needed for multioutput
        regresors.
        
        Parameters
        ----------        
        exog : np.ndarray, shape(samples,)
            Time series values

        Returns 
        -------
        exog_transformed: np.ndarray, shape(samples - self.max_lag, self.steps)
        '''

        exog_transformed = []

        for column in range(exog.shape[1]):

            exog_column_transformed = []

            for i in range(exog.shape[0] - (self.steps -1)):
                exog_column_transformed.append(exog[i:i + self.steps, column])

            if len(exog_column_transformed) > 1:
                exog_column_transformed = np.vstack(exog_column_transformed)

            exog_transformed.append(exog_column_transformed)

        if len(exog_transformed) > 1:
            exog_transformed = np.hstack(exog_transformed)
        else:
            exog_transformed = exog_column_transformed

        return exog_transformed
    
    
    def set_params(self, **params: dict) -> None:
        '''
        Set new values to the parameters of the scikit learn model stored in the
        forecaster. A separate model is created for each forecast time step using
        `sklearn.multioutput.MultiOutputRegressor`. This scikit learn class is a
        wrapper that automize fitting one regressor per target. It is important
        to note that all models share the same configuration of parameters and 
        hiperparameters.
        
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
        

    def get_coef(self, step) -> np.ndarray:
        '''      
        Return estimated coefficients for the linear regression model stored in
        the forecaster for a specific step. Since a separate model is created for
        each forecast time step, it is necessary to select the model from which
        retireve information.
        
        Only valid when the forecaster has been trained using as `regressor:
        `LinearRegression()`, `Lasso()` or `Ridge()`.
        
        Parameters
        ----------
        step : int
            Model from which retireve information (a separate model is created for
            each forecast time step).

        Returns 
        -------
        coef : 1D np.ndarray
            Value of the coefficients associated with each predictor (lag).
            Coefficients are aligned so that `coef[i]` is the value associated
            with `self.lags[i]`.
        
        '''
        
        if step > self.steps:
            raise Exception(
                f"Forecaster traied for {self.steps} steps. Got step={step}."
            )
            
        
        valid_instances = (sklearn.linear_model._base.LinearRegression,
                           sklearn.linear_model._coordinate_descent.Lasso,
                           sklearn.linear_model._ridge.Ridge
                           )
        
        if not isinstance(self.regressor.estimator, valid_instances):
            warnings.warn(
                ('Only forecasters with `regressor` `LinearRegression()`, ' +
                 ' `Lasso()` or `Ridge()` have coef.')
            )
            return
        else:
            coef = self.regressor.estimators_[step-1].coef_
            
        return coef

    
    def get_feature_importances(self, step) -> np.ndarray:
        '''      
        Return impurity-based feature importances of the model stored in
        the forecaster for a specific step. Since a separate model is created for
        each forecast time step, it is necessary to select the model from which
        retireve information.
        
        Only valid when the forecaster has been trained using
        `regressor=GradientBoostingRegressor()` or `regressor=RandomForestRegressor`.

        Parameters
        ----------
        step : int
            Model from which retireve information (a separate model is created for
            each forecast time step).

        Returns 
        -------
        feature_importances : 1D np.ndarray
        Impurity-based feature importances associated with each predictor (lag).
        Values are aligned so that `feature_importances[i]` is the value
        associated with `self.lags[i]`.
        '''
        
        if step > self.steps:
            raise Exception(
                f"Forecaster traied for {self.steps} steps. Got step={step}."
            )
        
        valid_instances = (sklearn.ensemble._forest.RandomForestRegressor,
                           sklearn.ensemble._gb.GradientBoostingRegressor)

        if not isinstance(self.regressor.estimator, valid_instances):
            warnings.warn(
                ('Only forecasters with `regressor=GradientBoostingRegressor()` '
                 'or `regressor=RandomForestRegressor`.')
            )
            return
        else:
            feature_importances = self.regressor.estimators_[step-1].feature_importances_

        return feature_importances