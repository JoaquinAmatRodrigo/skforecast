################################################################################
#                                 Sarimax                                      #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under the BSD 3-Clause License.                                              #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional
import warnings
import logging
import inspect
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.statespace.sarimax import SARIMAX

import skforecast

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def check_fitted(func):
    def wrapper(self, *args, **kwargs):

        if not self.fitted:
            raise NotFittedError(
                ("This Forecaster instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using predict.")
            )
        
        result = func(self, *args, **kwargs)
        
        return result
    
    return wrapper


class Sarimax(BaseEstimator, RegressorMixin):
    """
    A universal sklearn-style wrapper for statsmodels SARIMAX.

    Parameters
    ----------


    Attributes
    ----------

    """

    def __init__(
        self,
        order: tuple=(1, 0, 0),
        seasonal_order: tuple=(0, 0, 0, 0),
        trend: str=None,
        measurement_error: bool=False,
        time_varying_regression: bool=False,
        mle_regression: bool=True,
        simple_differencing: bool=False,
        enforce_stationarity: bool=True,
        enforce_invertibility: bool=True,
        hamilton_representation: bool=False,
        concentrate_scale: bool=False,
        trend_offset: int=1,
        use_exact_diffuse: bool=False,
        dates = None,
        freq = None,
        missing = 'none',
        validate_specification: bool=True,
        method: str='lbfgs',
        maxiter: int=50,
        start_params: np.ndarray= None,
        disp: bool=False,
        sm_init_kwargs: dict={},
        sm_fit_kwargs: dict={},
        sm_predict_kwargs: dict={}
    ) -> None:

        self.order                   = order
        self.seasonal_order          = seasonal_order
        self.trend                   = trend
        self.measurement_error       = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression          = mle_regression
        self.simple_differencing     = simple_differencing
        self.enforce_stationarity    = enforce_stationarity
        self.enforce_invertibility   = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale       = concentrate_scale
        self.trend_offset            = trend_offset
        self.use_exact_diffuse       = use_exact_diffuse
        self.dates                   = dates
        self.freq                    = freq
        self.missing                 = missing
        self.validate_specification  = validate_specification
        self.method                  = method
        self.maxiter                 = maxiter
        self.start_params            = start_params
        self.disp                    = disp

        # Create the dictionaries with the additional statsmodels parameters to be  
        # used during the init, fit and predict methods. Note that the statsmodels 
        # SARIMAX.fit parameters `method`, `max_iter`, `start_params` and `disp` 
        # have been moved to the initialization of this model and will have 
        # priority over those provided by the user using via `sm_fit_kwargs`.
        self.sm_init_kwargs    = sm_init_kwargs
        self.sm_fit_kwargs     = sm_fit_kwargs
        self.sm_predict_kwargs = sm_predict_kwargs

        # Params that can be set with the `set_params` method
        _, _, _, _sarimax_params = inspect.getargvalues(inspect.currentframe())
        _sarimax_params.pop("self")
        self._sarimax_params = _sarimax_params

        self._consolidate_kwargs()

        # Create Results Attributes 
        self.set_output     = None
        self.sarimax        = None
        self.fitted         = False
        self.sarimax_res    = None
        self.training_index = None


    def __repr__(self):

        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        
        return f"Sarimax({p},{d},{q})({P},{D},{Q})[{m}]"


    def _consolidate_kwargs(
        self
    ) -> None:
        """
        """
        
        # statsmodels.tsa.statespace.SARIMAX parameters
        _init_kwargs = self.sm_init_kwargs.copy()
        _init_kwargs.update({
           'order': self.order,
           'seasonal_order': self.seasonal_order,
           'trend': self.trend,
           'measurement_error': self.measurement_error,
           'time_varying_regression': self.time_varying_regression,
           'mle_regression': self.mle_regression,
           'simple_differencing': self.simple_differencing,
           'enforce_stationarity': self.enforce_stationarity,
           'enforce_invertibility': self.enforce_invertibility,
           'hamilton_representation': self.hamilton_representation,
           'concentrate_scale': self.concentrate_scale,
           'trend_offset': self.trend_offset,
           'use_exact_diffuse': self.use_exact_diffuse,
           'dates': self.dates,
           'freq': self.freq,
           'missing': self.missing,
           'validate_specification': self.validate_specification
        })
        self._init_kwargs = _init_kwargs

        # statsmodels.tsa.statespace.SARIMAX.fit parameters
        _fit_kwargs = self.sm_fit_kwargs.copy()
        _fit_kwargs.update({
           'method': self.method,
           'maxiter': self.maxiter,
           'start_params': self.start_params,
           'disp': self.disp,
        })        
        self._fit_kwargs = _fit_kwargs

        # statsmodels.tsa.statespace.SARIMAXResults.get_forecast parameters
        self._predict_kwargs = self.sm_predict_kwargs.copy()

        return
    
        
    def _create_sarimax(
        self,
        endog: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None
    ) -> None:
        """
        A helper function to create a new statsmodel.SARIMAX.

        Parameters
        ----------
        endog : pandas.Series
            The endogenous variable.
        exog : pandas.DataFrame
            The exogenous variables.
        
        Returns
        -------
        None

        """

        self.sarimax = SARIMAX(endog=endog, exog=exog, **self._init_kwargs)

        return


    def fit(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None,
        X: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None
    ) -> None:
        """
        Fit the model to the data.

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

        exog = X if X is not None else exog
        self.set_output = 'numpy' if isinstance(y, np.ndarray) else 'pandas'
        
        self._create_sarimax(endog=y, exog=exog)
        self.sarimax_res = self.sarimax.fit(**self._fit_kwargs)
        self.fitted = True
        
        if self.set_output == 'pandas':
            self.training_index = y.index

        return
    
    @check_fitted
    def predict(
        self,
        steps: int,
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None, 
        return_conf_int: bool=False,
        alpha: float=0.05
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        """
        predictions = self.sarimax_res.get_forecast(
                          steps = steps,
                          exog  = exog,
                          **self._predict_kwargs
                      )
        
        if not return_conf_int:
            predictions = predictions.predicted_mean
            if self.set_output == 'pandas':
                predictions = predictions.rename("pred").to_frame()
        else:
            if self.set_output == 'numpy':
                predictions = np.column_stack(
                                  [predictions.predicted_mean,
                                   predictions.conf_int(alpha=alpha)]
                              )
            else:
                predictions = pd.concat((
                                  predictions.predicted_mean.rename("pred"),
                                  predictions.conf_int(alpha=alpha)),
                                  axis = 1
                              )
                predictions.columns = ['pred', 'lower_bound', 'upper_bound']

        return predictions


    @check_fitted
    def append(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None,
        refit: bool=False,
        copy_initialization: bool=False,
        **kwargs
    ) -> None:
        """
        """
        fit_kwargs = self._fit_kwargs if refit else None

        self.sarimax_res = self.sarimax_res.append(
                               endog               = y,
                               exog                = exog,
                               refit               = refit,
                               copy_initialization = copy_initialization,
                               fit_kwargs          = fit_kwargs,
                               **kwargs
                           )


    @check_fitted
    def apply(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None,
        refit: bool=False,
        copy_initialization: bool=False,
        **kwargs
    ) -> None:
        """
        """
        fit_kwargs = self._fit_kwargs if refit else None

        self.sarimax_res = self.sarimax_res.apply(
                               endog               = y,
                               exog                = exog,
                               refit               = refit,
                               copy_initialization = copy_initialization,
                               fit_kwargs          = fit_kwargs,
                               **kwargs
                           )


    @check_fitted
    def extend(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]=None,
        **kwargs
    ) -> None:
        """
        """

        self.sarimax_res = self.sarimax_res.extend(
                               endog               = y,
                               exog                = exog,
                               **kwargs
                           )


    def set_params(
        self, 
        **params: dict
    ) -> None:
        """
        Set new values to the parameters of the regressor.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns
        -------
        None

        """
        
        params = {k:v for k,v in params.items() if k in self._sarimax_params}
        print(params)
        for key, value in params.items():
            setattr(self, key, value)

        self._consolidate_kwargs()

        self.fitted = False
        self.sarimax_res = None
        self.training_index = None

        # Create the dictionaries with the additional statsmodels parameters to be  
        # used during the init, fit and predict methods. Note that the statsmodels 
        # SARIMAX.fit parameters `method`, `max_iter`, `start_params` and `disp` 
        # have been moved to the initialization of this model and will have 
        # priority over those provided by the user using via `sm_fit_kwargs`.
        
        # statsmodels.tsa.statespace.SARIMAX parameters


    @check_fitted
    def params(self):
        """
        Get the parameters of the model. The order of variables is the trend
        coefficients and the :func:`k_exog` exogenous coefficients, then the
        :func:`k_ar` AR coefficients, and finally the :func:`k_ma` MA
        coefficients.

        Returns
        -------
        params : array-like
            The parameters of the model.
        """

        return self.sarimax_res.params


    @check_fitted
    def summary(self):
        """
        Get a summary of the ARIMA model
        """

        return self.sarimax_res.summary()
