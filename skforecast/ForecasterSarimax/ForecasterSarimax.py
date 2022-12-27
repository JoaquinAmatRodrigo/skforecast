################################################################################
#                            ForecasterSarimax                                 #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional
import warnings
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
import pmdarima
from pmdarima.arima import ARIMA

import skforecast
from ..utils import transform_series
from ..utils import transform_dataframe

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class ForecasterSarimax():
    """
    This class turns ARIMA model from pmdarima library into a Forecaster compatible with 
    the skforecast API.
    
    Parameters
    ----------
    regressor : pmdarima.arima.ARIMA
        An instance of an ARIMA from pmdarima library. This model internally wraps the
        statsmodels SARIMAX class.

    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster. 
        **New in version 0.5.0**

    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    
    Attributes
    ----------
    regressor : pmdarima.arima.ARIMA
        An instance of an ARIMA from pmdarima library. The model internally wraps the
        statsmodels SARIMAX class

    params: dict
        Parameters of the sarimax model.
        
    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.

    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
   
    window_size : int
        Size of the window needed to create the predictors. It is equal to
        max value of the parameters p,q,P,D of the ARIMA.

    last_window : pandas Series
        Last window the forecaster has seen during trained. It stores the
        values needed to predict the next `step` right after the training data.
        
    fitted : Bool
        Tag to identify if the regressor has been fitted (trained).
        
    index_type : type
        Type of index of the input used in training.
        
    index_freq : str
        Frequency of Index of the input used in training.
        
    training_range : pandas Index
        First and last values of index of the data used during training.
        
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
        
    exog_type : type
        Type of exogenous variable/s used in training.
        
    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    creation_date : str
        Date of creation.

    fit_date : str
        Date of last fit.

    skforcast_version : str
        Version of skforecast library used to create the forecaster.

    python_version : str
        Version of python used to create the forecaster.
        **New in version 0.5.0**
     
    """
    
    def __init__(
        self,
        regressor: ARIMA,
        transformer_y: Optional[object]=None,
        transformer_exog: Optional[object]=None,
    ) -> None:
        
        self.regressor               = regressor
        self.transformer_y           = transformer_y
        self.transformer_exog        = transformer_exog
        self.index_freq              = None
        self.training_range          = None
        self.last_window             = None
        self.included_exog           = False
        self.exog_type               = None
        self.exog_col_names          = None
        self.fitted                  = False
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                = None
        self.skforcast_version       = skforecast.__version__
        self.python_version          = sys.version.split(" ")[0]
        self.index_type              = None
        
        if not isinstance(self.regressor, pmdarima.arima.ARIMA):
            raise ValueError(
                f"regressor must be an instance of type pmdarima.arima.ARIMA. "
                f"Got {type(regressor)}."
            )

        self.params = self.regressor.get_params(deep=True)
        self.window_size = max([
                            self.params['order'][0], self.params['order'][2],
                            self.params['seasonal_order'][0], self.params['seasonal_order'][2]
                           ])


    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterSarimax object is printed.
        """

        info = (
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"{str(type(self)).split('.')[1]} \n"
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"Regressor: {self.regressor} \n"
            f"Regressor parameters: {self.params} \n"
            f"Window size: {self.window_size} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Exogenous included: {self.included_exog} \n"
            f"Type of exogenous variable: {self.exog_type} \n"
            f"Exogenous variables names: {self.exog_col_names} \n"
            f"Training range: {self.training_range.to_list() if self.fitted else None} \n"
            f"Training index type: {str(self.index_type).split('.')[-1][:-2] if self.fitted else None} \n"
            f"Training index frequency: {self.index_freq if self.fitted else None} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforcast_version} \n"
            f"Python version: {self.python_version} \n"
        )

        return info

        
    def fit(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
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
        self.X_train_col_names    = None
        self.in_sample_residuals  = None
        self.fitted               = False
        self.training_range       = None
        
        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else exog.name

        y = transform_series(
                series            = y,
                transformer       = self.transformer_y,
                fit               = True,
                inverse_transform = False
            )

        if exog is not None:
            if isinstance(exog, pd.Series):
                exog = transform_series(
                            series            = exog,
                            transformer       = self.transformer_exog,
                            fit               = True,
                            inverse_transform = False
                       )
                # pmdarima.arima.ARIMA only accepts DataFrames or 2d-arrays as exog       
                exog = exog.to_frame(name=exog.name)
            else:
                exog = transform_dataframe(
                            df                = exog,
                            transformer       = self.transformer_exog,
                            fit               = True,
                            inverse_transform = False
                       )

        self.regressor.fit(y=y, X=exog)
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = y.index[[0, -1]]
        self.index_type = type(y.index)
        if isinstance(y.index, pd.DatetimeIndex):
            self.index_freq = y.index.freqstr
        else: 
            self.index_freq = y.index.step
        self.last_window = y.iloc[-self.window_size:].copy()
    
          
    def predict(
        self,
        steps: int,
        last_window: Optional[pd.Series]=None,
        last_window_exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> pd.Series:
        """
        Predict n steps ahead. It is a recursive process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
        last_window : pandas Series, default `None`
            Values of the series used to create the predictors needed in the 
            first iteration of prediction (t + 1). This is used when the predictions
            do not follow directly from the end of the training data.

        last_window_exog : pandas Series, pandas DataFrame, default `None`
            Values of the exogenous variables aligned with `last_window`. Only
            need when `last_window` is not None and the forecaster has been
            trained including exogenous variables. This is used when the predictions
            do not follow directly from the end of the training data.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Value of the exogenous variable/s for the next steps.

        Returns 
        -------
        predictions : pandas Series
            Predicted values.
            
        """

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
                # pmdarima.arima.ARIMA only accepts DataFrames or 2d-arrays as exog
                exog = exog.to_frame(name=exog.name)
            
            exog_values = exog.iloc[:steps, ]
        else:
            exog_values = None


        if last_window is None:
        # Predictions follow directly from the end of the training data
            predictions = self.regressor.predict(
                              n_periods = steps,
                              X         = exog_values
                          )
        else:
        # Predictions do not follow directly from the end of the training data. The
        # internal statsmodels SARIMAX model need to be updated using the apply method.
            last_window = transform_series(
                            series            = last_window,
                            transformer       = self.transformer_y,
                            fit               = False,
                            inverse_transform = False
                        )

            if last_window_exog is not None:
                if isinstance(last_window_exog, pd.DataFrame):
                    exog = transform_dataframe(
                               df                = last_window_exog,
                               transformer       = self.transformer_exog,
                               fit               = False,
                               inverse_transform = False
                           )
                else:
                    exog = transform_series(
                               series            = last_window_exog,
                               transformer       = self.transformer_exog,
                               fit               = False,
                               inverse_transform = False
                           )
                    exog = exog.to_frame(name=exog.name)

            if self.included_exog:
                self.regressor.arima_res_ = self.regressor.arima_res_.apply(
                                                endog = last_window,
                                                exog  = last_window_exog,
                                                refit = False
                                            )
            else:
                self.regressor.arima_res_ = self.regressor.arima_res_.apply(
                                                endog = last_window,
                                                refit = False
                                            )
            
            predictions = self.regressor.predict(
                              n_periods   = steps,
                              X           = exog_values
                          )

        # Reverse the transformation if needed
        predictions = transform_series(
                          series            = predictions,
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = True
                      )
        predictions.name = 'pred'
        
        return predictions
    
        
    def predict_interval(
        self,
        steps: int,
        last_window: Optional[pd.Series]=None,
        last_window_exog: Optional[pd.Series]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
        alpha: float=0.05,
    ) -> pd.DataFrame:
        """
        Iterative process in which, each prediction, is used as a predictor
        for the next step. Both, predictions and intervals, are returned.
        
        Parameters
        ---------- 
        steps : int
            Number of future steps predicted.
            
        last_window : pandas Series, default `None`
            Values of the series used to create the predictors needed in the 
            first iteration of prediction (t + 1).

        last_window_exog : pandas Series, pandas DataFrame, default `None`
            Values of the exogenous variables aligned with `last_window`. Only
            need when `last_window` is not None and the forecaster has been
            trained including exogenous variables.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
            
        alpha : float, default=0.05
            The confidence intervals for the forecasts are (1 - alpha) %

        Returns 
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval:

            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound interval of the interval.

        """
        
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
                # pmdarima.arima.ARIMA only accepts DataFrames or 2d-arrays as exog
                exog = exog.to_frame(name=exog.name)
            
            exog_values = exog.iloc[:steps, ]
        else:
            exog_values = None


        if last_window is None:
            # Predictions follow directly from the end of the training data
            predicted_mean, conf_int = self.regressor.predict(
                                        n_periods       = steps,
                                        X               = exog_values,
                                        alpha           = alpha,
                                        return_conf_int = True
                                      )
        else:
            # Predictions do not follow directly from the end of the training data. The
            # internal statsmodels SARIMAX model need to be updated using the apply method.
            last_window = transform_series(
                            series            = last_window,
                            transformer       = self.transformer_y,
                            fit               = False,
                            inverse_transform = False
                        )

            if last_window_exog is not None:
                if isinstance(last_window_exog, pd.DataFrame):
                    exog = transform_dataframe(
                            df                = last_window_exog,
                            transformer       = self.transformer_exog,
                            fit               = False,
                            inverse_transform = False
                        )
                else:
                    exog = transform_series(
                            series            = last_window_exog,
                            transformer       = self.transformer_exog,
                            fit               = False,
                            inverse_transform = False
                        )
                    exog = exog.to_frame(name=exog.name)

            if self.included_exog:
                self.regressor.arima_res_ = self.regressor.arima_res_.apply(
                                                endog = last_window,
                                                exog  = last_window_exog,
                                                refit  = False
                                            )
            else:
                self.regressor.arima_res_ = self.regressor.arima_res_.apply(
                                                endog = last_window,
                                                refit  = False
                                            )

            predicted_mean, conf_int = self.regressor.predict(
                                            n_periods       = steps,
                                            X               = exog_values,
                                            alpha           = alpha,
                                            return_conf_int = True
                                    )

                                    
        predictions = predicted_mean.to_frame(name="pred")
        predictions['lower_bound'] = conf_int[:, 0]
        predictions['upper_bound'] = conf_int[:, 1]

        for col in predictions.columns:
            predictions[col] = transform_series(
                                   series            = predictions[col],
                                   transformer       = self.transformer_y,
                                   fit               = False,
                                   inverse_transform = True
                               )

        return predictions

    
    def set_params(
        self, 
        **params: dict
    ) -> None:
        """
        Set new values to the parameters of the model stored in the forecaster.
        
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
        
    
    def get_feature_importance(
        self
    ) -> pd.DataFrame:
        """      
        Return feature importance of the regressor stored in the
        forecaster.

        Parameters
        ----------
        self

        Returns
        -------
        feature_importance : pandas DataFrame
            Feature importance associated with each predictor.

        """

        try:
            feature_importance = self.regressor.params().to_frame().reset_index()
            feature_importance.columns = ['feature', 'importance']
        except:   
            warnings.warn(
                f"Impossible to access feature importance for regressor of type {type(self.regressor)}."
            )
            feature_importance = None

        return feature_importance