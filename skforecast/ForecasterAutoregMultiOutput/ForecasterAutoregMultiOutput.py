################################################################################
#                         ForecasterAutoregMultiOutput                         #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional
import warnings
import logging
import numpy as np
import sklearn

import skforecast
from ..ForecasterAutoregDirect import ForecasterAutoregDirect


logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class ForecasterAutoregMultiOutput(ForecasterAutoregDirect):
    '''
    This class turns any regressor compatible with the scikit-learn API into a
    autoregressive direct multi-step forecaster. A separate model is created for
    each forecast time step. See documentation for more details.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        
    lags : int, list, 1d numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list`, `numpy ndarray` or range: include only lags present in `lags`.
            
    steps : int
        Maximum number of future steps the forecaster will predict when using
        method `predict()`. Since a different model is created for each step,
        this value should be defined before training.

    transformer_y : transformer (preprocessor) compatible with the scikit-learn
                    preprocessing API, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.

    transformer_exog : transformer (preprocessor) compatible with the scikit-learn
                       preprocessing API, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        One instance of this regressor is trainned for each step. All
        them are stored in `self.regressors_`.
        
    regressors_ : dict
        Dictionary with regressors trained for each step.
        
    steps : int
        Number of future steps the forecaster will predict when using method
        `predict()`. Since a different model is created for each step, this value
        should be defined before training.
        
    lags : numpy ndarray
        Lags used as predictors.
        
    max_lag : int
        Maximum value of lag included in `lags`.

    last_window : pandas Series
        Last window the forecaster has seen during trained. It stores the
        values needed to predict the next `step` right after the training data.
        
    window_size: int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.
        
    fitted: Bool
        Tag to identify if the regressor has been fitted (trained).
        
    index_type : type
        Type of index of the input used in training.
        
    index_freq : str
        Frequency of Index of the input used in training.
        
    training_range: pandas Index
        First and last index of samples used during training.
        
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
        
    exog_type : type
        Type of exogenous variable/s used in training.
        
    exog_col_names : tuple
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    X_train_col_names : tuple
        Names of columns of the matrix created internally for training.

    creation_date: str
        Date of creation.

    fit_date: str
        Date of last fit.

    skforcast_version: str
        Version of skforecast library used to create the forecaster.

    python_version: str
        Version of python used to create the forecaster.
        
    Notes
    -----
    A separate model is created for each forecast time step. It is important to
    note that all models share the same configuration of parameters and
    hyperparameters.
     
    '''

    def __init__(self, regressor, steps: int,
                 lags: Union[int, np.ndarray, list],
                 transformer_y = None,
                 transformer_exog = None) -> None:
    
        # Add warning to __init__ ForecasterAutoregDirect
        ForecasterAutoregDirect.__init__(self, regressor, steps, lags, transformer_y, transformer_exog)

        warnings.warn(
            f'ForecasterAutoregMultiOutput has been renamed to ForecasterAutoregDirect.'
            f'This class will be removed in skforecast 0.6.0.'
        )
        
    def __repr__(self) -> str:
        '''
        Information displayed when a ForecasterAutoregMultiOutput object is printed.
        '''
           
        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            name_pipe_steps = tuple(name + "__" for name in self.regressor.named_steps.keys())
            params = {key : value for key, value in self.regressor.get_params().items() \
                     if key.startswith(name_pipe_steps)}
        else:
            params = self.regressor.get_params()

        info = (
            f"{'=' * len(str(type(self)).split('.')[-1].replace(''''>''', ''))} \n"
            f"{str(type(self)).split('.')[-1].replace(''''>''', '')} \n"
            f"{'=' * len(str(type(self)).split('.')[-1].replace(''''>''', ''))} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Window size: {self.window_size} \n"
            f"Maximum steps predicted: {self.steps} \n"
            f"Included exogenous: {self.included_exog} \n"
            f"Type of exogenous variable: {self.exog_type} \n"
            f"Exogenous variables names: {self.exog_col_names} \n"
            f"Training range: {self.training_range.to_list() if self.fitted else None} \n"
            f"Training index type: {str(self.index_type).split('.')[-1][:-2] if self.fitted else None} \n"
            f"Training index frequency: {self.index_freq if self.fitted else None} \n"
            f"Regressor parameters: {params} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforcast_version} \n"
            f"Python version: {self.python_version} \n"
        )
        
        return info
