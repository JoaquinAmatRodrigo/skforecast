################################################################################
#                         ForecasterAutoregMultiOutput                         #
#                                                                              #
# This work by Joaquin Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

from typing import Union, Dict, List, Tuple, Any, Optional
import warnings
import logging
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from sklearn.base import clone

import skforecast
from ..ForecasterBase import ForecasterBase
from ..utils import check_y
from ..utils import check_exog
from ..utils import preprocess_y
from ..utils import preprocess_last_window
from ..utils import preprocess_exog
from ..utils import exog_to_multi_output
from ..utils import expand_index
from ..utils import check_predict_input


logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)



class ForecasterAutoregMultiOutput(ForecasterBase):
    '''
    This class turns any regressor compatible with the scikit-learn API into a
    autoregressive multi-output forecaster. A separate model is created for each
    forecast time step. See Notes for more details.
    
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
        
    Notes
    -----
    A separate model is created for each forecast time step. It is important to
    note that all models share the same configuration of parameters and
    hyperparameters.
     
    '''
    
    def __init__(self, regressor, steps: int,
                 lags: Union[int, np.ndarray, list]) -> None:
        
        self.regressor     = regressor
        self.steps         = steps
        self.regressors_   = {step: clone(self.regressor) for step in range(steps)}
        self.index_type           = None
        self.index_freq           = None
        self.training_range       = None
        self.last_window          = None
        self.included_exog        = False
        self.exog_type            = None
        self.exog_col_names       = None
        self.X_train_col_names    = None
        self.fitted               = False
        self.creation_date        = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date             = None
        self.skforcast_version    = skforecast.__version__

        if isinstance(lags, int) and lags < 1:
            raise Exception('Minimum value of lags allowed is 1')
            
        if isinstance(lags, (list, range, np.ndarray)) and min(lags) < 1:
            raise Exception('Minimum value of lags allowed is 1')

        if isinstance(lags, (list, np.ndarray)):
            for lag in lags:
                if not isinstance(lag, (int, np.int64, np.int32)):
                    raise Exception('Values in lags must be int.')
            
        if isinstance(lags, int):
            self.lags = np.arange(lags) + 1
        elif isinstance(lags, (list, range)):
            self.lags = np.array(lags)
        elif isinstance(lags, np.ndarray):
            self.lags = lags
        else:
            raise Exception(
                '`lags` argument must be int, 1d numpy ndarray, range or list. '
                f"Got {type(lags)}"
            )
            
        self.max_lag  = max(self.lags)
        self.window_size = self.max_lag
                
        
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
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"{str(type(self)).split('.')[1]} \n"
            f"{'=' * len(str(type(self)).split('.')[1])} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
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
        )

        return info
    
    
    def _create_lags(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''       
        Transforms a 1d array into a 2d array (X) and a 1d array (y). Each row
        in X is associated with a value of y and it represents the lags that
        precede it.
        
        Notice that, the returned matrix X_data, contains the lag 1 in the first
        column, the lag 2 in the second column and so on.
        
        Parameters
        ----------        
        y : 1d numpy ndarray
            Training time series.

        Returns 
        -------
        X_data : 2d numpy ndarray, shape (samples - max(self.lags), len(self.lags))
            2d numpy array with the lagged values (predictors).
        
        y_data : 2d numpy ndarray, shape (samples - max(self.lags),)
            Values of the time series related to each row of `X_data` for each step.
            
        '''

        n_splits = len(y) - self.max_lag - (self.steps -1)
        X_data  = np.full(shape=(n_splits, self.max_lag), fill_value=np.nan, dtype=float)
        y_data  = np.full(shape=(n_splits, self.steps), fill_value=np.nan, dtype= float)

        for i in range(n_splits):
            X_index = np.arange(i, self.max_lag + i)
            y_index = np.arange(self.max_lag + i, self.max_lag + i + self.steps)

            X_data[i, :] = y[X_index]
            y_data[i, :] = y[y_index]
            
        X_data = X_data[:, -self.lags] # Only keep needed lags
            
        return X_data, y_data


    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Create training matrices from univariate time series and exogenous
        variables. The resulting matrices contain the target variable and predictors
        needed to train all the forecaster (one per step).      
        
        Parameters
        ----------        
        y : pandas Series
            Training time series.
            
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.


        Returns 
        -------
        X_train : pandas DataFrame, shape (len(y) - self.max_lag, len(self.lags) + exog.shape[1]*steps)
            Pandas DataFrame with the training values (predictors) for each step.
            
        y_train : pd.DataFrame, shape (len(y) - self.max_lag, )
            Values (target) of the time series related to each row of `X_train` 
            for each step.
        
        '''

        check_y(y=y)
        y_values, y_index = preprocess_y(y=y)

        if len(y_values) < self.max_lag + self.steps:
            raise Exception(
                f"Minimum length of `y` for training this forecaster is "
                f"{self.max_lag + self.steps}. Got {len(y_values)}"
            )
        if exog is not None:
            if len(exog) != len(y):
                raise Exception(
                    "`exog` must have same number of samples as `y`."
                )
            check_exog(exog=exog)
            exog_values, exog_index = preprocess_exog(exog=exog)
            if not (exog_index[:len(y_index)] == y_index).all():
                raise Exception(
                ('Different index for `y` and `exog`. They must be equal '
                'to ensure the correct alignment of values.')      
                )
      
        X_lags, y_train = self._create_lags(y=y_values)
        y_train_col_names = [f"y_step_{i}" for i in range(self.steps)]
        X_train_col_names = [f"lag_{i}" for i in self.lags]

        if exog is None:
            X_train = X_lags
        else:
            col_names_exog = exog.columns if isinstance(exog, pd.DataFrame) else [exog.name]
            # Transform exog to match multi output format
            X_exog = exog_to_multi_output(exog=exog_values, steps=self.steps)
            col_names_exog = [f"{col_name}_step_{i+1}" for col_name in col_names_exog for i in range(self.steps)]
            X_train_col_names.extend(col_names_exog)
            # The first `self.max_lag` positions have to be removed from X_exog
            # since they are not in X_lags.
            X_exog = X_exog[-X_lags.shape[0]:, ]
            X_train = np.column_stack((X_lags, X_exog))

        X_train = pd.DataFrame(
                    data    = X_train,
                    columns = X_train_col_names,
                    index   = y_index[self.max_lag + (self.steps -1): ]
                  )
        self.X_train_col_names = X_train_col_names
        y_train = pd.DataFrame(
                    data    = y_train,
                    index   = y_index[self.max_lag + (self.steps -1): ],
                    columns = y_train_col_names,
                 )
                        
        return X_train, y_train

    
    def filter_train_X_y_for_step(
        self,
        step: int,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:

        '''
        Select columns needed to train a forcaster for a specific step. The input
        matrices should be created with created with `create_train_X_y()`.         

        Parameters
        ----------
        step : int
            step for which columns must be selected selected. Starts at 0.

        X_train : pandas DataFrame
            Pandas DataFrame with the training values (predictors).
            
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.


        Returns 
        -------
        X_train_step : pandas DataFrame
            Pandas DataFrame with the training values (predictors) for step.
            
        y_train_step : pandas Series, shape (len(y) - self.max_lag)
            Values (target) of the time series related to each row of `X_train`.

        '''

        if step > self.steps - 1:
            raise Exception(
                f"Invalid value `step`. For this forecaster, the maximum step is {self.steps-1}."
            )

        y_train_step = y_train.iloc[:, step]

        if not self.included_exog:
            X_train_step = X_train
        else:
            idx_columns_lags = np.arange(len(self.lags))
            idx_columns_exog = np.arange(X_train.shape[1])[len(self.lags) + step::self.steps]
            idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
            X_train_step = X_train.iloc[:, idx_columns]

        return  X_train_step, y_train_step
    
    
    def fit(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
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

        # Reset values in case the forecaster has already been fitted.
        self.index_type           = None
        self.index_freq           = None
        self.last_window          = None
        self.included_exog        = False
        self.exog_type            = None
        self.exog_col_names       = None
        self.X_train_col_names    = None
        self.fitted               = False
        self.training_range       = None

        if exog is not None:
            self.included_exog = True
            self.exog_type = type(exog)
            self.exog_col_names = \
                 exog.columns.to_list() if isinstance(exog, pd.DataFrame) else exog.name

        X_train, y_train = self.create_train_X_y(y=y, exog=exog)
        
        # Train one regressor for each step 
        for step in range(self.steps):

            X_train_step, y_train_step = self.filter_train_X_y_for_step(
                                            step    = step,
                                            X_train = X_train,
                                            y_train = y_train
                                         )
            if not str(type(self.regressor)) == "<class 'xgboost.sklearn.XGBRegressor'>":
                self.regressors_[step].fit(X_train_step, y_train_step)
            else:
                self.regressors_[step].fit(X_train_step.to_numpy(), y_train_step.to_numpy())
        
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = preprocess_y(y=y)[1][[0, -1]]
        self.index_type = type(X_train.index)
        if isinstance(X_train.index, pd.DatetimeIndex):
            self.index_freq = y_train.index.freqstr
            self.last_window = y.loc[y_train.index[-1] - self.max_lag * y_train.index.freq: ]
        else: 
            self.index_freq = y_train.index.step
            self.last_window = y.loc[y_train.index[-1] - self.max_lag * y_train.index.step: ]
        

    
    def predict(
        self,
        steps: Optional[Union[int, None]]=None,
        last_window: Optional[pd.Series]=None,
        exog: Optional[Union[pd.Series, pd.DataFrame]]=None
    ) -> np.ndarray:
        '''
        Predict n steps ahead.

        Parameters
        ----------
        steps : int, None, default `None`
            Predict n steps ahead. `steps` must lower or equal to the value of
            steps defined when initializing the forecaster. If `None`, as many
            steps as defined in the initialization are predicted.

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

        if steps is None:
            steps = self.steps

        check_predict_input(
            steps          = steps,
            fitted         = self.fitted,
            included_exog  = self.included_exog,
            index_type     = self.index_type,
            index_freq     = self.index_freq,
            window_size    = self.window_size,
            last_window    = last_window,
            exog           = exog,
            exog_type      = self.exog_type,
            exog_col_names = self.exog_col_names,
            max_steps      = self.steps,
        ) 

        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                exog_values, _ = preprocess_exog(
                                    exog = exog[self.exog_col_names].iloc[:steps, ]
                                 )
            else: 
                exog_values, _ = preprocess_exog(
                                        exog = exog.iloc[:steps, ]
                                 )
            exog_values = exog_to_multi_output(exog=exog_values, steps=steps)

        else:
            exog_values = None

        if last_window is not None:
            last_window_values, last_window_index = preprocess_last_window(
                                                        last_window = last_window
                                                    )  
        else:
            last_window_values, last_window_index = preprocess_last_window(
                                                        last_window = self.last_window
                                                    )

        predictions = np.full(shape=steps, fill_value=np.nan)
        X_lags = last_window_values[-self.lags].reshape(1, -1)

        for step in range(steps):
            regressor = self.regressors_[step]
            if exog is None:
                X = X_lags
            else:
                # Only columns from exog related with the current step are selected.
                X = np.hstack([X_lags, exog_values[0][step::steps].reshape(1, -1)])
            with warnings.catch_warnings():
                # Suppress scikitlearn warning: "X does not have valid feature names,
                # but NoOpTransformer was fitted with feature names".
                warnings.simplefilter("ignore")
                predictions[step] = regressor.predict(X)

        predictions = pd.Series(
                        data  = predictions.reshape(-1),
                        index = expand_index(
                                    index = last_window_index,
                                    steps = steps
                                ),
                        name = 'pred'
                      )

        return predictions    
    
    
    def set_params(self, **params: dict) -> None:
        '''
        Set new values to the parameters of the scikit learn model stored in the
        forecaster. It is important to note that all models share the same 
        configuration of parameters and hyperparameters.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns 
        -------
        self
        
        '''
        
        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)
        self.regressors_ = {step: clone(self.regressor) for step in range(self.steps)}
        
        
        
    def set_lags(self, lags: Union[int, list, np.ndarray, range]) -> None:
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
        

    def get_coef(self, step) -> np.ndarray:
        '''      
        Return estimated coefficients for the regressor stored in the forecaster
        for a specific step. Since a separate model is created for each forecast
        time step, it is necessary to select the model from which retrieve the
        information.
        
        Only valid when regressor stores internally the feature coefficients in
        the attribute `coef_`.
        
        Parameters
        ----------
        step : int
            Model from which retrieve information (a separate model is created for
            each forecast time step). First step is 1.

        Returns 
        -------
        coef : pandas DataFrame
            Value of the coefficients associated with each predictor.
        
        '''
        warnings.warn(
            f'This method was deprecated in version 0.4.3 in favor of the get_feature_importance. '
            f'This method will be removed in 0.4.4', DeprecationWarning
        )

        if step > self.steps:
            raise Exception(
                f"Forecaster trained for {self.steps} steps. Got step={step}."
            )
        if step < 1:
            raise Exception("Minimum step is 1.")
        
        # Stored regressors start at index 0
        step = step - 1
            
        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            estimator = self.regressors_[step][-1]
        else:
            estimator = self.regressors_[step]

        try:
            idx_columns_lags = np.arange(len(self.lags))
            idx_columns_exog = np.array([], dtype=int)
            if self.included_exog:
                idx_columns_exog = np.arange(len(self.X_train_col_names))[len(self.lags) + step::self.steps]
            idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
            feature_names = [self.X_train_col_names[i] for i in idx_columns]
            feature_names = [name.replace(f"_step_{step+1}", "") for name in feature_names]
            coef = pd.DataFrame({
                        'feature': feature_names,
                        'coef' : estimator.coef_
                   })
        except:
            warnings.warn(
                f"Impossible to access feature coefficients for regressor of type {type(estimator)}. "
                f"This method is only valid when the regressor stores internally "
                f" the coefficients in the attribute `coef_`."
            )

            coef = None
            
        return coef

    
    def get_feature_importance(self, step) -> np.ndarray:
        '''      
        Return impurity-based feature importance of the model stored in
        the forecaster for a specific step. Since a separate model is created for
        each forecast time step, it is necessary to select the model from which
        retrieve information.

        Only valid when the forecaster has been trained using 
        `GradientBoostingRegressor`, `RandomForestRegressor` or 
        `HistGradientBoostingRegressor` as regressor.

        Parameters
        ----------
        step : int
            Model from which retrieve information (a separate model is created for
            each forecast time step). First step is 1.

        Returns 
        -------
        feature_importance : pandas DataFrame
            Impurity-based feature importance associated with each predictor.
        '''
        if step > self.steps:
            raise Exception(
                f"Forecaster trained for {self.steps} steps. Got step={step}."
            )
        if step < 1:
            raise Exception("Minimum step is 1.")

        # Stored regressors start at index 0
        step = step - 1

        if isinstance(self.regressor, sklearn.pipeline.Pipeline):
            estimator = self.regressors_[step][-1]
        else:
            estimator = self.regressors_[step]

        try:
            idx_columns_lags = np.arange(len(self.lags))
            idx_columns_exog = np.array([], dtype=int)
            if self.included_exog:
                idx_columns_exog = np.arange(len(self.X_train_col_names))[len(self.lags) + step::self.steps]
            idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
            feature_names = [self.X_train_col_names[i] for i in idx_columns]
            feature_names = [name.replace(f"_step_{step+1}", "") for name in feature_names]
            feature_importance = pd.DataFrame({
                                    'feature': feature_names,
                                    'importance' : estimator.feature_importances_
                                })
        except:   
            try:
                idx_columns_lags = np.arange(len(self.lags))
                idx_columns_exog = np.array([], dtype=int)
                if self.included_exog:
                    idx_columns_exog = np.arange(len(self.X_train_col_names))[len(self.lags) + step::self.steps]
                idx_columns = np.hstack((idx_columns_lags, idx_columns_exog))
                feature_names = [self.X_train_col_names[i] for i in idx_columns]
                feature_names = [name.replace(f"_step_{step+1}", "") for name in feature_names]
                feature_importance = pd.DataFrame({
                                        'feature': feature_names,
                                        'importance' : estimator.coef_
                                    })
            except:
                warnings.warn(
                    f"Impossible to access feature importance for regressor of type {type(estimator)}. "
                    f"This method is only valid when the regressor stores internally "
                    f"the feature importance in the attribute `feature_importances_` "
                    f"or `coef_`."
                )

                feature_importance = None

        return feature_importance
