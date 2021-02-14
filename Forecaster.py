################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8


import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid


logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


################################################################################
#                               Forecaster                                     #
################################################################################

class Forecaster():
    '''
    Convierte un regresor de scikitlearn en un forecaster recursivo (multi-step).
    
    Parameters
    ----------
    regressor : scikitlearn regressor
        Modelo de regresión de scikitlearn.
        
    fun_predictors: function, default `None`
        Función que recibe una ventana temporal de una serie temporal y devuelve
        un np.array con los predictores asociados a esa ventana.
        
    window_size: int
        Tamaño de la ventana temporal a pasado que necesita `fun_predictors`
        para crear los predictores.

    
    Attributes
    ----------
    regressor : scikitlearn regressor
        Modelo de regresión de scikitlearn.
        
    last_window : array-like, shape (1, n_lags)
        Última ventana temporal que ha visto el Forecaster al ser entrenado.
        Se corresponde con los valores de la serie temporal necesarios para 
        predecir el siguiente `step` después de los datos de entrenamiento.
        
    included_exog: bool, default `False`.
        Si el Forecaster se ha entrenado utilizando variable exógena.
        
    fun_predictors: function, default `None`
        Función que recibe una serie temporal y devuelve una matriz de predictores
        y el valor de la serie temporal asociado.
        
    window_size: int, default `None`
        Tamaño de la ventana temporal a pasado que necesita `fun_predictors`
        para crear los predictores.
     
    '''
    
    def __init__(self, regressor, fun_predictors, window_size):
        
        self.regressor         = regressor
        self.last_window       = None
        self.included_exog     = False
        self.create_predictors = fun_predictors
        self.window_size       = window_size

        
    def fit(self, y, exog=None):
        '''
        Entrenamiento del Forecaster
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Serie temporal de entrenamiento.
            
        exog : 1D np.ndarray, pd.Series, default `None`
            Variable exógena a la serie temporal que se incluye como
            predictor.


        Returns 
        -------
        self : object
               objeto Forecaster entrenado
        
        '''
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)
        
        if exog is not None:
            self._check_exog(exog=exog)
            exog = self._preproces_exog(exog=exog)
            
            
        X_train  = []
        y_train  = []

        for i in range(len(y) - self.window_size):

            train_index = np.arange(i, self.window_size + i)
            test_index  = [self.window_size + i]

            X_train.append(self.create_predictors(y=y[train_index]))
            y_train.append(y[test_index])
        
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        
        if exog is not None:
            self.included_exog = True
            self.regressor.fit(
                # Se tiene que eliminar de exog las primeras self.window_size
                # posiciones ya que son están en X_train.
                X = np.column_stack((X_train, exog[self.window_size:])),
                y = y_train
            )
            
        else:
            self.regressor.fit(X=X_train, y=y_train)
        
        # Se guarda la última ventana temporal de `y` para poder crear los 
        # predictores en la primera iteración del predict().
        self.last_window = y[-self.window_size:].copy()
        
            
            
    def predict(self, steps, y=None, exog=None):
        '''
        Proceso iterativo en la que, cada predicción, se utiliza como
        predictor de la siguiente predicción.
        
        Parameters
        ----------
               
        steps : int
            Número de pasos a futuro que se predicen.
            
        X : 1D np.ndarray, pd.Series, default `None`
            ventana temporal para crear los predictores con los que se inicia el
            proceso iterativo de predicción.
    
            Si `y=None`, se utilizan los valores almacenados en `self.last_window`,
            y las predicciones se inician a continuación de los datos de
            entrenamiento.
            
        exog : 1D np.ndarray, pd.Series, default `None`
            Variable exógena a la serie temporal que se incluye como
            predictor.

        Returns 
        -------
        predicciones : 1D np.array, shape (steps,)
            Predicciones del modelo.
            
        '''
        
        if y is None:
            y = self.last_window
            
        self._check_y(y=y)
        y = self._preproces_y(y=y)
        
        if exog is None and self.included_exog:
            raise Exception(
                "Forecaster entrenado con variable exogena. Se tiene " \
                + "que aportar esta misma variable en el predict()."
            )
        
        if exog is not None:
            self._check_exog(exog=exog)
            exog = self._preproces_exog(exog=exog)
            

        predicciones = np.full(shape=steps, fill_value=np.nan)

        for i in range(steps):
            
            X = self.create_predictors(y=y)
            
            if exog is None:
                prediccion = self.regressor.predict(X=X)
            else:
                prediccion = self.regressor.predict(X=np.column_stack((X, exog[i])))
            
            predicciones[i] = prediccion.ravel()[0]

            # Actualizar valores de `y`. Se descarta la primera posición y se
            # añade la nueva predicción al final.
            y = np.append(y[1:], prediccion)

        return np.array(predicciones)
    
    
    
    def _check_y(self, y):
        '''
        Comprueba que `y` es un `np.ndarray` o `pd.Series`.
        
        Parameters
        ----------        
        y : array-like
            valores de la serie temporal

        Returns 
        -------
        bool
        '''
        
        if not isinstance(y, (np.ndarray, pd.Series)):
            
            raise Exception('`y` tiene que ser `1D np.ndarray` o `pd.Series`.')
            
        elif isinstance(y, np.ndarray) and y.ndim != 1:
            
            raise Exception(
                f"`y` tiene que ser `1D np.ndarray` o `pd.Series`, "
                f"detectado `np.ndarray` con dimensiones {y.ndim}."
            )
        else:
            return
        
        
    def _check_exog(self, exog):
        '''
        Comprueba que `exog` es un `np.ndarray` o `pd.Series`.
        
        Parameters
        ----------        
        exog : array-like
            valores de la serie temporal

        Returns 
        -------
        bool
        '''
        
        if not isinstance(exog, (np.ndarray, pd.Series)):
            
            raise Exception('`exog` tiene que ser `1D np.ndarray` o `pd.Series`.')
            
        elif isinstance(exog, np.ndarray) and exog.ndim != 1:
            
            raise Exception(
                f"`exog` tiene que ser `1D np.ndarray` o `pd.Series`, "
                f"detectado `np.ndarray` con dimensiones {exog.ndim}."
            )
        else:
            return
        
    def _preproces_y(self, y):
        
        '''
        Si `y` es `pd.Series`, lo convierte en `np.ndarray` de 1 dimensión.
        
        Parameters
        ----------        
        y :1D np.ndarray, pd.Series
            valores de la serie temporal

        Returns 
        -------
        y: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(y, pd.Series):
            return y.to_numpy()
        else:
            return y
        
    def _preproces_exog(self, exog):
        
        '''
        Si `exog` es `pd.Series`, lo convierte en `np.ndarray` de 1 dimensión.
        
        Parameters
        ----------        
        exog : 1D np.ndarray, pd.Series
            valores de la serie temporal.

        Returns 
        -------
        exog: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(exog, pd.Series):
            return exog.to_numpy()
        else:
            return exog
    
    
    def set_params(self, **params):
        '''
        Asignar nuevos valores a los parámetros del modelo scikitlearn contenido
        en el Forecaster.
        
        Parameters
        ----------
        params : dict
            Valor de los parámetros.

        Returns 
        -------
        self
        
        '''
        
        self.regressor.set_params(**params)
        
        
    def set_n_lags(self, n_lags):
        '''
        Asignar nuevo valor al atributo `n_lags` del Forecaster.
        
        Parameters
        ----------
        n_lags : int
            Número de lags utilizados cómo predictores.

        Returns 
        -------
        self
        
        '''
        
        self.n_lags = n_lags
        
        
