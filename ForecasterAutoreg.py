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
#                             ForecasterAutoreg                                #
################################################################################

class ForecasterAutoreg():
    '''
    Convierte un regresor de scikitlearn en un forecaster autoregresivo 
    recursivo (multi-step).
    
    Parameters
    ----------
    regressor : scikitlearn regressor
        Modelo de regresión de scikitlearn.
        
    n_lags : int
        Número de lags utilizados cómo predictores.

    
    Attributes
    ----------
    regressor : scikitlearn regressor
        Modelo de regresión de scikitlearn.
        
    last_window : array-like, shape (1, n_lags)
        Última ventana temporal que ha visto el Forecaster al ser entrenado.
        Se corresponde con el valor de los lags necesarios para predecir el
        siguiente `step` después de los datos de entrenamiento.
        
    included_exog: bool, default `False`.
        Si el Forecaster se ha entrenado utilizando variable exógena.
     
    '''
    
    def __init__(self, regressor, n_lags):
        
        self.regressor     = regressor
        self.n_lags        = n_lags
        self.last_window   = None
        self.included_exog = False
    
    
    def create_lags(self, y):
        '''
        Convierte una serie temporal en una matriz donde, cada valor de `y`,
        está asociado a los lags temporales que le preceden.
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Serie temporal de entrenamiento.

        Returns 
        -------
        X_data : 2D np.ndarray, shape (nº observaciones, n_lags)
            Matriz con el valor de los lags.
        
        y_data : 1D np.ndarray, shape (nº observaciones - n_lags,)
            Valores de la variable respuesta
            
        '''
        
        # Comprobaciones
        self._check_y(y=y)
        y = self._preproces_y(y=y)        
            
        # Número de divisiones
        n_splits = len(y) - self.n_lags

        # Matriz donde almacenar los valore de cada lag
        X_data  = np.full(shape=(n_splits, self.n_lags), fill_value=np.nan, dtype= float)
        y_data  = np.full(shape=(n_splits, 1), fill_value=np.nan, dtype= float)

        for i in range(n_splits):

            train_index = np.arange(i, self.n_lags + i)
            test_index  = [self.n_lags + i]

            X_data[i, :] = y[train_index]
            y_data[i]    = y[test_index]

        return X_data, y_data.ravel()

        
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
        
        # Comprobaciones
        self._check_y(y=y)
        y = self._preproces_y(y=y)
        
        if exog is not None:
            self._check_exog(exog=exog)
            exog = self._preproces_exog(exog=exog)
        
        
        X_train, y_train = self.create_lags(y=y)
        
        if exog is not None:
            self.included_exog = True
            self.regressor.fit(
                # Se tiene que eliminar de exog las primeras self.n_lags
                # posiciones ya que son están en X_train.
                X = np.column_stack((X_train, exog[self.n_lags:])),
                y = y_train
            )
            
        else:
            self.regressor.fit(X=X_train, y=y_train)
        
        # Se guarda la última ventana temporal de lags de entrenamiento para que 
        # puedan utilizarse como predictores en la primera iteración del predict().
        self.last_window = np.hstack((X_train[-1, 1:], y_train[-1])).reshape(1, -1)
        
            
    def predict(self, steps, X=None, exog=None):
        '''
        Proceso iterativo en la que, cada predicción, se utiliza como
        predictor de la siguiente predicción.
        
        Parameters
        ----------
               
        steps : int
            Número de pasos a futuro que se predicen.
            
        X : 2D np.ndarray, shape (1, n_lags), default `None`
            Valor de los predictores con los que se inicia el proceso iterativo
            de predicción. Es decir, el valor de los lags en t+1.
    
            Si `X==None`, se utilizan como predictores iniciales los valores
            almacenados en `self.last_lags`, y las predicciones se inician a 
            continuación de los datos de entrenamiento.
            
        exog : 1D np.ndarray, pd.Series, default `None`
            Variable exógena a la serie temporal que se incluye como
            predictor.

        Returns 
        -------
        predicciones : 1D np.array, shape (steps,)
            Predicciones del modelo.
            
        '''
        
        if exog is None and self.included_exog:
            raise Exception(
                "Forecaster entrenado con variable exogena. Se tiene " \
                + "que aportar esta misma variable en el predict()."
            )
                
        if X is None:
            X = self.last_window
        
        if exog is not None:
            self._check_exog(exog=exog)
            exog = self._preproces_exog(exog=exog)
            

        predicciones = np.full(shape=steps, fill_value=np.nan)

        for i in range(steps):
            if exog is None:
                prediccion = self.regressor.predict(X=X)
            else:
                prediccion = self.regressor.predict(X=np.column_stack((X, exog[i])))
                
            predicciones[i] = prediccion.ravel()[0]

            # Actualizar valores de X. Se descarta la primera posición y se
            # añade la nueva predicción al final.
            X = X.flatten()
            X = np.append(X[1:], prediccion)
            X = X.reshape(1, -1)


        return predicciones
    
    
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
        
        
