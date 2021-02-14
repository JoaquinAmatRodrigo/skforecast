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
    
            Si `X=None`, se utilizan como predictores iniciales los valores
            almacenados en `self.last_window`, y las predicciones se inician a 
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
        
        

################################################################################
#      time_series_spliter,  ts_cv_forecaster,   grid_search_forecaster        #
################################################################################

def time_series_spliter(y, initial_train_size, steps, allow_incomplete_fold=True):
    '''
    
    Dividir los índices de una serie temporal en múltiples pares de train-test.
    Se mantiene el orden temporal de los datos y se incrementa en cada iteración
    el conjunto de entrenamiento.
    
    Parameters
    ----------        
    y : 1D np.ndarray, pd.Series
        Serie temporal de entrenamiento. 
    
    initial_train_size: int 
        Número de observaciones de entrenamiento en la partición inicial.
        
    steps : int
        Número de pasos a futuro que se predicen.
        
    allow_incomplete_fold : bool, default `True`
        Se permite que el último conjunto de test esté incompleto si este
        no alcanza `steps` observaciones. De lo contrario, se descartan las
        últimas observaciones.

    Yields
    ------
    train : 1D np.ndarray
        Índices del conjunto de entrenamiento.
        
    test : 1D np.ndarray
        Índices del conjunto de test.
        
    '''
    
    if not isinstance(y, (np.ndarray, pd.Series)):

        raise Exception('`y` tiene que ser `1D np.ndarray` o `pd.Series`.')

    elif isinstance(y, np.ndarray) and y.ndim != 1:

        raise Exception(
            f"`y` tiene que ser `1D np.ndarray` o `pd.Series`, "
            f"detectado `np.ndarray` con dimensiones {y.ndim}."
        ) 
        
    if isinstance(y, pd.Series):
        y = y.to_numpy().copy()
    
  
    folds     = (len(y) - initial_train_size) // steps  + 1
    remainder = (len(y) - initial_train_size) % steps    
    
    for i in range(folds):
          
        if i < folds - 1:
            train_end     = initial_train_size + i * steps    
            train_indices = range(train_end)
            test_indices  = range(train_end, train_end + steps)
            
        else:
            if remainder != 0 and allow_incomplete_fold:
                train_end     = initial_train_size + i * steps  
                train_indices = range(train_end)
                test_indices  = range(train_end, train_end + remainder)
            else:
                break
        
        yield train_indices, test_indices
        

def ts_cv_forecaster(forecaster, y, initial_train_size, steps, metric, exog=None,
                     allow_incomplete_fold=True):
    '''
    Validación cruzada de un objeto Forecaster, manteniendo el orden temporal de
    los datos e incrementando en cada iteración el conjunto de entrenamiento.
    
    Parameters
    ----------
    forecaster : object 
        Objeto de clase Forecaster.
        
    y : 1D np.ndarray, pd.Series
        Serie temporal de entrenamiento.
            
    exog : 1D np.ndarray, pd.Series, default `None`
        Variable exógena a la serie temporal que se incluye como
        predictor.
    
    initial_train_size: int 
        Número de observaciones de entrenamiento utilizadas en la primera iteración
        de validación.
        
    steps : int
        Número de pasos a futuro que se predicen.
        
    allow_incomplete_fold : bool, default `False`
        Se permite que el último conjunto de test esté incompleto si este
        no alcanza `steps` observaciones. De lo contrario, se descartan las
        últimas observaciones.
        
    metric : {'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'}
        Métrica utilizada para cuantificar la bondad de ajuste del modelo.

    Returns 
    -------
    ts_cv_results: 
        Valor de la métrica obtenido en cada partición.

    '''
    
    forecaster._check_y(y=y)
    y = forecaster._preproces_y(y=y)
    
    if exog is not None:
        forecaster._check_exog(exog=exog)
        exog = forecaster._preproces_exog(exog=exog)
    
    ts_cv_results = []
    
    metrics = {
        'neg_mean_squared_error': mean_squared_error,
        'neg_mean_absolute_error': mean_absolute_error,
        'neg_mean_absolute_percentage_error': mean_absolute_percentage_error
    }
    
    metric = metrics[metric]
    
    splits = time_series_spliter(
                y                     = y,
                initial_train_size    = initial_train_size,
                steps                 = steps,
                allow_incomplete_fold = allow_incomplete_fold
             )
    
    for train_index, test_index in splits:
        
        if exog is None:
            
            forecaster.fit(y=y[train_index])      
            pred = forecaster.predict(steps=len(test_index))
            
        else:
            
            forecaster.fit(y=y[train_index], exog=exog[train_index])      
            pred = forecaster.predict(steps=len(test_index), exog=exog[test_index])
               
        metric_value = metric(
                            y_true = y[test_index],
                            y_pred = pred
                       )
        
        ts_cv_results.append(metric_value)
                          
    return np.array(ts_cv_results)


def grid_search_forecaster(forecaster, y, param_grid, initial_train_size, steps,
                           metric, exog=None, n_lags_grid=None,
                           allow_incomplete_fold=False, return_best=True):
    '''
    Búsqueda exhaustiva sobre los hiperparámetros de un Forecaster mediante
    validación cruzada.
    
    Parameters
    ----------
    forecaster : object 
        Objeto de clase Forecaster.
        
    y : 1D np.ndarray, pd.Series
        Serie temporal de entrenamiento.
            
    exog : 1D np.ndarray, pd.Series, default `None`
        Variable exógena a la serie temporal que se incluye como
        predictor.
    
    param_grid : dict
        Valores de hiperparámetros sobre los que hacer la búsqueda
    
    initial_train_size: int 
        Número de observaciones de entrenamiento utilizadas en la primera iteración
        de validación.
        
    steps : int
        Número de pasos a futuro que se predicen.
        
    metric : {'neg_mean_squared_error'}
        Métrica utilizada para cuantificar la bondad de ajuste del modelo.
        
    n_lags_grid : array-like
        Valores de `n_lags` sobre los que hacer la búsqueda.
        
    allow_incomplete_fold : bool, default `False`
        Se permite que el último conjunto de test esté incompleto si este
        no alcanza `steps` observaciones. De lo contrario, se descartan las
        últimas observaciones.
        
    return_best : devuelve modifica el Forecaster y lo entrena con los mejores
                  hiperparámetros encontrados.

    Returns 
    -------
    results: pandas.DataFrame
        Valor de la métrica obtenido en cada combinación de hiperparámetros.

    '''
    
    forecaster._check_y(y=y)
    y = forecaster._preproces_y(y=y)
    
    if exog is not None:
        forecaster._check_exog(exog=exog)
        exog = forecaster._preproces_exog(exog=exog)
        
    if n_lags_grid is None:
        n_lags_grid = [forecaster.n_lags]
        
    n_lags_list = []
    params_list = []
    metric_list = []
    
    param_grid =  list(ParameterGrid(param_grid))
    
    for n_lags in tqdm.tqdm(n_lags_grid, desc='loop n_lags_grid', position=0):
        
        forecaster.set_n_lags(n_lags)
        
        for params in tqdm.tqdm(param_grid, desc='loop param_grid', position=1, leave=False):

            forecaster.set_params(**params)

            cv_metrics = ts_cv_forecaster(
                            forecaster = forecaster,
                            y          = y,
                            exog       = exog,
                            initial_train_size = initial_train_size,
                            steps  = steps,
                            metric = metric
                         )
            
            n_lags_list.append(n_lags)
            params_list.append(params)
            metric_list.append(cv_metrics.mean())
            
    resultados = pd.DataFrame({
                    'n_lags': n_lags_list,
                    'params': params_list,
                    'metric': metric_list})
    
    resultados = resultados.sort_values(by='metric', ascending=True)
    
    if return_best:
        
        best_n_lags = resultados['n_lags'].iloc[0]
        best_params = resultados['params'].iloc[0]
        logging.info("Entrenando Forecaster con los mejores parámetros encontrados:")
        logging.info(f"n_lags: {best_n_lags}, params: {best_params}")
        forecaster.set_n_lags(best_n_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog)
            
    return resultados 
