################################################################################
#                               skforecast.utils                               #
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
import matplotlib.pyplot as plt
import tqdm

logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)

def time_series_spliter(y: Union[np.ndarray, pd.Series],
                        initial_train_size: int, steps: int,
                        allow_incomplete_fold: bool=True) -> Dict[np.ndarray, np.ndarray]:
    '''
    
    Split indices of a time series into multiple train-test pairs. The order of
    is maintained and the training set increases in each iteration.
    
    Parameters
    ----------        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    allow_incomplete_fold : bool, default `True`
        The last test set is allowed to be incomplete if it does not reach `steps`
        observations. Otherwise, the latest observations are discarded.

    Yields
    ------
    train : 1D np.ndarray
        Training indices.
        
    test : 1D np.ndarray
        Test indices.
        
    '''
    
    if not isinstance(y, (np.ndarray, pd.Series)):

        raise Exception('`y` must be `1D np.ndarray` o `pd.Series`.')

    elif isinstance(y, np.ndarray) and y.ndim != 1:

        raise Exception(
            f"`y` must be `1D np.ndarray` o `pd.Series`, "
            f"got `np.ndarray` with {y.ndim} dimensions."
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
