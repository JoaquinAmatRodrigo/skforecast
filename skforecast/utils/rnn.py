################################################################################
#                             skforecast.utils                                 #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under the BSD 3-Clause License.                                              #
################################################################################
# coding=utf-8

from typing import Union, Any, Optional, Tuple, Callable
import warnings
import importlib
import joblib
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import inspect
from copy import deepcopy

from ..exceptions import MissingValuesExogWarning
from ..exceptions import DataTypeWarning
from ..exceptions import IgnoredArgumentWarning

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.callbacks import EarlyStopping

def create_model_lstm(
    series, 
    n_lags, 
    n_steps, 
    levels, 
    lstm_unit:int=100, 
    dense_units:list=[64]
):
    
    # ----------------------------- Start parameters ----------------------------- #
    n_series = series.shape[1]
    n_levels = len(levels)
    
    # ---------------------------------- Layers ---------------------------------- #
    # Define the input layer
    input_layer = Input(shape=(n_lags, n_series))

    # LSTM layer
    x = LSTM(lstm_unit, activation='relu')(input_layer)

    # Dense layers
    for nn in dense_units:
        x = Dense(nn, activation='relu')(x)
    
    x = Dense(n_levels * n_steps, activation='linear')(x)

    # Reshape layer
    output_layer = Reshape((n_steps, n_levels))(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model