################################################################################
#                             skforecast.utils                                 #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under the BSD 3-Clause License.                                              #
################################################################################
# coding=utf-8

from typing import Union, Any, Optional, Tuple, Callable
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, LSTM, SimpleRNN
from keras.optimizers import Adam
from keras.losses import MeanSquaredError


def create_and_compile_model(
    series: pd.DataFrame,
    lags: Union[int, list],
    steps: Union[int, list],
    levels: Union[str, int, list] = None,
    recurrent_layer: str = "LSTM",
    recurrent_units: Union[int, list] = 100,
    dense_units: list = [64],
    activation: str = "relu",
    optimizer: object = Adam(learning_rate=0.01),
    loss: object = MeanSquaredError(),
) -> tf.keras.models.Model:
    """
    Creates a neural network model for time series prediction with flexible recurrent layers.

    Parameters
    ----------
    series : np.ndarray
        Input time series data represented as a NumPy array.
    lags : int or list
        Number of lagged time steps to consider in the input, or a list of specific lag indices.
    steps : int or list
        Number of steps to predict into the future, or a list of specific step indices.
    levels : str, int, or list, optional
        Number of output levels (features) to predict, or a list of specific level indices.
        If None, defaults to the number of input series. Default is None.
    recurrent_layer : str, optional
        Type of recurrent layer to be used ('LSTM' or 'RNN'). Default is 'LSTM'.
    recurrent_units : int or list, optional
        Number of units in the recurrent layer(s). Can be an integer or a list of integers for multiple layers.
        Default is 100.
    dense_units : list, optional
        List of integers representing the number of units in each dense layer. Default is [64].
    activation : str, optional
        Activation function for the recurrent and dense layers. Default is 'relu'.
    optimizer : object, optional
        Optimization algorithm and learning rate. Default is Adam(learning_rate=0.01).
    loss : object, optional
        Loss function for model training. Default is MeanSquaredError().

    Returns
    -------
    model : tf.keras.models.Model
        Compiled neural network model.
    """

    n_series = series.shape[1]

    if isinstance(lags, list):
        lags = len(lags)
    if isinstance(steps, list):
        steps = len(steps)
    if isinstance(levels, list):
        levels = len(levels)
    elif isinstance(levels, (str)):
        levels = 1
    elif isinstance(levels, type(None)):
        levels = series.shape[1]
    else:
        raise TypeError(
            f"`levels` argument must be a string, list or int. Got {type(levels)}."
        )

    input_layer = Input(shape=(lags, n_series))
    x = input_layer

    # Dynamically create multiple recurrent layers if recurrent_units is a list
    if isinstance(recurrent_units, list):
        for units in recurrent_units[:-1]:  # All layers except the last one
            if recurrent_layer == "LSTM":
                x = LSTM(units, activation=activation, return_sequences=True)(x)
            elif recurrent_layer == "RNN":
                x = SimpleRNN(units, activation=activation, return_sequences=True)(x)
            else:
                raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")
        # Last layer without return_sequences
        if recurrent_layer == "LSTM":
            x = LSTM(recurrent_units[-1], activation=activation)(x)
        elif recurrent_layer == "RNN":
            x = SimpleRNN(recurrent_units[-1], activation=activation)(x)
        else:
            raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")
    else:
        # Single recurrent layer
        if recurrent_layer == "LSTM":
            x = LSTM(recurrent_units, activation=activation)(x)
        elif recurrent_layer == "RNN":
            x = SimpleRNN(recurrent_units, activation=activation)(x)
        else:
            raise ValueError(f"Invalid recurrent layer: {recurrent_layer}")

    # Dense layers
    for nn in dense_units:
        x = Dense(nn, activation=activation)(x)

    # Output layer
    x = Dense(levels * steps, activation="linear")(x)
    output_layer = tf.keras.layers.Reshape((steps, levels))(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    if loss is not None:
        model.compile(optimizer=optimizer, loss=loss)

    return model
