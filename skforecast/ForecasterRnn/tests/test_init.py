from skforecast.ForecasterRnn.ForecasterRnn import ForecasterRnn
from skforecast.ForecasterRnn.utils import create_and_compile_model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import MeanSquaredError
import pytest


# test levels
@pytest.mark.parametrize(
    "levels",
    [
        "t+1",
        ["t+1"],
        ["t+1", "t+2"],
    ],
)
def test_init(levels):
    print(f"levels: {levels}")
    # Generate dummy data for testing
    series = pd.DataFrame(np.random.randn(100, 3))
    series.columns = [f"t+{step}" for step in range(1, series.shape[1] + 1)]
    lags = 10
    steps = 5
    recurrent_units = 100
    dense_units = [64]
    activation = "relu"
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.MeanSquaredError()

    # Call the function to create and compile the model
    model = create_and_compile_model(
        series=series,
        lags=lags,
        steps=steps,
        levels=levels,
        recurrent_units=recurrent_units,
        dense_units=dense_units,
        activation=activation,
        optimizer=optimizer,
        loss=loss,
    )

    forecaster = ForecasterRnn(
        regressor=model,
        levels=levels,
    )

    # Assert that the forecaster is an instance of ForecasterAutoreg
    assert isinstance(forecaster, ForecasterRnn)

    # Assert that the forecaster is not fitted
    assert forecaster.fitted == False

    # Assert that the model is compiled with the correct optimizer and loss
    assert forecaster.regressor.optimizer == optimizer
    assert forecaster.regressor.loss == loss

    # Assert that the model has the correct number of layers
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    if isinstance(recurrent_units, int):
        recurrent_units = [recurrent_units]
    if isinstance(levels, str):
        levels = [levels]

    assert (
        len(forecaster.regressor.layers)
        == 5 + len(dense_units) - 1 + len(recurrent_units) - 1
    )

    # Assert that the forecaster has the correct number of levels
    assert forecaster.levels == levels

    # Assert that the forecaster has the correct number of lags
    assert len(forecaster.lags) == lags

    # Assert that the forecaster has the correct number of steps
    assert forecaster.steps == steps
