import pandas as pd
import numpy as np
import tensorflow as tf
import pytest
from skforecast.ForecasterRnn.utils import create_and_compile_model

# test with several parameters for dense_units and recurrent_units

@pytest.mark.parametrize("dense_units, recurrent_units", [
    (64, 100),
    ([64], 100),
    ([64, 32], 100),
    (64, [100]),
    (64, [100, 50]),
    ([64], [100]),
    ([64], [100, 50]),
    ([64, 32], [100]),
    ([64, 32], [100, 50])
])
def test_dense_units(dense_units, recurrent_units):
    print(f"dense_units: {dense_units}")
    print(f"recurrent_units: {recurrent_units}")
    # Generate dummy data for testing
    series = pd.DataFrame(np.random.randn(100, 3))
    lags = 10
    steps = 5
    levels = 1
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
        loss=loss
    )
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    if isinstance(recurrent_units, int):
        recurrent_units = [recurrent_units]
        
    # Assert that the model is an instance of tf.keras.models.Model
    assert isinstance(model, tf.keras.models.Model)

    # Assert that the model has the correct number of layers
    assert len(model.layers) == 5 + len(dense_units) - 1 + len(recurrent_units) - 1

    # Assert that the model is compiled with the correct optimizer and loss
    assert model.optimizer == optimizer
    assert model.loss == loss