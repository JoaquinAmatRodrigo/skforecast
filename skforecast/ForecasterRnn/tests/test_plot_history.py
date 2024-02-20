# Unit test plot history method
# ==============================================================================
import pandas as pd
import numpy as np
import pytest
from skforecast.ForecasterRnn import ForecasterRnn
import tensorflow as tf
from skforecast.ForecasterRnn.utils import create_and_compile_model
import matplotlib.pyplot as plt

series = pd.DataFrame(
    {
        "1": pd.Series(np.arange(50)),
        "2": pd.Series(np.arange(50)),
        "3": pd.Series(np.arange(50)),
    }
)
lags = 3
steps = 4
levels = ["1", "2"]
activation = "relu"
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.MeanSquaredError()
recurrent_units = 100
dense_units = [128, 64]


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
forecaster = ForecasterRnn(model, levels)
forecaster.fit(series)


def test_plot_history_with_val_loss():
    """
    Test case for the plot_history method
    """
    # Call the plot_history method
    fig = forecaster.plot_history()

    # Assert that the figure is of type matplotlib.figure.Figure
    assert isinstance(fig, plt.Figure)

    # Assert that the plot contains the training loss curve
    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].lines[0].get_label() == "Training Loss"

    # Assert that the plot contains the validation loss curve
    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].lines[0].get_label() == "Training Loss"
