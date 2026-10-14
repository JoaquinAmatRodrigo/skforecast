import os
import torch
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
os.environ.setdefault("KERAS_BACKEND", "torch")
import contextlib
import io
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import keras
from keras import layers

from skforecast.deep_learning import create_and_compile_model

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def windows(arr, L, H, origins):
    """arr (T, C). origin t: X = arr[t-L:t], Y = arr[t:t+H]."""
    X = np.stack([arr[t - L:t] for t in origins])
    Y = np.stack([arr[t:t + H] for t in origins])
    return X, Y


def compile_(m, lr=1e-3):
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return m


def train(model, x_tr, y_tr, x_va, y_va, epochs=40, batch=256, seed=0, regime=None):
    regime = regime or os.environ.get("REGIME", "default")
    keras.utils.set_random_seed(seed)
    if regime == "default":
        cb = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    else:  # robust: longer patience + LR schedule
        epochs = 100
        cb = [keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)]
    t0 = time.time()
    h = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs,
                  batch_size=batch, verbose=0, callbacks=cb, shuffle=True)
    return time.time() - t0, len(h.history["loss"])


def skf_model(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        m = create_and_compile_model(**kw)
    return compile_(m)


def mlp(L, H, n_feat_in=0, hidden=128, n_exog=0, exog_proj=4):
    """TiDE-lite: flattened lags (+ projected future exog) -> MLP -> H, plus linear skip."""
    lag_in = keras.Input((L, 1), name="series_input")
    x_l = layers.Flatten()(lag_in)
    inputs = [lag_in]
    feats = [x_l]
    if n_exog:
        ex_in = keras.Input((H, n_exog), name="exog_input")
        inputs.append(ex_in)
        p = layers.TimeDistributed(layers.Dense(exog_proj))(ex_in)
        feats.append(layers.Flatten()(p))
    x = layers.Concatenate()(feats) if len(feats) > 1 else feats[0]
    h = layers.Dense(hidden, activation="relu")(x)
    h = layers.Dense(hidden, activation="relu")(h)
    out = layers.Add()([layers.Dense(H)(h), layers.Dense(H)(x_l)])
    out = layers.Reshape((H, 1))(out)
    return compile_(keras.Model(inputs, out))
