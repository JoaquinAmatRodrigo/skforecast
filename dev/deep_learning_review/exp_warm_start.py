"""ForecasterRnn.fit does not reset the weights: backtesting starts from
the current weights of the forecaster (warm start)."""
import os
os.environ["KERAS_BACKEND"] = "torch"
import contextlib
import io
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import keras
from skforecast.deep_learning import ForecasterRnn, create_and_compile_model
from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold
from common import DATA
L, H = 24, 24
y = pd.read_pickle(f"{DATA}/bike_sharing_extended_features.pkl")[["users"]].loc["2012-01-01":]
n_train = len(y.loc[:"2012-08-31"])
n = n_train + (len(y) - n_train) // H * H
y = y.iloc[:n]

def new_forecaster(seed=0):
    keras.utils.set_random_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        m = create_and_compile_model(series=y, levels="users", lags=L, steps=H, recurrent_units=64,
                                     compile_kwargs={"optimizer": "adam", "loss": "mse"})
    return ForecasterRnn(m, levels="users", lags=L, fit_kwargs={"epochs": 30, "batch_size": 128, "verbose": 0})

cv = TimeSeriesFold(steps=H, initial_train_size=n_train, refit=False)
def bt(f):
    with contextlib.redirect_stdout(io.StringIO()):
        m, _ = backtesting_forecaster_multiseries(f, series=y, cv=cv, metric="mean_absolute_error", show_progress=False)
    return m["mean_absolute_error"].iloc[0]

for seed in [0, 1]:
    fresh = bt(new_forecaster(seed))
    f = new_forecaster(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        f.fit(y)  # trained with ALL the data, test period included
    leaked = bt(f)
    f = new_forecaster(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        f.fit(y.iloc[:n_train])  # control: same number of epochs, no test data
    control = bt(f)
    print(f"seed {seed}: backtest MAE | fresh forecaster = {fresh:.1f} | "
          f"previously fitted on all data = {leaked:.1f} | "
          f"previously fitted on train only (control) = {control:.1f}", flush=True)
