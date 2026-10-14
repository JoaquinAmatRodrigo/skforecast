"""Zero-code neural baseline: sklearn MLPRegressor inside existing skforecast forecasters."""
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from skforecast.recursive import ForecasterRecursiveMultiSeries, ForecasterRecursive
from skforecast.model_selection import backtesting_forecaster_multiseries, backtesting_forecaster, TimeSeriesFold
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def mlp(seed):
    return MLPRegressor(hidden_layer_sizes=(128, 128), early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=15, max_iter=300, learning_rate_init=1e-3, random_state=seed)

# --- store_sales N=50, L=28, H=14 (same split as exp_levels2)
L, H, N = 28, 14, 50
df = pd.read_pickle(f"{DATA}/store_sales.pkl")
df["id"] = "s" + df["store"].astype(str) + "_i" + df["item"].astype(str)
wide = df.pivot_table(index=df.index, columns="id", values="sales").asfreq("D")
cols = np.random.default_rng(0).permutation(wide.columns)[:N]
idx = wide.index; T = len(wide)
t_val = idx.get_loc(pd.Timestamp("2016-10-01")); t_test = idx.get_loc(pd.Timestamp("2017-01-01"))
n_te = (T - t_test) // H
data = wide[cols].iloc[:t_test + n_te * H]
# seasonal naive MAE per series on the same folds
y = data.to_numpy(float)
sn = np.array([np.abs(y[t:t+H] - np.concatenate([y[t-7:t]]*2)[:H]).mean(0) for t in range(t_test, t_test+n_te*H, H)]).mean(0)
for seed in [0, 1, 2]:
    f = ForecasterRecursiveMultiSeries(mlp(seed), lags=L, encoding="onehot", transformer_series=MinMaxScaler())
    cv = TimeSeriesFold(steps=H, initial_train_size=t_val + (t_test - t_val) % H, refit=False)
    t0 = time.time()
    _, p = backtesting_forecaster_multiseries(f, series=data, cv=cv, metric="mean_absolute_error", show_progress=False)
    p = p.loc[p.index >= idx[t_test]]
    P = p.pivot_table(index=p.index, columns="level", values="pred")[cols].to_numpy()
    mae = np.abs(P - y[t_test:]).mean(0)
    print(json.dumps(dict(data="store_sales N=50", model="sklearn MLP recursive multiseries", seed=seed,
                          rmae=round(float(np.mean(mae / sn)), 3), secs=round(time.time()-t0, 1))), flush=True)

# --- bike sharing, users, L=48, H=24 (same split as exp_exog_torch)
L, H = 48, 24
b = pd.read_pickle(f"{DATA}/bike_sharing_extended_features.pkl").drop(columns="weather")
small = ["temp", "holiday", "hour_day_sin", "hour_day_cos", "week_day_sin", "week_day_cos"]
idx = b.index; T = len(b)
t_val = idx.get_loc(pd.Timestamp("2012-05-01")); t_test = idx.get_loc(pd.Timestamp("2012-09-01"))
n_te = len(np.arange(t_test, T - H + 1, H))
data = b.iloc[:t_test + n_te * H]
for set_name, ex in [("none", None), ("small(6)", small), ("all(88)", [c for c in b.columns if c != "users"])]:
    for seed in [0, 1]:
        f = ForecasterRecursive(mlp(seed), lags=L, transformer_y=MinMaxScaler(),
                                transformer_exog=MinMaxScaler() if ex else None)
        cv = TimeSeriesFold(steps=H, initial_train_size=t_val + (t_test - t_val) % H, refit=False)
        t0 = time.time()
        _, p = backtesting_forecaster(f, y=data["users"], exog=data[ex] if ex else None, cv=cv,
                                      metric="mean_absolute_error", show_progress=False)
        p = p.loc[p.index >= idx[t_test], "pred"].to_numpy()
        mae = np.abs(p - data["users"].to_numpy()[t_test:]).mean()
        print(json.dumps(dict(data=f"bike exog={set_name}", model="sklearn MLP recursive", seed=seed,
                              mae=round(float(mae), 1), secs=round(time.time()-t0, 1))), flush=True)
