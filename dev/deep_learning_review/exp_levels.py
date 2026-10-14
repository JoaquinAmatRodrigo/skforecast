"""Exp 1: many levels. store_sales (daily, 500 series). Current ForecasterRnn
architecture (all series as channels of one sample) vs the same LSTM trained
globally / channel-independent (each series is a sample, shared weights)."""
import sys
import json
import time
import numpy as np
import pandas as pd
from common import DATA, windows, train, skf_model, mlp

L, H = 28, 14
Ns = [int(n) for n in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 10, 50, 200]
seeds = [0, 1, 2]

df = pd.read_pickle(f"{DATA}/store_sales.pkl")
df["id"] = "s" + df["store"].astype(str) + "_i" + df["item"].astype(str)
wide = df.pivot_table(index=df.index, columns="id", values="sales").asfreq("D")
order = np.random.default_rng(0).permutation(wide.columns)
T = len(wide)
idx = wide.index
t_val = idx.get_loc(pd.Timestamp("2016-10-01"))
t_test = idx.get_loc(pd.Timestamp("2017-01-01"))
tr_orig = np.arange(L, t_val - H + 1)
va_orig = np.arange(t_val, t_test - H + 1)
te_orig = np.arange(t_test, T - H + 1, H)

results = []
for N in Ns:
    cols = order[:N]
    raw = wide[cols].to_numpy(float)
    mn, mx = raw[:t_val].min(0), raw[:t_val].max(0)
    sc = (raw - mn) / (mx - mn)
    Xtr, Ytr = windows(sc, L, H, tr_orig)
    Xva, Yva = windows(sc, L, H, va_orig)
    Xte, Yte_raw = windows(sc, L, H, te_orig)
    Yte_raw = Yte_raw * (mx - mn) + mn
    # seasonal naive (repeat last week)
    Xte_raw = Xte * (mx - mn) + mn
    snaive = np.concatenate([Xte_raw[:, -7:]] * 2, axis=1)[:, :H]
    mae_sn = np.abs(snaive - Yte_raw).mean((0, 1))

    def score(pred_sc, name, secs=None, ep=None, seed=None):
        pred = pred_sc * (mx - mn) + mn
        mae = np.abs(pred - Yte_raw).mean((0, 1))
        r = dict(N=N, model=name, seed=seed, rmae=float(np.mean(mae / mae_sn)),
                 mae=float(mae.mean()), secs=secs, epochs=ep)
        results.append(r)
        print(json.dumps(r), flush=True)

    score(snaive.astype(float).__sub__(mn).__truediv__(mx - mn), "seasonal_naive")

    # stacked (channel independent) samples
    def stack(X, Y):
        return (X.transpose(0, 2, 1).reshape(-1, L, 1),
                Y.transpose(0, 2, 1).reshape(-1, H, 1))
    Xtr_s, Ytr_s = stack(Xtr, Ytr)
    Xva_s, Yva_s = stack(Xva, Yva)
    Xte_s = Xte.transpose(0, 2, 1).reshape(-1, L, 1)

    def unstack(P):
        return P.reshape(len(te_orig), N, H).transpose(0, 2, 1)

    for seed in seeds:
        import keras
        for regime in ["default", "robust"]:
            keras.utils.set_random_seed(seed)
            m = skf_model(series=wide[cols], levels=list(cols), lags=L, steps=H)
            s, ep = train(m, Xtr, Ytr, Xva, Yva, batch=64, seed=seed, regime=regime)
            score(m.predict(Xte, verbose=0), f"rnn_current [{regime}]", round(s, 1), ep, seed)
        bs = 64 if N == 1 else 512
        if N <= 10:
            keras.utils.set_random_seed(seed)
            m = skf_model(series=wide[cols[:1]], levels=list(cols[:1]), lags=L, steps=H)
            s, ep = train(m, Xtr_s, Ytr_s, Xva_s, Yva_s, batch=bs, seed=seed, regime="robust")
            score(unstack(m.predict(Xte_s, verbose=0, batch_size=4096)), "rnn_global [robust]", round(s, 1), ep, seed)
        keras.utils.set_random_seed(seed)
        m = mlp(L, H)
        s, ep = train(m, Xtr_s, Ytr_s, Xva_s, Yva_s, batch=bs, seed=seed, regime="robust")
        score(unstack(m.predict(Xte_s, verbose=0, batch_size=4096)), "mlp_global [robust]", round(s, 1), ep, seed)

    # LightGBM global recursive (skforecast)
    from lightgbm import LGBMRegressor
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold
    f = ForecasterRecursiveMultiSeries(LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=0),
                                       lags=L)
    data = wide[cols].iloc[:T - ((T - t_test) % H)]
    cv = TimeSeriesFold(steps=H, initial_train_size=t_val + (t_test - t_val) % H, refit=False, fixed_train_size=True)
    t0 = time.time()
    _, preds = backtesting_forecaster_multiseries(f, series=data, cv=cv, metric="mean_absolute_error",
                                                  show_progress=False)
    preds = preds.loc[preds.index >= idx[t_test]]
    P = preds.pivot_table(index=preds.index, columns="level", values="pred")[cols].to_numpy()
    P = P.reshape(len(te_orig), H, N)
    score((P - mn) / (mx - mn), "lgbm_recursive_multiseries", round(time.time() - t0, 1))

pd.DataFrame(results).to_csv(f"res_levels_{'_'.join(map(str, Ns))}.csv", index=False)
