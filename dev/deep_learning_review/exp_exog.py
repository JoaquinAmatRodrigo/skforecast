"""Exp 2 (native PyTorch for speed): exogenous variables. bike_sharing hourly,
target `users`, L=72, H=24. `rnn_current` is a faithful replica of
create_and_compile_model (LSTM 100 -> [RepeatVector + concat future exog ->
TimeDistributed Dense 64 relu -> Dense 1] or [Dense 64 relu -> Dense H])."""
import sys
import json
import time
import copy
import os
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
L, H = 48, 24
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
sets_arg = sys.argv[1].split(",") if len(sys.argv) > 1 else None


def windows(arr, origins, L, H):
    return (np.stack([arr[t - L:t] for t in origins]).astype("float32"),
            np.stack([arr[t:t + H] for t in origins]).astype("float32"))


def keras_like_init(m):
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)
        if isinstance(mod, nn.LSTM):
            for name, p in mod.named_parameters():
                if "weight_ih" in name: nn.init.xavier_uniform_(p)
                elif "weight_hh" in name: nn.init.orthogonal_(p)
                else:
                    nn.init.zeros_(p)
                    if "bias_ih" in name: p.data[mod.hidden_size:2 * mod.hidden_size] = 1.0


class RnnCurrent(nn.Module):
    def __init__(self, k, step=False, past_exog=False):
        super().__init__()
        self.k, self.step, self.past = k, step, past_exog
        self.lstm = nn.LSTM(1 + (k if past_exog else 0), 100, batch_first=True)
        if k == 0:
            self.head = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, H))
        else:
            d = 100 + k + (H if step else 0)
            self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        keras_like_init(self)

    def forward(self, x, ep, ef):
        inp = torch.cat([x, ep], -1) if self.past else x
        h = self.lstm(inp)[0][:, -1]
        if self.k == 0:
            return self.head(h)
        z = [h.unsqueeze(1).expand(-1, H, -1), ef]
        if self.step:
            z.append(torch.eye(H).expand(len(x), H, H))
        return self.head(torch.cat(z, -1)).squeeze(-1)


class TideLite(nn.Module):
    def __init__(self, k, hidden=128, proj=4):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(k, proj) if k else None
        d = L + (H * proj if k else 0)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, H))
        self.skip = nn.Linear(L, H)
        keras_like_init(self)

    def forward(self, x, ep, ef):
        xl = x[..., 0]
        z = [xl] + ([self.proj(ef).flatten(1)] if self.k else [])
        return self.mlp(torch.cat(z, -1)) + self.skip(xl)


def fit(model, tr, va, regime, seed, batch=256):
    torch.manual_seed(seed)
    max_ep, pat = (40, 5) if regime == "default" else (100, 15)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5) if regime == "robust" else None
    best, best_state, bad, n = np.inf, None, 0, len(tr[0])
    t0 = time.time()
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = ((model(tr[0][b], tr[1][b], tr[2][b]) - tr[3][b]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((model(*va[:3]) - va[3]) ** 2).mean().item()
        if sched: sched.step(vl)
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= pat: break
    model.load_state_dict(best_state)
    return round(time.time() - t0, 1), ep + 1


b = pd.read_pickle(f"{DATA}/bike_sharing_extended_features.pkl").drop(columns="weather")
rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.normal(size=(len(b), 50)), index=b.index, columns=[f"noise_{i}" for i in range(50)])
small = ["temp", "holiday", "hour_day_sin", "hour_day_cos", "week_day_sin", "week_day_cos"]
all_cols = [c for c in b.columns if c != "users"]
EXOG_SETS = {"none": [], "temp": ["temp"], "small(6)": small, "all(88)": all_cols,
             "small+50noise": small + list(noise.columns)}
b = pd.concat([b, noise], axis=1)
idx, T = b.index, len(b)
t_val = idx.get_loc(pd.Timestamp("2012-05-01")); t_test = idx.get_loc(pd.Timestamp("2012-09-01"))
orig = {"tr": np.arange(L, t_val - H + 1), "va": np.arange(t_val, t_test - H + 1), "te": np.arange(t_test, T - H + 1, H)}
y = b[["users"]].to_numpy(float)
ymn, ymx = y[:t_val].min(), y[:t_val].max()
ys = (y - ymn) / (ymx - ymn)
Yte_raw = windows(y, orig["te"], L, H)[1][..., 0]
results = []


def score(p_sc, set_name, model, secs=None, ep=None, seed=None):
    p = np.asarray(p_sc).reshape(len(orig["te"]), H) * (ymx - ymn) + ymn
    err = np.abs(p - Yte_raw)
    r = dict(exog=set_name, model=model, seed=seed, mae=round(float(err.mean()), 2), secs=secs, epochs=ep,
             mae_h1=round(float(err.mean(0)[0]), 1), mae_h24=round(float(err.mean(0)[-1]), 1))
    results.append(r); print(json.dumps(r), flush=True)


snaive = windows(y, orig["te"], L, H)[0][:, -H:, 0]
score((snaive - ymn) / (ymx - ymn), "-", "seasonal_naive_24h")

for set_name, cols in EXOG_SETS.items():
    if sets_arg and set_name not in sets_arg:
        continue
    k = len(cols)
    if k:
        e = b[cols].to_numpy(float)
        emn, emx = e[:t_val].min(0), e[:t_val].max(0)
        es = (e - emn) / np.where(emx - emn == 0, 1, emx - emn)
    else:
        es = np.zeros((T, 1))
    D = {}
    for s, o in orig.items():
        X, Y = windows(ys, o, L, H)
        P, F = windows(es, o, L, H)
        D[s] = tuple(torch.tensor(a) for a in (X, P, F, Y[..., 0]))
    specs = [("rnn_current [default]", lambda: RnnCurrent(k), "default"),
             ("rnn_current [robust]", lambda: RnnCurrent(k), "robust")]
    if k:
        specs += [("rnn_current+step_onehot [robust]", lambda: RnnCurrent(k, step=True), "robust"),
                  ("rnn_s2s_past_exog+step [robust]", lambda: RnnCurrent(k, step=True, past_exog=True), "robust")]
    specs += [("mlp_tide_lite [robust]", lambda: TideLite(k), "robust")]
    for seed in seeds:
        for name, make, regime in specs:
            torch.manual_seed(seed)
            m = make()
            secs, ep = fit(m, D["tr"], D["va"], regime, seed)
            m.eval()
            with torch.no_grad():
                score(m(*D["te"][:3]).numpy(), set_name, name, secs, ep, seed)

    from lightgbm import LGBMRegressor
    from skforecast.recursive import ForecasterRecursive
    from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
    f = ForecasterRecursive(LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1, random_state=0), lags=L)
    data = b.iloc[:t_test + len(orig["te"]) * H]
    cv = TimeSeriesFold(steps=H, initial_train_size=t_val + (t_test - t_val) % H, refit=False)
    t0 = time.time()
    _, preds = backtesting_forecaster(f, y=data["users"], exog=data[cols] if k else None, cv=cv,
                                      metric="mean_absolute_error", show_progress=False)
    p = preds.loc[preds.index >= idx[t_test], "pred"].to_numpy()
    score((p - ymn) / (ymx - ymn), set_name, "lgbm_recursive", round(time.time() - t0, 1))

pd.DataFrame(results).to_csv(f"res_exog_{'_'.join(sets_arg) if sets_arg else 'all'}.csv", index=False)
