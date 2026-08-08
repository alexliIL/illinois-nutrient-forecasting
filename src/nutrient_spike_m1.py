"""
Illinois nutrient-spike FORECASTING — M1 baseline (CSV-only features).

Question: given information available at day t, predict whether a PHOSPHORUS
spike occurs on any day in t+1 .. t+7, on a river the model was NOT trained on.

This is the M1 tier of the ablation:
    M1 = river-state only (this file)
    M2 = + weather        (add later)
    M3 = + watershed/geo  (add later)
    M4 = + satellite       (add later)

Target is defined PER RIVER in log space (concentrations are ~log-normal;
the USGS uncertainty is a *geometric* standard error, so log-space is correct).

Evaluation is LEAVE-ONE-RIVER-OUT: train on 7 rivers, test on the 8th, rotate.
Spikes are rare -> we score with PR-AUC (average precision), not accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CSV = "data/Illinois_nutrient_and_sediment_concentrations_wy2016-2021.csv"
TARGET_COL = "total_p_mg_per_l"   # start with phosphorus; swap to no3_mg_per_l later
SPIKE_PCTILE = 90                 # a "spike day" = above this per-river percentile
SPIKE_WARMUP = 60                 # min days of history before a river's threshold is trusted
HORIZON = 7                       # predict a spike within the next 1..7 days

# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------
df = pd.read_csv(CSV, parse_dates=["datetime"])
df = df.sort_values(["USGS_site", "datetime"]).reset_index(drop=True)
df["site"] = df["USGS_site"]  # keep a data copy that survives groupby-apply

# log-transform the concentration + flow (log1p on q since flow can be near 0;
# clip avoids log1p(x<-1) -> NaN on the handful of negative-discharge readings)
df["log_p"] = np.log(df[TARGET_COL])
df["log_q"] = np.log1p(df["q"].clip(lower=0))

# ----------------------------------------------------------------------
# TARGET: per-river spike, then "spike in next HORIZON days"
# ----------------------------------------------------------------------
def build_target(g):
    g = g.sort_values("datetime").reset_index(drop=True)
    # expanding (causal) per-river threshold: uses only data up to and including day t,
    # so "spike" is defined the way a live system would see it -- not with hindsight
    # over the river's full 2015-2021 record.
    thr = g["log_p"].expanding(min_periods=SPIKE_WARMUP).quantile(SPIKE_PCTILE / 100)
    g["is_spike"] = (g["log_p"] > thr).astype(float)
    g.loc[thr.isna(), "is_spike"] = np.nan   # no reliable threshold yet (warmup period)
    # future window: does a spike occur on any of days t+1..t+HORIZON?
    fut = pd.concat([g["is_spike"].shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    g["y"] = (fut.sum(axis=1) > 0).astype(float)
    g.loc[fut.isna().any(axis=1), "y"] = np.nan   # drop rows without full future window
    return g

df = pd.concat([build_target(g.copy()) for _, g in df.groupby("site")], ignore_index=True)

# ----------------------------------------------------------------------
# FEATURES (M1 — only things known at time t, from the CSV alone)
# ----------------------------------------------------------------------
def build_features(g):
    for lag in [1, 2, 3, 7, 14]:
        g[f"log_p_lag{lag}"] = g["log_p"].shift(lag)
        g[f"log_q_lag{lag}"] = g["log_q"].shift(lag)
    # rates of change / short-term dynamics
    g["dlog_q_1"]  = g["log_q"] - g["log_q"].shift(1)
    g["dlog_q_3"]  = g["log_q"] - g["log_q"].shift(3)
    g["dlog_p_1"]  = g["log_p"] - g["log_p"].shift(1)
    g["q_roll7"]   = g["log_q"].rolling(7).mean()
    g["p_roll7"]   = g["log_p"].rolling(7).mean()
    # seasonality
    doy = g["datetime"].dt.dayofyear
    g["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    g["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return g

df = pd.concat([build_features(g.copy()) for _, g in df.groupby("site")], ignore_index=True)

FEATURES = [c for c in df.columns if c.startswith(("log_p_lag", "log_q_lag"))] + [
    "log_q", "dlog_q_1", "dlog_q_3", "dlog_p_1",
    "q_roll7", "p_roll7", "sin_doy", "cos_doy",
]

model_df = df.dropna(subset=FEATURES + ["y"]).copy()
print(f"Usable rows: {len(model_df)}  |  spike-rate: {model_df['y'].mean():.3f}")

# ----------------------------------------------------------------------
# LEAVE-ONE-RIVER-OUT EVALUATION
# ----------------------------------------------------------------------
sites = sorted(model_df["site"].unique())
rows = []
for test_site in sites:
    tr = model_df[model_df.site != test_site]
    te = model_df[model_df.site == test_site]
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=0
    )
    clf.fit(tr[FEATURES], tr["y"])
    p = clf.predict_proba(te[FEATURES])[:, 1]
    ap  = average_precision_score(te["y"], p)
    auc = roc_auc_score(te["y"], p)
    base = te["y"].mean()   # PR-AUC of random guessing = base rate
    rows.append((test_site, len(te), base, ap, auc))
    print(f"site {test_site}: n={len(te):5d}  base={base:.3f}  PR-AUC={ap:.3f}  ROC-AUC={auc:.3f}")

res = pd.DataFrame(rows, columns=["site", "n", "base_rate", "pr_auc", "roc_auc"])
print("\nMean PR-AUC: {:.3f}  (mean base rate {:.3f})".format(res.pr_auc.mean(), res.base_rate.mean()))
print("Mean ROC-AUC: {:.3f}".format(res.roc_auc.mean()))
print("\nLift over base (PR-AUC / base_rate):")
print((res.pr_auc / res.base_rate).round(2).tolist())