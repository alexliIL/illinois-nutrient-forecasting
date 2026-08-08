"""
Illinois nutrient-spike FORECASTING — M2 ablation for TWO nutrients.

Runs the M1 (river-state) vs M2 (river-state + weather) leave-one-river-out
ablation for BOTH phosphorus and nitrate, and prints a comparison so you can
say which nutrient is more forecastable and where weather helps most.

Files expected (run from project root):
    data/Illinois_nutrient_and_sediment_concentrations_wy2016-2021.csv
    data/weather_daymet.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

NUT_CSV = "data/Illinois_nutrient_and_sediment_concentrations_wy2016-2021.csv"
WX_CSV  = "data/weather_daymet.csv"
SPIKE_PCTILE = 90
SPIKE_WARMUP = 60                 # min days of history before a river's threshold is trusted
HORIZON = 7

TARGETS = {
    "phosphorus": "total_p_mg_per_l",
    "nitrate":    "no3_mg_per_l",
}

# ----------------------------------------------------------------------
# WEATHER (built once, shared across nutrients)
# ----------------------------------------------------------------------
wx = pd.read_csv(WX_CSV).rename(columns={
    "time": "datetime", "prcp (mm/day)": "prcp",
    "tmax (degrees C)": "tmax", "tmin (degrees C)": "tmin",
})
wx["datetime"] = pd.to_datetime(wx["datetime"])
wx = wx.sort_values(["site", "datetime"]).reset_index(drop=True)

def weather_features(g):
    g = g.sort_values("datetime")
    p = g["prcp"]
    for w in [1, 3, 7, 14, 30]:
        g[f"rain_{w}d"] = p.rolling(w, min_periods=1).sum()
    wet = (p > 1.0).astype(int)
    days_since, c = [], 0
    for v in wet:
        c = 0 if v == 1 else c + 1
        days_since.append(c)
    g["dry_days"] = days_since
    g["tmean"] = (g["tmax"] + g["tmin"]) / 2.0
    return g

wx = pd.concat([weather_features(g.copy()) for _, g in wx.groupby("site")], ignore_index=True)
WX_FEATS = ["rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "dry_days", "tmean"]

# ----------------------------------------------------------------------
# BASE NUTRIENT FRAME (loaded once)
# ----------------------------------------------------------------------
base = pd.read_csv(NUT_CSV, parse_dates=["datetime"])
base = base.sort_values(["USGS_site", "datetime"]).reset_index(drop=True)
base["site"] = base["USGS_site"]
base["log_q"] = np.log1p(base["q"].clip(lower=0))

def build_target(g, log_col):
    g = g.sort_values("datetime").reset_index(drop=True)
    # expanding (causal) per-river threshold: uses only data up to and including day t,
    # not hindsight over the river's full record.
    thr = g[log_col].expanding(min_periods=SPIKE_WARMUP).quantile(SPIKE_PCTILE / 100)
    g["is_spike"] = (g[log_col] > thr).astype(float)
    g.loc[thr.isna(), "is_spike"] = np.nan
    fut = pd.concat([g["is_spike"].shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    g["y"] = (fut.sum(axis=1) > 0).astype(float)
    g.loc[fut.isna().any(axis=1), "y"] = np.nan
    return g

def build_features(g, log_col):
    for lag in [1, 2, 3, 7, 14]:
        g[f"logc_lag{lag}"] = g[log_col].shift(lag)
        g[f"log_q_lag{lag}"] = g["log_q"].shift(lag)
    g["dlog_q_1"] = g["log_q"] - g["log_q"].shift(1)
    g["dlog_q_3"] = g["log_q"] - g["log_q"].shift(3)
    g["dlogc_1"]  = g[log_col] - g[log_col].shift(1)
    g["q_roll7"]  = g["log_q"].rolling(7).mean()
    g["c_roll7"]  = g[log_col].rolling(7).mean()
    doy = g["datetime"].dt.dayofyear
    g["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    g["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return g

def evaluate(mdf, feats):
    aps, bases = [], []
    for test_site in sorted(mdf["site"].unique()):
        tr = mdf[mdf.site != test_site]; te = mdf[mdf.site == test_site]
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=0)
        clf.fit(tr[feats], tr["y"])
        p = clf.predict_proba(te[feats])[:, 1]
        aps.append(average_precision_score(te["y"], p)); bases.append(te["y"].mean())
    return np.mean(aps), np.mean(bases)

# ----------------------------------------------------------------------
# RUN BOTH NUTRIENTS
# ----------------------------------------------------------------------
print(f"{'nutrient':<12}{'model':<10}{'PR-AUC':>8}{'base':>8}{'lift':>7}")
print("-" * 45)
summary = []
for name, col in TARGETS.items():
    df = base.copy()
    df["log_c"] = np.log(df[col])
    df = pd.concat([build_target(g.copy(), "log_c") for _, g in df.groupby("site")], ignore_index=True)
    df = pd.concat([build_features(g.copy(), "log_c") for _, g in df.groupby("site")], ignore_index=True)
    df = df.merge(wx[["site", "datetime"] + WX_FEATS], on=["site", "datetime"], how="left")

    M1 = [c for c in df.columns if c.startswith(("logc_lag", "log_q_lag"))] + \
         ["log_q", "dlog_q_1", "dlog_q_3", "dlogc_1", "q_roll7", "c_roll7", "sin_doy", "cos_doy"]
    M2 = M1 + WX_FEATS

    # evaluate M1 and M2 on the identical row set (M2's superset requirement)
    # so the comparison isolates the effect of adding weather
    common_rows = df.dropna(subset=M2 + ["y"]).copy()
    ap1, b1 = evaluate(common_rows, M1)
    ap2, b2 = evaluate(common_rows, M2)
    print(f"{name:<12}{'M1':<10}{ap1:>8.3f}{b1:>8.3f}{ap1/b1:>6.2f}x")
    print(f"{name:<12}{'M2':<10}{ap2:>8.3f}{b2:>8.3f}{ap2/b2:>6.2f}x")
    print(f"{'':12}{'weather Δ':<10}{ap2-ap1:>+8.3f}")
    print("-" * 45)
    summary.append((name, ap1, ap2, ap2 - ap1))

print("\nSummary:")
for name, ap1, ap2, d in summary:
    print(f"  {name:<12} M1={ap1:.3f}  M2={ap2:.3f}  weather adds {d:+.3f}")