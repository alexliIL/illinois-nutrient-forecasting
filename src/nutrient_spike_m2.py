"""
Illinois nutrient-spike FORECASTING — M2 (river-state + WEATHER).

Same target and evaluation as M1, but adds Daymet weather features:
antecedent rainfall over 1/3/7/14/30 days, antecedent dry period, temperature.

Run M1 and M2 and compare mean PR-AUC on unseen rivers -> does weather help?

Files expected (run from project root):
    data/Illinois_nutrient_and_sediment_concentrations_wy2016-2021.csv
    data/weather_daymet.csv   (columns: time, prcp (mm/day), tmax (degrees C), tmin (degrees C), site)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
NUT_CSV = "data/Illinois_nutrient_and_sediment_concentrations_wy2016-2021.csv"
WX_CSV  = "data/weather_daymet.csv"
TARGET_COL = "total_p_mg_per_l"
SPIKE_PCTILE = 90
HORIZON = 7

# ----------------------------------------------------------------------
# LOAD NUTRIENTS
# ----------------------------------------------------------------------
df = pd.read_csv(NUT_CSV, parse_dates=["datetime"])
df = df.sort_values(["USGS_site", "datetime"]).reset_index(drop=True)
df["site"] = df["USGS_site"]
df["log_p"] = np.log(df[TARGET_COL])
df["log_q"] = np.log1p(df["q"].clip(lower=0))   # clip avoids log1p warning on tiny negatives

# ----------------------------------------------------------------------
# LOAD + CLEAN WEATHER, then build weather features per site
# ----------------------------------------------------------------------
wx = pd.read_csv(WX_CSV)
wx = wx.rename(columns={
    "time": "datetime",
    "prcp (mm/day)": "prcp",
    "tmax (degrees C)": "tmax",
    "tmin (degrees C)": "tmin",
})
wx["datetime"] = pd.to_datetime(wx["datetime"])
wx = wx.sort_values(["site", "datetime"]).reset_index(drop=True)

def weather_features(g):
    g = g.sort_values("datetime")
    p = g["prcp"]
    for w in [1, 3, 7, 14, 30]:
        g[f"rain_{w}d"] = p.rolling(w, min_periods=1).sum()
    # antecedent dry period: days since last rain > 1mm
    wet = (p > 1.0).astype(int)
    days_since = []
    c = 0
    for v in wet:
        c = 0 if v == 1 else c + 1
        days_since.append(c)
    g["dry_days"] = days_since
    g["tmean"] = (g["tmax"] + g["tmin"]) / 2.0
    return g

wx = pd.concat([weather_features(g.copy()) for _, g in wx.groupby("site")],
               ignore_index=True)

WX_FEATS = ["rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "dry_days", "tmean"]

# merge weather onto nutrient rows by site + date (inner: keeps overlapping dates only)
df = df.merge(wx[["site", "datetime"] + WX_FEATS], on=["site", "datetime"], how="left")
print("Weather merge coverage: {:.1%} of nutrient rows have rain_7d"
      .format(df["rain_7d"].notna().mean()))

# ----------------------------------------------------------------------
# TARGET
# ----------------------------------------------------------------------
def build_target(g):
    thr = np.percentile(g["log_p"], SPIKE_PCTILE)
    g["is_spike"] = (g["log_p"] > thr).astype(int)
    fut = pd.concat([g["is_spike"].shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    g["y"] = (fut.sum(axis=1) > 0).astype(float)
    g.loc[fut.isna().any(axis=1), "y"] = np.nan
    return g

df = pd.concat([build_target(g.copy()) for _, g in df.groupby("site")], ignore_index=True)

# ----------------------------------------------------------------------
# RIVER-STATE FEATURES (same as M1)
# ----------------------------------------------------------------------
def build_features(g):
    for lag in [1, 2, 3, 7, 14]:
        g[f"log_p_lag{lag}"] = g["log_p"].shift(lag)
        g[f"log_q_lag{lag}"] = g["log_q"].shift(lag)
    g["dlog_q_1"] = g["log_q"] - g["log_q"].shift(1)
    g["dlog_q_3"] = g["log_q"] - g["log_q"].shift(3)
    g["dlog_p_1"] = g["log_p"] - g["log_p"].shift(1)
    g["q_roll7"]  = g["log_q"].rolling(7).mean()
    g["p_roll7"]  = g["log_p"].rolling(7).mean()
    doy = g["datetime"].dt.dayofyear
    g["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    g["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return g

df = pd.concat([build_features(g.copy()) for _, g in df.groupby("site")], ignore_index=True)

M1_FEATS = [c for c in df.columns if c.startswith(("log_p_lag", "log_q_lag"))] + [
    "log_q", "dlog_q_1", "dlog_q_3", "dlog_p_1", "q_roll7", "p_roll7", "sin_doy", "cos_doy",
]
M2_FEATS = M1_FEATS + WX_FEATS

# ----------------------------------------------------------------------
# EVALUATION: leave-one-river-out, for a given feature set
# ----------------------------------------------------------------------
def evaluate(feats, label):
    mdf = df.dropna(subset=feats + ["y"]).copy()
    sites = sorted(mdf["site"].unique())
    aps, bases = [], []
    for test_site in sites:
        tr = mdf[mdf.site != test_site]
        te = mdf[mdf.site == test_site]
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=0)
        clf.fit(tr[feats], tr["y"])
        p = clf.predict_proba(te[feats])[:, 1]
        aps.append(average_precision_score(te["y"], p))
        bases.append(te["y"].mean())
    print(f"{label}: mean PR-AUC={np.mean(aps):.3f}  (base {np.mean(bases):.3f}, "
          f"lift {np.mean(aps)/np.mean(bases):.2f}x)  n_rows={len(mdf)}")
    return np.mean(aps)

print()
m1 = evaluate(M1_FEATS, "M1 (river-state)      ")
m2 = evaluate(M2_FEATS, "M2 (river + weather)  ")
print(f"\nWeather adds {m2 - m1:+.3f} PR-AUC on unseen rivers.")