"""03 Features: use past observations and references fitted only on prior years."""
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from config import *
from integrity import development_allowed, assert_before, write_json, sha256
from step02_clean import load_development

def calendar_features(times, enhanced=True, trend=False):
    idx = pd.DatetimeIndex(times)
    local = idx.tz_convert(LOCAL_TZ)
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    days = np.where(local.is_leap_year, 366.0, 365.0)
    phase = 2 * np.pi * ((local.dayofyear.to_numpy() - 1) + hour / 24) / days
    result = {"hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
              "year_sin": np.sin(phase), "year_cos": np.cos(phase)}
    if enhanced:
        result.update({"hour2_sin": np.sin(4*np.pi*hour/24), "hour2_cos": np.cos(4*np.pi*hour/24),
                       "year2_sin": np.sin(2*phase), "year2_cos": np.cos(2*phase)})
        for daily in ["hour_sin", "hour_cos"]:
            for annual in ["year_sin", "year_cos"]:
                result[daily + "_x_" + annual] = result[daily] * result[annual]
    if trend:
        result["year_trend"] = np.asarray((idx - START) / pd.Timedelta(days=365.25))
    return pd.DataFrame(result)

def station_frames(df):
    return {s: g.set_index("time").sort_index() for s, g in df.groupby("station")}

def fit_reference(frame, year):
    # 03.1 Fixed, causal annual climatology for the residual-learning experiment.
    # A simulated origin in 2020 NEVER uses 2020+ labels in its climatology.
    boundary = origin(f"{year}-01-01")
    history = frame.loc[(frame.index < boundary) &
                        (frame.index >= boundary - pd.DateOffset(years=8))].dropna(subset=["tmpf"])
    if len(history) < 1000:
        raise ValueError(f"Insufficient pre-{year} history")
    assert_before(history.index, boundary, "annual reference labels")
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(calendar_features(history.index), history.tmpf)
    return model

def references_for_year(frames, year):
    return {s: fit_reference(frame, year) for s, frame in frames.items()}

def context_at(frames, refs, issue_time):
    # 03.2 Only origin-known measurements are eligible. No bfill / centered rolling.
    state = {}
    latest_sources = []
    for station in STATIONS:
        frame = frames[station]
        past = frame.loc[(frame.index < issue_time) &
                         (frame.index >= issue_time - pd.Timedelta(hours=168)) &
                         (frame.valid + pd.Timedelta(minutes=AVAILABILITY_BUFFER_MINUTES) <= issue_time)].copy()
        if len(past):
            latest_sources.append(past.valid.max())
            past["anomaly"] = past.tmpf.to_numpy() - refs[station].predict(calendar_features(past.index))
        else:
            past["anomaly"] = pd.Series(dtype=float)
        if station != "RDU":
            recent = past.loc[past.index >= issue_time - pd.Timedelta(hours=24)]
            state[f"neighbor_{station}_anom24"] = recent.anomaly.mean()
            state[f"neighbor_{station}_coverage24"] = recent.tmpf.notna().sum() / 24
            continue
        for hours in [24, 72, 168]:
            p = past.loc[past.index >= issue_time - pd.Timedelta(hours=hours)]
            state[f"state_anom{hours}"] = p.anomaly.mean()
            state[f"state_coverage{hours}"] = p.tmpf.notna().sum() / hours
        p = past.loc[past.index >= issue_time - pd.Timedelta(hours=24)]
        observed = past.dropna(subset=["tmpf"])
        state["state_last_anomaly"] = observed.anomaly.iloc[-1] if len(observed) else np.nan
        state["state_last_age_hours"] = float((issue_time - observed.index[-1]) / pd.Timedelta(hours=1)) if len(observed) else np.nan
        state["state_anomaly_sd24"] = p.anomaly.std()
        state["state_range24"] = p.tmpf.max() - p.tmpf.min()
        state["state_trend24_72"] = state["state_anom24"] - state["state_anom72"]
        state["weather_dewpoint_depression24"] = (p.tmpf - p.dwpf).mean()
        state["weather_pressure24"] = p.mslp.mean()
        older = past.loc[(past.index >= issue_time - pd.Timedelta(hours=48)) &
                         (past.index < issue_time - pd.Timedelta(hours=24))]
        state["weather_pressure_change24"] = p.mslp.mean() - older.mslp.mean()
        angle = np.deg2rad(p.drct)
        state["weather_wind_u24"] = (-p.sknt * np.sin(angle)).mean()
        state["weather_wind_v24"] = (-p.sknt * np.cos(angle)).mean()
    latest = max(latest_sources) if latest_sources else pd.NaT
    if pd.notna(latest) and latest + pd.Timedelta(minutes=AVAILABILITY_BUFFER_MINUTES) > issue_time:
        raise ValueError("Context used an unavailable observation")
    return state, latest

def features_at(frames, refs, issue_time):
    times = pd.date_range(issue_time, periods=HORIZON, freq="h")
    x = calendar_features(times)
    base = refs["RDU"].predict(calendar_features(times))
    x["base_f"] = base
    x["lead_hours"] = np.arange(1, HORIZON+1)
    x["lead_log"] = np.log1p(x.lead_hours)
    state, latest = context_at(frames, refs, issue_time)
    for key, value in state.items():
        x[key] = value
    # Smooth horizon interactions make the DIRECT Ridge correction horizon-aware.
    for tau in [24, 72, 168]:
        decay = np.exp(-(x.lead_hours.to_numpy() - 1) / tau)
        for name in ["state_anom24", "state_anom72", "state_last_anomaly", "state_trend24_72"]:
            x[f"decay{tau}_{name}"] = state[name] * decay
    return x, times, base, latest

def columns_for(group, columns, direct_linear=False):
    selected = []
    for c in columns:
        if c.startswith("decay"):
            if direct_linear:
                selected.append(c)
        elif c.startswith("weather_"):
            if group in ["weather", "neighbors"]:
                selected.append(c)
        elif c.startswith("neighbor_"):
            if group == "neighbors":
                selected.append(c)
        else:
            selected.append(c)
    return selected

def training_mask(meta, boundary):
    # 03.3 Purge WHOLE 336-hour trajectories whose labels cross the fold boundary.
    mask = (meta["origin"] + pd.Timedelta(hours=HORIZON) <= boundary) & (meta["target_time"] < boundary)
    assert_before(meta.loc[mask, "target_time"], boundary, "direct-model train labels")
    return mask

def run():
    development_allowed()
    frames = station_frames(load_development())
    issues = pd.date_range("2018-01-01", CUTOFF.tz_convert(LOCAL_TZ).tz_localize(None) - pd.Timedelta(days=14),
                           freq="7D", tz=LOCAL_TZ).tz_convert("UTC")
    arrays, metadata, targets, bases = [], [], [], []
    year, refs = None, None
    for i, issue in enumerate(issues):
        if issue.tz_convert(LOCAL_TZ).year != year:
            year = issue.tz_convert(LOCAL_TZ).year
            refs = references_for_year(frames, year)
        x, times, base, latest = features_at(frames, refs, issue)
        y = frames["RDU"].tmpf.reindex(times).to_numpy()
        known = np.isfinite(y)
        if known.mean() < MIN_TARGET_COVERAGE or x.state_coverage72.iloc[0] < 0.8:
            continue
        arrays.append(x.loc[known].reset_index(drop=True))
        metadata.append(pd.DataFrame({"origin": issue, "target_time": times[known],
            "source_max": latest, "reference_train_end_exclusive": origin(f"{year}-01-01")}))
        targets.append(y[known]); bases.append(base[known])
        if i % 100 == 0:
            print(f"Causal feature origins: {i+1}/{len(issues)}", flush=True)
    dataset = {"X": pd.concat(arrays, ignore_index=True), "meta": pd.concat(metadata, ignore_index=True),
               "y": np.concatenate(targets), "base": np.concatenate(bases)}
    assert_before(dataset["meta"].target_time, CUTOFF, "all development supervised labels")
    if not (dataset["meta"].source_max < dataset["meta"].origin).all():
        raise ValueError("Causal feature audit failed")
    if not (dataset["meta"].reference_train_end_exclusive <= dataset["meta"].origin).all():
        raise ValueError("Climatology audit failed")
    joblib.dump(dataset, DEV / "causal_features.joblib", compress=3)
    pd.DataFrame({"feature": dataset["X"].columns,
        "group": ["neighbors" if c.startswith("neighbor_") else "weather" if c.startswith("weather_")
                  else "horizon_interaction" if c.startswith("decay") else "state" if c.startswith("state_")
                  else "calendar_or_horizon" for c in dataset["X"].columns]}).to_csv(ART / "03_feature_dictionary.csv", index=False)
    write_json(ART / "03_causality_audit.json", {"supervised_rows": len(dataset["y"]),
        "origins": int(dataset["meta"].origin.nunique()), "max_target_time": dataset["meta"].target_time.max(),
        "feature_sources_precede_origin": True, "annual_reference_uses_prior_years_only": True,
        "scaling_or_imputation_fitted_here": False, "final_test_read": False,
        "development_data_sha256": sha256(DEV / "hourly.csv")})
    print(f"Built {len(dataset['y']):,} causal training rows", flush=True)

if __name__ == "__main__":
    run()
