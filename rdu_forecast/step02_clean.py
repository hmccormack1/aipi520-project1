"""02 Cleaning: fixed rules, no cross-time interpolation and no fabricated labels."""
import numpy as np
import pandas as pd
from config import *
from integrity import development_allowed, assert_before, write_json

def clean_reports(raw, lower, upper):
    # 02.1 Parse, deduplicate and apply PREDECLARED quality bounds, not test-derived bounds.
    df = raw.copy()
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df = df.loc[(df.valid >= lower - pd.Timedelta(minutes=20)) & (df.valid < upper)].copy()
    df["station"] = df.station.str.replace("^K", "", regex=True)
    bounds = {"tmpf": (-60, 130), "dwpf": (-80, 100), "mslp": (850, 1100),
              "sknt": (0, 150), "drct": (0, 360)}
    for name, (lo, hi) in bounds.items():
        df[name] = pd.to_numeric(df[name], errors="coerce")
        df.loc[~df[name].between(lo, hi), name] = np.nan
    # 02.2 :45--:59 routine report represents the following nominal hour.
    # Preserve actual observation timestamp for causal feature construction.
    df = df.loc[df.valid.dt.minute >= REPORT_MINUTE_MIN].copy()
    df["time"] = df.valid.dt.ceil("h")
    df = df.loc[(df.time >= lower) & (df.time < upper)]
    df = df.sort_values("valid").drop_duplicates(["station", "time"], keep="last")
    return df[["station", "time", "valid"] + VARIABLES].sort_values(["station", "time"]).reset_index(drop=True)

def run():
    development_allowed()
    files = sorted((DEV / "raw").glob("iem_*.csv"))
    if len(files) != CUTOFF.year - START.year + 1:
        raise RuntimeError("Development downloads incomplete")
    raw = pd.concat([pd.read_csv(p, comment="#", na_values=["M", "null"]) for p in files], ignore_index=True)
    assert_before(pd.to_datetime(raw.valid, utc=True), CUTOFF, "raw observations")
    df = clean_reports(raw, START, CUTOFF)
    assert_before(df.time, CUTOFF, "development labels")
    df.to_csv(DEV / "hourly.csv", index=False)
    rdu = df.loc[df.station == "RDU"].set_index("time").sort_index()
    # Fail loudly if latest data are unavailable, rather than forecasting from stale years.
    recent = pd.date_range(CUTOFF - pd.Timedelta(days=14), periods=336, freq="h")
    coverage = rdu.tmpf.reindex(recent).notna().mean()
    if coverage < MIN_TARGET_COVERAGE:
        raise RuntimeError(f"Latest development 14 days coverage only {coverage:.1%}; require authentic newer data.")
    quality = []
    for (station, year), g in df.groupby(["station", df.time.dt.year]):
        first = max(START, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        last = min(CUTOFF, pd.Timestamp(f"{year+1}-01-01", tz="UTC"))
        expected = int((last - first) / pd.Timedelta(hours=1))
        quality.append({"station": station, "year": int(year), "expected_hours": expected,
                        "observed_targets": int(g.tmpf.notna().sum()),
                        "coverage": float(g.tmpf.notna().sum() / expected),
                        **{f"missing_{v}": float(g[v].isna().mean()) for v in VARIABLES}})
    pd.DataFrame(quality).to_csv(ART / "02_data_quality.csv", index=False)
    write_json(ART / "02_cleaning_audit.json", {"raw_rows": len(raw), "hourly_rows": len(df),
        "last_label": df.time.max(), "last_observation": df.valid.max(), "test_rows_read": 0,
        "alignment": "Routine observation at minute >=45 mapped to next hour; latest report wins.",
        "label_imputation": "NONE", "recent_rdu_coverage": coverage})
    print(f"Cleaned {len(df):,} station-hours. Latest 14-day RDU coverage: {coverage:.1%}", flush=True)

def load_development():
    df = pd.read_csv(DEV / "hourly.csv")
    for name in ["time", "valid"]:
        df[name] = pd.to_datetime(df[name], utc=True)
    assert_before(df.time, CUTOFF, "loaded development labels")
    assert_before(df.valid, CUTOFF, "loaded development observations")
    return df

if __name__ == "__main__":
    run()
