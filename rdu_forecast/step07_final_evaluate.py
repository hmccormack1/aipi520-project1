"""07 Final evaluation: score frozen predictions without fitting or selection."""
import numpy as np
import pandas as pd
from config import *
from integrity import verify_freeze, write_json, read_json, now, metrics, sha256
from step01_download import fetch
from step02_clean import clean_reports

def run():
    frozen = verify_freeze()
    result_file = ART / "07_test_access_audit.json"
    if result_file.exists():
        record = read_json(result_file)
        for path, digest in record["result_hashes"].items():
            if sha256(ROOT / path) != digest:
                raise RuntimeError(f"Stored evaluation changed: {path}")
        print("Final test already evaluated; reusing sealed evaluation without fitting or selecting.", flush=True)
        return
    TEST.mkdir(parents=True, exist_ok=True)
    if not (TEST / "access_started.json").exists():
        write_json(TEST / "access_started.json", {"started_at_utc": now(), "freeze_sha256": sha256(FREEZE),
            "predictions_sha256": sha256(ART / "06_frozen_predictions.csv")})
    # 07.1 Only NOW may the target-window observations be downloaded.
    fetch(CUTOFF-pd.Timedelta(minutes=20), END, TEST / "iem_final_test.csv", ["RDU"], "final_test")
    raw = pd.read_csv(TEST / "iem_final_test.csv", comment="#", na_values=["M", "null"])
    labels = clean_reports(raw, CUTOFF, END)
    labels = labels.loc[labels.station == "RDU"].set_index("time").tmpf
    pred = pd.read_csv(ART / "06_frozen_predictions.csv")
    pred["time_utc"] = pd.to_datetime(pred.time_utc, utc=True)
    expected = pd.date_range(CUTOFF, periods=HORIZON, freq="h")
    if not pd.DatetimeIndex(pred.time_utc).equals(expected):
        raise ValueError("Frozen predictions are not the exact expected 336-hour grid")
    pred["actual_f"] = labels.reindex(expected).to_numpy()
    coverage = pred.actual_f.notna().mean()
    if coverage < MIN_TARGET_COVERAGE:
        raise RuntimeError(f"Only {coverage:.1%} authentic test labels. Need user help; no labels will be invented.")
    pred["forecast_day"] = (pred.lead_hour-1)//24+1
    scores, daily = [], []
    # Preserve PRE-TEST declared order, do not select the winner from this table.
    for name in frozen["final_models"]:
        scores.append({"model": name, "selected_before_test": name == frozen["champion"],
                       **metrics(pred.actual_f, pred[name])})
        for day, g in pred.groupby("forecast_day"):
            daily.append({"model": name, "forecast_day": day, **metrics(g.actual_f, g[name])})
    pred.to_csv(ART / "07_final_predictions_and_actuals.csv", index=False)
    pd.DataFrame(scores).to_csv(ART / "07_final_metrics.csv", index=False)
    pd.DataFrame(daily).to_csv(ART / "07_final_metrics_by_day.csv", index=False)
    ok = pred.actual_f.notna()
    coverage80 = ((pred.loc[ok, "actual_f"] >= pred.loc[ok, "champion_lower80_f"]) &
                  (pred.loc[ok, "actual_f"] <= pred.loc[ok, "champion_upper80_f"])).mean()
    sealed = [ART / "07_final_predictions_and_actuals.csv", ART / "07_final_metrics.csv",
              ART / "07_final_metrics_by_day.csv", TEST / "iem_final_test.csv", TEST / "iem_final_test.json"]
    verify_freeze()
    write_json(result_file, {"evaluated_at_utc": now(), "freeze_sha256": sha256(FREEZE),
        "labels_available": int(ok.sum()), "expected_labels": HORIZON, "test_coverage": coverage,
        "champion": frozen["champion"], "empirical_80_band_test_coverage": float(coverage80),
        "fit_calls": 0, "feature_selection_calls": 0, "model_selection_calls": 0,
        "result_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sealed}})
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print("Final evaluation complete. Model selection remains unchanged.", flush=True)

if __name__ == "__main__":
    run()
