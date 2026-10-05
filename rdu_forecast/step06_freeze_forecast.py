"""06 Final fit and freeze: save predictions before unlocking final test labels."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from config import *
from integrity import development_allowed, read_json, write_json, now, source_hashes, sha256, versions
from step02_clean import load_development
from step03_features import station_frames, references_for_year, features_at
from step04_models import fit_model, predict_model
from step05_backtest_select import experiment_signature

def run():
    development_allowed()
    selection = read_json(ART / "05_selection.json")
    if selection["experiment_signature"] != experiment_signature():
        raise RuntimeError("Source/data changed after CV. Re-run DEVELOPMENT before freezing.")
    frames = station_frames(load_development())
    dataset = joblib.load(DEV / "causal_features.joblib")
    refs = references_for_year(frames, CUTOFF.tz_convert(LOCAL_TZ).year)
    x, times, base, source_max = features_at(frames, refs, CUTOFF)
    output = pd.DataFrame({"time_utc": times, "time_local": times.tz_convert(LOCAL_TZ),
                           "lead_hour": np.arange(1,337), "reference_f": base})
    bundles = {}
    for name in selection["final_models"]:
        spec = next(s for s in selection["specs"] if s["id"] == name)
        with threadpool_limits(limits=4):
            model = fit_model(spec, frames["RDU"], dataset, CUTOFF)
            output[name] = predict_model(model, x, times, base)
        bundles[name] = model
        print("Final pre-test fit:", name, flush=True)
    # 06.1 Empirical 80% marginal bands from historical validation residuals only.
    cv = pd.read_csv(ART / "05_cv_predictions.csv")
    cv = cv.loc[cv.model == selection["champion"]].copy()
    cv["error"] = cv.actual_f - cv.prediction_f
    cv["band"] = pd.cut(cv.forecast_day, [0,3,7,14], labels=["1-3", "4-7", "8-14"])
    bands = cv.groupby("band", observed=True).error.quantile([0.1,0.9]).unstack()
    group = pd.cut((output.lead_hour-1)//24+1, [0,3,7,14], labels=["1-3", "4-7", "8-14"])
    output["champion_lower80_f"] = output[selection["champion"]] + group.map(bands[0.1]).astype(float)
    output["champion_upper80_f"] = output[selection["champion"]] + group.map(bands[0.9]).astype(float)
    output.to_csv(ART / "06_frozen_predictions.csv", index=False)
    x.to_csv(ART / "06_future_features.csv", index=False)
    joblib.dump({"models": bundles, "annual_references": refs}, ART / "06_models.joblib", compress=3)
    bands.to_csv(ART / "06_validation_error_quantiles.csv")
    # 06.2 Save linear coefficients in original feature units for explanation.
    rows = []
    for name, bundle in bundles.items():
        if bundle["spec"]["kind"] not in ["ols", "ridge"]:
            continue
        scaler, model = bundle["model"].steps[0][1], bundle["model"].steps[-1][1]
        original_coef = model.coef_/scaler.scale_
        intercept = model.intercept_ - np.dot(scaler.mean_, original_coef)
        rows.append({"model": name, "feature": "intercept", "coefficient_f": intercept})
        rows.extend({"model": name, "feature": key, "coefficient_f": value}
                    for key, value in zip(bundle["columns"], original_coef))
    pd.DataFrame(rows).to_csv(ART / "06_linear_coefficients.csv", index=False)
    sealed = [DEV / "hourly.csv", DEV / "causal_features.joblib", ART / "05_selection.json",
              ART / "05_cv_metrics.csv", ART / "05_cv_predictions.csv", ART / "05_model_selection.csv",
              ART / "05_feature_group_selection.csv", ART / "05_fold_leakage_audit.csv",
              ART / "06_frozen_predictions.csv", ART / "06_future_features.csv", ART / "06_models.joblib",
              ART / "06_validation_error_quantiles.csv", ART / "06_linear_coefficients.csv"]
    sealed += sorted((DEV / "raw").glob("*.csv")) + sorted((DEV / "raw").glob("*.json"))
    # Write manifest LAST. The test downloader refuses to run without this.
    write_json(FREEZE, {"frozen_at_utc": now(), "cutoff": str(CUTOFF), "test_end_exclusive": str(END),
        "champion": selection["champion"], "final_models": selection["final_models"],
        "source_max_for_final_features": str(source_max), "source_hashes": source_hashes(),
        "artifact_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sealed},
        "versions": versions(), "prediction_rows": len(output), "test_access_before_freeze": False,
        "interval_note": "Development-residual empirical bands; selection and time dependence prevent formal coverage guarantees.",
        "availability_note": "Retrospective IEM archive. 15-minute operational buffer is an assumption, not verified issuance provenance."})
    print("FROZEN. Test labels have not been read. Champion:", selection["champion"], flush=True)

if __name__ == "__main__":
    run()
