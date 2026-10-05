"""05 Historical backtesting, feature groups and tuning: development data only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from config import *
from integrity import development_allowed, metrics, write_json, sha256, now
from step02_clean import load_development
from step03_features import station_frames, references_for_year, features_at, training_mask
from step04_models import candidates, fit_model, predict_model

def experiment_signature():
    files = ["config.py", "integrity.py", "step02_clean.py", "step03_features.py", "step04_models.py", "step05_backtest_select.py"]
    obj = {name: sha256(ROOT / name) for name in files}
    obj["hourly"] = sha256(DEV / "hourly.csv")
    obj["features"] = sha256(DEV / "causal_features.joblib")
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def choose(summary, specs):
    # 05.1 Predeclared 1% near-tie rule: prefer simpler models only within 1% CV RMSE.
    best = summary.mean_rmse_f.min()
    acceptable = summary.loc[summary.mean_rmse_f <= best * 1.01]
    champion = acceptable.sort_values(["complexity", "mean_rmse_f"]).iloc[0].model
    kinds = {s["id"]: s["kind"] for s in specs}
    def best_of(allowed):
        return summary.loc[summary.model.map(kinds).isin(allowed)].sort_values("mean_rmse_f").iloc[0].model
    selected = {"champion": champion, "best_ols": best_of(["ols"]),
                "best_ridge": best_of(["ridge", "direct_ridge"]), "best_nonlinear": best_of(["hgb", "rf"])}
    selected["final_models"] = list(dict.fromkeys(["baseline_climatology", "baseline_yesterday", *selected.values()]))
    return selected

def run():
    development_allowed()
    frames = station_frames(load_development())
    dataset = joblib.load(DEV / "causal_features.joblib")
    specs = candidates()
    signature = experiment_signature()
    results, all_predictions, audits = [], [], []
    checkpoint = ART / "cv_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    for date in FOLD_DATES:
        issue = origin(date)
        if issue + pd.Timedelta(hours=HORIZON) > CUTOFF:
            raise ValueError("Development validation window overlaps final test")
        refs = references_for_year(frames, issue.tz_convert(LOCAL_TZ).year)
        x, times, base, source_max = features_at(frames, refs, issue)
        actual = frames["RDU"].tmpf.reindex(times).to_numpy()
        if np.isfinite(actual).mean() < MIN_TARGET_COVERAGE:
            raise RuntimeError(f"Insufficient validation labels for {date}")
        mask = training_mask(dataset["meta"], issue)
        audits.append({"fold": date, "origin_utc": issue, "train_target_max": dataset["meta"].loc[mask, "target_time"].max(),
                       "feature_source_max": source_max, "validation_end_exclusive": times[-1]+pd.Timedelta(hours=1),
                       "purged_training_rows": int(mask.sum()), "validation_observed": int(np.isfinite(actual).sum()),
                       "test_access": False})
        for spec in specs:
            stamp = checkpoint / f"{date}_{spec['id']}.joblib"
            if stamp.exists():
                saved = joblib.load(stamp)
            else:
                saved = {}
            if saved.get("signature") == signature:
                row, p = saved["metrics"], saved["predictions"]
            else:
                started = time.perf_counter()
                with threadpool_limits(limits=4):
                    model = fit_model(spec, frames["RDU"], dataset, issue)
                    prediction = predict_model(model, x, times, base)
                row = {"fold": date, "model": spec["id"], "kind": spec["kind"],
                       "complexity": spec["complexity"], **metrics(actual, prediction),
                       "fit_predict_seconds": time.perf_counter()-started}
                p = pd.DataFrame({"fold": date, "model": spec["id"], "origin_utc": issue,
                    "time_utc": times, "lead_hour": np.arange(1,337), "forecast_day": np.arange(336)//24+1,
                    "actual_f": actual, "prediction_f": prediction, "reference_f": base})
                joblib.dump({"signature": signature, "metrics": row, "predictions": p}, stamp)
            results.append(row); all_predictions.append(p)
            print(f"{date} | {spec['id']:<25} RMSE {row['rmse_f']:.3f} F | {row['fit_predict_seconds']:.1f}s", flush=True)
        pd.DataFrame(results).to_csv(ART / "05_cv_metrics_progress.csv", index=False)
    scores = pd.DataFrame(results)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = scores.groupby("model", as_index=False).agg(mean_rmse_f=("rmse_f", "mean"),
        sd_rmse_f=("rmse_f", "std"), mean_mae_f=("mae_f", "mean"), worst_fold_rmse_f=("rmse_f", "max"),
        mean_fit_seconds=("fit_predict_seconds", "mean"), folds=("fold", "nunique"), complexity=("complexity", "first"))
    summary = summary.sort_values("mean_rmse_f")
    scores.to_csv(ART / "05_cv_metrics.csv", index=False)
    predictions.to_csv(ART / "05_cv_predictions.csv", index=False)
    summary.to_csv(ART / "05_model_selection.csv", index=False)
    pd.DataFrame(audits).to_csv(ART / "05_fold_leakage_audit.csv", index=False)
    selection = choose(summary, specs)
    selection.update(selected_at_utc=now(), experiment_signature=signature,
                     rule="Smallest complexity within 1% of minimum mean fold RMSE; then lower RMSE.",
                     final_test_used=False, specs=specs)
    write_json(ART / "05_selection.json", selection)
    # 05.2 Paired feature-group ablations, same hyperparameters and same validation windows.
    paired = []
    for leaves in [7,15]:
        pivot = scores.pivot(index="fold", columns="model", values="rmse_f")
        for a,b in [("state", "weather"), ("weather", "neighbors")]:
            delta = pivot[f"hgb_{b}_l{leaves}"] - pivot[f"hgb_{a}_l{leaves}"]
            paired.append({"leaves": leaves, "added_group": b, "against": a,
                "mean_rmse_change_f": delta.mean(), "improved_folds": int((delta<0).sum()), "folds": len(delta)})
    pd.DataFrame(paired).to_csv(ART / "05_feature_group_selection.csv", index=False)
    print("Selected on DEVELOPMENT ONLY:", selection["champion"], flush=True)

if __name__ == "__main__":
    run()
