"""08 Visualization: read existing results without training or tuning."""
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import *
from integrity import read_json, verify_freeze
from step02_clean import load_development

COLORS = ["#145a7c", "#d77435", "#399b84", "#8b6ab3", "#7e8b93", "#bf5268"]

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)

def run():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 13, "font.size": 10, "figure.facecolor": "white"})
    selected = read_json(ART / "05_selection.json")
    summary = pd.read_csv(ART / "05_model_selection.csv")
    cv = pd.read_csv(ART / "05_cv_predictions.csv")
    models = selected["final_models"]
    # 08.1 EDA uses ONLY development labels.
    df = load_development()
    rdu = df.loc[df.station == "RDU"].copy()
    local = rdu.time.dt.tz_convert(LOCAL_TZ)
    rdu["month"], rdu["hour"] = local.dt.month, local.dt.hour
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    monthly = rdu.groupby("month").tmpf.agg(["mean", "std"])
    ax[0].plot(monthly.index, monthly["mean"], color=COLORS[0], marker="o")
    ax[0].fill_between(monthly.index, monthly["mean"]-monthly["std"], monthly["mean"]+monthly["std"], alpha=.15)
    ax[0].set(title="Annual pattern | development only", xlabel="Month", ylabel="Temperature (F)")
    for month, color in zip([1,4,7,9], COLORS):
        by_hour = rdu.loc[rdu.month == month].groupby("hour").tmpf.mean()
        ax[1].plot(by_hour.index, by_hour, label=f"Month {month}", color=color)
    ax[1].set(title="Daily patterns change with season", xlabel="Local hour", ylabel="Temperature (F)")
    ax[1].legend(ncol=2)
    save(fig, "01_development_patterns.png")
    # 08.2 Cross-validation variability is not a formal confidence interval.
    ordered = summary.sort_values("mean_rmse_f", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = [COLORS[2] if m == selected["champion"] else COLORS[0] for m in ordered.model]
    ax.barh(ordered.model, ordered.mean_rmse_f, xerr=ordered.sd_rmse_f, color=bars, alpha=.85, capsize=3)
    ax.set(title="Model selection | 8 historical 14-day windows", xlabel="Mean window RMSE (F); whiskers = SD across windows")
    save(fig, "02_model_comparison.png")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for name, color in zip(models, COLORS):
        g = cv.loc[cv.model == name].copy()
        g["sq_error"] = (g.prediction_f-g.actual_f)**2
        curve = np.sqrt(g.groupby("forecast_day").sq_error.mean())
        ax.plot(curve.index, curve, label=name, color=color, linewidth=2)
    ax.set(title="Historical error by lead day", xlabel="Forecast day", ylabel="RMSE (F)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "03_cv_error_by_day.png")
    feature = pd.read_csv(ART / "05_feature_group_selection.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{r.against} + {r.added_group} | {r.leaves} leaves" for r in feature.itertuples()]
    ax.barh(labels, feature.mean_rmse_change_f, color=[COLORS[2] if d<0 else COLORS[1] for d in feature.mean_rmse_change_f])
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set(title="Feature-group ablation | validation only", xlabel="Change in mean RMSE (F); negative is better")
    save(fig, "04_feature_ablation.png")
    quality = pd.read_csv(ART / "02_data_quality.csv")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for station, g in quality.groupby("station"):
        ax.plot(g.year, 100*g.coverage, marker="o", label=station)
    ax.set(title="Authentic hourly target coverage", xlabel="Year", ylabel="Available observations (%)", ylim=(90,101))
    ax.legend()
    save(fig, "05_data_coverage.png")
    images = ["01_development_patterns.png", "02_model_comparison.png", "03_cv_error_by_day.png",
              "04_feature_ablation.png", "05_data_coverage.png"]
    test_table = "<p>Final test has not been evaluated.</p>"
    extra = ""
    if (ART / "07_test_access_audit.json").exists():
        verify_freeze()
        evaluation = read_json(ART / "07_test_access_audit.json")
        predictions = pd.read_csv(ART / "07_final_predictions_and_actuals.csv")
        dates = pd.to_datetime(predictions.time_utc, utc=True).dt.tz_convert(LOCAL_TZ)
        scores = pd.read_csv(ART / "07_final_metrics.csv")
        champion = selected["champion"]
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.fill_between(dates, predictions.champion_lower80_f, predictions.champion_upper80_f,
                        color=COLORS[0], alpha=.12, label="Historical residual 80% band")
        ax.plot(dates, predictions.actual_f, color="#222222", linewidth=1.7, label="KRDU observed")
        ax.plot(dates, predictions[champion], color=COLORS[0], linewidth=1.7, label=f"Preselected: {champion}")
        nonlinear = selected["best_nonlinear"]
        if nonlinear != champion:
            ax.plot(dates, predictions[nonlinear], color=COLORS[1], alpha=.7, linewidth=1, label=nonlinear)
        ax.set(title="Unseen final test | fixed 336-hour forecast", ylabel="Temperature (F)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=dates.dt.tz))
        ax.legend(fontsize=8, ncol=2)
        save(fig, "06_final_forecast.png")
        daily = pd.read_csv(ART / "07_final_metrics_by_day.csv")
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, color in zip(models, COLORS):
            g = daily.loc[daily.model == name]
            ax.plot(g.forecast_day, g.rmse_f, color=color, label=name, marker=".")
        ax.set(title="Final test error by day | diagnostic, not model selection", xlabel="Forecast day", ylabel="RMSE (F)")
        ax.legend(fontsize=8, ncol=2)
        save(fig, "07_final_error_by_day.png")
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(dates, predictions.reference_f, color=COLORS[0])
        ax[0].set(title="Nonlinear model decomposition: causal seasonal reference", ylabel="Reference (F)")
        ax[1].plot(dates, predictions[nonlinear]-predictions.reference_f, color=COLORS[1])
        ax[1].axhline(0, color="#555555", linewidth=.6)
        ax[1].set(title=f"Learned correction: {nonlinear}", ylabel="Correction (F)")
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=dates.dt.tz))
        save(fig, "08_model_decomposition.png")
        images += ["06_final_forecast.png", "07_final_error_by_day.png", "08_model_decomposition.png"]
        test_table = scores.to_html(index=False, float_format=lambda v: f"{v:.3f}", classes="results")
        extra = f"<p>Authentic test labels: {evaluation['labels_available']}/336. Historical 80% band test coverage: {evaluation['empirical_80_band_test_coverage']:.1%}. No post-test selection.</p>"
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>RDU Forecast | Results</title>
    <style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#173042}}
    h1,h2{{line-height:1.25}}.note{{padding:18px;background:#edf5f7;border-left:4px solid #145a7c}}
    table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;text-align:right;border-bottom:1px solid #d8e0e5}}
    th:first-child,td:first-child{{text-align:left}}img{{width:100%;margin:22px 0}}code{{background:#edf5f7;padding:2px 5px}}
    </style><h1>RDU hourly temperature forecast</h1>
    <p>One forecast issued at 2026-09-17 00:00 EDT, covering 336 hours. Station observations, Fahrenheit.</p>
    <div class="note"><b>Selected before test: {html.escape(selected['champion'])}</b><br>
    Feature groups and hyperparameters were selected on eight historical windows. Final labels were fetched only after predictions,
    models, source code and data hashes were frozen. This is a retrospective forecast experiment, not a forecast actually issued in September.</div>
    <h2>Final evaluation</h2>{test_table}{extra}
    <h2>Historical validation and model selection</h2>
    <p>These development scores were used for selection and are not an independent final-test estimate.</p>
    {summary.to_html(index=False, float_format=lambda v: f"{v:.3f}")}
    <h2>Feature selection by paired ablation</h2>{feature.to_html(index=False, float_format=lambda v: f"{v:.3f}")}
    <h2>Evidence</h2>{''.join(f'<img src="figures/{name}" alt="{name}">' for name in images)}
    <h2>Limitations</h2><p>Routine :45–:59 reports are mapped to the following nominal hour. An original-publication archive is unavailable,
    so the 15-minute availability buffer is an assumption. Historical corrections may differ from observations available in real time.
    Empirical intervals do not have guaranteed conditional coverage. Station-only 14-day forecasting has limited information about incoming weather.
    Effects and ablations explain predictions, not causal effects. Missing evaluation labels are excluded, never interpolated.</p>
    <p>Data: <a href="https://mesonet.agron.iastate.edu/request/download.phtml">Iowa Environmental Mesonet</a>.
    Audit files: <a href="freeze_manifest.json">freeze manifest</a> and <a href="07_test_access_audit.json">test access audit</a>.</p></html>'''
    (ART / "results.html").write_text(page, encoding="utf-8")
    print("Saved charts and artifacts/results.html", flush=True)

if __name__ == "__main__":
    run()
