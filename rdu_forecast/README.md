# RDU hourly temperature forecast — complete pipeline

One fixed forecast origin: **September 17, 2026 00:00 America/New_York** (`04:00 UTC`). Forecast all **336 hours through September 30 23:00**. Temperatures are in Fahrenheit.

The complete pipeline has been executed: acquisition, cleaning, causal features, feature-group selection, model selection, frozen final predictions, independent evaluation and eight figures. Active Python comments and docstrings are English.

This is a retrospective experiment run in October, not a forecast actually issued in September. The final labels were withheld until selection and forecasts were frozen. Published test results must not be reused for tuning.

## 1. Recorded results

There are **336/336 genuine observed test labels**. RMSE and MAE should be low; R² should be high. **R² is not classification accuracy.**

| Model | Historical-window mean RMSE (°F) | Test RMSE (°F) | Test MAE (°F) | Test R² | Role fixed before test |
|---|---:|---:|---:|---:|---|
| `anomaly_tau72` | 5.768 | **7.307** | **6.097** | **0.492** | Selected primary model |
| `ols_harmonics` | 5.863 | 7.653 | 6.398 | 0.443 | Required linear regression |
| `ridge_a1000_trend0` | 5.854 | 7.669 | 6.423 | 0.440 | Best Ridge candidate |
| `hgb_neighbors_l15` | 5.879 | **6.969** | **5.298** | **0.538** | Best nonlinear validation candidate |
| `baseline_climatology` | 5.855 | 7.654 | 6.317 | 0.443 | Seasonal baseline |
| `baseline_yesterday` | 8.202 | 8.745 | 7.057 | 0.272 | Repeat-last-day baseline |

The primary model has test RMSE **4.06°C** and MAE **3.39°C**. The nonlinear comparator has RMSE **3.87°C** and MAE **2.94°C**. Its lower test error is reported, but it did **not** replace the primary model after test access.

As a descriptive tolerance statistic, **32.14%** of primary-model predictions and **46.13%** of nonlinear predictions fall within ±2°C of observations. This statistic was added after evaluation for reporting; it was never a selection criterion. It is not classification accuracy.

The nominal 80% historical residual band covered **64.58%** of the final test. Do not claim calibrated 80% coverage.

Start with [the results page](reference_results/artifacts/results.html), [frozen forecasts](reference_results/artifacts/06_frozen_predictions.csv), and [test metrics](reference_results/artifacts/07_final_metrics.csv).

## 2. Numbered files

| Step | File | Purpose |
|---|---|---|
| 00 | `00_run_pipeline.py` | Entry point, leakage tests, ordered execution, frozen-run resume |
| 00 | `config.py`, `integrity.py` | Dates, rules, timezone, hashes and access locks |
| 01 | `step01_download.py` | Bounded public API requests, caching and provenance |
| 02 | `step02_clean.py` | Routine report alignment, duplicates, fixed cleaning rules |
| 03 | `step03_features.py` | Calendar features, past weather state, causal training rows |
| 04 | `step04_models.py` | Registry, fit and prediction for 19 configurations |
| 05 | `step05_backtest_select.py` | Eight historical windows, feature ablation and model selection |
| 06 | `step06_freeze_forecast.py` | Development-only final fit, forecasts and freeze record |
| 07 | `step07_final_evaluate.py` | Test download after freeze; scoring without fitting or selection |
| 08 | `step08_visualize.py` | Eight PNG figures and an HTML results page |
| 09 | `tools/09_verify_delivery.py` | Verify source equivalence, freeze chain and scores |
| Tests | `tests/test_leakage.py` | Nine synthetic-data temporal isolation/access tests |

Code sections carry numbered comments. Step 04 is an imported library called by 05/06, not a separate training command. For plain-language explanations of functions, variables and Python conventions, read [CODE_GUIDE.md](CODE_GUIDE.md).

## 3. Install

Tested with **Python 3.10.8 on Windows**; Python 3.10 or 3.11 is recommended. Open this folder in VS Code, then use PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements.txt` pins direct dependencies; `requirements-lock.txt` pins the full tested environment. The local folder already has its own installed virtual environment. Virtual environments are excluded from Git/ZIP.

| Library | Purpose |
|---|---|
| numpy, pandas | Numeric computation, hourly tables and timezones |
| scipy, scikit-learn | OLS, Ridge, tree ensembles and preprocessing |
| matplotlib | Figures |
| joblib | Model and feature storage |
| threadpoolctl | CPU thread limits |
| tzdata | Windows timezone support |

Downloads use standard-library `urllib`; tests use `unittest`. No API key, GPU, requests or PyTorch is needed.

## 4. Run

**Inspect without training**, even in a fresh Git clone, using standard-library Python only:

```powershell
python tools/09_verify_delivery.py
```

Open `reference_results/artifacts/results.html` in a browser. The verifier authenticates the original freeze-to-access chain, source translation, included artifacts, evaluation hashes and recomputed metrics.

**Resume the existing local folder:**

```powershell
.\.venv\Scripts\python.exe 00_run_pipeline.py --final-eval
```

The local folder contains data, fitted models and a working freeze manifest. The command verifies hashes, reuses sealed evaluation and regenerates plots. **No retraining or reselection occurs.**

**Reproduce from a fresh Git clone:** install dependencies and run the same command. Git excludes large development data and runtime caches, so it executes the predetermined pipeline and downloads public observations. It freezes predictions before fetching final labels. Network and several minutes of CPU time are needed.

That is a reproduction of an already evaluated specification, **not a new unseen evaluation**. Do not modify features, candidates or tuning based on these published test results. New development needs a new untouched test period. Later archive corrections may prevent byte-identical fresh downloads.

Omit `--final-eval` for development-only execution in a fresh experiment. Run synthetic leakage tests independently:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If final download fails, preserve the freeze/access locks and rerun after restoring connectivity. Never delete the locks to tune again.

## 5. Data and hour convention

- Source: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml), [API documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help).
- History: January 2015 through the cutoff. Target station `RDU`; neighboring candidates `GSO` and `CLT`.
- Variables: temperature, dew point, sea-level pressure, wind speed/direction.
- Request routine `report_type=3` only. Reports at minutes :45–:59 map to the **following nominal hour**; retain the latest report per station/hour. Do not average the hour.
- The mapping follows the course's station-report convention. Verify the timestamp convention if an official instructor label file becomes available; these scores use the documented IEM mapping.
- Missing targets stay missing and are excluded, never interpolated into artificial truth. Coverage is reported.
- Raw-file sidecars record URL, request bounds, download time, row count and SHA-256.
- Clean development data: **307,323 station-hours**. Causal supervised data: **151,960 labeled rows, 453 historical origins**.

The development request may contain the pre-origin raw observation that maps to the first withheld nominal hour. Cleaning removes every nominal label at/after cutoff before fitting, features or selection. That boundary observation is excluded from model context. Final feature observations end at `2026-09-17 02:54 UTC`.

The archive has no complete original publication/revision timestamps. A 15-minute availability buffer is an explicit assumption, not verified real-time issuance history. Computational test isolation is enforced; retrospective archive corrections remain a limitation.

## 6. Leakage prevention

1. Development and test use separate directories. Steps 01–06 never read test files.
2. Development requests cannot end after cutoff; cleaned observation and target timestamps must precede it.
3. Each simulated origin uses only its own past, including the availability buffer. No backward filling or centered rolling.
4. Annual references are fitted only on years before each simulated origin's year. A 2020 feature never uses a reference fitted on 2021–2026 labels.
5. Purge entire 336-hour training label trajectories crossing a validation origin.
6. Fit imputers/scalers inside the fold's training pipeline. Disable HGB random early stopping.
7. Select features, hyperparameters, models and interval quantiles using historical validation only.
8. Freeze source/data/model/prediction hashes before test requests are allowed.
9. Test access creates a lock; development entry points then refuse to run.
10. Final scoring uses no fitting or selection functions. Sealed evaluation records zero such calls and result hashes.

These are accidental-leakage controls and an audit trail, not a security sandbox against malicious rewriting of all files and hashes.

## 7. Modeling and selection

The 19 fixed configurations cover two baselines, two OLS models, four harmonic Ridge models, three decaying-anomaly models, direct residual Ridge, six HGB models and one random forest.

The primary model combines a learned harmonic Ridge seasonal reference with the latest 72-hour temperature anomaly, decaying with a validation-selected 72-hour time constant. Both components are interpretable.

Nonlinear prediction = **causal seasonal reference + learned correction**. Features include daily/annual harmonics and interactions; forecast lead; past temperature anomaly/variation/age/coverage; dew-point depression, pressure change, wind vectors; neighboring-station anomalies. No actual weather from inside the forecast window becomes a predictor.

Feature selection is paired validation-only group ablation, plus harmonic/trend candidates. HGB compares 7/15 leaves and state/weather/neighbors groups. There is no full-data correlation screening. Training origins are seven days apart; their overlapping label windows are purged at validation boundaries.

Eight local-midnight validation origins: September 17 in 2021/2022/2023; August 15 and September 17 in 2024/2025; August 15 in 2026. Each predicts all 336 hours without new observations.

The fixed selection rule chooses the lowest-complexity model within **1% of minimum mean window RMSE**, then lower RMSE. This is an engineering tie rule, not a significance test. Required OLS and the best validation nonlinear candidate are separately retained.

Historical validation is used for selection and is not an independent performance estimate. Small validation differences do not prove statistical superiority.

Bands use validation-residual 10th/90th percentiles for days 1–3, 4–7, 8–14, frozen before test. Time dependence, selection and distribution change prevent guaranteed conditional coverage.

## 8. Outputs

Runtime outputs are in `artifacts/`. Original recorded outputs included in Git are in `reference_results/artifacts/`:

- `02_data_quality.csv`, `03_feature_dictionary.csv`, `03_causality_audit.json`.
- `05_model_selection.csv`, `05_cv_metrics.csv`, `05_cv_predictions.csv`.
- `05_feature_group_selection.csv`, `05_fold_leakage_audit.csv`.
- `06_frozen_predictions.csv`, `06_future_features.csv`, `06_linear_coefficients.csv`.
- `07_final_metrics.csv`, `07_final_metrics_by_day.csv`, `07_final_predictions_and_actuals.csv`.
- `freeze_manifest.json`, `07_test_access_audit.json`.
- `results.html` and eight PNG figures in `figures/`.

Presentation sequence: target/time split → OLS/nonlinear comparison → validation feature ablation → untouched test forecast → error by lead day and limitations. Reference/correction plots and original-unit coefficients explain predictions, not causal effects. The cool period in the test shows the limits of a station-only 14-day forecast.

## 9. Translation provenance

After the original experiment, module docstrings and exactly two exception messages were translated. The **computational AST is unchanged**, verified file by file. No training, selection, forecasts or scores changed during packaging.

- `reference_results/artifacts/freeze_manifest.json`: original, unmodified freeze record.
- `reference_results/audit/original_frozen_source.zip`: original-source audit evidence, read without execution. Original-language text remains only inside this archive and translation metadata.
- `reference_results/delivery_manifest.json`: old/new hashes, normalized AST hashes and translations.
- Local `artifacts/freeze_manifest.json`: explicitly a **derived delivery manifest**, authenticating English source and the same artifacts, linked to the original freeze. It does not pretend translation occurred before evaluation.
- Historical development files are excluded from Git; their original hashes remain recorded. The existing local data support full offline verification.

## 10. Merge integration

Directory `rdu_forecast/`, branch `codex/rdu-station-forecast`, based on upstream main `034b83a8e748c836a6ab4c7c79c0534156f11899`. The root README now provides a team entry point. Original `project1_pipeline.py`, predictions CSV and `.gitignore` are unchanged.

The original pipeline still runs as before. After `cd rdu_forecast`, the new pipeline uses separate dependencies and outputs. Both forecast files have the same 336-hour `time_utc`/`time_local` grid. Model columns keep truthful names: HGB is not relabeled as random forest.

The merge is checked against the fetched upstream commit; future contributor edits can still introduce conflicts. The root Open-Meteo pipeline does not reproduce these station-data scores.

## 11. References and scope

- [Scikit-learn lagged time-series features](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html): preprocessing and boosting principles; this project additionally enforces a fixed 14-day origin.
- [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504): motivation for strong simple baselines; this is not a paper reproduction.
- [Accurate Intelligible Models with Pairwise Interactions](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf): interpretability background; EBM is not trained here.
- Harmonics, interactions, linear models, regularization, generalization and leakage checks follow the relevant course trajectory. No physical forecast model or future weather product is used.

This README is engineering documentation. Team members should write their own assignment slides and narrative.
