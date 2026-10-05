# AIPI 520 Project 1 — RDU Temperature Forecast

Predict 336 hourly temperatures from September 17–30, 2026, using information available before the forecast starts.

## Start here

The current implementation is in **`rdu_forecast/`**.

1. Read [installation and results](rdu_forecast/README.md).
2. Follow the [course-style code guide](rdu_forecast/CODE_GUIDE.md).
3. Open the [recorded results and eight plots](rdu_forecast/reference_results/artifacts/results.html).
4. Review [Git handoff instructions](rdu_forecast/INTEGRATION.md).
5. Share the [English methods and results report](AIPI520_Project1_Methods_and_Results_EN.pdf) or the [Chinese methods and results report](AIPI520_Project1_Methods_and_Results_ZH.pdf).

```powershell
cd rdu_forecast
python tools/09_verify_delivery.py
```

This checks the recorded experiment without installing ML libraries or training a model. Installation and full pipeline commands are in the implementation README.

## Directory map

| Path | Purpose |
|---|---|
| `rdu_forecast/step01_...` through `step08_...` | Download, clean, build features, compare models, freeze, evaluate and visualize |
| `rdu_forecast/tests/` | Synthetic tests for leakage prevention |
| `rdu_forecast/reference_results/` | Original forecasts, metrics, figures and audit evidence tracked in Git |
| `rdu_forecast/data/`, `artifacts/`, `logs/`, `.venv/` | Local working data, results and environment; excluded from Git |
| `project1_pipeline.py` | Original teammate baseline, retained unchanged for comparison |
| `predictions_sep17_sep30.csv` | Original baseline predictions, retained unchanged |

The original root pipeline uses Open-Meteo. The current implementation uses documented IEM station observations. Run the current implementation to reproduce its reported scores; do not mix the two data definitions.

## Test isolation and results

The test was opened only after model selection and predictions were frozen. It was never used for training, preprocessing fits, feature selection or tuning.

The preselected primary model achieved test RMSE **7.307°F**, MAE **6.097°F**, R² **0.492**. The predefined gradient-boosting comparator achieved RMSE **6.969°F**, MAE **5.298°F**, R² **0.538**. The primary model was not switched after seeing test results. All 336 test hours have observed labels. This is a retrospective experiment, not a forecast issued in September.

## Team workflow

Work on `codex/rdu-station-forecast` and review it against `main`. The original baseline code and CSV are unchanged; the root README now identifies the current implementation. Large data and virtual environments stay out of Git.

Do not tune against the published test period or remove the freeze locks. A new modeling iteration needs a new untouched test period.
