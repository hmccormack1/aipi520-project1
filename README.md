# AIPI 520 Project 1 - RDU Temperature Forecasting

## Project Objective

Forecast hourly temperature at RDU for a 336-hour horizon, from 2026-09-17 00:00 EDT through 2026-09-30 23:00 EDT. The forecast is issued once and uses only information available before the strict origin cutoff (2026-09-17 04:00 UTC); no observations arriving inside the forecast window are used as predictors.

## Modeling Workflow

```text
Data -> Target Construction -> Feature Engineering -> Baselines
     -> Linear Regression -> Interpretable Anomaly Model -> Nonlinear HGB
     -> Temporal Validation -> Model Selection & Freeze
     -> Final Untouched Test -> Conclusion
```

## Main Findings

- Seasonality is a strong baseline, and OLS adds almost no improvement beyond it.
- Recent temperature anomaly provides a meaningful, simple, interpretable improvement.
- HGB captures additional nonlinear structure.
- The primary model was selected using historical validation only; the final test was not used to change model selection.
- HGB achieved lower error on this specific final test period, but that result does not invalidate the validation-based selection process.
- The conclusion is a performance-versus-interpretability/generalization tradeoff, not "the most complex model wins."

## Start Here

1. Project overview: [docs/PROJECT_FLOW_EN.pdf](docs/PROJECT_FLOW_EN.pdf)
2. Code walkthrough: [rdu_forecast/CODE_GUIDE.md](rdu_forecast/CODE_GUIDE.md)
3. Technical setup and reproduction: [rdu_forecast/README.md](rdu_forecast/README.md)
4. Recorded figures and tables: [results.html](rdu_forecast/reference_results/artifacts/results.html)
5. Verify without retraining:

```powershell
cd rdu_forecast
python tools/09_verify_delivery.py
```

## Key Results

| Model | Validation RMSE (°F) | Test RMSE (°F) | Test MAE (°F) | Test R² | Test bias (°F) |
|---|---:|---:|---:|---:|---:|
| Seasonal baseline | - | 7.654 | 6.317 | 0.443 | -0.124 |
| OLS harmonics | 5.863 | 7.653 | 6.398 | 0.443 | -0.695 |
| `anomaly_tau72` | **5.768** | **7.307** | **6.097** | **0.492** | **-0.272** |
| `hgb_neighbors_l15` | **5.879** | **6.969** | **5.298** | **0.538** | **+3.386** |

`anomaly_tau72` had the minimum validation RMSE. HGB was approximately 1.9% worse on validation, placing it outside the predefined 1% near-tie region. The selected primary model therefore remains `anomaly_tau72`; HGB's lower final-test error is reported as an untouched-test result, not used for retrospective reselection.

## Repository Map

| Stage | Existing implementation |
|---|---|
| Data acquisition | `rdu_forecast/step01_download.py` |
| Cleaning and target construction | `rdu_forecast/step02_clean.py` |
| Feature engineering | `rdu_forecast/step03_features.py` |
| Baselines and model definitions | `rdu_forecast/step04_models.py` |
| Backtesting and model selection | `rdu_forecast/step05_backtest_select.py` |
| Final development-only fit and freeze | `rdu_forecast/step06_freeze_forecast.py` |
| Final untouched-test evaluation | `rdu_forecast/step07_final_evaluate.py` |
| Visualization and results page | `rdu_forecast/step08_visualize.py` |

## Verification

### A. Verify the frozen submitted result without retraining

```powershell
cd rdu_forecast
python tools/09_verify_delivery.py
```

This checks source equivalence, frozen artifacts, the freeze-before-test chain, test-access evidence, and recorded metrics using only the standard library.

### B. Reproduce the full experiment from scratch, if desired

Follow [rdu_forecast/README.md](rdu_forecast/README.md). A fresh reproduction downloads public data and runs the fixed pipeline, so it requires network access and the pinned dependencies. The published final period is no longer an unseen test for future model development.
