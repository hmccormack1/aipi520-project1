# AIPI 520 Project 1: Technical Walkthrough

## Research story

Seasonality is already a strong baseline. OLS adds almost no improvement beyond it. A recent-temperature anomaly adjustment gives a meaningful, simple, interpretable improvement. HGB captures additional nonlinear structure and achieved lower error on the final test period, but it is more complex and has substantial warm bias. The primary model was chosen using historical validation and was not changed after final-test access.

## 1. Problem and forecasting setup

- **Task:** one 336-hour forecast for RDU airport temperature.
- **Issue time:** 2026-09-17 00:00 EDT (2026-09-17 04:00 UTC).
- **Forecast window:** every hour through 2026-09-30 23:00 EDT.
- **Constraint:** no new observations are supplied after the issue time.

```text
Past station history -> one forecast origin -> 336 future hourly predictions
        development                           untouched final test
```

## 2. Data and cutoff

The implementation downloads public routine ASOS/METAR reports from the Iowa Environmental Mesonet (IEM). RDU is the target station; GSO and CLT are optional neighboring-station inputs. Variables are temperature, dew point, sea-level pressure, wind speed, and wind direction. Development history starts on 2015-01-01 and ends strictly before the forecast cutoff.

The development and final-test directories are separate. Development API requests cannot cross the cutoff, and loaded observation and target timestamps are checked again before use.

## 3. Target construction

Routine reports at minutes **:45-:59** are mapped to the following nominal hour, and the latest eligible report for each station-hour is kept. The actual report timestamp is retained for availability checks. This is the project's documented operational mapping of the course hourly target; an official instructor label file would be required to verify exact timestamp equivalence.

Fixed plausibility bounds convert invalid fields to missing. Target temperatures are never interpolated or fabricated. The cleaned development set contains **307,323 station-hours**; the most recent 14-day RDU development coverage is 100%.

## 4. Feature engineering

| Group | Main features | Availability rule |
|---|---|---|
| Calendar | daily/yearly sine and cosine, second harmonics, interactions | future timestamps are known |
| Lead | lead hour, log lead, smooth decay interactions | forecast horizon is known |
| RDU state | 24/72/168-hour anomaly, coverage, last value age, range, trend | only observations available before origin |
| Local weather | dew-point depression, pressure level/change, wind components | past 24-48 hours only |
| Neighbors | GSO/CLT 24-hour anomaly and coverage | past reports only |

A seasonal Ridge reference is fitted separately for each simulated year using only earlier years. For each historical origin, weather-state features are fixed at origin-known values; actual weather inside the future 336-hour window is never used as an input. Feature groups are compared by paired historical-validation ablation, not by screening against the final test.

## 5. Baselines

1. **Seasonal climatology:** average temperature for the same hour in a nearby calendar-day window, computed from training data only.
2. **Previous-day persistence:** repeat the last complete 24-hour temperature pattern.

The seasonal baseline is strong: final-test RMSE is **7.654°F**. The persistence baseline is weaker at 8.745°F.

## 6. Required linear regression

OLS uses known calendar harmonics and their interactions. It is transparent: each coefficient changes the prediction through a known periodic component. Its final-test RMSE is **7.653°F**, essentially identical to the seasonal baseline's 7.654°F. Adding an unrestricted linear fit to seasonality provides almost no extra final-test accuracy here.

## 7. Interpretable anomaly model

The selected `anomaly_tau72` model is:

```text
forecast = causal seasonal reference
         + recent 72-hour temperature anomaly x exp(-lead / 72)
```

It measures whether the last three days were warmer or cooler than the learned seasonal pattern, then gradually reduces that adjustment farther into the forecast.

- Validation RMSE: **5.768°F**
- Test RMSE / MAE / R² / bias: **7.307°F / 6.097°F / 0.492 / -0.272°F**

## 8. Nonlinear HGB comparison

`hgb_neighbors_l15` uses small histogram-gradient-boosted trees to learn a nonlinear correction to the same causal seasonal reference. It includes RDU state, local weather, neighboring-station features, and forecast lead.

- Validation RMSE: **5.879°F**
- Test RMSE / MAE / R² / bias: **6.969°F / 5.298°F / 0.538 / +3.386°F**

HGB had the lowest error on this final test, but it is more complex and substantially warm-biased. That result was not used to change the selected primary model.

## 9. Temporal validation and leakage prevention

Eight historical 14-day windows are used: September 17 in 2021-2023; August 15 and September 17 in 2024-2025; and August 15 in 2026.

- Each fold predicts all 336 hours without receiving future observations.
- Training keeps only trajectories whose complete 336-hour target window ends before the validation origin.
- Imputation and scaling are fitted inside each training fold; annual references use prior years only.
- Development and final-test files are separated.
- Predictions, models, source, and development artifacts are hashed before test access.
- Final evaluation performs zero fitting, feature selection, or model selection.

## 10. Evaluation metrics and why RMSE is primary

| Metric | Meaning |
|---|---|
| RMSE | square root of mean squared error; penalizes large hourly misses and is the primary selection metric |
| MAE | average absolute hourly error |
| Bias | average prediction minus observation; positive means too warm |
| R² | variation explained relative to the test-period mean; not classification accuracy |

RMSE is primary because large temperature errors matter disproportionately and one consistent metric was needed for validation-only selection. All 336 final-test hours have authentic labels.

## 11. Validation-only model selection

Nineteen configurations were compared using mean RMSE across the eight validation windows. The predefined rule selects the least complex model only **within 1% of the minimum validation RMSE**, then uses lower RMSE as the remaining tiebreaker.

`anomaly_tau72` had the minimum validation RMSE of 5.768°F. HGB's 5.879°F was approximately **1.9% worse**, so HGB was **outside** the predefined 1% near-tie region. The anomaly model won on validation performance under the rule; it was not chosen over HGB merely because it was simpler.

## 12. Freeze

Step 06 fitted only the predeclared final comparison set, saved 336 predictions, and froze code/data/model/prediction hashes before step 07 accessed test labels. The final evaluator refuses to proceed if the frozen evidence changes.

## 13. Final untouched-test results

| Model | RMSE °F | MAE °F | Bias °F | R² | Role fixed before test |
|---|---:|---:|---:|---:|---|
| Seasonal baseline | 7.654 | 6.317 | -0.124 | 0.443 | baseline |
| OLS harmonics | 7.653 | 6.398 | -0.695 | 0.443 | required linear model |
| `anomaly_tau72` | **7.307** | **6.097** | **-0.272** | **0.492** | selected primary model |
| `hgb_neighbors_l15` | **6.969** | **5.298** | **+3.386** | **0.538** | nonlinear comparator |

The final table is for evaluation, not reselection. The selected primary model remains `anomaly_tau72`.

## 14. Conclusions, tradeoffs, and limitations

- Seasonality explains much of the predictable pattern; OLS adds almost nothing beyond it.
- Recent anomaly improves the seasonal reference with a clear, low-complexity mechanism.
- HGB captures additional nonlinear structure and performed best on this test, but its complexity and warm bias weaken interpretability and robustness claims.
- The conclusion is a performance-versus-interpretability/generalization tradeoff, not "the most complex model wins."
- A station-only 14-day forecast cannot know future fronts, cloud, precipitation, or large-scale circulation.
- The retrospective IEM archive may include later corrections; the 15-minute availability buffer is an explicit assumption.
- The final test is now published. Any model change requires a new untouched test period.

## 15. Code map

| Stage | File |
|---|---|
| Configuration and integrity | `config.py`, `integrity.py` |
| 1-2. Data download and target cleaning | `step01_download.py`, `step02_clean.py` |
| 3. Feature engineering | `step03_features.py` |
| 4-7. Baselines and model definitions | `step04_models.py` |
| 8-10. Backtesting, ablation, selection | `step05_backtest_select.py` |
| 10. Final development-only fit and freeze | `step06_freeze_forecast.py` |
| 11. Untouched-test scoring | `step07_final_evaluate.py` |
| Figures and results page | `step08_visualize.py` |
| Ordered entry point | `00_run_pipeline.py` |

Data source: Iowa Environmental Mesonet ASOS/METAR archive, https://mesonet.agron.iastate.edu/request/download.phtml
