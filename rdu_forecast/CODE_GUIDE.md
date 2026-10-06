# Reading the code

Start with `00_run_pipeline.py`, then read steps 01–08 in order. Each file handles one stage. Read `tools/09_verify_delivery.py` last: it checks the audit evidence, not the modeling method.

## Follow the course workflow

| Stage | Main function | Input → output | Main idea |
|---|---|---|---|
| Acquire | `step01_download.fetch` | Dates and stations → raw CSV | Download only the allowed period |
| Clean | `step02_clean.clean_reports` | Raw reports → hourly table | Align timestamps, remove duplicates, preserve missing targets |
| Features | `step03_features.features_at` | Past observations and forecast start → predictor table | Only use information available at the forecast start |
| Train | `step04_models.fit_model` | Model settings and past data → fitted model | Learn from training data only |
| Predict | `step04_models.predict_model` | Fitted model and predictors → temperatures | Produce the whole 14-day forecast without new observations |
| Select | `step05_backtest_select.run` | Historical forecast windows → selected models | Compare validation RMSE and feature groups |
| Freeze | `step06_freeze_forecast.run` | Selected models → saved predictions and hashes | Finish all decisions before opening the test |
| Evaluate | `step07_final_evaluate.run` | Frozen predictions and test observations → error metrics | Score only; do not fit or select |
| Explain | `step08_visualize.run` | Saved results → plots and HTML | Compare predictions, errors and model components |

## Names used throughout

- `X`: a pandas table of predictors; each row represents one future hour.
- `y`: the actual temperature for that row, available for historical training only.
- `base`: the seasonal reference prediction learned from earlier years.
- `origin` / `issue_time`: the moment when the entire forecast is made.
- `boundary`: the time separating a training period from its validation period.
- `spec`: a plain Python dictionary containing one model's settings.
- `bundle`: a dictionary containing the settings, fitted estimator and feature names.
- `mask`: a True/False filter selecting eligible rows.
- `frames`: a dictionary of station tables, indexed by timestamp.

## The main models in familiar terms

**OLS:** temperature = intercept + weighted predictors. Sine/cosine features represent daily and yearly cycles. Interaction features let the daily cycle change with season.

**Ridge:** the same regression idea, with a penalty on large coefficients to reduce overfitting.

**Selected anomaly model:** seasonal reference + recent temperature departure × a decreasing weight. The departure is measured over the previous 72 hours. Its influence decreases farther into the future.

**Gradient boosting:** several small decision trees learn corrections to the seasonal reference. It is retained as the nonlinear comparison. Feature-group ablation checks whether adding pressure/wind/dew point or neighboring stations improves historical validation.

**Random forest:** an additional tree-based comparison. It is not the same model as gradient boosting.

## Useful Python and scikit-learn conventions

- `df.loc[mask, columns]` selects rows and columns. It does not train anything.
- `groupby(...).mean()` computes means separately for each group.
- A list comprehension is a compact `for` loop that builds a list.
- `make_pipeline(imputer, scaler, model)` applies those steps in order. Calling `fit` learns every step from the supplied training rows only.
- A function named `run()` is the entry point for that stage. The `if __name__ == "__main__"` block runs it when that file is launched directly.
- `joblib` saves fitted objects; SHA-256 hashes detect changed files. These are reproducibility tools, not extra forecasting algorithms.

## The test rule to remember

Historical validation chooses the model. The final test only measures it. Even choosing features or fitting a scaler on test data would violate this rule.

For a 14-day forecast, tomorrow's observed temperature cannot be used to predict the following day. All predictors must be available at the original start. Training windows crossing a validation boundary are excluded in full.

For the completed experiment, use `python tools/09_verify_delivery.py` to inspect results without training. Keep the existing locks. Any model changes motivated by the published final results need a different, untouched test period.
