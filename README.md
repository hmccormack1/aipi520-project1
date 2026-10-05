# AIPI 520 Project 1: Hourly Temperature Forecast for RDU

Predicts the hourly temperature at RDU airport for 12am Sept 17 - 11pm Sept 30, 2026 (336 hours).

## How to run

1. Install the packages:
   ```
   pip install pandas numpy scikit-learn requests
   ```
2. Run the script from inside this folder:
   ```
   python project1_pipeline.py
   ```
3. The first run downloads historical hourly temperatures and saves them to `rdu_hourly.csv` (not tracked by git). Later runs reuse that file. Delete it to download fresh data.

## What the script does

1. Loads hourly temperature data (Open-Meteo archive, RDU coordinates, degrees F).
2. Removes every row at or after the cutoff (12am Sept 17, 2026 local time = 04:00 UTC) and stops with an error if any remain.
3. Builds features from the timestamp only (hour of day, day of year, long-term trend).
4. Validates by time: trains on data before the final 14 days and tests on those 14 days. No random splits.
5. Trains two models: linear regression and a random forest.
6. Refits both on all data before the cutoff and writes predictions for every hour of Sept 17-30.

## Files

- `project1_pipeline.py`: the full pipeline
- `predictions_sep17_sep30.csv`: hourly predictions from both models

## Avoiding future data

All data passes through a single `CUTOFF` value, and assertions check that no training or feature data comes from after it.