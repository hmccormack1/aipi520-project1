# Project 1: predict hourly temperature at RDU for Sept 17 - Sept 30, 2026
# RULE: the model may only see data from BEFORE 12am Sept 17 (local time).
# Written one step per line with a comment per step.

import os
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------
# STEP 0: the cutoff. EVERYTHING is filtered through this one value.
# 12am Sept 17 in Raleigh (EDT, UTC-4) is 04:00 UTC.
# We work in UTC so daylight saving time cannot shift the cutoff.
# ---------------------------------------------------------------
CUTOFF = pd.Timestamp("2026-09-17 04:00:00", tz="UTC")
LOCAL_TZ = "America/New_York"
CACHE_FILE = "rdu_hourly.csv"


# ---------------------------------------------------------------
# STEP 1: get the data (cached so we only download once)
# ---------------------------------------------------------------
def load_raw_data():
    # if we already downloaded the data, just read it from the file
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
    else:
        # otherwise download hourly temperature from Open-Meteo (RDU lat/lon)
        # end_date is Sept 16, but we ALSO filter below in case the API gives extra
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 35.8776,
            "longitude": -78.7875,
            "start_date": "2015-01-01",
            "end_date": "2026-09-17",
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
        }
        response = requests.get(url, params=params, timeout=120)
        data = response.json()
        df = pd.DataFrame({
            "time": data["hourly"]["time"],
            "temp": data["hourly"]["temperature_2m"],
        })
        # save to a file so we do not need to download again
        df.to_csv(CACHE_FILE, index=False)
    # turn the time column into real timestamps in UTC
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ---------------------------------------------------------------
# STEP 2: leakage guard. Keep only rows strictly before the cutoff,
# then CHECK that nothing after the cutoff is left.
# ---------------------------------------------------------------
def remove_future_data(df):
    # keep only rows before the cutoff
    df = df[df["time"] < CUTOFF].copy()
    # drop rows with missing temperatures
    df = df.dropna(subset=["temp"])
    # safety check: this stops the program if any future row survived
    assert df["time"].max() < CUTOFF, "LEAK: data at or after the cutoff is in the dataset!"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# STEP 3: features. These use ONLY the timestamp (calendar info),
# so we know them for any future hour. No lag features, because
# "temperature 1 hour ago" is unknown for most of the forecast window.
# ---------------------------------------------------------------
def make_features(times):
    # convert UTC times to local time so "hour" means hour in Raleigh
    local = times.dt.tz_convert(LOCAL_TZ)
    hour = local.dt.hour + local.dt.minute / 60.0
    day_of_year = local.dt.dayofyear
    features = pd.DataFrame(index=times.index)
    # hour of day as a circle (so 11pm and 12am are close together)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # day of year as a circle (so Dec 31 and Jan 1 are close together)
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # years since 2015 captures slow warming trends
    features["year_trend"] = (times - pd.Timestamp("2015-01-01", tz="UTC")).dt.days / 365.25
    return features


# ---------------------------------------------------------------
# STEP 4: simple error measures, written out step by step
# ---------------------------------------------------------------
def rmse(actual, predicted):
    errors = actual - predicted
    squared = errors ** 2
    return float(np.sqrt(squared.mean()))


def mae(actual, predicted):
    errors = actual - predicted
    return float(np.abs(errors).mean())


# ---------------------------------------------------------------
# STEP 5: build the models
# ---------------------------------------------------------------
def make_models():
    models = {}
    # required model 1: linear regression
    models["linear_regression"] = LinearRegression()
    # required model 2 (other model): random forest
    models["random_forest"] = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    return models


def main():
    # load, then immediately remove anything from the future
    raw = load_raw_data()
    df = remove_future_data(raw)
    print("Rows available:", len(df), "| last timestamp:", df["time"].max())

    # ---------------------------------------------------------------
    # STEP 6: honest validation that mimics the real task.
    # The real task: forecast 336 hours (14 days) after the last known data.
    # So: pretend the data ends 14 days earlier, train before that point,
    # and test on the final 14 days. Split by TIME, never randomly.
    # ---------------------------------------------------------------
    val_start = CUTOFF - pd.Timedelta(days=14)
    train_df = df[df["time"] < val_start]
    val_df = df[df["time"] >= val_start]
    # check the train rows all come before the validation rows
    assert train_df["time"].max() < val_df["time"].min(), "LEAK: train and validation overlap!"
    print("Train rows:", len(train_df), "| Validation rows:", len(val_df))

    x_train = make_features(train_df["time"])
    y_train = train_df["temp"]
    x_val = make_features(val_df["time"])
    y_val = val_df["temp"]

    # train each model on the training part and score it on the validation part
    models = make_models()
    for name in models:
        model = models[name]
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        print(name, "| RMSE:", round(rmse(y_val.values, predictions), 2),
              "| MAE:", round(mae(y_val.values, predictions), 2))

    # ---------------------------------------------------------------
    # STEP 7: final models. Refit on ALL data before the cutoff,
    # then predict every hour from Sept 17 12am to Sept 30 11pm local.
    # ---------------------------------------------------------------
    x_all = make_features(df["time"])
    y_all = df["temp"]

    # build the list of the 336 hours we need to predict (in UTC)
    future_times = pd.date_range(start=CUTOFF, periods=14 * 24, freq="h")
    future_df = pd.DataFrame({"time": future_times})
    x_future = make_features(future_df["time"])

    output = pd.DataFrame({"time_utc": future_times})
    output["time_local"] = future_df["time"].dt.tz_convert(LOCAL_TZ)
    final_models = make_models()
    for name in final_models:
        model = final_models[name]
        model.fit(x_all, y_all)
        output["pred_" + name] = model.predict(x_future)

    output.to_csv("predictions_sep17_sep30.csv", index=False)
    print("Saved predictions_sep17_sep30.csv with", len(output), "rows")


if __name__ == "__main__":
    main()
