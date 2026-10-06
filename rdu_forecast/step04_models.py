"""04 Model registry: declare all candidates before accessing final test labels."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from config import *
from integrity import assert_before
from step03_features import calendar_features, columns_for, training_mask

def candidates():
    # 04.1 Baselines, required OLS, regularization, anomaly adjustment and nonlinear challengers.
    specs = [dict(id="baseline_climatology", kind="climatology", complexity=0),
             dict(id="baseline_yesterday", kind="persistence", complexity=0),
             dict(id="ols_basic", kind="ols", enhanced=False, trend=False, complexity=1),
             dict(id="ols_harmonics", kind="ols", enhanced=True, trend=False, complexity=2)]
    for alpha in [10, 1000]:
        for trend in [False, True]:
            specs.append(dict(id=f"ridge_a{alpha}_trend{int(trend)}", kind="ridge", enhanced=True,
                              trend=trend, alpha=alpha, complexity=2))
    for tau in [24, 72, 168]:
        specs.append(dict(id=f"anomaly_tau{tau}", kind="anomaly", tau=tau, complexity=3))
    specs.append(dict(id="direct_ridge", kind="direct_ridge", group="state", alpha=1000, complexity=4))
    for group in ["state", "weather", "neighbors"]:
        for leaves in [7, 15]:
            specs.append(dict(id=f"hgb_{group}_l{leaves}", kind="hgb", group=group,
                              leaves=leaves, iterations=180, complexity=6))
    specs.append(dict(id="rf_state", kind="rf", group="state", complexity=7))
    return specs

def fit_model(spec, rdu, dataset, boundary):
    # 04.2 No estimator, imputer, scaler or feature selector can access >=boundary labels.
    past = rdu.loc[rdu.index < boundary].dropna(subset=["tmpf"])
    assert_before(past.index, boundary, "calendar model labels")
    bundle = {"spec": spec, "boundary": boundary}
    kind = spec["kind"]
    if kind in ["ols", "ridge"]:
        x = calendar_features(past.index, spec["enhanced"], spec["trend"])
        reg = LinearRegression() if kind == "ols" else Ridge(alpha=spec["alpha"])
        model = make_pipeline(StandardScaler(), reg)
        model.fit(x, past.tmpf)
        bundle.update(model=model, columns=list(x.columns))
    elif kind == "climatology":
        # Pooled 31-day calendar neighborhood, computed from this fold's training data only.
        local = past.index.tz_convert(LOCAL_TZ)
        day = local.dayofyear.to_numpy()
        table = np.full((366, 24), np.nan)
        for d in range(1, 367):
            distance = np.abs(day-d)
            selected = (np.minimum(distance, 366-distance) <= 15)
            values = pd.DataFrame({"hour": local.hour[selected], "temp": past.tmpf.to_numpy()[selected]})
            table[d-1] = values.groupby("hour").temp.mean().reindex(range(24)).to_numpy()
        bundle.update(table=table, fallback=float(past.tmpf.mean()))
    elif kind == "persistence":
        last_day = pd.date_range(boundary-pd.Timedelta(hours=24), periods=24, freq="h")
        values = past.tmpf.reindex(last_day).to_numpy()
        bundle.update(last_day=np.where(np.isfinite(values), values, past.tmpf.tail(72).mean()))
    elif kind == "anomaly":
        pass
    else:
        mask = training_mask(dataset["meta"], boundary)
        cols = columns_for(spec["group"], dataset["X"].columns, kind == "direct_ridge")
        x = dataset["X"].loc[mask, cols]
        y = (dataset["y"] - dataset["base"])[mask.to_numpy()]
        if kind == "direct_ridge":
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=spec["alpha"]))
        elif kind == "hgb":
            # Explicitly disable the estimator's default random early-stopping split.
            model = HistGradientBoostingRegressor(max_iter=spec["iterations"], learning_rate=0.05,
                max_leaf_nodes=spec["leaves"], min_samples_leaf=300, l2_regularization=20,
                early_stopping=False, random_state=SEED)
        elif kind == "rf":
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=80,
                                      max_samples=0.7, n_jobs=4, random_state=SEED))
        else:
            raise ValueError(kind)
        model.fit(x, y)
        bundle.update(model=model, columns=cols, train_rows=len(y),
                      train_target_max=str(dataset["meta"].loc[mask, "target_time"].max()))
    return bundle

def predict_model(bundle, x, times, base):
    spec = bundle["spec"]; kind = spec["kind"]
    if kind in ["ols", "ridge"]:
        pred = bundle["model"].predict(calendar_features(times, spec["enhanced"], spec["trend"]))
    elif kind == "climatology":
        local = pd.DatetimeIndex(times).tz_convert(LOCAL_TZ)
        pred = bundle["table"][local.dayofyear.to_numpy()-1, local.hour.to_numpy()]
        pred = np.where(np.isfinite(pred), pred, bundle["fallback"])
    elif kind == "persistence":
        pred = np.tile(bundle["last_day"], 14)
    elif kind == "anomaly":
        adjustment = float(x.state_anom72.iloc[0])
        if not np.isfinite(adjustment):
            adjustment = 0.0
        pred = base + adjustment * np.exp(-np.arange(HORIZON)/spec["tau"])
    else:
        pred = base + bundle["model"].predict(x[bundle["columns"]])
    if not np.isfinite(pred).all():
        raise ValueError(f"Nonfinite predictions: {spec['id']}")
    return np.asarray(pred)
