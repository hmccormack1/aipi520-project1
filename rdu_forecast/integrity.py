"""00 Integrity: hashes, state locks, time boundaries and reproducibility records."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from config import ROOT, FREEZE, CUTOFF, TEST

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def now():
    return datetime.now(timezone.utc).isoformat()

def development_allowed():
    if FREEZE.exists() or (TEST / "access_started.json").exists():
        raise RuntimeError("Experiment frozen or test accessed: feature/model selection is disabled. Review existing results; do not remove locks to tune again.")

def assert_before(values, cutoff, name):
    series = pd.DatetimeIndex(values).dropna()
    if len(series) and not bool((series < cutoff).all()):
        raise ValueError(f"LEAKAGE: {name} contains timestamps >= {cutoff}")

def source_hashes():
    files = sorted(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]
    files += sorted((ROOT / "tests").glob("*.py"))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in files}

def versions():
    result = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]:
        result[package] = importlib.metadata.version(package)
    return result

def verify_freeze():
    if not FREEZE.exists():
        raise RuntimeError("Freeze models, predictions and source code before accessing final test data.")
    manifest = read_json(FREEZE)
    for name, digest in manifest["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, digest in manifest["artifact_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Frozen artifact changed: {name}")
    if manifest["cutoff"] != str(CUTOFF):
        raise RuntimeError("Frozen cutoff mismatch")
    return manifest

def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        raise ValueError("No observed labels to evaluate; never substitute synthetic labels.")
    error = pred[ok] - y[ok]
    sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    return {"n": int(ok.sum()), "rmse_f": float(np.sqrt(np.mean(error ** 2))),
            "mae_f": float(np.mean(np.abs(error))), "bias_f": float(error.mean()),
            "r2": float(1 - np.sum(error ** 2) / sst) if sst else None}
