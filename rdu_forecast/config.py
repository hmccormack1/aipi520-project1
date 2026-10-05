"""00 Configuration: fix all experiment rules before accessing final test labels."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOCAL_TZ = "America/New_York"
CUTOFF = pd.Timestamp("2026-09-17 04:00", tz="UTC")
END = CUTOFF + pd.Timedelta(hours=336)
START = pd.Timestamp("2015-01-01", tz="UTC")
HORIZON = 336
STATIONS = ["RDU", "GSO", "CLT"]
VARIABLES = ["tmpf", "dwpf", "mslp", "sknt", "drct"]
SEED = 520
# Routine reports near :51 are mapped to the FOLLOWING hour, never hourly averaged.
REPORT_MINUTE_MIN = 45
AVAILABILITY_BUFFER_MINUTES = 15
MIN_TARGET_COVERAGE = 0.95
# Every model is evaluated on exactly these development windows.
FOLD_DATES = ["2021-09-17", "2022-09-17", "2023-09-17",
              "2024-08-15", "2024-09-17", "2025-08-15",
              "2025-09-17", "2026-08-15"]
DATA = ROOT / "data"
DEV = DATA / "development"
TEST = DATA / "test_locked"
ART = ROOT / "artifacts"
FIG = ART / "figures"
FREEZE = ART / "freeze_manifest.json"

def origin(date):
    return pd.Timestamp(date, tz=LOCAL_TZ).tz_convert("UTC")

def make_dirs():
    for path in [DEV / "raw", ART, FIG, ROOT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
