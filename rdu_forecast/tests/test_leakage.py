"""Leakage regression tests: synthetic data only, never real final test labels."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CUTOFF, STATIONS, VARIABLES, HORIZON
from integrity import assert_before, verify_freeze, development_allowed
from step01_download import fetch
from step02_clean import clean_reports
from step03_features import training_mask, context_at, calendar_features
from step04_models import fit_model, candidates

class ConstantReference:
    def predict(self, x):
        return np.full(len(x), 60.0)

class LeakageTests(unittest.TestCase):
    def test_cutoff_is_strict(self):
        with self.assertRaises(ValueError):
            assert_before([CUTOFF], CUTOFF, "test")

    def test_download_rejects_test_before_network(self):
        with patch("urllib.request.urlopen") as network:
            with self.assertRaises(ValueError):
                fetch(CUTOFF-pd.Timedelta(days=1), CUTOFF+pd.Timedelta(hours=1), Path("unused.csv"), ["RDU"], "development")
            network.assert_not_called()

    def test_no_test_without_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("integrity.FREEZE", Path(folder)/"missing.json"):
                with self.assertRaises(RuntimeError):
                    verify_freeze()

    def test_development_blocked_after_test_access(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p/"access_started.json").write_text("{}")
            with patch("integrity.FREEZE", p/"no_freeze"), patch("integrity.TEST", p):
                with self.assertRaises(RuntimeError):
                    development_allowed()

    def test_hour_mapping_and_no_label_imputation(self):
        raw = pd.DataFrame({"station": ["RDU"]*3,
             "valid": ["2026-09-16 21:51Z", "2026-09-16 22:51Z", "2026-09-16 23:51Z"],
             "tmpf": [60, np.nan, 999], "dwpf": [40]*3, "mslp": [1010]*3, "sknt": [5]*3, "drct": [90]*3})
        result = clean_reports(raw, CUTOFF-pd.Timedelta(days=1), pd.Timestamp("2026-09-17 00:00Z"))
        self.assertEqual(len(result), 2)
        self.assertEqual(result.time.iloc[0], pd.Timestamp("2026-09-16 22:00Z"))
        self.assertTrue(pd.isna(result.tmpf.iloc[1]))

    def test_purge_entire_future_label_window(self):
        boundary = pd.Timestamp("2024-09-17", tz="UTC")
        meta = pd.DataFrame({"origin": [boundary-pd.Timedelta(days=14), boundary-pd.Timedelta(days=13)],
                             "target_time": [boundary-pd.Timedelta(hours=1)]*2})
        self.assertEqual(training_mask(meta, boundary).tolist(), [True, False])

    def test_future_observation_poison_cannot_change_context(self):
        issue = pd.Timestamp("2024-09-17", tz="UTC")
        times = pd.date_range(issue-pd.Timedelta(hours=168), periods=200, freq="h")
        frame = pd.DataFrame({"valid": times-pd.Timedelta(minutes=9), "tmpf": np.arange(200)/10+50,
            "dwpf": 40., "mslp": 1010., "sknt": 5., "drct": 90.}, index=times)
        frames = {s: frame.copy() for s in STATIONS}
        refs = {s: ConstantReference() for s in STATIONS}
        a, latest_a = context_at(frames, refs, issue)
        for s in STATIONS:
            frames[s].loc[frames[s].index >= issue, VARIABLES] = -99999
        b, latest_b = context_at(frames, refs, issue)
        np.testing.assert_allclose(list(a.values()), list(b.values()), equal_nan=True)
        self.assertEqual(latest_a, latest_b)
        self.assertLess(latest_a, issue)

    def test_future_training_labels_and_features_cannot_change_fit(self):
        boundary = pd.Timestamp("2024-09-17", tz="UTC")
        rng = np.random.default_rng(520)
        n = 700
        meta = pd.DataFrame({"origin": [boundary-pd.Timedelta(days=30)]*600+[boundary]*100,
            "target_time": [boundary-pd.Timedelta(days=20)]*600+[boundary+pd.Timedelta(days=1)]*100})
        x = pd.DataFrame({"state_anom24": rng.normal(size=n), "lead_hours": rng.integers(1,337,size=n)})
        data = {"X": x, "y": 60+2*x.state_anom24.to_numpy(), "base": np.full(n,60.), "meta":meta}
        history = pd.DataFrame({"tmpf": [60.,61.]}, index=pd.DatetimeIndex([boundary-pd.Timedelta(days=2),boundary-pd.Timedelta(days=1)]))
        spec = next(s for s in candidates() if s["id"] == "direct_ridge")
        a = fit_model(spec, history, data, boundary)
        data["y"][600:] = 1e9
        data["X"].iloc[600:] = -1e9
        b = fit_model(spec, history, data, boundary)
        before = a["model"].predict(x.iloc[:30])
        after = b["model"].predict(x.iloc[:30])
        np.testing.assert_allclose(before, after)
        np.testing.assert_allclose(a["model"][0].statistics_, b["model"][0].statistics_)
        np.testing.assert_allclose(a["model"][1].mean_, b["model"][1].mean_)

    def test_calendar_features_need_no_labels(self):
        x = calendar_features(pd.date_range(CUTOFF, periods=336, freq="h"))
        self.assertEqual(x.shape, (336,12))
        self.assertTrue(np.isfinite(x).all().all())

if __name__ == "__main__":
    unittest.main()
