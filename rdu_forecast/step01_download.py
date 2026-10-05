"""01 Acquisition: development requests only; step 07 controls final test access."""
import io
import time
import urllib.parse
import urllib.request
import pandas as pd
from config import *
from integrity import development_allowed, write_json, sha256, now, verify_freeze

API = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

def fetch(start, end, target, stations, purpose):
    # 01.1 Independent request guard: a programming error cannot fetch test early.
    if purpose == "development":
        if end > CUTOFF or start >= CUTOFF:
            raise ValueError("Development request intersects the test period")
    elif purpose == "final_test":
        verify_freeze()
        if start != CUTOFF - pd.Timedelta(minutes=20) or end != END:
            raise ValueError("Unexpected final-test request bounds")
    else:
        raise ValueError("Unknown data access purpose")
    if target.exists():
        manifest = target.with_suffix(".json")
        from integrity import read_json
        if not manifest.exists() or read_json(manifest)["sha256"] != sha256(target):
            raise RuntimeError(f"Cache integrity failure: {target}")
        return
    params = [("station", s) for s in stations]
    params += [("data", v) for v in VARIABLES]
    params += [("sts", start.isoformat()), ("ets", end.isoformat()), ("tz", "UTC"),
               ("format", "onlycomma"), ("missing", "M"), ("report_type", "3"),
               ("latlon", "no"), ("elev", "no")]
    url = API + "?" + urllib.parse.urlencode(params)
    error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AIPI520-course-project/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            frame = pd.read_csv(io.BytesIO(payload), comment="#", na_values=["M", "null"])
            if not {"station", "valid", "tmpf"}.issubset(frame.columns) or frame.empty:
                raise RuntimeError("API did not return non-empty observation CSV")
            valid = pd.to_datetime(frame["valid"], utc=True)
            if valid.min() < start or valid.max() >= end:
                raise RuntimeError("API returned out-of-range observations")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            write_json(target.with_suffix(".json"), {
                "purpose": purpose, "url": url, "requested_start": start, "requested_end_exclusive": end,
                "downloaded_at_utc": now(), "rows": len(frame), "observed_min": valid.min(),
                "observed_max": valid.max(), "sha256": sha256(target),
                "note": "Retrospective routine METAR archive; original publication/revision timestamps unavailable."})
            print(f"Downloaded {target.name}: {len(frame):,} reports", flush=True)
            time.sleep(1.1)  # Respect IEM's one request per second limit.
            return
        except Exception as exc:
            error = exc
            print(f"Download attempt {attempt+1}/4 failed: {exc}", flush=True)
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Public API download failed. Need user help with network or authentic CSV. {error}")

def run():
    development_allowed()
    make_dirs()
    for year in range(START.year, CUTOFF.year + 1):
        start = max(START, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        end = min(CUTOFF, pd.Timestamp(f"{year+1}-01-01", tz="UTC"))
        fetch(start, end, DEV / "raw" / f"iem_{year}.csv", STATIONS, "development")

if __name__ == "__main__":
    run()
