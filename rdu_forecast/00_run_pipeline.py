"""00 Pipeline entry: develop by default; unlock final evaluation only after freezing."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import subprocess
import sys
from config import ROOT, FREEZE, ART, make_dirs
from integrity import verify_freeze, now

def execute(script):
    print(f"\n{'='*12} {script} {'='*12}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-eval", action="store_true", help="Freeze predictions, THEN download and evaluate unseen final labels.")
    args = parser.parse_args()
    make_dirs()
    if FREEZE.exists():
        verify_freeze()
        print("Existing frozen experiment verified. Development/refitting is disabled.", flush=True)
    else:
        # Always run leakage tests before training or accessing the network.
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                cwd=ROOT, text=True, capture_output=True)
        (ROOT / "logs" / "leakage_tests.log").write_text(now()+"\n"+result.stdout+result.stderr, encoding="utf-8")
        print(result.stdout+result.stderr, flush=True)
        result.check_returncode()
        for script in ["step01_download.py", "step02_clean.py", "step03_features.py", "step05_backtest_select.py"]:
            execute(script)
        if args.final_eval:
            execute("step06_freeze_forecast.py")
    if args.final_eval:
        execute("step07_final_evaluate.py")
    execute("step08_visualize.py")
    print("Results:", ART / "results.html", flush=True)

if __name__ == "__main__":
    run()
