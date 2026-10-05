# Repository handoff

The feature branch is `codex/rdu-station-forecast`. It adds only the `rdu_forecast/` directory. It leaves the four original tracked root files unchanged.

The checked upstream base is `034b83a8e748c836a6ab4c7c79c0534156f11899` (`main`). The merge check applies to that fetched state, not arbitrary future edits. The new pipeline and the original pipeline run independently; their data sources and resulting scores are different.

## Review before sharing

From the repository root:

```powershell
git status
git diff main...codex/rdu-station-forecast --stat
cd rdu_forecast
python tools/09_verify_delivery.py
```

The verifier needs only the standard library. For full execution, install the dependencies listed in the README.

## Publish this branch when ready

The branch is committed locally; it has not been pushed automatically. From the repository root, with permission to push:

```powershell
git push -u origin codex/rdu-station-forecast
```

Then create a pull request from this branch into `main`. The directory addition does not replace the teammate's root README, source script or forecast CSV.

## Transfer without GitHub write access

The separately supplied `rdu_forecast.bundle` contains the feature-branch history after the checked base. In another clone of the teammate repository that contains that base:

```powershell
git bundle verify PATH_TO_BUNDLE/rdu_forecast.bundle
git fetch PATH_TO_BUNDLE/rdu_forecast.bundle codex/rdu-station-forecast:codex/rdu-station-forecast
git switch main
git merge codex/rdu-station-forecast
```

If working from the same machine, `git fetch` can use the local `teammate-review` repository path instead of the bundle path. Any uncommitted teammate work should be preserved before merging. Never force-push or reset their branch for this integration.

## Data and outputs

- Git includes English source, pinned dependencies, original reference results, figures and an audit verifier.
- Git excludes virtual environments, large development data and runtime caches.
- The portable ZIP additionally includes the exact downloaded data, fitted models and working frozen state. Install dependencies after extracting, then run `00_run_pipeline.py --final-eval` to verify/replot without training.
- `reference_results/` remains original evidence. Do not copy it over runtime directories and then delete locks to tune against the final test.
- Folder-scoped `.gitattributes` preserves exact bytes across operating systems, preventing CRLF conversion from breaking audited hashes.

## Verification completed

- 19 configurations evaluated on eight historical 14-day windows.
- 336 final forecasts generated and frozen before final-label access.
- Nine synthetic leakage regression tests passed.
- Installed dependencies passed `pip check`.
- English source computational AST matches the source used for the original experiment, except docstrings and two explicitly mapped exception-message translations.
- Final metric recomputation and all sealed evaluation hashes passed.
- The delivered frozen pipeline completed without fitting or selecting any model.
- Eight plots were generated; final forecast, daily error and decomposition figures were visually reviewed.

The primary model remains `anomaly_tau72`. The lower test RMSE of the predefined nonlinear comparator is reported without retrospective reselection.
