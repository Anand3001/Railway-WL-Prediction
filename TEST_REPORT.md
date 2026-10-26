# Railway Waitlist Prediction — Test and Fix Report

Date: 2026-10-23

## Summary

- Python syntax compilation: **PASS** (`main.py`, `app.py`, `src/`, `tests/`).
- Automated tests: **PASS — 9 passed**.
- Full pipeline (`python main.py`): **PASS**.
- Standalone training entry point (`python src/train_model.py`): **PASS**.
- Dashboard code-path smoke test: **PASS for PNR Lookup, Manual Prediction, and Analytics using a lightweight Streamlit mock**.
- Actual Streamlit server/browser run: **NOT VERIFIED in this environment**. Streamlit is not installed in the isolated runtime, and package installation failed because external network/DNS access is unavailable. `streamlit>=1.36` is now included in `requirements.txt` for local installation.

## Data checks

- Raw rows loaded: 1,110
- Rows after cleaning: 1,086
- Rows removed due to unsupported/cancelled status: 24
- Cleaned outcomes: CNF 787, RAC 123, WL 176
- Waitlisted records (position > 0): 745
- Duplicate PNRs after cleaning: 0
- Journey-date / booking-date lead-time mismatches after cleaning: 0

These counts describe the included dataset only. They do not establish that it is official or representative Indian Railways data.

## Model evaluation from the included dataset

- Held-out accuracy: 73.83%
- Held-out ROC-AUC: 0.766
- Five-fold CV ROC-AUC: 0.725 ± 0.037
- Rule-based ROC-AUC: 0.673

These are sample-dataset evaluation results only. The dataset is relatively small, the source/collection methodology is not documented sufficiently to validate real-world representativeness, and no independent real-world evaluation was available. Do not describe these results as proven real-world performance.

## Fixes applied

1. Added Streamlit and Plotly to `requirements.txt`.
2. Preserved PNR values as strings during CSV loading and cleaning, so leading zeroes are not dropped.
3. Added duplicate-PNR protection to avoid silently returning the first of multiple matching rows.
4. Added numeric/range validation to the ML prediction function and rule-based estimator.
5. Prevented waitlist-only predictors from assigning a score to position `0`; lookup still returns the ticket's stored details and clearly marks the prediction as not applicable.
6. Escaped dataset-provided values before inserting them into dashboard HTML; only internally generated status-pill HTML is allowed through.
7. Made `python src/train_model.py` work as well as `python -m src.train_model`.
8. Changed dashboard labels to say “score” rather than imply a calibrated probability.
9. Added reproducible tests and Windows setup/run instructions.
10. Excluded the bundled virtual environment from the fixed ZIP; virtual environments are machine- and Python-version-specific and should be recreated locally.

## Automated test coverage

The test suite checks:

- raw CSV loading and required PNR data;
- cleaning and target consistency;
- PNR preservation with leading zeroes;
- rule-based input validation and risk labels;
- known and unknown PNR lookup;
- position-zero lookup behavior;
- duplicate PNR handling;
- ML training, prediction, and invalid input rejection;
- generation of all six EDA plots;
- expected project paths.

## Remaining limitations

- No live Indian Railways API integration exists; PNR lookup searches the local CSV only.
- The model was evaluated on the included dataset, not independently verified real railway records.
- Actual Streamlit browser interaction could not be tested here due to the unavailable dependency/network; install requirements and launch locally to complete that final check.
- A passing test suite cannot guarantee absence of every possible bug, but the listed automated checks and pipeline runs passed.
