# Railway Waitlist Prediction System

A Streamlit dashboard for exploring a local sample dataset and estimating whether a **waitlisted** ticket may be confirmed. It includes data cleaning, exploratory charts, a Random Forest classifier, a rule-based heuristic, and local PNR lookup.

> **Important limitation:** This project does not connect to Indian Railways or retrieve live PNR status. The included dataset is a small sample dataset and its predictions are for demonstration/academic use only. The model's output is a score, not a guaranteed or independently validated real-world probability. Do not use it to make travel decisions.

## Requirements

- Python 3.10+ (Python 3.13 was used in the development environment)
- pip

## Windows setup (PowerShell)

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the full data and model pipeline

```powershell
python main.py
```

This loads `data/raw/railway_data.csv`, cleans it, saves `data/processed/cleaned_data.csv`, generates six charts in `plots/`, trains and saves a model in `models/`, and runs sample lookups/predictions.

To train the model alone:

```powershell
python -m src.train_model
```

## Start the dashboard

Run the pipeline once, then:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). The PNR Lookup page searches only the included local CSV; it is not a live PNR checker.

## Run tests

```powershell
python -m pytest -q
```

The test suite covers data loading/cleaning, PNR string preservation, prediction input validation, duplicate PNR handling, model training/prediction, and plot generation.

## Project structure

```text
railway_wl_prediction/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/railway_data.csv
│   └── processed/cleaned_data.csv
├── src/
│   ├── analyze.py
│   ├── clean_data.py
│   ├── config.py
│   ├── load_data.py
│   ├── ml_model.py
│   ├── pnr_lookup.py
│   ├── predict.py
│   └── train_model.py
├── assets/styles.css
├── models/
├── plots/
└── tests/test_project.py
```

## Key implementation notes

- PNRs are treated as string identifiers to preserve leading zeroes.
- Duplicate PNR matches raise a clear error rather than returning an arbitrary row.
- The model is trained only on records with `waitlist_position > 0`.
- The rule-based score is explicitly a heuristic, not a calibrated probability.
- Tickets with waitlist position `0` are not scored by the waitlist-only predictors.
- `requirements.txt` includes Streamlit and Plotly, which the dashboard imports.
