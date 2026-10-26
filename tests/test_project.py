from pathlib import Path

import pandas as pd
import pytest

from src.clean_data import clean_data
from src.config import RAW_CSV, CLEAN_CSV, PLOTS_DIR
from src.load_data import load_csv
from src.ml_model import build_pipeline, predict_ml, train_model
from src.pnr_lookup import lookup_pnr
from src.predict import classify_risk, predict_confirmation


@pytest.fixture(scope="module")
def raw_data():
    return load_csv(RAW_CSV)


@pytest.fixture(scope="module")
def cleaned_data(raw_data):
    return clean_data(raw_data, verbose=False)


def test_raw_dataset_loads_and_has_required_columns(raw_data):
    assert not raw_data.empty
    assert "pnr" in raw_data.columns
    assert raw_data["pnr"].notna().all()


def test_cleaning_removes_cancelled_and_creates_target(cleaned_data):
    assert not cleaned_data.empty
    assert set(cleaned_data["final_status"].unique()) <= {"CNF", "RAC", "WL"}
    assert set(cleaned_data["confirmed"].unique()) <= {0, 1}
    assert (cleaned_data["confirmed"] == (cleaned_data["final_status"] == "CNF").astype(int)).all()
    assert cleaned_data["pnr"].is_unique


def test_pnr_is_preserved_as_string_with_leading_zero():
    tiny = pd.DataFrame({
        "pnr": ["0000000123"], "train_number": [10001],
        "train_name": ["Test Express"], "from_station": ["AAA"],
        "to_station": ["BBB"], "journey_date": ["01/10/2026"],
        "class": ["3A"], "quota": ["GNWL"], "waitlist_position": [4],
        "booking_date": ["01/01/2026"], "days_before_journey": [9],
        "confirmation_status": ["CNF"], "final_status": ["CNF"],
        "passenger_count": [1], "is_tatkal": [0], "is_peak_season": [0],
        "is_holiday": [0], "train_type": ["EXPRESS"], "distance_km": [100],
    })
    result = clean_data(tiny, verbose=False)
    assert result.loc[0, "pnr"] == "0000000123"


def test_rule_prediction_boundaries_and_invalid_values():
    score = predict_confirmation(5, 14, "3A")
    assert 0 <= score <= 0.95
    assert classify_risk(score)
    with pytest.raises(ValueError):
        predict_confirmation(0, 10, "3A")
    with pytest.raises(ValueError):
        predict_confirmation(-1, 10, "3A")
    with pytest.raises(ValueError):
        predict_confirmation("not-a-number", 10, "3A")
    with pytest.raises(ValueError):
        classify_risk(float("nan"))


def test_lookup_known_unknown_and_zero_position(cleaned_data):
    known = cleaned_data.iloc[0]
    result = lookup_pnr(str(known["pnr"]), cleaned_data)
    assert result is not None
    assert result["pnr"] == str(known["pnr"])
    assert result["actual_status"] in {"CNF", "RAC", "WL"}
    assert lookup_pnr("DOES-NOT-EXIST", cleaned_data) is None

    zero = cleaned_data.loc[cleaned_data["waitlist_position"] == 0].iloc[0]
    zero_result = lookup_pnr(str(zero["pnr"]), cleaned_data)
    assert zero_result["predicted_prob"] is None
    assert "not waitlisted" in zero_result["risk_label"].lower()


def test_duplicate_pnr_lookup_is_rejected(cleaned_data):
    duplicate = pd.concat([cleaned_data.iloc[[0]], cleaned_data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="appears 2 times"):
        lookup_pnr(str(duplicate.iloc[0]["pnr"]), duplicate)


def test_ml_training_and_prediction(cleaned_data):
    model, metrics = train_model(cleaned_data, save=False, waitlisted_only=True)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["cv_roc_auc_mean"] <= 1
    probability = predict_ml(model, 10, 14, "3A", "SUPERFAST", "GNWL", 1000, 1, 0, 0, 0)
    assert 0 <= probability <= 1
    with pytest.raises(ValueError):
        predict_ml(model, 0, 14)
    with pytest.raises(ValueError):
        predict_ml(model, -1, 14)
    with pytest.raises(ValueError):
        predict_ml(model, 10, 14, is_tatkal=2)


def test_plot_generation_runs(cleaned_data, tmp_path):
    from src.analyze import analyze
    analyze(cleaned_data, save_dir=tmp_path)
    images = list(Path(tmp_path).glob("*.png"))
    assert len(images) == 6


def test_expected_project_paths_exist():
    assert RAW_CSV.is_file()
    assert CLEAN_CSV.parent.is_dir()
    assert PLOTS_DIR.is_dir()
