
import pandas as pd

# Support package imports and direct script execution
try:
    from .ml_model import predict_ml
    from .predict import classify_risk, predict_confirmation
except ImportError:
    from ml_model import predict_ml
    from predict import classify_risk, predict_confirmation


def lookup_pnr(pnr, df, model=None):
    """
    Find a ticket by PNR and return its details and predictions.

    If a trained ML model is supplied, include its prediction too.
    Return None if the PNR is not found.
    """

    # Validate the dataset
    required_columns = [
        "pnr",
        "train_number",
        "train_name",
        "from_station",
        "to_station",
        "journey_date",
        "class",
        "quota",
        "waitlist_position",
        "days_before_journey",
        "final_status",
        "train_type",
        "distance_km",
        "passenger_count",
        "is_tatkal",
        "is_peak_season",
        "is_holiday",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Normalize the input PNR
    pnr = str(pnr).strip()

    if not pnr:
        raise ValueError("Please provide a valid PNR.")

    # Compare as strings to support numeric and string PNR columns
    rows = df[df["pnr"].astype(str).str.strip() == pnr]

    if rows.empty:
        return None

    # A PNR should identify one record. Avoid silently returning an arbitrary
    # row if the source dataset contains duplicate identifiers.
    if len(rows) > 1:
        raise ValueError(
            f"PNR {pnr} appears {len(rows)} times in the dataset; "
            "remove duplicate PNR records before lookup."
        )

    row = rows.iloc[0]

    # Extract basic ticket information
    wl = int(row["waitlist_position"])
    days = int(row["days_before_journey"])

    # Predictions are defined only for genuinely waitlisted tickets.
    # A zero position is a different ticket state, not WL position zero.
    prob = None
    if wl > 0:
        prob = predict_confirmation(wl, days, row["class"])

    # Prepare ticket details
    journey_date = pd.to_datetime(
        row["journey_date"],
        errors="coerce",
    )

    result = {
        "pnr": pnr,
        "train_number": int(row["train_number"]),
        "train_name": row["train_name"],
        "from": row["from_station"],
        "to": row["to_station"],
        "journey_date": (
            journey_date.strftime("%Y-%m-%d")
            if pd.notna(journey_date)
            else None
        ),
        "class": row["class"],
        "quota": row["quota"],
        "waitlist_position": wl,
        "days_before": days,
        "actual_status": row["final_status"],
        "predicted_prob": float(prob) if prob is not None else None,
        "risk_label": classify_risk(prob) if prob is not None else "Not applicable: ticket is not waitlisted",
    }

    # The saved ML model is trained on waitlisted tickets only.
    if model is not None and wl > 0:
        ml_probability = predict_ml(
            model,
            wl,
            days,
            travel_class=row["class"],
            train_type=row["train_type"],
            quota=row["quota"],
            distance_km=int(row["distance_km"]),
            passenger_count=int(row["passenger_count"]),
            is_tatkal=int(row["is_tatkal"]),
            is_peak_season=int(row["is_peak_season"]),
            is_holiday=int(row["is_holiday"]),
        )

        result["ml_prob"] = float(ml_probability)

    return result


# Example usage
if __name__ == "__main__":

    try:
        from .config import CLEAN_CSV
        from .ml_model import load_model
    except ImportError:
        from config import CLEAN_CSV
        from ml_model import load_model

    # Load cleaned railway data
    df = pd.read_csv(CLEAN_CSV)

    # Load the trained model if available
    try:
        model = load_model()
        print("ML model loaded successfully.")
    except FileNotFoundError:
        model = None
        print("ML model not found. Using rule-based prediction only.")

    # Enter a PNR to search
    pnr_input = input("Enter PNR number: ").strip()

    result = lookup_pnr(
        pnr_input,
        df,
        model=model,
    )

    if result is None:
        print("PNR not found in the dataset.")
    else:
        print("\nPNR Lookup Result")
        print("=" * 40)

        for key, value in result.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
