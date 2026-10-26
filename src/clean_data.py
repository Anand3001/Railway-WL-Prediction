
from pathlib import Path

import pandas as pd

# Support both package execution and direct script execution
try:
    from .config import RAW_CSV, CLEAN_CSV
    from .load_data import load_csv
except ImportError:
    from config import RAW_CSV, CLEAN_CSV
    from load_data import load_csv


# Text columns
TEXT_COLS = [
    "class",
    "quota",
    "final_status",
    "confirmation_status",
    "train_type"
]

# Columns that should contain numeric values
NUMERIC_COLS = [
    "train_number",
    "waitlist_position",
    "passenger_count",
    "distance_km",
    "days_before_journey",
    "is_tatkal",
    "is_peak_season",
    "is_holiday"
]

# Binary flag columns
FLAG_COLS = [
    "is_tatkal",
    "is_peak_season",
    "is_holiday"
]

# Standardize train type names
TRAIN_TYPE_ALIASES = {
    "MAIL": "MAIL/EXPRESS",
    "EXPRESS": "MAIL/EXPRESS"
}

# Map ticket statuses to numeric codes
STATUS_MAP = {
    "CNF": 2,
    "RAC": 1,
    "WL": 0
}


def normalize_train_type(value):
    """Convert train type to uppercase and standardize aliases."""

    value = str(value).strip().upper()

    return TRAIN_TYPE_ALIASES.get(value, value)


def _parse_dates(series):
    """Parse M/D/YYYY dates and handle other common date formats."""

    # First, try the expected month/day/year format
    parsed = pd.to_datetime(
        series,
        format="%m/%d/%Y",
        errors="coerce"
    )

    # Retry values that could not be parsed
    bad = parsed.isna() & series.notna()

    if bad.any():
        parsed.loc[bad] = pd.to_datetime(
            series.loc[bad],
            errors="coerce"
        )

    return parsed


def clean_data(df, verbose=True):
    """Clean railway waiting-list data."""

    df = df.copy()
    start = len(df)

    # --------------------------------------------------
    # 0. Validate required columns
    # --------------------------------------------------
    required_cols = [
        "pnr",
        "train_number",
        "journey_date",
        "booking_date",
        "days_before_journey",
        "waitlist_position",
        "passenger_count",
        "distance_km",
        "class",
        "quota",
        "final_status",
        "confirmation_status",
        "train_type",
        "is_tatkal",
        "is_peak_season",
        "is_holiday"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_cols)
        )

    # --------------------------------------------------
    # 1. Convert numeric columns
    # --------------------------------------------------
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # 2. Standardize text columns
    # --------------------------------------------------
    for col in TEXT_COLS:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    # Keep PNR identifiers as strings; they are identifiers, not numbers.
    df["pnr"] = df["pnr"].astype("string").str.strip()

    # Standardize train type aliases
    df["train_type"] = df["train_type"].map(
        lambda value: (
            value
            if pd.isna(value)
            else normalize_train_type(value)
        )
    )

    # --------------------------------------------------
    # 3. Parse dates and recalculate journey lead time
    # --------------------------------------------------
    df["journey_date"] = _parse_dates(
        df["journey_date"]
    )

    df["booking_date"] = _parse_dates(
        df["booking_date"]
    )

    # Calculate the correct number of days between dates
    recalculated = (
        df["journey_date"] - df["booking_date"]
    ).dt.days

    # Count rows where the old value differs
    changed = (
        recalculated.notna()
        & df["days_before_journey"].notna()
        & (recalculated != df["days_before_journey"])
    )

    if verbose:
        print(
            "Corrected days_before_journey on "
            f"{int(changed.sum())} rows"
        )

    # Replace old values with recalculated values
    df["days_before_journey"] = recalculated

    # --------------------------------------------------
    # 4. Remove rows with missing critical data
    # --------------------------------------------------
    critical_cols = [
        "pnr",
        "waitlist_position",
        "final_status",
        "days_before_journey",
        "class",
        "quota",
        "train_type",
        "distance_km",
        "passenger_count"
    ]

    before = len(df)

    df = df.dropna(
        subset=critical_cols
    ).copy()

    if verbose:
        print(
            "Dropped rows with missing critical values: "
            f"{before - len(df)}"
        )

    # --------------------------------------------------
    # 5. Remove duplicate PNRs
    # --------------------------------------------------
    before = len(df)

    # Keep the first valid record per PNR after normalization.
    df = df.drop_duplicates(
        subset=["pnr"]
    ).copy()

    if verbose:
        print(
            "Removed duplicate PNRs: "
            f"{before - len(df)}"
        )

    # --------------------------------------------------
    # 6. Remove impossible values
    # --------------------------------------------------
    before = len(df)

    df = df[
        (df["days_before_journey"] >= 0)
        & (df["waitlist_position"] >= 0)
        & (df["passenger_count"] >= 1)
        & (df["distance_km"] >= 0)
    ].copy()

    if verbose:
        print(
            "Removed rows with impossible values: "
            f"{before - len(df)}"
        )

    # --------------------------------------------------
    # 7. Keep only known ticket outcomes
    # --------------------------------------------------
    before = len(df)

    invalid_outcomes = ~df["final_status"].isin(
        STATUS_MAP
    )

    outcome_counts = (
        df.loc[invalid_outcomes, "final_status"]
        .value_counts()
        .to_dict()
    )

    df = df[
        df["final_status"].isin(STATUS_MAP)
    ].copy()

    if verbose:
        removed = before - len(df)

        if removed:
            print(
                "Removed rows without a CNF/RAC/WL outcome: "
                f"{removed}"
            )
            print(f"Unknown outcomes: {outcome_counts}")

    # --------------------------------------------------
    # 8. Create target and helper columns
    # --------------------------------------------------
    df["status_code"] = (
        df["final_status"]
        .map(STATUS_MAP)
        .astype(int)
    )

    df["confirmed"] = (
        df["final_status"] == "CNF"
    ).astype(int)

    df["is_waitlisted"] = (
        df["waitlist_position"] > 0
    ).astype(int)

    # --------------------------------------------------
    # 9. Clean flags and integer columns
    # --------------------------------------------------
    # Missing flag values are treated as zero
    for col in FLAG_COLS:
        df[col] = (
            df[col]
            .fillna(0)
            .astype(int)
        )

    integer_cols = [
        "train_number",
        "waitlist_position",
        "passenger_count",
        "distance_km",
        "days_before_journey"
    ]

    for col in integer_cols:
        df[col] = df[col].astype(int)

    # Convert text columns to regular Python-compatible
    # object columns for downstream analysis
    for col in TEXT_COLS:
        df[col] = df[col].astype(object)

    # Reset row numbering after removing records
    df = df.reset_index(drop=True)

    # --------------------------------------------------
    # 10. Print summary
    # --------------------------------------------------
    if verbose:
        print(f"\nCleaned: {len(df)} of {start} rows kept")

        print("\nStatus distribution:")
        print(
            df["final_status"]
            .value_counts()
            .reindex(["CNF", "RAC", "WL"], fill_value=0)
        )

        never_waitlisted = (
            df["waitlist_position"] == 0
        ).sum()

        print(
            f"\nTickets with waitlist position 0: "
            f"{never_waitlisted}"
        )

    return df


# --------------------------------------------------
# Run this file directly
# --------------------------------------------------
if __name__ == "__main__":

    # Load the raw railway dataset
    raw_data = load_csv(RAW_CSV)

    # Clean the dataset
    cleaned = clean_data(raw_data)

    # Create the output directory if needed
    output_path = Path(CLEAN_CSV)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the cleaned dataset
    cleaned.to_csv(
        output_path,
        index=False
    )

    print(f"\nCleaned dataset saved to: {output_path.resolve()}")
