
from pathlib import Path

import pandas as pd

# Support package and direct script execution
try:
    from .config import RAW_CSV
except ImportError:
    from config import RAW_CSV


# Columns expected in the raw railway dataset
REQUIRED_COLUMNS = [
    "pnr",
    "train_number",
    "train_name",
    "from_station",
    "to_station",
    "journey_date",
    "class",
    "quota",
    "waitlist_position",
    "booking_date",
    "days_before_journey",
    "final_status",
    "passenger_count",
    "is_tatkal",
    "is_peak_season",
    "is_holiday",
    "train_type",
    "distance_km",
]


def load_csv(file_path=RAW_CSV):
    """Load the railway CSV and validate its required columns."""

    # Convert the supplied path to a Path object
    file_path = Path(file_path)

    # Check whether the file exists
    if not file_path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {file_path.resolve()}"
        )

    # Read the CSV
    try:
        df = pd.read_csv(file_path, dtype={"pnr": "string"})
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The CSV file is empty: {file_path}"
        ) from exc

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    # Check required columns
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    # Check whether the CSV contains any data rows
    if df.empty:
        raise ValueError(
            f"The CSV contains column headers but no data rows: {file_path}"
        )

    # Display information about the loaded dataset
    print(
        f"Loaded {df.shape[0]} rows x "
        f"{df.shape[1]} columns"
    )

    print("Columns:", list(df.columns))

    return df


if __name__ == "__main__":
    data = load_csv()

    print("\nFirst five rows:")
    print(data.head())
