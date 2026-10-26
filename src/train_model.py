
"""Train and save the model on its own (main.py also does this)."""

try:
    from .clean_data import clean_data
    from .config import RAW_CSV, setup_console
    from .load_data import load_csv
    from .ml_model import train_model
except ImportError:  # Support `python src/train_model.py` from the project root.
    from clean_data import clean_data
    from config import RAW_CSV, setup_console
    from load_data import load_csv
    from ml_model import train_model


def main():
    setup_console()

    # Load the original railway dataset
    print("📂 Loading raw railway data...")
    raw_df = load_csv(RAW_CSV)

    # Clean the data before training
    # The model requires the 'confirmed' target column
    # and properly formatted categorical values.
    print("\n🧹 Cleaning railway data...")
    df = clean_data(raw_df)

    print(f"\n📥 {len(df)} cleaned rows ready for training")

    if df.empty:
        raise ValueError(
            "No cleaned data available. Check your raw CSV file and cleaning rules."
        )

    # Train and save the machine learning model
    print("\n🤖 Training the model...")
    train_model(df, save=True)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
