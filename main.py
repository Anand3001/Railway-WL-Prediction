from src.analyze import analyze
from src.clean_data import clean_data
from src.config import CLEAN_CSV, PLOTS_DIR, RAW_CSV, setup_console
from src.load_data import load_csv
from src.ml_model import predict_ml, train_model
from src.pnr_lookup import lookup_pnr
from src.predict import predict_confirmation


def main():
    setup_console()
    print("=" * 60)
    print("🚂 RAILWAY WAITING LIST PREDICTION — FULL PIPELINE")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading data...")
    df = load_csv(RAW_CSV)

    # 2. Clean
    print("\n[2/6] Cleaning data...")
    df = clean_data(df)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"✅ Saved cleaned data to {CLEAN_CSV}")

    # 3. Analyze
    print("\n[3/6] Generating plots...")
    analyze(df, save_dir=PLOTS_DIR)

    # 4. Train ML
    print("\n[4/6] Training ML model...")
    model, _metrics = train_model(df, save=True)

    # 5. Sample predictions
    print("\n[5/6] Sample Predictions (Rule vs ML)")
    print("-" * 60)
    for wl, days, cls in [(5, 14, "3A"), (45, 14, "3A"), (156, 10, "SL")]:
        rp = predict_confirmation(wl, days, cls)
        mp = predict_ml(model, wl, days, cls)
        print(f"WL {wl:<4} | {days}d | {cls} | "
              f"Rule: {rp*100:>3.0f}%  ML: {mp*100:>3.0f}%")

    # 6. PNR sample
    print("\n[6/6] 🎫 PNR Sample:")
    wanted = "1234567890"
    res = lookup_pnr(wanted, df, model)
    if res is None:  # that PNR was removed in cleaning, so show the first one
        res = lookup_pnr(df["pnr"].iloc[0], df, model)
    print(f"  PNR {res['pnr']}: {res['train_name']}")
    rule_text = (
        f"{res['predicted_prob'] * 100:.0f}%"
        if res.get("predicted_prob") is not None else "N/A (not waitlisted)"
    )
    ml_text = (
        f"{res['ml_prob'] * 100:.0f}%"
        if res.get("ml_prob") is not None else "N/A (not waitlisted)"
    )
    print(f"  Actual: {res['actual_status']} | Rule score: {rule_text} | ML score: {ml_text}")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()