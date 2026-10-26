
from pathlib import Path
import math

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Support package imports and direct script execution
try:
    from .clean_data import normalize_train_type
    from .config import MODEL_DIR, CLEAN_CSV
    from .predict import predict_confirmation
except ImportError:
    from clean_data import normalize_train_type
    from config import MODEL_DIR, CLEAN_CSV
    from predict import predict_confirmation


# --------------------------------------------------
# 1. Feature configuration
# --------------------------------------------------

NUMERIC_FEATURES = [
    "waitlist_position",
    "days_before_journey",
    "distance_km",
    "passenger_count",
    "is_tatkal",
    "is_peak_season",
    "is_holiday",
]

CATEGORICAL_FEATURES = [
    "class",
    "train_type",
    "quota",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_PATH = MODEL_DIR / "rf_model.pkl"


# --------------------------------------------------
# 2. Build the machine-learning pipeline
# --------------------------------------------------

def build_pipeline():
    """Create a preprocessing and Random Forest pipeline."""

    # Convert categorical values into one-hot encoded columns
    prep = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # Create the Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Combine preprocessing and model into one pipeline
    return Pipeline([
        ("prep", prep),
        ("rf", rf),
    ])


# --------------------------------------------------
# 3. Prepare input features
# --------------------------------------------------

def _features(df):
    """Select features and standardize categorical values."""

    missing = [
        col for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = df[FEATURE_COLS].copy()

    # Ensure numeric features contain valid numeric values
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    # Clean categorical features
    for col in CATEGORICAL_FEATURES:
        X[col] = (
            X[col]
            .astype("string")
            .str.strip()
            .str.upper()
            .astype(object)
        )

    # Standardize train types consistently
    X["train_type"] = X["train_type"].map(
        normalize_train_type
    )

    # Avoid silently training with missing features
    if X.isna().any().any():
        missing_counts = (
            X.isna()
            .sum()
            .loc[lambda values: values > 0]
            .to_dict()
        )

        raise ValueError(
            "Missing or invalid feature values: "
            f"{missing_counts}. Clean the data first."
        )

    return X


# --------------------------------------------------
# 4. Train and evaluate the model
# --------------------------------------------------

def train_model(df, save=True, waitlisted_only=True):
    """
    Train and evaluate the Random Forest classifier.

    If waitlisted_only=True, only tickets with
    waitlist_position > 0 are included.
    """

    required_cols = FEATURE_COLS + [
        "confirmed",
        "final_status",
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required training columns: {missing}"
        )

    data = df.copy()

    # Select genuinely waitlisted tickets if requested
    if waitlisted_only:
        positions = pd.to_numeric(
            data["waitlist_position"],
            errors="coerce",
        )

        data = data.loc[positions > 0].copy()

    # Clean target values
    data["confirmed"] = pd.to_numeric(
        data["confirmed"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["confirmed"]
    ).copy()

    if not data["confirmed"].isin([0, 1]).all():
        raise ValueError(
            "'confirmed' must contain only 0 and 1."
        )

    data["confirmed"] = data["confirmed"].astype(int)

    # A binary classification model needs both classes
    class_counts = data["confirmed"].value_counts()

    if len(data) < 50 or len(class_counts) < 2:
        raise ValueError(
            f"Not enough data to train: {len(data)} rows "
            f"and {len(class_counts)} target classes."
        )

    # Stratified 5-fold CV requires at least 5 rows per class
    if class_counts.min() < 5:
        raise ValueError(
            "At least 5 samples from each target class are "
            "required for 5-fold stratified cross-validation. "
            f"Class counts: {class_counts.to_dict()}"
        )

    print(
        f"\nTraining on {len(data)} rows "
        f"({'waitlisted tickets only' if waitlisted_only else 'all tickets'})"
    )

    print(
        f"Confirmation rate: "
        f"{data['confirmed'].mean():.1%}"
    )

    # Separate features and target
    X = _features(data)
    y = data["confirmed"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train the model
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Make predictions on unseen test data
    preds = model.predict(X_test)

    # Find the probability corresponding to class 1
    rf = model.named_steps["rf"]
    positive_index = list(rf.classes_).index(1)

    proba = model.predict_proba(X_test)[:, positive_index]

    # Compare with the existing rule-based predictor
    rule_scores = [
        predict_confirmation(
            waitlist_position,
            days_before,
            travel_class,
        )
        for waitlist_position, days_before, travel_class
        in zip(
            X_test["waitlist_position"],
            X_test["days_before_journey"],
            X_test["class"],
        )
    ]

    # Evaluate rule-based output as a probability or score
    rule_auc = roc_auc_score(
        y_test,
        rule_scores,
    )

    # Five-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_auc = cross_val_score(
        build_pipeline(),
        X,
        y,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
    )

    # Calculate performance metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "rule_based_roc_auc": float(rule_auc),
    }

    # Display performance results
    print("\nMODEL PERFORMANCE (held-out 20%)")
    print("=" * 50)

    print(f"Accuracy             : {metrics['accuracy']:.2%}")
    print(f"ROC-AUC (ML)         : {metrics['roc_auc']:.3f}")
    print(f"ROC-AUC (rule-based) : {metrics['rule_based_roc_auc']:.3f}")
    print(
        f"5-fold CV ROC-AUC    : "
        f"{cv_auc.mean():.3f} +/- {cv_auc.std():.3f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            preds,
            labels=[0, 1],
            target_names=["Not CNF", "CNF"],
            zero_division=0,
        )
    )

    # Display the ten most important features
    print("FEATURE IMPORTANCE (Top 10)")
    print("=" * 50)

    names = model.named_steps[
        "prep"
    ].get_feature_names_out()

    importances = pd.Series(
        rf.feature_importances_,
        index=names,
    )

    for name, value in (
        importances.sort_values(ascending=False)
        .head(10)
        .items()
    ):
        print(f"{name:<30} {value:.3f}")

    # Save model and metadata
    if save:
        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLS,
                "waitlisted_only": waitlisted_only,
            },
            MODEL_PATH,
        )

        print(f"\nModel saved to: {MODEL_PATH}")

    return model, metrics


# --------------------------------------------------
# 5. Load a saved model
# --------------------------------------------------

def load_model():
    """Load the saved Random Forest pipeline."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train the model first by running train_model.py."
        )

    saved = joblib.load(MODEL_PATH)

    return saved["model"]


# --------------------------------------------------
# 6. Predict ticket confirmation probability
# --------------------------------------------------

def predict_ml(
    model,
    waitlist_position,
    days_before,
    travel_class="3A",
    train_type="Superfast",
    quota="GN",
    distance_km=1000,
    passenger_count=1,
    is_tatkal=0,
    is_peak_season=0,
    is_holiday=0,
):
    """Return the model's estimated CNF score for a valid waitlisted ticket.

    The returned score is not guaranteed to be calibrated to real-world data.
    """
    numeric_values = {
        "waitlist_position": waitlist_position,
        "days_before": days_before,
        "distance_km": distance_km,
        "passenger_count": passenger_count,
        "is_tatkal": is_tatkal,
        "is_peak_season": is_peak_season,
        "is_holiday": is_holiday,
    }
    for name, value in numeric_values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        if name in {"waitlist_position", "days_before", "passenger_count", "distance_km"} and not numeric.is_integer():
            raise ValueError(f"{name} must be a whole number")
    if float(waitlist_position) < 1:
        raise ValueError("waitlist_position must be at least 1 for a waitlisted-ticket prediction")
    if float(passenger_count) < 1:
        raise ValueError("passenger_count must be at least 1")
    for name in ("is_tatkal", "is_peak_season", "is_holiday"):
        if float(numeric_values[name]) not in (0.0, 1.0):
            raise ValueError(f"{name} must be either 0 or 1")

    X = pd.DataFrame([{
        "waitlist_position": waitlist_position,
        "days_before_journey": days_before,
        "distance_km": distance_km,
        "passenger_count": passenger_count,
        "is_tatkal": is_tatkal,
        "is_peak_season": is_peak_season,
        "is_holiday": is_holiday,
        "class": str(travel_class).strip().upper(),
        "train_type": normalize_train_type(train_type),
        "quota": str(quota).strip().upper(),
    }])[FEATURE_COLS]

    # Validate and prepare the prediction input
    X = _features(X)

    # Select the probability for confirmed class (1)
    rf = model.named_steps["rf"]
    positive_index = list(rf.classes_).index(1)

    probability = model.predict_proba(X)[0][positive_index]

    return round(float(probability), 3)


# --------------------------------------------------
# 7. Run training directly
# --------------------------------------------------

if __name__ == "__main__":

    try:
        from .load_data import load_csv
    except ImportError:
        from load_data import load_csv

    # Read the cleaned dataset
    df = pd.read_csv(CLEAN_CSV)

    # Train and save the model
    trained_model, model_metrics = train_model(
        df,
        save=True,
        waitlisted_only=True,
    )

    # Example prediction
    probability = predict_ml(
        trained_model,
        waitlist_position=10,
        days_before=15,
        travel_class="3A",
        train_type="Superfast",
        quota="GN",
    )

    print(
        f"\nExample predicted confirmation probability: "
        f"{probability:.1%}"
    )
