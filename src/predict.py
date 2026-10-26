
"""Rule-based confirmation score (not a calibrated probability)."""

CLASS_BONUS = {
    "2A": 0.10,
    "3A": 0.08,
    "CC": 0.10,
    "EC": 0.12,
    "SL": 0.0
}

MAX_PROB = 0.95


def predict_confirmation(waitlist_position, days_before, travel_class="3A"):
    """Return a rule-based score between 0.0 and 1.0 (not a calibrated probability)."""

    if waitlist_position is None or days_before is None:
        raise ValueError("waitlist_position and days_before are required")

    try:
        waitlist_position = float(waitlist_position)
        days_before = float(days_before)
    except (TypeError, ValueError) as exc:
        raise ValueError("waitlist_position and days_before must be numeric") from exc

    if not (waitlist_position >= 0 and days_before >= 0):
        raise ValueError("waitlist_position and days_before cannot be negative")

    if not (waitlist_position.is_integer() and days_before.is_integer()):
        raise ValueError("waitlist_position and days_before must be whole numbers")

    # This heuristic is trained/designed for actual waitlisted tickets only.
    # Position zero is not a waitlist position and should not receive a score.
    if waitlist_position == 0:
        raise ValueError("Waitlist position 0 is not a waitlisted ticket; prediction is not applicable")

    waitlist_position = int(waitlist_position)
    days_before = int(days_before)

    score = 0.0

    # Waitlist position: lower is generally better
    if waitlist_position <= 10:
        score += 0.55
    elif waitlist_position <= 30:
        score += 0.40
    elif waitlist_position <= 60:
        score += 0.25
    elif waitlist_position <= 100:
        score += 0.15
    elif waitlist_position <= 200:
        score += 0.05

    # Days remaining before the journey
    if days_before >= 30:
        score += 0.25
    elif days_before >= 15:
        score += 0.18
    elif days_before >= 7:
        score += 0.10
    elif days_before >= 3:
        score += 0.05

    # Travel class bonus
    travel_class = str(travel_class).strip().upper()
    score += CLASS_BONUS.get(travel_class, 0.0)

    # Keep heuristic score within the configured maximum
    return min(round(score, 2), MAX_PROB)


def classify_risk(prob):
    """Return a human-friendly label for a score between 0 and 1."""
    try:
        prob = float(prob)
    except (TypeError, ValueError) as exc:
        raise ValueError("Score must be numeric and between 0.0 and 1.0") from exc

    if not (prob == prob and 0.0 <= prob <= 1.0):
        raise ValueError("Score must be finite and between 0.0 and 1.0")

    if prob >= 0.75:
        return "🟢 HIGH chance of confirmation"
    elif prob >= 0.45:
        return "🟡 MEDIUM chance — keep checking"
    else:
        return "🔴 LOW chance — have backup plan"


if __name__ == "__main__":
    try:
        from .config import setup_console
    except ImportError:
        from config import setup_console

    setup_console()

    print("RULE-BASED PREDICTIONS\n")

    tests = [
        (5, 14, "3A"),
        (25, 10, "3A"),
        (56, 11, "3A"),
        (145, 12, "3A"),
        (312, 15, "SL"),
    ]

    for wl, days, cls in tests:
        probability = predict_confirmation(wl, days, cls)

        print(
            f"WL={wl:<4} Days={days:<3} Class={cls} "
            f"-> {probability * 100:>3.0f}%  "
            f"{classify_risk(probability)}"
        )
