
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save graphs without opening display windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Import configuration settings
# Supports both package execution and direct script execution
try:
    from .config import CLEAN_CSV, PLOTS_DIR
except ImportError:
    from config import CLEAN_CSV, PLOTS_DIR


sns.set_style("whitegrid")

# Colors for different ticket statuses
COLORS = {
    "CNF": "#2ecc71",
    "RAC": "#f39c12",
    "WL": "#e74c3c"
}


def _save(fig, path):
    """Adjust layout, save the graph, and close the figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _overlay_hist(ax, data, column, bins=20):
    """Plot histograms for each ticket status using common bins."""

    if data.empty:
        ax.text(
            0.5, 0.5, "No data available",
            ha="center", va="center",
            transform=ax.transAxes
        )
        ax.set_ylabel("Count")
        return

    # Convert the selected column into numeric values
    values = pd.to_numeric(data[column], errors="coerce").dropna()

    if values.empty:
        ax.text(
            0.5, 0.5, "No valid numeric data",
            ha="center", va="center",
            transform=ax.transAxes
        )
        ax.set_ylabel("Count")
        return

    # Use the same bin edges for every status
    edges = np.histogram_bin_edges(values, bins=bins)

    for status, color in COLORS.items():
        subset = pd.to_numeric(
            data.loc[data["final_status"] == status, column],
            errors="coerce"
        ).dropna()

        if not subset.empty:
            ax.hist(
                subset,
                bins=edges,
                alpha=0.6,
                label=f"{status} (n={len(subset)})",
                color=color
            )

    ax.set_ylabel("Count")
    ax.legend()


def _rate_bar(ax, data, by, color):
    """Plot confirmation rate for each group."""

    if data.empty:
        ax.text(
            0.5, 0.5, "No data available",
            ha="center", va="center",
            transform=ax.transAxes
        )
        ax.set_ylabel("Confirmation Rate (%)")
        return

    # Calculate the mean confirmation rate and group size
    grp = (
        data.groupby(by)["confirmed"]
        .agg(["mean", "size"])
        .sort_values("mean", ascending=False)
    )

    if grp.empty:
        ax.text(
            0.5, 0.5, "No groups available",
            ha="center", va="center",
            transform=ax.transAxes
        )
        ax.set_ylabel("Confirmation Rate (%)")
        return

    labels = [
        f"{idx}\n(n={n})"
        for idx, n in zip(grp.index, grp["size"])
    ]

    rates = grp["mean"] * 100

    bars = ax.bar(labels, rates, color=color)

    ax.set_ylabel("Confirmation Rate (%)")
    ax.set_ylim(0, 110)

    # Display the percentage above each bar
    for bar, rate in zip(bars, rates):
        if pd.notna(rate):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 1.5,
                f"{rate:.0f}%",
                ha="center",
                fontweight="bold"
            )


def analyze(df, save_dir=None, waitlisted_only=True):
    """Generate six exploratory data analysis plots."""

    # Validate required columns
    required_cols = [
        "final_status",
        "waitlist_position",
        "class",
        "days_before_journey",
        "passenger_count",
        "distance_km",
        "status_code",
        "confirmed",
        "train_type"
    ]

    missing_cols = [
        col for col in required_cols if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_cols)
        )

    if df.empty:
        raise ValueError("The input CSV contains no data rows.")

    # Make a copy so the original DataFrame is not modified
    df = df.copy()

    # Convert numeric columns safely
    numeric_cols = [
        "waitlist_position",
        "days_before_journey",
        "passenger_count",
        "distance_km",
        "status_code",
        "confirmed"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Set the output directory
    if save_dir is None:
        save_dir = PLOTS_DIR

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter tickets that had a waiting-list position
    if waitlisted_only:
        wl = df[df["waitlist_position"] > 0].copy()
    else:
        wl = df.copy()

    tag = (
        "waitlisted tickets only"
        if waitlisted_only
        else "all tickets"
    )

    # --------------------------------------------------
    # 1. Final Status Distribution
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    counts = df["final_status"].value_counts()

    ax.bar(
        counts.index.astype(str),
        counts.values,
        color=[
            COLORS.get(status, "gray")
            for status in counts.index
        ]
    )

    ax.set_title(
        "Final Status Distribution",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Final Status")
    ax.set_ylabel("Number of Tickets")

    for i, value in enumerate(counts.values):
        ax.text(
            i,
            value + counts.max() * 0.01,
            str(value),
            ha="center",
            fontweight="bold"
        )

    _save(fig, save_dir / "1_status_distribution.png")

    # --------------------------------------------------
    # 2. Waitlist Position vs Status
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    _overlay_hist(ax, wl, "waitlist_position", bins=20)

    ax.set_title(
        f"Waitlist Position vs Status ({tag})",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Waitlist Position")

    _save(fig, save_dir / "2_wl_position_vs_status.png")

    # --------------------------------------------------
    # 3. Confirmation Rate by Class
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    _rate_bar(ax, wl, "class", "steelblue")

    ax.set_title(
        f"Confirmation Rate by Class ({tag})",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Ticket Class")

    _save(fig, save_dir / "3_class_confirmation.png")

    # --------------------------------------------------
    # 4. Days Before Journey vs Status
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    _overlay_hist(ax, wl, "days_before_journey", bins=15)

    ax.set_title(
        f"Days Before Journey vs Status ({tag})",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Days Before Journey")

    _save(fig, save_dir / "4_days_before_vs_status.png")

    # --------------------------------------------------
    # 5. Feature Correlation Heatmap
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    correlation = df[numeric_cols].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title(
        "Feature Correlation",
        fontsize=14,
        fontweight="bold"
    )

    _save(fig, save_dir / "5_correlation.png")

    # --------------------------------------------------
    # 6. Confirmation Rate by Train Type
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    _rate_bar(ax, wl, "train_type", "mediumseagreen")

    ax.set_title(
        f"Confirmation Rate by Train Type ({tag})",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Train Type")

    plt.setp(ax.get_xticklabels(), fontsize=8)

    _save(fig, save_dir / "6_train_type_confirmation.png")

    # --------------------------------------------------
    # Display Quick Statistics
    # --------------------------------------------------
    print(f"\nAll 6 plots saved to: {save_dir.resolve()}")

    print("\nQUICK STATISTICS")
    print("=" * 50)

    print(f"Total tickets           : {len(df)}")

    for status in COLORS:
        count = (df["final_status"] == status).sum()
        print(f"{status:<25}: {count}")

    # Calculate confirmation rate using valid values only
    overall_rate = df["confirmed"].mean() * 100

    print(f"Overall CNF rate         : {overall_rate:.1f}%")

    if waitlisted_only:
        print(f"Waitlisted tickets       : {len(wl)}")

        if not wl.empty:
            wl_rate = wl["confirmed"].mean() * 100
            print(f"CNF rate (waitlisted)    : {wl_rate:.1f}%")
        else:
            print("CNF rate (waitlisted)    : No data")


if __name__ == "__main__":
    # Read the cleaned railway dataset
    data = pd.read_csv(CLEAN_CSV)

    # Generate plots and statistics
    analyze(data)
