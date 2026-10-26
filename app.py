import html
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CLEAN_CSV, ROOT
from src.ml_model import load_model, predict_ml
from src.pnr_lookup import lookup_pnr
from src.predict import classify_risk, predict_confirmation


# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# Keep this before other Streamlit UI commands.
# ---------------------------------------------------------
st.set_page_config(
    page_title="Railway WL Predictor",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 2. CSS
# Expected path: <project root>/assets/styles.css
# ---------------------------------------------------------
def load_css():
    css_file = ROOT / "assets" / "styles.css"

    if css_file.exists():
        try:
            css = css_file.read_text(encoding="utf-8")
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except OSError as exc:
            st.warning(f"Could not read CSS file: {exc}")
    else:
        st.info(f"Custom CSS was not found at: {css_file}. Using default styling.")


# ---------------------------------------------------------
# 3. DATA AND MODEL LOADING
# ---------------------------------------------------------
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
    "confirmed",
    "passenger_count",
    "is_tatkal",
    "is_peak_season",
    "is_holiday",
    "train_type",
    "distance_km",
]


@st.cache_data(show_spinner="Loading cleaned railway dataset...")
def load_data(csv_path: str, file_modified: float) -> pd.DataFrame:
    # file_modified is included in the cache key so that changes to the
    # CSV file cause Streamlit to load it again.
    del file_modified

    data = pd.read_csv(csv_path, dtype={"pnr": "string"})

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "The cleaned CSV is missing these required columns: "
            + ", ".join(missing)
        )

    data["journey_date"] = pd.to_datetime(
        data["journey_date"], errors="coerce"
    )
    data["booking_date"] = pd.to_datetime(
        data["booking_date"], errors="coerce"
    )

    for column in [
        "waitlist_position",
        "days_before_journey",
        "passenger_count",
        "distance_km",
        "is_tatkal",
        "is_peak_season",
        "is_holiday",
        "confirmed",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["final_status"] = data["final_status"].astype("string").str.upper().str.strip()
    data["class"] = data["class"].astype("string").str.strip()
    data["quota"] = data["quota"].astype("string").str.strip()
    data["train_type"] = data["train_type"].astype("string").str.strip()
    data["train_name"] = data["train_name"].astype("string").fillna("Unknown train")

    data = data.dropna(subset=["waitlist_position", "confirmed", "final_status"])

    if data.empty:
        raise ValueError("The cleaned CSV has no usable rows after validation.")

    data["waitlist_position"] = data["waitlist_position"].astype(int)
    data["confirmed"] = data["confirmed"].astype(int)

    return data


@st.cache_resource
def load_ml_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Could not load the ML model")
        return None


# ---------------------------------------------------------
# 4. REUSABLE UI HELPERS
# ---------------------------------------------------------
def safe_text(value) -> str:
    if pd.isna(value):
        return "—"
    return html.escape(str(value))


def hero(title, subtitle, badges):
    badge_html = "".join(
        f'<span class="badge">{safe_text(badge)}</span>' for badge in badges
    )
    st.markdown(
        f'<div class="hero"><h1>{safe_text(title)}</h1>'
        f'<p>{safe_text(subtitle)}</p>'
        f'<div class="badge-row">{badge_html}</div></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(
        f'<div class="section-header">{safe_text(title)}</div>',
        unsafe_allow_html=True,
    )


def metric(label, value, delta=None, delta_dir="neutral"):
    delta_html = (
        f'<div class="metric-delta {safe_text(delta_dir)}">'
        f'{safe_text(delta)}</div>'
        if delta is not None
        else ""
    )
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{safe_text(label)}</div>'
        f'<div class="metric-value">{safe_text(value)}</div>'
        f'{delta_html}</div>',
        unsafe_allow_html=True,
    )


def info_row(label, value, trusted_html=False):
    """Render a row, escaping data values unless explicitly generated HTML."""
    rendered_value = str(value) if trusted_html else safe_text(value)
    return (
        f'<div class="info-row"><span class="label">{safe_text(label)}</span>'
        f'<span class="value">{rendered_value}</span></div>'
    )


def result_card(prob):
    try:
        prob = float(prob)
    except (TypeError, ValueError):
        st.error("The prediction model returned an invalid probability.")
        return

    if not 0 <= prob <= 1:
        st.error(f"Probability must be between 0 and 1; received {prob}.")
        return

    if prob >= 0.75:
        css_class, message = "high", "Likely to be Confirmed ✅"
    elif prob >= 0.45:
        css_class, message = "medium", "Moderate Chances — Monitor Closely ⚠️"
    else:
        css_class, message = "low", "Low Score — Have a Backup 🔴"

    st.markdown(
        f'<div class="result-card {css_class}">'
        f'<div class="result-label">Estimated Confirmation Score</div>'
        f'<div class="big-number">{prob * 100:.0f}%</div>'
        f'<div class="result-message">{message}</div></div>',
        unsafe_allow_html=True,
    )


def pill(status):
    status_text = str(status).upper().strip()
    css_class = {"CNF": "cnf", "RAC": "rac", "WL": "wl"}.get(status_text, "")
    return (
        f'<span class="pill {css_class}">{safe_text(status_text)}</span>'
    )


def choices(column, preferred):
    values = df[column].dropna().astype(str).unique().tolist()
    options = sorted(values)
    if not options:
        options = [preferred]
    index = options.index(preferred) if preferred in options else 0
    return options, index


def show_startup_error(message, exc=None):
    st.error(message)
    if exc is not None:
        with st.expander("Technical details"):
            st.exception(exc)


# ---------------------------------------------------------
# 5. STARTUP
# ---------------------------------------------------------
load_css()

if not CLEAN_CSV.exists():
    st.error("Cleaned dataset not found.")
    st.code("python main.py", language="bash")
    st.caption(f"Expected CSV path: {CLEAN_CSV}")
    st.stop()

try:
    df = load_data(str(CLEAN_CSV), CLEAN_CSV.stat().st_mtime)
except Exception as exc:
    logger.exception("Could not load cleaned data")
    show_startup_error("The dashboard could not load the cleaned dataset.", exc)
    st.stop()

model = load_ml_model()


# ---------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:0.75rem 0 1.25rem;">
            <div style="
                display:inline-flex; align-items:center; justify-content:center;
                width:60px; height:60px; border-radius:16px;
                background:linear-gradient(135deg,#6366f1,#8b5cf6);
                box-shadow:0 10px 24px rgba(99,102,241,.25);
                font-size:1.75rem; margin-bottom:.75rem;">
                🚂
            </div>
            <div style="font-weight:800;font-size:1.1rem;color:#0f172a;">
                Railway WL Predictor
            </div>
            <div style="color:#64748b;font-size:.78rem;margin-top:.2rem;">
                Waitlist forecast dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Navigation",
        ["🎫 PNR Lookup", "🎯 Manual Prediction", "📊 Analytics"],
        label_visibility="collapsed",
    )

    st.divider()
    waitlisted_count = int((df["waitlist_position"] > 0).sum())
    confirmation_rate = float(df["confirmed"].mean() * 100)

    st.markdown(
        f"""
        <div style="
            background:rgba(99,102,241,.06);
            border:1px solid rgba(99,102,241,.15);
            border-radius:12px;padding:.9rem 1rem;
            font-size:.8rem;color:#475569;line-height:1.8;">
            <div style="font-weight:700;color:#0f172a;font-size:.72rem;
                        text-transform:uppercase;letter-spacing:.08em;
                        margin-bottom:.35rem;">Dataset</div>
            <b style="color:#0f172a;">{len(df):,}</b> tickets<br>
            <b style="color:#0f172a;">{waitlisted_count:,}</b> waitlisted<br>
            <b style="color:#0f172a;">{confirmation_rate:.1f}%</b> confirmed rate<br>
            <b style="color:#0f172a;">{df['class'].nunique()}</b> travel classes
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Streamlit · Pandas · Plotly · scikit-learn")

if model is None:
    st.sidebar.warning(
        "ML model is unavailable. Manual Prediction can still use the rule-based estimate."
    )


# ---------------------------------------------------------
# 7. PNR LOOKUP
# ---------------------------------------------------------
if mode == "🎫 PNR Lookup":
    hero(
        "PNR Lookup",
        "Search for a PNR contained in your dataset and view its stored details and estimate.",
        ["Rule-based + ML when available", f"{len(df):,} records"],
    )

    input_col, button_col = st.columns([3, 1])
    with input_col:
        pnr_input = st.text_input(
            "PNR Number",
            placeholder="Enter a PNR from your dataset",
        )
    with button_col:
        st.write("")
        search_clicked = st.button(
            "🔍 Look Up PNR", type="primary", use_container_width=True
        )

    if search_clicked:
        if not pnr_input.strip():
            st.warning("Please enter a PNR.")
        else:
            try:
                result = lookup_pnr(pnr_input.strip(), df, model)
            except Exception as exc:
                logger.exception("PNR lookup failed")
                show_startup_error("PNR lookup failed.", exc)
                result = None

            if result is not None:
                section("🚆 Journey Information")
                left, right = st.columns([1.2, 1])

                with left:
                    rows = [
                        info_row("PNR", result.get("pnr", "—")),
                        info_row("Train No.", result.get("train_number", "—")),
                        info_row("Train Name", result.get("train_name", "—")),
                        info_row("From", result.get("from", "—")),
                        info_row("To", result.get("to", "—")),
                        info_row("Journey Date", result.get("journey_date", "—")),
                        info_row("Class", result.get("class", "—")),
                        info_row("Quota", result.get("quota", "—")),
                        info_row("Waitlist Position", result.get("waitlist_position", "—")),
                        info_row("Days Before Journey", result.get("days_before", "—")),
                        info_row("Stored Status", pill(result.get("actual_status", "Unknown")), trusted_html=True),
                    ]
                    st.markdown(
                        f'<div class="card">{"".join(rows)}</div>',
                        unsafe_allow_html=True,
                    )
                with right:
                    probability = result.get("ml_prob")
                    if probability is None:
                        probability = result.get("predicted_prob")

                    if probability is not None:
                        result_card(probability)
                        st.markdown(
                            f'<div class="card" style="text-align:center;margin-top:1rem;">'
                            f'<div class="metric-label">Risk Assessment</div>'
                            f'<div style="font-size:1rem;font-weight:600;margin-top:.5rem;">'
                            f'{safe_text(classify_risk(float(probability)))}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("No prediction score is available for this ticket.")
            else:
                st.error(
                    f"PNR {pnr_input.strip()} was not found in the dataset, "
                    "or the lookup could not be completed. Check the terminal for details."
                )

    st.caption(
        "This app searches a local sample dataset only. It does not retrieve live Indian Railways PNR status. The displayed heuristic score is not a calibrated probability."
    )


# ---------------------------------------------------------
# 8. MANUAL PREDICTION
# ---------------------------------------------------------
elif mode == "🎯 Manual Prediction":
    hero(
        "Manual Prediction",
        "Enter booking details to estimate a confirmation score.",
        ["Rule-based estimate", "Random Forest if available", "Interactive"],
    )

    use_ml = st.toggle(
        "🤖 Use Machine Learning Model",
        value=(model is not None),
        disabled=(model is None),
        help="Uses the saved model loaded from src.ml_model.",
    )

    class_options, class_index = choices("class", "3A")
    quota_options, quota_index = choices("quota", "GN")
    train_type_options, train_type_index = choices("train_type", "SUPERFAST")

    section("📝 Booking Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        waitlist_position = st.number_input(
            "Waitlist Position", min_value=1, max_value=500, value=45, step=1
        )
        travel_class = st.selectbox(
            "Travel Class", class_options, index=class_index
        )

    with col2:
        days_before = st.number_input(
            "Days Before Journey", min_value=0, max_value=120, value=14, step=1
        )
        quota = st.selectbox("Quota", quota_options, index=quota_index)

    with col3:
        train_type = st.selectbox(
            "Train Type", train_type_options, index=train_type_index
        )
        distance_km = st.number_input(
            "Distance (km)", min_value=0, max_value=5000, value=1000, step=50
        )

    col4, col5, col6, col7 = st.columns(4)
    with col4:
        passenger_count = st.number_input(
            "Passengers", min_value=1, max_value=6, value=2, step=1
        )
    with col5:
        is_tatkal = st.checkbox("Tatkal Booking")
    with col6:
        is_peak_season = st.checkbox("Peak Season")
    with col7:
        is_holiday = st.checkbox("Holiday Travel")

    if st.button(
        "🔮 Predict Confirmation", type="primary", use_container_width=True
    ):
        try:
            if use_ml and model is not None:
                probability = predict_ml(
                    model,
                    waitlist_position=waitlist_position,
                    days_before=days_before,
                    travel_class=travel_class,
                    train_type=train_type,
                    quota=quota,
                    distance_km=distance_km,
                    passenger_count=passenger_count,
                    is_tatkal=int(is_tatkal),
                    is_peak_season=int(is_peak_season),
                    is_holiday=int(is_holiday),
                )
                method = "Random Forest model"
            else:
                probability = predict_confirmation(
                    waitlist_position, days_before, travel_class
                )
                method = "Rule-based heuristic"

            probability = float(probability)
            section("🎯 Prediction Result")
            result_col, details_col = st.columns(2)

            with result_col:
                result_card(probability)

            with details_col:
                st.markdown(
                    f'<div class="card">'
                    f'<div class="metric-label">Prediction Method</div>'
                    f'<div style="font-size:1rem;font-weight:600;margin:.5rem 0 1rem;">'
                    f'{safe_text(method)}</div>'
                    f'<div class="metric-label">Estimated Score</div>'
                    f'<div style="font-size:2rem;font-weight:800;margin-top:.25rem;">'
                    f'{probability * 100:.1f}%</div>'
                    f'<div style="margin-top:.75rem;">'
                    f'{safe_text(classify_risk(probability))}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "This is an estimate from your local dataset/model, not an official "
                "Railway prediction or live PNR status. Model scores are not guaranteed "
                "to be calibrated real-world probabilities."
            )
        except Exception as exc:
            logger.exception("Manual prediction failed")
            show_startup_error("Could not generate the prediction.", exc)


# ---------------------------------------------------------
# 9. ANALYTICS
# ---------------------------------------------------------
elif mode == "📊 Analytics":
    hero(
        "Analytics Dashboard",
        "Explore confirmation patterns across travel classes, train types, and waitlist positions.",
        ["Exploratory analysis", "Interactive charts", f"{len(df):,} records"],
    )

    waitlisted_only = st.checkbox(
        "Waitlisted tickets only (position > 0)", value=True
    )
    analytics_data = (
        df[df["waitlist_position"] > 0].copy()
        if waitlisted_only
        else df.copy()
    )

    if analytics_data.empty:
        st.warning("No rows match the selected analytics filter.")
    else:
        section("📈 Key Metrics")
        metric1, metric2, metric3, metric4 = st.columns(4)

        confirmed_count = int((analytics_data["final_status"] == "CNF").sum())
        rac_count = int((analytics_data["final_status"] == "RAC").sum())
        wl_count = int((analytics_data["final_status"] == "WL").sum())
        rate = float(analytics_data["confirmed"].mean() * 100)

        with metric1:
            metric("Total Tickets", f"{len(analytics_data):,}")
        with metric2:
            metric("Confirmed", f"{confirmed_count:,}", f"{rate:.1f}% rate", "up")
        with metric3:
            metric("RAC", f"{rac_count:,}", "Stored RAC status", "neutral")
        with metric4:
            metric("Waiting List", f"{wl_count:,}", "Stored WL status", "down")

        def rate_table(group_column, label):
            table = (
                analytics_data.groupby(group_column, dropna=False)["confirmed"]
                .agg(["mean", "size"])
                .reset_index()
            )
            table.columns = [label, "Rate", "Tickets"]
            table["Rate"] = table["Rate"] * 100
            return table

        plot_style = {
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#475569"},
            "margin": {"t": 25, "b": 25, "l": 10, "r": 10},
        }
        color_scale = ["#fee2e2", "#fef3c7", "#d1fae5"]

        section("📊 Confirmation by Class")
        class_rates = rate_table("class", "Class")
        if not class_rates.empty:
            fig = px.bar(
                class_rates,
                x="Class",
                y="Rate",
                color="Rate",
                color_continuous_scale=color_scale,
                text_auto=".0f",
                hover_data={"Tickets": True},
                range_y=[0, 115],
            )
            fig.update_layout(
                showlegend=False,
                height=380,
                coloraxis_showscale=False,
                xaxis={"showgrid": False, "title": ""},
                yaxis={"gridcolor": "#f1f5f9", "title": "Confirmation rate (%)"},
                **plot_style,
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        section("🚆 Confirmation by Train Type")
        train_rates = rate_table("train_type", "Train Type").sort_values("Rate")
        if not train_rates.empty:
            fig2 = px.bar(
                train_rates,
                x="Rate",
                y="Train Type",
                orientation="h",
                color="Rate",
                color_continuous_scale=color_scale,
                text_auto=".0f",
                hover_data={"Tickets": True},
                range_x=[0, 115],
            )
            fig2.update_layout(
                showlegend=False,
                height=420,
                coloraxis_showscale=False,
                xaxis={"gridcolor": "#f1f5f9", "title": "Confirmation rate (%)"},
                yaxis={"showgrid": False, "title": ""},
                **plot_style,
            )
            st.plotly_chart(fig2, use_container_width=True)

        section("📉 Waitlist Position vs Status")
        histogram_data = analytics_data.dropna(
            subset=["waitlist_position", "final_status"]
        )
        if not histogram_data.empty:
            fig3 = px.histogram(
                histogram_data,
                x="waitlist_position",
                color="final_status",
                nbins=25,
                barmode="stack",
                category_orders={"final_status": ["CNF", "RAC", "WL"]},
                color_discrete_map={
                    "CNF": "#10b981",
                    "RAC": "#f59e0b",
                    "WL": "#ef4444",
                },
                labels={
                    "waitlist_position": "Waitlist position",
                    "final_status": "Status",
                },
            )
            fig3.update_layout(
                height=420,
                legend_title_text="",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                xaxis={"gridcolor": "#f1f5f9"},
                yaxis={"gridcolor": "#f1f5f9"},
                **plot_style,
            )
            st.plotly_chart(fig3, use_container_width=True)

    section("🗂️ Dataset Explorer")
    filter1, filter2, filter3 = st.columns([1, 1, 2])

    with filter1:
        status_options = ["CNF", "RAC", "WL"]
        status_filter = st.multiselect(
            "Final status", status_options, default=status_options
        )
    with filter2:
        class_options_all = sorted(df["class"].dropna().astype(str).unique().tolist())
        class_filter = st.multiselect("Class", class_options_all)
    with filter3:
        search_query = st.text_input(
            "Search PNR or train name", placeholder="e.g. Rajdhani"
        )

    view = df[df["final_status"].isin(status_filter)].copy()

    if class_filter:
        view = view[view["class"].astype(str).isin(class_filter)]

    if search_query.strip():
        query = search_query.strip()
        pnr_match = view["pnr"].astype(str).str.contains(query, regex=False, na=False)
        train_match = view["train_name"].astype(str).str.contains(
            query, case=False, regex=False, na=False
        )
        view = view[pnr_match | train_match]

    st.caption(f"Showing {len(view):,} of {len(df):,} tickets")
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)

    st.download_button(
        "⬇️ Download filtered rows as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="railway_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------
# 10. FOOTER
# ---------------------------------------------------------
st.markdown(
    '<div class="footer">🚂 <b>Railway WL Predictor</b> · '
    'Student project · Built with Streamlit, Pandas, scikit-learn &amp; Plotly</div>',
    unsafe_allow_html=True,
)
