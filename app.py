"""Streamlit frontend for the Commodity Price Prediction Dashboard.

MLOps Phase 2: this UI is a thin client. All ML inference is delegated to the
Flask API specified by the API_BASE_URL setting (Streamlit secret or env var).
"""

import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_API_BASE_URL = "http://localhost:5000"


def get_api_base_url() -> str:
    """Resolve API base URL from Streamlit secrets or environment variables."""
    try:
        if "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


API_BASE_URL = get_api_base_url()
REQUEST_TIMEOUT = 10  # seconds


# --------------------------------------------------------------------------- #
# API client helpers
# --------------------------------------------------------------------------- #
def api_health() -> Tuple[bool, str]:
    """Call GET / to verify API health."""
    try:
        resp = requests.get(f"{API_BASE_URL}/", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return True, str(data.get("status", "ok"))
    except requests.exceptions.RequestException as exc:
        return False, f"Unreachable: {exc}"
    except ValueError:
        return False, "Invalid JSON in health response."


def api_predict(days: float) -> Tuple[Optional[float], Optional[str]]:
    """Call POST /predict and return (price, error)."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            json={"days": days},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        return None, f"Network error: {exc}"

    try:
        data: Dict[str, Any] = resp.json()
    except ValueError:
        return None, f"Invalid JSON response (HTTP {resp.status_code})."

    if resp.status_code != 200:
        return None, str(data.get("error", f"HTTP {resp.status_code}"))
    if "predicted_price" not in data:
        return None, "Response missing 'predicted_price'."
    try:
        return float(data["predicted_price"]), None
    except (TypeError, ValueError):
        return None, "Invalid 'predicted_price' value."


# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Commodity Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
)

st.sidebar.info(
    "Commodity Price Prediction Dashboard\n\n"
    "MLOps Phase 2 — Streamlit frontend + Flask API backend."
)
st.sidebar.markdown(f"**API URL:** `{API_BASE_URL}`")

healthy, health_msg = api_health()
if healthy:
    st.sidebar.success(f"API status: {health_msg}")
else:
    st.sidebar.error(f"API status: {health_msg}")


# --------------------------------------------------------------------------- #
# Title and description
# --------------------------------------------------------------------------- #
st.title("📈 Commodity Price Prediction Dashboard")
st.markdown(
    "An interactive dashboard backed by a Flask ML API. "
    "Streamlit no longer performs inference in-process — all predictions are "
    "served by the backend at the configured `API_BASE_URL`."
)


# --------------------------------------------------------------------------- #
# Data loading (for visualization only)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the commodity dataset for visualization."""
    candidates = ["data.csv"]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("data.csv not found in working directory.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        if "days" not in df.columns:
            df["days"] = (df["date"] - df["date"].min()).dt.days
    elif "days" not in df.columns:
        df["days"] = range(len(df))
    return df


try:
    df = load_data()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load dataset: {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# Dataset preview
# --------------------------------------------------------------------------- #
st.subheader("📂 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------------------------------- #
# Prediction controls
# --------------------------------------------------------------------------- #
st.subheader("🔮 Predict Future Price")
col_a, col_b = st.columns([3, 1])
with col_a:
    days_ahead = st.slider(
        "Days into the future",
        min_value=0,
        max_value=365,
        value=30,
        step=1,
    )
with col_b:
    predict_clicked = st.button("Predict", use_container_width=True)

if not healthy:
    st.warning(
        "API is unreachable. Configure `API_BASE_URL` in Streamlit secrets or "
        "environment variables to point to your Flask backend "
        "(e.g. a Railway deployment)."
    )

predicted_price: Optional[float] = None
if predict_clicked:
    if not healthy:
        st.error("Cannot run prediction while the API is unreachable.")
    else:
        last_day = float(df["days"].max()) if "days" in df.columns else 0.0
        target_day = last_day + float(days_ahead)
        with st.spinner("Calling Flask API..."):
            predicted_price, err = api_predict(target_day)
        if err:
            st.error(f"Prediction failed: {err}")
        else:
            st.success(
                f"Predicted price in {days_ahead} days: **{predicted_price:.4f}**"
            )


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
st.subheader("📊 Price Trend and Forecast")
fig, ax = plt.subplots(figsize=(10, 5))
if "date" in df.columns:
    ax.scatter(df["date"], df["price"], label="Historical price", color="#1f77b4")
    ax.set_xlabel("Date")
else:
    ax.scatter(df["days"], df["price"], label="Historical price", color="#1f77b4")
    ax.set_xlabel("Days")
ax.set_ylabel("price")
ax.set_title("Commodity Price Trend with Forecast")
ax.grid(True, alpha=0.3)

if predicted_price is not None:
    if "date" in df.columns:
        forecast_date = df["date"].max() + pd.Timedelta(days=int(days_ahead))
        ax.scatter(
            [forecast_date],
            [predicted_price],
            marker="*",
            s=200,
            color="green",
            label="Prediction",
        )
    else:
        ax.scatter(
            [df["days"].max() + days_ahead],
            [predicted_price],
            marker="*",
            s=200,
            color="green",
            label="Prediction",
        )

ax.legend()
st.pyplot(fig)


# --------------------------------------------------------------------------- #
# Analytical insights
# --------------------------------------------------------------------------- #
st.subheader("🧠 Analytical Insights")
latest_price = float(df["price"].iloc[-1])
st.markdown(f"- **Latest observed price:** {latest_price:.2f}")
if predicted_price is not None:
    change_pct = (predicted_price - latest_price) / latest_price * 100.0
    direction = "upward" if change_pct >= 0 else "downward"
    st.markdown(
        f"- **Predicted price ({days_ahead} days ahead):** {predicted_price:.2f}\n"
        f"- **Expected change:** {change_pct:+.2f}%\n"
        f"- **Trend assessment:** {direction} trend"
    )
else:
    st.caption("Click **Predict** to fetch a forecast from the Flask API.")


st.markdown("---")
st.caption(
    "© 2026 Commodity Price Prediction Demo — MLOps Phase 2 "
    "(Streamlit frontend → Flask API → ML model)."
)
