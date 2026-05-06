"""Flask backend API for Commodity Price Prediction Dashboard.

MLOps Phase 2: serves a Linear Regression model over HTTP.
- Trains and caches model.pkl on startup if it does not exist.
- Exposes GET / for health and POST /predict for inference.
"""

import json
import logging
import os
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("commodity-api")


# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")


# --------------------------------------------------------------------------- #
# Model training / loading
# --------------------------------------------------------------------------- #
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has a numeric `days` feature and `price` target."""
    if "days" not in df.columns:
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            df = df.assign(days=(dates - dates.min()).dt.days)
        else:
            df = df.assign(days=np.arange(len(df)))
    if "price" not in df.columns:
        raise ValueError("Dataset must contain a 'price' column.")
    return df


def train_and_cache_model() -> LinearRegression:
    """Train a LinearRegression on data.csv and persist it to model.pkl."""
    logger.info("Training new model from %s", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = _build_features(df)

    X = df[["days"]].astype(float).values
    y = df["price"].astype(float).values

    model = LinearRegression()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model trained and cached at %s", MODEL_PATH)
    return model


def load_or_train_model() -> LinearRegression:
    """Load model.pkl if it exists, otherwise train and cache it."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as fh:
                model = pickle.load(fh)
            logger.info("Loaded cached model from %s", MODEL_PATH)
            return model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load cached model (%s); retraining.", exc)
    return train_and_cache_model()


# --------------------------------------------------------------------------- #
# Flask app factory
# --------------------------------------------------------------------------- #
app = Flask(__name__)
MODEL = load_or_train_model()


def _validate_predict_payload(payload: Any) -> float:
    """Validate /predict JSON payload, return a validated `days` float."""
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    if "days" not in payload:
        raise ValueError("Missing required field: 'days'.")
    value = payload["days"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("'days' must be numeric.") from exc
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("'days' must be a finite number.")
    if value < 0:
        raise ValueError("'days' must be non-negative.")
    return value


@app.route("/", methods=["GET"])
def health() -> Any:
    """Health endpoint."""
    logger.info("GET / health check")
    return jsonify({"status": "API is running"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    """Predict the future price for a given `days` offset."""
    try:
        payload: Dict[str, Any] = request.get_json(silent=True)
        if payload is None:
            logger.warning("Invalid or missing JSON body on /predict")
            return jsonify({"error": "Request body must be valid JSON."}), 400

        days = _validate_predict_payload(payload)
        prediction = float(MODEL.predict(np.array([[days]]))[0])
        logger.info("Prediction for days=%s -> %.4f", days, prediction)
        return jsonify({"predicted_price": round(prediction, 4)}), 200

    except ValueError as exc:
        logger.warning("Validation error on /predict: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error on /predict")
        return jsonify({"error": "Internal server error."}), 500


@app.errorhandler(404)
def not_found(_e: Any) -> Any:
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_e: Any) -> Any:
    return jsonify({"error": "Method not allowed."}), 405


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting Flask API on %s:%s (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)
