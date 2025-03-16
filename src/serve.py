import os
import time
from datetime import datetime

from flask import Flask, jsonify, request

from src.model_service import IrisModelService
from src.utils.logging_config import iris_logger, log_request

app = Flask(__name__)

# Instantiate the model service at startup
model_service = IrisModelService(model_dir="models", model_name="iris_model")

@app.route("/predict", methods=["POST"])
@log_request
def predict():
    """
    Flask endpoint for predicting Iris species.

    Args:
        None directly. Expects a JSON body with key 'input' containing a list of feature lists.
        Each feature list should contain 4 numeric values representing sepal length, sepal width,
        petal length, and petal width.

    Returns:
        JSON response with structure:
        - On success: {"prediction": List[str]} where each str is a predicted species
        - On error: {"error": str} with appropriate HTTP status code
    """
    data = request.get_json(force=True)
    feature_inputs = data.get("input")

    if not feature_inputs:
        iris_logger.warning(
            "Invalid request: No input provided",
            extra={"request_id": request.headers.get('X-Request-ID', str(time.time()))}
        )
        return jsonify({"error": "No 'input' provided."}), 422  # Unprocessable Entity

    try:
        if not isinstance(feature_inputs, list):
            return jsonify({"error": "Input must be a list of feature lists"}), 400  # Bad Request

        if any(len(features) != 4 for features in feature_inputs):
            return jsonify({"error": "Each feature list must contain exactly 4 features"}), 422

        if any(not all(isinstance(x, (int, float)) for x in features) for features in feature_inputs):
            return jsonify({"error": "All features must be numeric values"}), 415  # Unsupported Media Type

        predicted_labels = model_service.predict(feature_inputs)

        iris_logger.info(
            "Prediction made",
            extra={
                "request_id": request.headers.get('X-Request-ID', str(time.time())),
                "predicted_labels": predicted_labels,
                "input_size": len(feature_inputs)
            }
        )

        return jsonify({"prediction": predicted_labels})

    except Exception as e:
        iris_logger.error(
            "Prediction error",
            extra={
                "error": str(e),
                "request_id": request.headers.get('X-Request-ID', str(time.time()))
            }
        )
        return jsonify({"error": str(e)}), 500  # Internal Server Error

@app.route("/predict-proba", methods=["POST"])
@log_request
def predict_proba():
    """
    Flask endpoint for predicting Iris species with probabilities.

    Args:
        None directly. Expects a JSON body with key 'input' containing a list of feature lists.
        Each feature list should contain 4 numeric values representing sepal length, sepal width,
        petal length, and petal width.

    Returns:
        JSON response with structure:
        - On success: {
            "prediction": List[str],
            "probabilities": List[List[float]]
          }
        - On error: {"error": str} with appropriate HTTP status code
    """
    data = request.get_json(force=True)
    feature_inputs = data.get("input")

    if not feature_inputs:
        return jsonify({"error": "No 'input' provided."}), 422  # Unprocessable Entity

    try:
        if not isinstance(feature_inputs, list):
            return jsonify({"error": "Input must be a list of feature lists"}), 400  # Bad Request

        if any(len(features) != 4 for features in feature_inputs):
            return jsonify({"error": "Each feature list must contain exactly 4 features"}), 422

        if any(not all(isinstance(x, (int, float)) for x in features) for features in feature_inputs):
            return jsonify({"error": "All features must be numeric values"}), 415  # Unsupported Media Type

        predicted_labels = model_service.predict(feature_inputs)
        probabilities = model_service.predict_proba(feature_inputs)

        iris_logger.info(
            "Prediction with probabilities",
            extra={
                "request_id": request.headers.get('X-Request-ID', str(time.time())),
                "predicted_labels": predicted_labels,
                "probabilities": probabilities
            }
        )

        return jsonify({
            "prediction": predicted_labels,
            "probabilities": probabilities
        })

    except Exception as e:
        iris_logger.error(
            "Prediction error",
            extra={
                "error": str(e),
                "request_id": request.headers.get('X-Request-ID', str(time.time()))
            }
        )
        return jsonify({"error": str(e)}), 500  # Internal Server Error

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint for monitoring.

    Args:
        None

    Returns:
        JSON response with structure:
        {
            "status": str,
            "timestamp": str,
            "service": str
        }
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "iris-prediction-api"
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)