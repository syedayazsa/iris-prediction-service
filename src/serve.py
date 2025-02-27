"""
Flask application exposing an endpoint for Iris model inference.
"""

import json
import os
from flask import Flask, request, jsonify
from src.model_service import IrisModelService
from datetime import datetime
from src.utils.logging_config import logger_config
import time

app = Flask(__name__)
loggers = logger_config.loggers
request_logger = logger_config.get_request_logger()

# Instantiate the model service at startup
model_service = IrisModelService(model_dir="src/models", model_name="iris_model")

@app.route("/predict", methods=["POST"])
@request_logger
def predict():
    data = request.get_json(force=True)
    feature_inputs = data.get("input")

    if not feature_inputs:
        loggers['error'].warning(
            "Invalid request: No input provided",
            extra={'request_id': request.headers.get('X-Request-ID', str(time.time()))}
        )
        return jsonify({"error": "No 'input' provided."}), 400

    try:
        if not isinstance(feature_inputs, list):
            return jsonify({"error": "Input must be a list of feature lists"}), 400
            
        if any(len(features) != 4 for features in feature_inputs):
            return jsonify({"error": "Each feature list must contain exactly 4 features"}), 400
            
        if any(not all(isinstance(x, (int, float)) for x in features) for features in feature_inputs):
            return jsonify({"error": "All features must be numeric values"}), 400

        predicted_labels = model_service.predict(feature_inputs)
        
        loggers['model'].info(
            "Prediction made",
            extra={
                'prediction_details': {
                    'input_size': len(feature_inputs),
                    'predictions': predicted_labels
                },
                'request_id': request.headers.get('X-Request-ID', str(time.time()))
            }
        )
        
        return jsonify({"prediction": predicted_labels})
        
    except Exception as e:
        loggers['error'].error(
            "Prediction error",
            extra={
                'error_details': str(e),
                'request_id': request.headers.get('X-Request-ID', str(time.time()))
            }
        )
        return jsonify({"error": str(e)}), 400

@app.route("/predict-proba", methods=["POST"])
@request_logger
def predict_proba():
    """
    Flask endpoint for predicting Iris species with probabilities.
    Expects a JSON body with a key 'input' containing a list of feature lists.

    Returns:
        JSON response containing both predicted labels and class probabilities.
    """
    data = request.get_json(force=True)
    feature_inputs = data.get("input")

    if not feature_inputs:
        return jsonify({"error": "No 'input' provided."}), 400

    predicted_labels = model_service.predict(feature_inputs)
    probabilities = model_service.predict_proba(feature_inputs)

    # Log prediction in structured JSON
    log_record = {
        "event": "prediction_with_probabilities",
        "inputs": feature_inputs,
        "predictions": predicted_labels,
        "probabilities": probabilities
    }
    loggers['model'].info(json.dumps(log_record))

    return jsonify({
        "prediction": predicted_labels,
        "probabilities": probabilities
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "iris-prediction-api"
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("GUNICORN_WORKERS", 4))
    threads = int(os.getenv("GUNICORN_THREADS", 2))
    timeout = int(os.getenv("GUNICORN_TIMEOUT", 30))
    
    app.run(host="0.0.0.0", port=port, debug=False)