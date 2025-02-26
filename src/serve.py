"""
Flask application exposing an endpoint for Iris model inference.
"""

import json
import logging
from flask import Flask, request, jsonify
from src.model_service import IrisModelService

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Instantiate the model service at startup
model_service = IrisModelService(model_dir="src/models", model_name="iris_model")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    feature_inputs = data.get("input")

    if not feature_inputs:
        return jsonify({"error": "No 'input' provided."}), 400

    try:
        if not isinstance(feature_inputs, list):
            return jsonify({"error": "Input must be a list of feature lists"}), 400
            
        if any(len(features) != 4 for features in feature_inputs):
            return jsonify({"error": "Each feature list must contain exactly 4 features"}), 400
            
        if any(not all(isinstance(x, (int, float)) for x in features) for features in feature_inputs):
            return jsonify({"error": "All features must be numeric values"}), 400

        predicted_labels = model_service.predict(feature_inputs)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Log prediction in structured JSON
    log_record = {
        "event": "prediction",
        "inputs": feature_inputs,
        "predictions": predicted_labels
    }
    logging.info(json.dumps(log_record))

    return jsonify({"prediction": predicted_labels})


@app.route("/predict-proba", methods=["POST"])
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
    logging.info(json.dumps(log_record))

    return jsonify({
        "prediction": predicted_labels,
        "probabilities": probabilities
    })


if __name__ == "__main__":
    # For local dev, can run "python serve.py" directly.
    # In production, we usually run via Gunicorn (see Dockerfile).
    app.run(host="0.0.0.0", port=8000, debug=False)