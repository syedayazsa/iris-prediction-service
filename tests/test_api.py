"""
Integration tests for the Flask API endpoints.
"""

from pathlib import Path

import numpy as np
import pytest

from src.serve import app
from src.train import train_and_save_model


@pytest.fixture(scope="session")
def test_model():
    """
    Pytest fixture to ensure we have a model for testing.
    Creates a test model in a separate directory.
    """
    test_model_dir = Path("test_models")
    test_model_dir.mkdir(exist_ok=True)
    train_and_save_model(
        model_dir=str(test_model_dir),
        model_name="test_iris_model",
        random_state=42
    )
    yield test_model_dir
    # Cleanup after tests
    for file in test_model_dir.glob("*"):
        file.unlink()
    test_model_dir.rmdir()

@pytest.fixture
def client(test_model):
    """
    Pytest fixture to create a Flask test client.
    
    Args:
        test_model: The test model fixture.
    """
    app.config['TESTING'] = True
    with app.test_client() as testing_client:
        yield testing_client

def test_prediction_endpoint_valid_input(client):
    """
    Test if the /predict endpoint returns correct predictions for valid input.
    """
    # Test case 1: Single sample (likely setosa)
    sample_input = {"input": [[5.1, 3.5, 1.4, 0.2]]}
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    response_data = response.get_json()
    assert "prediction" in response_data
    assert response_data["prediction"][0] in ["setosa", "versicolor", "virginica"]

    # Test case 2: Multiple samples
    multi_input = {"input": [
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.1, 4.4, 1.4],
        [6.3, 3.3, 6.0, 2.5]
    ]}
    response = client.post("/predict", json=multi_input)
    assert response.status_code == 200
    predictions = response.get_json()["prediction"]
    assert len(predictions) == 3
    assert all(pred in ["setosa", "versicolor", "virginica"] for pred in predictions)

def test_prediction_endpoint_invalid_input(client):
    """
    Test if the /predict endpoint handles invalid inputs appropriately with correct status codes.
    """
    # Test case 1: Empty input
    response = client.post("/predict", json={"input": []})
    assert response.status_code == 422  # Unprocessable Entity
    assert "error" in response.get_json()

    # Test case 2: Missing input key
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity
    assert "error" in response.get_json()

    # Test case 3: Invalid input format (not a list)
    response = client.post("/predict", json={"input": "not a list"})
    assert response.status_code == 400  # Bad Request
    assert "error" in response.get_json()

    # Test case 4: Wrong number of features
    response = client.post("/predict", json={"input": [[1.0, 2.0, 3.0]]})
    assert response.status_code == 422  # Unprocessable Entity
    assert "error" in response.get_json()

    # Test case 5: Non-numeric features
    response = client.post("/predict", json={"input": [["a", "b", "c", "d"]]})
    assert response.status_code == 415  # Unsupported Media Type
    assert "error" in response.get_json()

def test_prediction_proba_endpoint_invalid_input(client):
    """
    Test if the /predict-proba endpoint handles invalid inputs appropriately with correct status codes.
    """
    # Test case 1: Empty input
    response = client.post("/predict-proba", json={"input": []})
    assert response.status_code == 422  # Unprocessable Entity
    assert "error" in response.get_json()

    # Test case 2: Missing input key
    response = client.post("/predict-proba", json={})
    assert response.status_code == 422  # Unprocessable Entity
    assert "error" in response.get_json()

    # Test case 3: Invalid input format (not a list)
    response = client.post("/predict-proba", json={"input": "not a list"})
    assert response.status_code == 400  # Bad Request
    assert "error" in response.get_json()

    # Test case 4: Wrong number of features
    response = client.post("/predict-proba", json={"input": [[1.0, 2.0, 3.0]]})
    assert response.status_code == 422  # Unprocessable Entity
    assert "error" in response.get_json()

    # Test case 5: Non-numeric features
    response = client.post("/predict-proba", json={"input": [["a", "b", "c", "d"]]})
    assert response.status_code == 415  # Unsupported Media Type
    assert "error" in response.get_json()

def test_prediction_endpoint_boundary_values(client):
    """
    Test if the /predict endpoint handles boundary values correctly.
    """
    # Test case 1: Zero values
    response = client.post("/predict", json={"input": [[0.0, 0.0, 0.0, 0.0]]})
    assert response.status_code == 200

    # Test case 2: Very large values
    response = client.post("/predict", json={"input": [[100.0, 100.0, 100.0, 100.0]]})
    assert response.status_code == 200

    # Test case 3: Negative values
    response = client.post("/predict", json={"input": [[-1.0, -1.0, -1.0, -1.0]]})
    assert response.status_code == 200

def test_prediction_endpoint_performance(client):
    """
    Test the performance of the /predict endpoint with larger batches.
    """
    # Generate a large batch of random inputs
    np.random.seed(42)
    n_samples = 100
    batch_input = {
        "input": np.random.rand(n_samples, 4).tolist()
    }
    
    response = client.post("/predict", json=batch_input)
    assert response.status_code == 200
    predictions = response.get_json()["prediction"]
    assert len(predictions) == n_samples

def test_health_endpoint(client):
    """
    Test if the health check endpoint returns correct response.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "service" in data