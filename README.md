# Iris Species Prediction Service
A production-ready machine learning service that predicts Iris flower species using a RandomForest classifier. The service includes a REST API, a Gradio web interface, and comprehensive tests.

## Project Structure

```bash
iris-prediction-service/
├── src/
│   ├── __init__.py
│   ├── train.py            # Model training script
│   ├── serve.py            # Flask API server
│   ├── model_service.py    # Model inference service
│   ├── demo_gradio.py      # Gradio web interface
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logging_config.py
├── models/                 # Directory for saved models
│   ├── iris_model.joblib
│   └── iris_model_metadata.json
├── tests/                  # Tests for API and model
│   ├── __init__.py
│   ├── test_api.py         # API integration tests
│   └── test_model_service.py  # Model service unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Module Descriptions

### Core Modules

- `train.py`: Trains a RandomForest classifier on the Iris dataset and saves both the model and its metadata
- `serve.py`: Implements a Flask REST API for model inference
- `model_service.py`: Handles model loading and prediction logic
- `demo_gradio.py`: Provides a user-friendly web interface using Gradio

### Testing Modules

- `test_api.py`: Integration tests for the Flask API endpoints
- `test_model_service.py`: Unit tests for the model service

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iris-prediction-service.git
cd iris-prediction-service
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the model using default parameters:
```bash
python -m src.train
```

To specify model directory, model name, and test size:

```bash
python -m src.train --model-dir models --model-name custom_model --test-size 0.3
```

### Running the API Server

#### Local Development

```bash
python -m src.serve
```

#### Production (using Gunicorn)

```bash
gunicorn --bind 0.0.0.0:5000 src.serve:app
```

### Running the Gradio Interface

```bash
python -m src.demo_gradio
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t iris-inference-server .
```

2. Run the Docker container:
```bash
docker run -p 5000:5000 iris-inference-server
```

3. or using Docker Compose:
```bash
docker-compose up
```

## API Endpoints

### POST /predict
Predicts Iris species for given measurements.

Request format:
```json
{
  "input": [[5.1, 3.5, 1.4, 0.2]]  // Single prediction
  // Add more arrays for batch prediction
}
```

Response format:
```json
{
  "prediction": ["setosa"]  // Or multiple predictions for batch input
}
```


## Testing

Run all tests:

```bash
pytest tests/
```

Run specific test files:

```bash
pytest tests/test_api.py
pytest tests/test_model_service.py
```

### Run Tests with Coverage
```bash
pytest --cov=src tests/
```

### Code Style
- Follows **PEP 8** guidelines.
- Uses   **type hints**.
- Includes **docstrings** for all functions and classes.