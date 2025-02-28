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
- `serve.py`: Implements a Flask REST API for model inference. It provides an API for performing inference. It includes endpoints for predicting Iris species (/predict), returning class probabilities (/predict-proba), and a health check (/health). The application validates inputs, logs requests, and handles errors while running a preloaded model service. Also logs prediction requests, errors, and model inference details using a structured logging system for monitoring and debugging.
- `model_service.py`: Defines the IrisModelService class, which loads a trained RandomForest model for the Iris dataset. It provides methods to predict class labels (predict) and return class probabilities (predict_proba) based on input feature vectors. 
- `demo_gradio.py`: Provides a UI where users can input flower measurements via sliders and receive predictions from the API. The interface supports both single-label predictions and probability-based confidence scores.


### Testing Modules

- `test_api.py`: Holds integration tests validate the Flask API's endpoints by simulating real-world requests and checking responses. It uses pytest fixtures to create a temporary test model and a Flask test client for isolated testing. The tests cover valid and invalid input cases, boundary values, and API performance with large batches. Assertions verify correct responses, error handling, and consistency in predictions.


## CI/CD Pipeline

The project uses GitHub Actions for CI/CD. The pipeline includes:

- Testing with pytest
- Building and pushing the Docker image to Docker Hub

For modularity reason, each YAML file, `test-and-build.yml` and `publish.yml`, can focus on a specific task. The `test-and-build.yml` file is responsible for running tests and building the application, while the `publish.yml` file is dedicated to publishing the Docker image to Docker Hub. 

Each workflow can be triggered by different events. For example, the test-and-build.yml is triggered on every push or pull request on the `main` branch, while the publish.yml is triggered when a release is published.

If the test or build process fails, it won't affect the publish process ensuring that only tested and built images are published.

## Installation & Setup

1. Clone the repository:
```bash
git clone git@github.com:syedayazsa/iris-prediction-service.git
cd iris-prediction-service
```

2. Create and activate a virtual environment:

   **Using venv:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   **Using Conda:**
   ```bash
   conda create --name iris-env python=3.8
   conda activate iris-env
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
In production, we prefer Gunicorn is preferred since HTTP server can handle multiple requests concurrently, which helps in performance and responsiveness. It also provides better management of worker processes, allowing for efficient resource utilization and improved fault tolerance.
```bash
gunicorn --bind 0.0.0.0:8000 src.serve:app
```

### Running the Gradio Interface
You can playaround with the sliders and receive predictions from the API via a web interface.

```bash
python -m src.demo_gradio
```

### Docker Deployment

#### Local Docker Deployment

For simple local deployment with Docker:

1. Build the Docker image:
```bash
docker build -t iris-inference-server .
```

2. Run the Docker container:
```bash
docker run -p 8000:8000 \
  -e GUNICORN_WORKERS=4 \
  -e GUNICORN_THREADS=2 \
  -e FLASK_ENV=production \
  iris-inference-server
```

The container uses a Python 3.9 slim base image and runs the API server using Gunicorn with configurable workers and threads.

#### Production Deployment with Docker Compose

For production-like deployment using Docker Compose, which includes features like:
- Automatic container restart
- Health monitoring
- Log rotation
- Volume mounts for models and logs
- Environment variable configuration

1. Build and start the services:
```bash
docker-compose up --build -d
```

2. Check the service status (includes health check results):
```bash
docker-compose ps
```

3. View logs (JSON formatted with rotation):
```bash
docker-compose logs -f iris-api
```

4. Scale the service horizontally (if needed):
```bash
docker-compose up -d --scale iris-api=3
```

5. Stop the services:
```bash
docker-compose down
```

#### Environment Configuration

The service can be configured using the following environment variables:

- `PORT`: The port on which the API server listens (default: 8000)
- `GUNICORN_WORKERS`: Number of Gunicorn worker processes (default: 4)
- `GUNICORN_THREADS`: Number of threads per worker (default: 2)
- `GUNICORN_TIMEOUT`: Worker timeout in seconds (default: 30)
- `FLASK_ENV`: Flask environment setting (production/development)
- `LOG_LEVEL`: Logging level (default: INFO)

You can override these in docker-compose.yml or pass them directly to docker run:
```bash
docker run -p 8000:8000 \
  -e GUNICORN_WORKERS=8 \
  -e LOG_LEVEL=DEBUG \
  iris-inference-server
```

#### Container Health Monitoring

The service includes automatic health monitoring that:
- Checks the `/health` endpoint every 30 seconds
- Allows 10 seconds for each health check
- Retries 3 times before marking unhealthy
- Waits 10 seconds before starting checks on container startup
- Automatically restarts unhealthy containers

#### Volume Mounts

The Docker setup includes two important volume mounts:
- `./src/models:/app/src/models`: Persists trained models
- `./logs:/app/logs`: Stores application logs outside the container

## API Endpoints

The service provides three RESTful endpoints for model inference and health monitoring. All endpoints return JSON responses and include appropriate error handling.

### GET /health
Health check endpoint for monitoring the service status. Used by Docker for container health checks and general service monitoring.

**Sample Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Response format:**
```json
{
    "status": "healthy",
    "timestamp": "2024-03-14T12:00:00.000Z",
    "service": "iris-prediction-api"
}
```

### POST /predict
Predicts Iris species for given measurements. Accepts single or batch predictions.

**Sample Request:**
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[5.1, 3.5, 1.4, 0.2]]}'

# Batch prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}'
```

**Response format:**
```json
{
  "prediction": ["setosa"]  // Or multiple predictions for batch input
}
```

**Error Responses:**
- 400: Bad Request - Invalid input format (e.g., input is not a list)
- 415: Unsupported Media Type - Non-numeric features provided
- 422: Unprocessable Entity - Missing input data or wrong number of features (must be exactly 4)
- 500: Internal Server Error - Unexpected server-side errors during prediction

The error codes are chosen based on HTTP standards:
- 400 (Bad Request): Used when the request syntax is invalid
- 415 (Unsupported Media Type): Used when the input data format is not supported (non-numeric values)
- 422 (Unprocessable Entity): Used when the syntax is valid but the semantic validation fails
- 500 (Internal Server Error): Used for unexpected server-side errors

Each error response includes a JSON body with an "error" field containing a descriptive message:
```json
{
    "error": "Detailed error message explaining the issue"
}
```

### POST /predict-proba
Predicts Iris species with class probabilities for each input. Useful when confidence scores are needed.

**Sample Request:**
```bash
curl -X POST http://localhost:8000/predict-proba \
  -H "Content-Type: application/json" \
  -d '{"input": [[5.1, 3.5, 1.4, 0.2]]}'
```

**Response format:**
```json
{
  "prediction": ["setosa"],
  "probabilities": [
    [0.97, 0.02, 0.01]  // Probability for each class [setosa, versicolor, virginica]
  ]
}
```

### Request Headers

All endpoints support the following optional headers:
- `X-Request-ID`: Custom identifier for request tracking
- `Content-Type`: application/json (optional, as server always expects JSON data)

Using request headers can be beneficial for:
- **Tracking Requests**: The `X-Request-ID` header allows you to trace specific requests through logs, making it easier to debug issues or analyze performance.
- **API Documentation**: While the `Content-Type` header is not strictly required (the server always expects JSON), including it follows REST API best practices and makes the API usage more explicit.

**Sample Request with Headers:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: test-123" \
  -d '{"input": [[5.1, 3.5, 1.4, 0.2]]}'
```

## Testing

The project includes comprehensive integration tests that verify the API endpoints' functionality, error handling, and performance characteristics.

Run all tests:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_api.py
```

The test suite covers:
- **API Integration Tests**: Validates endpoint behavior with various inputs
- **Input Validation**: Tests handling of invalid, boundary, and malformed data
- **Batch Processing**: Verifies performance with large batches of predictions
- **Error Handling**: Ensures appropriate error responses for edge cases

### Run Tests with Coverage
To measure code coverage and generate a detailed report:
```bash
pytest --cov=src tests/
```

## Logging and Monitoring

The service implements comprehensive structured logging and monitoring capabilities to track model inference, performance metrics, and system health. The logging system is built using Python's built-in logging module with custom JSON formatting and rotating file handlers.

### Logging System

#### Log Types and Files
The service maintains separate log files for different concerns:
- `access.log`: API request access logs
- `error.log`: Error and warning messages
- `performance.log`: Request performance metrics
- `model.log`: Model inference details
- `security.log`: Security-related events

#### Structured JSON Logging
All logs are formatted as JSON objects with consistent fields:
```json
{
  "timestamp": "2024-03-14T12:00:00.000Z",
  "level": "INFO",
  "message": "Request processed: /predict",
  "module": "serve",
  "request_metrics": {
    "endpoint": "/predict",
    "method": "POST",
    "status_code": 200,
    "latency_ms": 45.23,
    "request_id": "1234-5678",
    "input_shape": [1, 4],
    "error": null
  }
}
```

#### Request Tracking
Each request is automatically logged with:
- Unique request ID (X-Request-ID header or timestamp if not provided)
- Endpoint path and HTTP method
- Remote address and user agent
- Response status code
- Request latency in milliseconds
- Input data shape for ML monitoring
- Error details (if any)

#### ML-Specific Metrics
The service tracks:
- Model prediction counts and results
- Input data shapes and distributions
- Prediction latencies
- Error rates and types
- Batch sizes
- Prediction probabilities (for /predict-proba endpoint)

### Monitoring Setup

#### Log Rotation
All log files are automatically rotated using Python's RotatingFileHandler:
- Maximum file size: 10MB (MAX_BYTES = 10485760)
- Backup count: 3 files
- Total maximum log storage: ~30MB per log type

#### Local Monitoring
Logs are written to both console (in development) and files:
```bash
# View real-time logs
docker-compose logs -f iris-api

# Check logs directory
ls -l logs/
```

#### Docker Configuration
The service uses Docker's json-file logging driver with rotation:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

#### Environment Variables
Logging can be configured via environment variables:
- `LOG_LEVEL`: Set logging level (default: INFO)
- `FLASK_ENV`: Environment setting affecting log verbosity and console output

### Health Monitoring

#### Health Check Endpoint
The service provides a `/health` endpoint that returns:
```json
{
    "status": "healthy",
    "timestamp": "2024-03-14T12:00:00.000Z",
    "service": "iris-prediction-api"
}
```

#### Docker Health Checks
The service includes Docker health monitoring that:
- Checks the `/health` endpoint every 30 seconds
- Times out after 10 seconds
- Retries 3 times before marking unhealthy
- Waits 10 seconds before starting checks on container startup
- Automatically restarts unhealthy containers

### Error Handling and Logging

The service implements comprehensive error handling with appropriate logging:

1. **Input Validation Errors**
   - Missing input data
   - Invalid feature counts
   - Non-numeric features
   - Malformed requests

2. **Model Inference Errors**
   - Model loading failures
   - Prediction errors
   - Shape mismatches

3. **System Errors**
   - Server errors
   - Resource constraints
   - Network issues

Each error is logged with:
- Error type and message
- Request context
- Stack trace (when appropriate)
- Request ID for tracing

### Request Tracing

To trace specific requests through the system:

1. Add an X-Request-ID header to your request:
```bash
curl -H "X-Request-ID: test-123" \
     -X POST http://localhost:8000/predict \
     -d '{"input": [[5.1, 3.5, 1.4, 0.2]]}'
```

2. Use the request ID to follow the request through different log files:
```bash
grep "test-123" logs/*.log
```

### Monitoring Best Practices

1. **Regular Log Review**
   - Monitor error rates and patterns
   - Track prediction latencies
   - Review input distributions
   - Check system health status

2. **Alert Configuration**
   - Set up alerts for high error rates
   - Monitor latency thresholds
   - Track resource utilization
   - Watch for repeated health check failures

3. **Log Management**
   - Regular log rotation
   - Backup of important logs
   - Periodic log analysis
   - Storage monitoring

4. **Performance Monitoring**
   - Track request latencies
   - Monitor concurrent requests
   - Watch resource usage
   - Analyze batch processing performance

## Code Style
- Follows **PEP 8** guidelines.
- Uses **type hints**.
- Includes **docstrings** for all functions and classes.
