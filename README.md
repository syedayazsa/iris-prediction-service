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

## CI/CD Pipeline

The project uses GitHub Actions for CI/CD. The pipeline includes:

- Testing with pytest
- Building and pushing the Docker image to Docker Hub

For modularity reason, each YAML file, `test-and-build.yml` and `publish.yml`, can focus on a specific task. The `test-and-build.yml` file is responsible for running tests and building the application, while the `publish.yml` file is dedicated to publishing the Docker image to Docker Hub. 

Each workflow can be triggered by different events. For example, the test-and-build.yml might be triggered on every push or pull request, while the publish.yml is triggered only when a release is published. This allows for more control over when certain actions are taken.

If the test or build process fails, it won't affect the publish process. This ensures that only tested and built images are published.

## Installation & Setup

1. Clone the repository:
```bash
git clone git@github.com:syedayazsa/iris-prediction-service.git
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
gunicorn --bind 0.0.0.0:8000 src.serve:app
```

### Running the Gradio Interface

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
docker run -p 8000:8000 iris-inference-server
```

#### Production Deployment with Docker Compose

For production-like deployment using Docker Compose:

1. Build and start the services:
```bash
docker-compose up --build -d
```

2. Check the service status:
```bash
docker-compose ps
```

3. View logs:
```bash
docker-compose logs -f iris-api
```

4. Scale the service (if needed):
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

#### Health Monitoring

The service includes a health check endpoint at `/health` that returns:
- Current service status
- Timestamp
- Service identification

This endpoint is used by Docker for container health monitoring.

## API Endpoints

### GET /health
Health check endpoint for monitoring.

Response format:
```json
{
    "status": "healthy",
    "timestamp": "2024-03-14T12:00:00.000Z",
    "service": "iris-prediction-api"
}
```

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

## Logging and Monitoring

The service implements comprehensive structured logging and monitoring capabilities to track model inference, performance metrics, and system health.

### Logging System

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
- Unique request ID (X-Request-ID header)
- Endpoint path and HTTP method
- Response status code
- Request latency in milliseconds
- Input data shape for ML monitoring
- Error details (if any)

#### ML-Specific Metrics
The service tracks:
- Model prediction counts
- Input data distributions
- Prediction latencies
- Error rates and types
- Batch sizes and shapes

### Monitoring Setup

#### Local Monitoring
Logs are written to both console and files:
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
- `FLASK_ENV`: Environment setting affecting log verbosity

### Integration with Monitoring Tools

The structured JSON logs can be easily integrated with:

#### ELK Stack (Elasticsearch, Logstash, Kibana)
1. **Logstash**: Configure Logstash to read logs from the Docker container or log files. Use the `json` codec to parse the structured logs.
   ```bash
   input {
     file {
       path => "/path/to/logs/*.log"
       start_position => "beginning"
       codec => "json"
     }
   }
   ```

2. **Elasticsearch**: Send the parsed logs to Elasticsearch for storage and indexing.

3. **Kibana**: Use Kibana to visualize the logs and create dashboards for monitoring request metrics, error rates, and system health.

#### Prometheus and Grafana
1. **Prometheus**: Use a Prometheus client library to expose metrics from your Flask application. Create a `/metrics` endpoint that Prometheus can scrape.
   ```python
   from prometheus_flask_exporter import PrometheusMetrics

   metrics = PrometheusMetrics(app)
   ```

2. **Grafana**: Connect Grafana to your Prometheus instance to visualize the metrics. Create dashboards to monitor request latencies, error rates, and other performance metrics.

#### Example Prometheus Metrics
You can track metrics such as:
- Total requests
- Successful requests
- Failed requests
- Request latencies
- Custom application metrics

### Log Directory Structure
```bash
logs/
├── access.log    # API request logs
├── error.log     # Error and exception logs
└── metrics.log   # Performance and ML metrics
```

### Request Tracing

To trace specific requests:
1. Add an X-Request-ID header to your request
2. Use the request ID to follow the request through logs:
```bash
curl -H "X-Request-ID: test-123" -X POST http://localhost:8000/predict -d '{"input": [[5.1, 3.5, 1.4, 0.2]]}'
```

### Monitoring Best Practices

1. **Regular Metrics Review**
   - Monitor prediction latencies
   - Track error rates
   - Review input data distributions

2. **Alert Configuration**
   - Set up alerts for high error rates
   - Monitor latency thresholds
   - Track resource utilization

3. **Log Retention**
   - Logs are rotated every 10MB
   - Keep last 3 log files
   - Archive older logs if needed

4. **Performance Monitoring**
   - Track request latencies
   - Monitor concurrent requests
   - Watch resource usage