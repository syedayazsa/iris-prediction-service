"""
Logging configuration for the Iris prediction service.
"""

import logging
import json
import os
from datetime import datetime
from typing import Any, Dict
from functools import wraps
import time
from dataclasses import dataclass, asdict
from flask import request, Response, has_request_context
from logging.handlers import RotatingFileHandler

class LoggerConfig:
    """Class to manage logging configuration and setup."""
    
    LOGS_DIR = "logs"
    MAX_BYTES = 10485760  # 10MB
    BACKUP_COUNT = 3
    
    def __init__(self):
        """Initialize logging configuration."""
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        self.loggers = self._setup_loggers()
    
    def _setup_file_handler(self, filename: str) -> RotatingFileHandler:
        """Setup a rotating file handler with JSON formatting."""
        handler = RotatingFileHandler(
            os.path.join(self.LOGS_DIR, filename),
            maxBytes=self.MAX_BYTES,
            backupCount=self.BACKUP_COUNT
        )
        handler.setFormatter(JsonFormatter())
        return handler
    
    def _setup_loggers(self) -> Dict[str, logging.Logger]:
        """Configure and return multiple loggers."""
        logger_names = {
            'access': 'access.log',
            'error': 'error.log',
            'performance': 'performance.log',
            'model': 'model.log',
            'security': 'security.log'
        }
        
        loggers = {
            name: logging.getLogger(f'iris_prediction_service.{name}')
            for name in logger_names
        }
        
        for name, logger in loggers.items():
            logger.setLevel(logging.INFO)
            logger.addHandler(self._setup_file_handler(logger_names[name]))
            
            if os.getenv('FLASK_ENV') == 'development':
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(JsonFormatter())
                logger.addHandler(console_handler)
        
        return loggers

    def get_request_logger(self):
        """Return a configured RequestLogger instance."""
        return RequestLogger(self.loggers)

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "extra"):
            log_object.update(record.extra)
        return json.dumps(log_object)

@dataclass
class RequestMetrics:
    """Data class for tracking request metrics."""
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    request_id: str
    input_shape: tuple = None
    error: str = None

class RequestLogger:
    """Class to handle request logging."""
    
    def __init__(self, loggers: Dict[str, logging.Logger]):
        """Initialize with configured loggers."""
        self.loggers = loggers
    
    def __call__(self, func):
        """Make the class callable as a decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            
            try:
                response = func(*args, **kwargs)
                # Handle both tuple returns (response, status_code) and direct response objects
                if isinstance(response, tuple):
                    status_code = response[1]
                    response_obj = response[0]
                else:
                    status_code = response.status_code
                    response_obj = response
                    
            except Exception as e:
                status_code = 500
                error = str(e)
                if has_request_context():
                    self._log_error(request.headers.get('X-Request-ID', str(time.time())), str(e))
                raise
            finally:
                if has_request_context():
                    request_id = request.headers.get('X-Request-ID', str(time.time()))
                    metrics = self._create_metrics(start_time, status_code, error, request_id)
                    self._log_metrics(metrics)
                    self._log_access(request_id)
            
            return response
        return wrapper
    
    def _log_access(self, request_id: str):
        """Log access information."""
        self.loggers['access'].info(
            "Request received",
            extra={
                'request_id': request_id,
                'endpoint': request.path,
                'method': request.method,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent')
            }
        )
    
    def _log_error(self, request_id: str, error: str):
        """Log error information."""
        self.loggers['error'].error(
            "Request failed",
            extra={
                'request_id': request_id,
                'error': error,
                'endpoint': request.path,
                'method': request.method
            }
        )
    
    def _create_metrics(self, start_time: float, status_code: int, 
                       error: str, request_id: str) -> RequestMetrics:
        """Create request metrics."""
        latency = (time.time() - start_time) * 1000
        input_shape = self._get_input_shape()
        
        return RequestMetrics(
            endpoint=request.path,
            method=request.method,
            status_code=status_code,
            latency_ms=latency,
            request_id=request_id,
            input_shape=input_shape,
            error=error
        )
    
    def _get_input_shape(self) -> tuple:
        """Get shape of input data if available."""
        if request.json and "input" in request.json:
            input_data = request.json["input"]
            return (len(input_data), len(input_data[0])) if input_data else None
        return None
    
    def _log_metrics(self, metrics: RequestMetrics):
        """Log various metrics."""
        self._log_performance_metrics(metrics)
        if request.path in ['/predict', '/predict-proba']:
            self._log_model_metrics(metrics)
    
    def _log_performance_metrics(self, metrics: RequestMetrics):
        """Log performance-related metrics."""
        self.loggers['performance'].info(
            "Request performance metrics",
            extra={
                'request_metrics': asdict(metrics),
                'system_metrics': {
                    'latency_ms': metrics.latency_ms,
                    'status_code': metrics.status_code
                }
            }
        )
    
    def _log_model_metrics(self, metrics: RequestMetrics):
        """Log model-specific metrics."""
        self.loggers['model'].info(
            "Model inference metrics",
            extra={
                'request_id': metrics.request_id,
                'input_shape': metrics.input_shape,
                'latency_ms': metrics.latency_ms,
                'endpoint': metrics.endpoint
            }
        )

# Initialize logger configuration
logger_config = LoggerConfig()
# Create request logger decorator
log_request = logger_config.get_request_logger()
