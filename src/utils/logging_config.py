"""
A simplified logging setup for the Iris prediction service.
Logs everything to stdout in JSON format and captures both 4xx and 5xx errors.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from typing import Optional

from flask import has_request_context, request


@dataclass
class RequestMetrics:
    """
    Track essential request details:
    - endpoint, method, status_code, latency, etc.
    - error is optional, set if an exception occurs or a 4xx/5xx response is sent.
    """
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    request_id: str
    error: Optional[str] = None

class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter that captures both standard log fields
    and any extra fields passed in the log call.
    """
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record (logging.LogRecord): The log record to format

        Returns:
            str: JSON-formatted log entry
        """
        # Base fields for the log
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        
        # These attributes are used internally by Python's logging;
        # we typically don't want them verbatim in the output.
        base_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process', 'message'
        }
        
        # Merge any "extra" fields or user-defined attributes.
        for key, value in record.__dict__.items():
            if key not in base_attrs:
                log_object[key] = value
        
        return json.dumps(log_object)

# Create a single logger for the entire app
iris_logger = logging.getLogger("iris_prediction_service")
iris_logger.setLevel(logging.INFO)

# Configure it to log to stdout in JSON
console_handler = logging.StreamHandler()
console_handler.setFormatter(JsonFormatter())
iris_logger.addHandler(console_handler)

def log_request(func):
    """
    Decorator to measure request latency and log errors & metrics.
    Captures and logs all 4xx or 5xx status codes, not just 500 exceptions.

    Args:
        func (callable): The function to wrap

    Returns:
        callable: The wrapped function that includes logging
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error_msg = None
        status_code = 200  # Default (assume success unless changed)
        
        try:
            response = func(*args, **kwargs)
            # If user returns (response, status_code), capture that
            if isinstance(response, tuple):
                status_code = response[1]
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            # Log unhandled exceptions
            iris_logger.error("Unhandled exception", extra={"error": error_msg})
            raise
        finally:
            if has_request_context():
                request_id = request.headers.get('X-Request-ID', str(time.time()))
                latency_ms = (time.time() - start_time) * 1000

                # If the function or route returned a 4xx or 5xx code, log it as an error.
                if status_code >= 400 and error_msg is None:
                    # Possibly there's a message in your route's JSON, but we don't parse it here.
                    error_msg = f"Request returned status code {status_code}"
                    iris_logger.error(
                        "Non-success response",
                        extra={"status_code": status_code, "request_id": request_id}
                    )
                
                # Construct final metrics to log
                metrics = RequestMetrics(
                    endpoint=request.path,
                    method=request.method,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_id=request_id,
                    error=error_msg
                )
                
                # Log request metrics (info-level)
                iris_logger.info(
                    "Request completed",
                    extra={"request_metrics": asdict(metrics)}
                )
                
        return response
    return wrapper