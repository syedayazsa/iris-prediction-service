"""
Core package for the Iris species prediction service.
"""

from .model_service import IrisModelService
from .app import GradioIrisDemo

__all__ = ['IrisModelService', 'GradioIrisDemo']