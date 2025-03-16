"""
Core package for the Iris species prediction service.
"""

from .app import GradioIrisDemo
from .model_service import IrisModelService

__all__ = ['IrisModelService', 'GradioIrisDemo']