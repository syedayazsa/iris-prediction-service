"""
Core package for the Iris species prediction service.
"""

from .model_service import IrisModelService
from .demo_gradio import GradioIrisDemo

__all__ = ['IrisModelService', 'GradioIrisDemo']