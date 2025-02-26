"""
Module that defines a class to handle loading and inference of the Iris model.
"""

import joblib
from typing import List
from pathlib import Path


class IrisModelService:
    """
    A service class to load a trained Iris RandomForest model and perform predictions.
    """

    def __init__(self, model_dir: str = "models", model_name: str = "iris_model") -> None:
        """
        Initialize the model service.

        Args:
            model_dir (str): Directory containing model artifacts.
            model_name (str): Name of the model files.
        """
        model_path = Path(model_dir) / f"{model_name}.joblib"
        self._model = joblib.load(model_path)
        self._class_names = ["setosa", "versicolor", "virginica"]

    def predict(self, inputs: List[List[float]]) -> List[str]:
        """
        Perform inference using the loaded RandomForest model.

        Args:
            inputs: A list of feature vectors, each containing 4 float values.
        
        Returns:
            List[str]: A list of predicted Iris class names.
        """
        predictions = self._model.predict(inputs)
        predicted_labels = [self._class_names[idx] for idx in predictions]
        return predicted_labels