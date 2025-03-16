"""
Train a RandomForest model on the Iris dataset and save it locally.
"""

import json
from pathlib import Path
from typing import Union

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_and_save_model(
    model_dir: Union[str, Path] = "models",
    model_name: str = "iris_model",
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Train a RandomForest on the Iris dataset, evaluate it, and save the model.
    
    Args:
        model_dir (Union[str, Path]): Directory to save model artifacts.
        model_name (str): Name of the model (used for files).
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        None. The function saves the following files:
        - {model_name}.joblib: The trained model
        - {model_name}_metadata.json: Model metadata including metrics and feature importance
    """
    # Create model directory if it doesn't exist
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Define paths for model artifacts
    model_path = model_dir / f"{model_name}.joblib"
    metadata_path = model_dir / f"{model_name}_metadata.json"

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train the model
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    
    # Generate classification report
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=target_names,
        output_dict=True
    )
    
    # Print formatted classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Feature importance analysis
    feature_importance = dict(zip(feature_names, clf.feature_importances_))
    print("\nFeature Importances:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")

    # Save model and metadata
    model_info = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "metrics": report,
        "feature_importance": feature_importance,
        "train_samples": len(y_train),
        "test_samples": len(y_test)
    }
    
    # Save model and metadata
    joblib.dump(clf, model_path)
    with open(metadata_path, "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model saved as {model_path}")
    print(f"Metadata saved as {metadata_path}")

if __name__ == "__main__":
    train_and_save_model()