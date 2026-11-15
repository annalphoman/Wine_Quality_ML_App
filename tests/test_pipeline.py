"""
Unit and Integration Tests for Wine Quality Prediction Pipeline
Run with: python tests/test_pipeline.py
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import load_wine_data, get_data_summary
from src.preprocessing import prepare_wine_data


def test_data_loader():
    print("Testing data_loader.py...")
    red = load_wine_data("red")
    white = load_wine_data("white")
    both = load_wine_data("both")

    assert len(red) == 1599, f"Expected 1599 red rows, got {len(red)}"
    assert len(white) == 4898, f"Expected 4898 white rows, got {len(white)}"
    assert len(both) == 6497, f"Expected 6497 combined rows, got {len(both)}"
    assert both.isnull().sum().sum() == 0, "Missing values found in combined dataset!"
    assert "quality" in both.columns, "'quality' column missing!"
    print("  [PASSED] data_loader.py works correctly.")


def test_preprocessing():
    print("Testing preprocessing.py...")
    data = prepare_wine_data(wine_type="both", quality_threshold=7, test_size=0.2, random_state=42)
    
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    scaler = data["scaler"]

    assert len(X_train) == 5197, f"Expected 5197 train rows, got {len(X_train)}"
    assert len(X_test) == 1300, f"Expected 1300 test rows, got {len(X_test)}"
    assert len(y_train) == 5197, f"Expected 5197 train targets, got {len(y_train)}"
    assert len(y_test) == 1300, f"Expected 1300 test targets, got {len(y_test)}"
    assert scaler is not None, "Scaler is None!"
    assert set(y_train.unique()).issubset({0, 1}), "Target values must be binary (0 or 1)"
    print("  [PASSED] preprocessing.py works correctly.")


def test_saved_model_artifacts():
    print("Testing saved model artifacts in models/...")
    model_path = os.path.join(project_root, "models", "best_model.joblib")
    scaler_path = os.path.join(project_root, "models", "scaler.joblib")
    meta_path = os.path.join(project_root, "models", "model_metadata.json")

    assert os.path.exists(model_path), f"Missing {model_path}"
    assert os.path.exists(scaler_path), f"Missing {scaler_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert hasattr(model, "predict"), "Loaded model does not have predict() method"
    assert hasattr(scaler, "transform"), "Loaded scaler does not have transform() method"
    assert "best_model_name" in meta, "Metadata missing best_model_name"
    assert meta["metrics"]["accuracy"] > 0.85, f"Expected accuracy > 0.85, got {meta['metrics']['accuracy']}"
    print(f"  [PASSED] Model artifacts verified ({meta['best_model_name']} - Acc: {meta['metrics']['accuracy']:.1%}).")


def test_end_to_end_prediction():
    print("Testing end-to-end inference prediction...")
    model = joblib.load(os.path.join(project_root, "models", "best_model.joblib"))
    scaler = joblib.load(os.path.join(project_root, "models", "scaler.joblib"))

    # Sample wine input with all 12 expected features
    sample_input = pd.DataFrame([{
        "fixed acidity": 8.0,
        "volatile acidity": 0.30,
        "citric acid": 0.35,
        "residual sugar": 3.0,
        "chlorides": 0.05,
        "free sulfur dioxide": 20.0,
        "total sulfur dioxide": 60.0,
        "density": 0.9950,
        "pH": 3.30,
        "sulphates": 0.65,
        "alcohol": 12.0,
        "is_white": 0
    }])

    # Transform preserving feature column names to prevent sklearn warnings
    scaled_input = pd.DataFrame(scaler.transform(sample_input), columns=sample_input.columns)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    assert prediction in [0, 1], f"Prediction {prediction} is not binary"
    assert len(probabilities) == 2, "Probabilities should have 2 classes"
    assert 0.0 <= probabilities[0] <= 1.0, "Probability out of range"
    assert 0.0 <= probabilities[1] <= 1.0, "Probability out of range"
    assert np.isclose(sum(probabilities), 1.0), "Probabilities must sum to 1.0"
    print(f"  [PASSED] Prediction output: class={prediction}, prob_high_quality={probabilities[1]:.2%}")


def test_generated_reports_and_figures():
    print("Testing reports and figures...")
    expected_figures = [
        "quality_distribution.png",
        "correlation_heatmap.png",
        "key_features_boxplots.png",
        "feature_distributions.png",
        "confusion_matrix.png",
        "roc_curves.png",
        "feature_importance.png",
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(project_root, "reports", "figures", fig_name)
        assert os.path.exists(fig_path), f"Missing figure: {fig_path}"
        assert os.path.getsize(fig_path) > 1000, f"Figure file is suspiciously small: {fig_path}"

    assert os.path.exists(os.path.join(project_root, "reports", "model_comparison.csv")), "Missing model_comparison.csv"
    assert os.path.exists(os.path.join(project_root, "reports", "eda_summary.txt")), "Missing eda_summary.txt"
    print("  [PASSED] All 7 figures and reports exist and are populated.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PIPELINE TEST SUITE")
    print("=" * 60)
    test_data_loader()
    test_preprocessing()
    test_saved_model_artifacts()
    test_end_to_end_prediction()
    test_generated_reports_and_figures()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (5/5 suites)")
    print("=" * 60)
