"""
Module: model_trainer.py
Description: Trains and compares multiple machine learning models for wine quality prediction,
evaluates them rigorously, logs real metrics, and saves the best model.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from src.preprocessing import prepare_wine_data
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance
)


def train_and_compare_models(wine_type="both", quality_threshold=7):
    """
    Train multiple classification models, evaluate their performance,
    and persist the best performing model to disk.

    Parameters:
    ----------
    wine_type : str
        'both', 'red', or 'white'.
    quality_threshold : int
        Quality score cutoff for High Quality vs Standard (default: 7).

    Returns:
    -------
    tuple:
        (best_model_name, best_model, comparison_df, scaler, feature_names)
    """
    print("=" * 65)
    print("STARTING MODEL TRAINING & EVALUATION PIPELINE")
    print(f"Dataset: Wine Type = '{wine_type}', Quality Threshold = {quality_threshold}")
    print("=" * 65)

    # 1. Preprocess data
    data = prepare_wine_data(wine_type=wine_type, quality_threshold=quality_threshold)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    scaler = data["scaler"]

    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    print(f"Class Balance (Training): Standard={sum(y_train==0)}, High Quality={sum(y_train==1)}")

    # 2. Define Candidate Models
    # We include Logistic Regression (baseline), Decision Tree, Random Forest,
    # Gradient Boosting, and Support Vector Machine.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(SVC(random_state=42), cv=3),
    }

    results = []
    trained_models = {}

    # 3. Train and Evaluate each model
    for name, model in models.items():
        print(f"\n--> Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append({
            "Model": metrics["Model"],
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"],
        })
        print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"    Precision: {metrics['Precision']:.4f}")
        print(f"    Recall:    {metrics['Recall']:.4f}")
        print(f"    F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"    ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    # 4. Create and save model comparison dataframe
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    os.makedirs("reports", exist_ok=True)
    comparison_path = "reports/model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModel comparison table saved to: {comparison_path}")

    # 5. Identify the Best Model (based on F1-Score on the imbalanced high-quality class)
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 65)
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"F1-Score: {best_row['F1-Score']:.4f} | Accuracy: {best_row['Accuracy']:.4f} | ROC-AUC: {best_row['ROC-AUC']:.4f}")
    print("=" * 65)

    # 6. Generate and save evaluation visual plots
    best_y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, best_y_pred, best_model_name)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_feature_importance(best_model, feature_names, best_model_name)

    # 7. Save Classification Report
    clf_report = classification_report(
        y_test, 
        best_y_pred, 
        target_names=["Standard (<7)", "High Quality (>=7)"]
    )
    report_file = "reports/best_model_classification_report.txt"
    with open(report_file, "w") as f:
        f.write(f"BEST MODEL: {best_model_name}\n\n")
        f.write(clf_report)
    print(f"Saved classification report to: {report_file}")

    # 8. Save Best Model, Scaler, and Metadata
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/best_model.joblib"
    scaler_save_path = "models/scaler.joblib"
    metadata_path = "models/model_metadata.json"

    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    metadata = {
        "best_model_name": best_model_name,
        "wine_type": wine_type,
        "quality_threshold": quality_threshold,
        "feature_names": feature_names,
        "metrics": {
            "accuracy": float(best_row["Accuracy"]),
            "precision": float(best_row["Precision"]),
            "recall": float(best_row["Recall"]),
            "f1_score": float(best_row["F1-Score"]),
            "roc_auc": float(best_row["ROC-AUC"]),
        },
        "all_models_summary": comparison_df.to_dict(orient="records"),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved best model to:   {model_save_path}")
    print(f"Saved scaler to:       {scaler_save_path}")
    print(f"Saved metadata to:     {metadata_path}")
    print("=" * 65)

    return best_model_name, best_model, comparison_df, scaler, feature_names


if __name__ == "__main__":
    train_and_compare_models(wine_type="both", quality_threshold=7)
