"""
Module: evaluate.py
Description: Model evaluation metrics, confusion matrix, ROC curves,
and feature importance visualization.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Compute core classification metrics for a trained model.

    Parameters:
    ----------
    model : estimator
        Trained scikit-learn classification model.
    X_test : pd.DataFrame or np.ndarray
        Testing features.
    y_test : pd.Series or np.ndarray
        Ground truth labels.
    model_name : str
        Human-readable name of the model.

    Returns:
    -------
    dict
        Dictionary containing all evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Calculate probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        auc = roc_auc_score(y_test, y_scores)
    else:
        y_proba = None
        auc = np.nan

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc,
        "y_pred": y_pred,
        "y_proba": y_proba if hasattr(model, "predict_proba") else None,
    }
    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name, output_path="reports/figures/confusion_matrix.png"):
    """
    Plot and save a styled confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Standard (<7)", "High Quality (>=7)"],
        yticklabels=["Standard (<7)", "High Quality (>=7)"],
        ax=ax,
        annot_kws={"size": 14, "fontweight": "bold"}
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved Confusion Matrix: {output_path}")


def plot_roc_curves(models_dict, X_test, y_test, output_path="reports/figures/roc_curves.png"):
    """
    Plot and save ROC curves for all compared models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved ROC Curve: {output_path}")


def plot_feature_importance(model, feature_names, model_name, output_path="reports/figures/feature_importance.png"):
    """
    Plot and save feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"Model {model_name} does not have feature_importances_ attribute.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    df_imp = pd.DataFrame({
        "Feature": [feature_names[i] for i in sorted_idx],
        "Importance": [importances[i] for i in sorted_idx]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_imp, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False, ax=ax)
    ax.set_title(f"Feature Importance ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Gini Importance Score", fontsize=11)
    ax.set_ylabel("Physicochemical Property", fontsize=11)

    # Annotate values
    for i, imp in enumerate(df_imp["Importance"]):
        ax.text(imp + 0.003, i, f"{imp:.3f}", va="center", fontsize=9, fontweight="semibold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved Feature Importance: {output_path}")
    return df_imp
