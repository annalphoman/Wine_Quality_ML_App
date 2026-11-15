"""
Module: preprocessing.py
Description: Data preprocessing, feature scaling, and train-test splitting
for Wine Quality prediction.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_wine_data


def prepare_wine_data(
    wine_type="both",
    quality_threshold=7,
    test_size=0.2,
    random_state=42,
    scale_features=True,
    drop_duplicates=False
):
    """
    Load, preprocess, and split the wine dataset.

    Parameters:
    ----------
    wine_type : str
        'red', 'white', or 'both'.
    quality_threshold : int
        Wines with quality >= threshold are labeled as High Quality (1), else Standard (0).
        Standard threshold is 7 (classifies scores 7, 8, 9 as Good/High Quality).
    test_size : float
        Proportion of dataset to include in the test split (default: 0.2).
    random_state : int
        Seed for reproducibility (default: 42).
    scale_features : bool
        Whether to standardize features with StandardScaler (default: True).
    drop_duplicates : bool
        Whether to drop duplicate samples (default: False, to preserve full sample distribution).

    Returns:
    -------
    dict:
        A dictionary containing:
        - 'X_train': Training features (scaled if scale_features=True)
        - 'X_test': Testing features (scaled if scale_features=True)
        - 'X_train_raw': Raw unscaled training features
        - 'X_test_raw': Raw unscaled testing features
        - 'y_train': Training target labels (0 or 1)
        - 'y_test': Testing target labels (0 or 1)
        - 'feature_names': List of feature column names
        - 'scaler': Fitted StandardScaler instance (or None)
    """
    # 1. Load data
    df = load_wine_data(wine_type=wine_type)

    # 2. Duplicate handling (optional)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    # 3. Create Binary Classification Target: 1 for Good/High Quality, 0 for Standard
    df["quality_label"] = (df["quality"] >= quality_threshold).astype(int)

    # 4. Handle categorical 'wine_type' column if combined
    if "wine_type" in df.columns:
        if wine_type == "both":
            # Map 'red': 0, 'white': 1
            df["is_white"] = (df["wine_type"] == "white").astype(int)
            df = df.drop(columns=["wine_type"])
        else:
            df = df.drop(columns=["wine_type"])

    # 5. Separate Features (X) and Target (y)
    # Drop both raw quality and binary label from features
    feature_cols = [col for col in df.columns if col not in ["quality", "quality_label"]]
    X = df[feature_cols].copy()
    y = df["quality_label"].copy()

    # 6. Train-Test Split (Stratified to maintain class balance)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 7. Feature Scaling (StandardScaler)
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        # IMPORTANT: Fit ONLY on training data to avoid data leakage
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        # Convert back to DataFrame for clean column tracking
        X_train = pd.DataFrame(X_train, columns=feature_cols, index=X_train_raw.index)
        X_test = pd.DataFrame(X_test, columns=feature_cols, index=X_test_raw.index)
    else:
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


if __name__ == "__main__":
    print("Testing preprocessing module...")
    data = prepare_wine_data(wine_type="both", quality_threshold=7)
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape:  {data['X_test'].shape}")
    print(f"y_train class balance:\n{data['y_train'].value_counts(normalize=True)}")
    print(f"Features ({len(data['feature_names'])}): {data['feature_names']}")
