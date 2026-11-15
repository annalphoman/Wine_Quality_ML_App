"""
Module: data_loader.py
Description: Beginner-friendly functions to load and inspect UCI Wine Quality datasets.
"""

import os
import pandas as pd


def load_wine_data(wine_type="both", data_dir="data"):
    """
    Load red, white, or both wine quality datasets from CSV files.

    Parameters:
    ----------
    wine_type : str
        Type of wine to load: 'red', 'white', or 'both' (default).
    data_dir : str
        Directory where CSV files are stored (default: 'data').

    Returns:
    -------
    pd.DataFrame
        Loaded pandas DataFrame with wine quality data.
    """
    wine_type = wine_type.lower()
    red_path = os.path.join(data_dir, "winequality-red.csv")
    white_path = os.path.join(data_dir, "winequality-white.csv")

    def _read_csv_safe(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found at '{path}'. Please ensure the file exists."
            )
        # UCI Wine Quality CSV files use semicolon ';' as delimiter
        return pd.read_csv(path, sep=";")

    if wine_type == "red":
        df = _read_csv_safe(red_path)
        df["wine_type"] = "red"
        return df

    elif wine_type == "white":
        df = _read_csv_safe(white_path)
        df["wine_type"] = "white"
        return df

    elif wine_type == "both":
        red_df = _read_csv_safe(red_path)
        red_df["wine_type"] = "red"

        white_df = _read_csv_safe(white_path)
        white_df["wine_type"] = "white"

        combined_df = pd.concat([red_df, white_df], ignore_index=True)
        return combined_df

    else:
        raise ValueError("Invalid wine_type. Choose from 'red', 'white', or 'both'.")


def get_data_summary(df):
    """
    Generate a quick summary of dataset characteristics.

    Parameters:
    ----------
    df : pd.DataFrame
        Dataset to summarize.

    Returns:
    -------
    dict
        Dictionary containing shape, missing values count, duplicate count, and columns.
    """
    summary = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "quality_distribution": df["quality"].value_counts().sort_index().to_dict(),
    }
    return summary


if __name__ == "__main__":
    # Self-test when executed directly
    print("Testing data_loader module...")
    df_combined = load_wine_data("both")
    summary = get_data_summary(df_combined)
    print(f"Total Rows: {summary['num_rows']}")
    print(f"Total Columns: {summary['num_columns']}")
    print(f"Total Missing Values: {summary['missing_values']}")
    print(f"Total Duplicates: {summary['duplicates']}")
    print(f"Quality Distribution: {summary['quality_distribution']}")
