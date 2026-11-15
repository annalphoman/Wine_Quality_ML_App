import pandas as pd
import os

# Define paths to your CSV files (adjust file names if yours are slightly different)
data_dir = "data"
red_path = os.path.join(data_dir, "winequality-red.csv")
white_path = os.path.join(data_dir, "winequality-white.csv")

# Function to test and inspect a dataset
def inspect_wine_data(file_path, name):
    print(f"\n================ INSPECTING: {name} ================")
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}. Check if the file is in the 'data' folder!")
        return

    # 1. Read the CSV file (Handling potential semicolon separators common in UCI datasets)
    # Note: UCI Wine Quality datasets often use ';' as a separator instead of ','
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception:
        df = pd.read_csv(file_path) # fallback to default comma if needed

    # 2. Display the first five rows
    print("\n--- First 5 Rows ---")
    print(df.head())

    # 3. Display the number of rows and columns
    print("\n--- Shape (Rows, Columns) ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 4. Display column names
    print("\n--- Column Names ---")
    print(list(df.columns))

    # 5. Check missing values
    print("\n--- Missing Values Per Column ---")
    print(df.isnull().sum())

    # 6. Check duplicate records
    print("\n--- Duplicate Records Count ---")
    print(f"Duplicates: {df.duplicated().sum()}")

# Run the inspection for both datasets
inspect_wine_data(red_path, "Red Wine Dataset")
inspect_wine_data(white_path, "White Wine Dataset")