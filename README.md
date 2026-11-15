# 🍷 Wine Quality Prediction - Machine Learning Internship Project

An end-to-end Machine Learning project to predict wine quality based on physicochemical properties using the standard UCI Wine Quality dataset. This project includes modular data pipelines, exploratory analysis, model benchmarking, serialized artifacts, and an interactive Streamlit web application.

---

## 📌 Project Overview
- **Title:** Wine Quality Prediction
- **Objective:** Classify whether a wine is **High Quality (Rating $\ge$ 7)** or **Standard Quality** using its laboratory chemical attributes.
- **Dataset:** UCI Machine Learning Repository — Wine Quality Dataset (Red and White wine, 6,497 total samples).
- **Core Technologies:** Python 3.10+, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib.

---

## 📁 Project Structure

```text
Wine_Quality/
├── data/                                  # Datasets
│   ├── winequality-red.csv                # UCI Red Wine dataset (1,599 samples)
│   └── winequality-white.csv              # UCI White Wine dataset (4,898 samples)
├── notebooks/                             # Interactive Jupyter Notebooks
│   └── wine_quality_analysis.ipynb        # Step-by-step exploratory & modeling notebook
├── src/                                   # Clean modular Python source code
│   ├── __init__.py                        # Package initializer
│   ├── data_loader.py                     # Dataset ingestion and validation
│   ├── eda.py                             # Exploratory Data Analysis & visual charts
│   ├── preprocessing.py                   # Data cleaning, scaling, and train/test split
│   ├── evaluate.py                        # Performance metrics, ROC, and confusion matrices
│   └── model_trainer.py                   # Training pipeline & model comparison
├── models/                                # Saved ML artifacts
│   ├── best_model.joblib                  # Serialized best trained model (Random Forest)
│   ├── scaler.joblib                      # Fitted StandardScaler to prevent data leakage
│   └── model_metadata.json                # Model parameters, feature names, and metrics
├── reports/                               # Generated reports and visual figures
│   ├── eda_summary.txt                    # Detailed summary of dataset statistics
│   ├── model_comparison.csv               # Empirical benchmark across all 5 models
│   ├── best_model_classification_report.txt
│   └── figures/                           # High-resolution saved plots
│       ├── quality_distribution.png
│       ├── correlation_heatmap.png
│       ├── key_features_boxplots.png
│       ├── feature_distributions.png
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       └── feature_importance.png
├── tests/                                 # Automated test suite
│   └── test_pipeline.py                   # Integration & unit tests for the pipeline
├── app.py                                 # Interactive Streamlit Web Application
├── requirements.txt                       # Project dependencies
├── test_data.py                           # Quick dataset inspection test
└── README.md                              # Project documentation
```

---

## 🔬 Dataset & Physicochemical Features

The dataset comprises **11 laboratory features** and 1 target sensory rating:

| Feature | Description | Practical Interpretation |
| :--- | :--- | :--- |
| **Fixed Acidity** | Non-volatile acids (tartaric acid) | Provides structure and crispness. |
| **Volatile Acidity** | Gaseous acids (acetic acid) | Negative indicator. High values cause a vinegar off-taste. |
| **Citric Acid** | Natural citric acid | Adds fresh, citrus notes to wine. |
| **Residual Sugar** | Natural sugar remaining after fermentation | Determines wine dryness or sweetness. |
| **Chlorides** | Sodium chloride (salt concentration) | Imparts salty notes; lower levels are preferred. |
| **Free Sulfur Dioxide** | Active dissolved $\text{SO}_2$ preservative | Prevents microbial growth and oxidation. |
| **Total Sulfur Dioxide** | Bound + free $\text{SO}_2$ | Excess levels are pungent in aroma. |
| **Density** | Density relative to water | Influenced by sugar and alcohol levels. |
| **pH** | Acid-base balance | Typical wines range between 3.0 and 3.8. |
| **Sulphates** | Potassium sulphate additive | Promotes $\text{SO}_2$ levels; acts as an antioxidant. |
| **Alcohol** | Percent alcohol by volume | Strongest positive driver of perceived quality. |
| **Quality** *(Target)* | Expert sensory score (0–10) | Formulated as: **$\ge$ 7 $\rightarrow$ High Quality (1)**, **$<$ 7 $\rightarrow$ Standard (0)**. |

---

## 📊 Empirical Model Benchmark Results

Models were evaluated on a held-out test split of **1,300 samples** (20% stratified test set):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Winner)** | **89.23%** | **83.33%** | **56.64%** | **0.6744** | **0.9120** |
| **Gradient Boosting** | 84.62% | 68.92% | 39.84% | 0.5050 | 0.8593 |
| **Decision Tree** | 83.15% | 64.80% | 31.64% | 0.4252 | 0.8269 |
| **Support Vector Machine (SVC)** | 82.69% | 63.96% | 27.73% | 0.3869 | 0.8424 |
| **Logistic Regression (Baseline)** | 82.23% | 61.47% | 26.17% | 0.3671 | 0.8048 |

> **Key Finding:** Ensemble methods (Random Forest) significantly outperform single decision trees and linear models by combining diverse bagging trees, reducing variance, and capturing non-linear interactions among chemical properties.

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Setup
Clone the repository and navigate into the project folder:
```bash
cd Wine_Quality
```

Activate the virtual environment:
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

If creating a new virtual environment from scratch:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run Exploratory Data Analysis (EDA)
Inspect dataset distributions, correlations, and automatically save plots into `reports/figures/`:
```bash
python src/eda.py
```
*Generated charts:*
- `reports/figures/quality_distribution.png`
- `reports/figures/correlation_heatmap.png`
- `reports/figures/key_features_boxplots.png`
- `reports/figures/feature_distributions.png`
- `reports/eda_summary.txt`

---

### 3. Train Models & Evaluate Pipeline
Train all 5 models, benchmark test results, and export the best model to disk:
```bash
python src/model_trainer.py
```
*Artifacts created:*
- `models/best_model.joblib`
- `models/scaler.joblib`
- `models/model_metadata.json`
- `reports/model_comparison.csv`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curves.png`
- `reports/figures/feature_importance.png`

---

### 4. Launch the Interactive Streamlit Web App
Run the web application locally:
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

**Features of the Redesigned Streamlit App:**
- 🍷 **CellarAI Luxury Dark UI:** Wine-inspired palette (Burgundy `#9E2A2B`, Cabernet `#540B0E`, Obsidian Charcoal `#0F0C0E`, and Gold `#D4AF37`).
- 🧭 **Modern Sidebar Navigation:** Direct access to *Quality Predictor*, *Model Benchmarks*, *Exploratory Analytics*, and *Physicochemical Guide*.
- 📋 **Curated Presets:** Instant loading of real dataset profiles (*Grand Reserve Red*, *Everyday Table Red*, *Crisp Reserve White*, *Commercial Table White*).
- 🔮 **Sensory Classification Verdict:** Premium badge, classification status, probability confidence gauge, and custom enological profile assessment.
- 📊 **Interactive Altair Charts:** Grouped interactive model comparison bars and Gini feature importance plots with hover tooltips.
- 📈 **Exploratory Visualizations:** Interactive distributions, alcohol vs quality boxplots, and high-resolution diagnostic charts.
- 🚀 **Zero Deprecation Warnings:** Modern Streamlit 1.39+ compatible (`width="stretch"`).

---

### 5. Run the Automated Test Suite
Verify that data loaders, preprocessors, saved models, and inference run properly:
```bash
python tests/test_pipeline.py
```
*Expected output: `ALL TESTS PASSED SUCCESSFULLY! (5/5 suites)`*

---

### 6. Open the Step-by-Step Jupyter Notebook
To review or present the internship analysis interactively:
```bash
jupyter notebook notebooks/wine_quality_analysis.ipynb
```

---

## 💡 Practical Insights for Internship Presentation
1. **Alcohol is the strongest quality predictor:** High-quality wines show an average alcohol level of ~11.5% to 12.5%, compared to 9.5% to 10.2% in standard wines.
2. **Volatile Acidity hurts quality:** Excess acetic acid leads to unpleasant vinegar aromas.
3. **Data Leakage Prevention:** Feature scaling (`StandardScaler`) was strictly fitted on the training split only, preserving true out-of-sample evaluation validity.
4. **Ensemble Power:** Random Forest achieved an **ROC-AUC of 0.912**, proving strong discriminative capacity despite natural class imbalance.
# Wine_Quality_ML_App
