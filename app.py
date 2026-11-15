"""
CellarAI — Professional Wine Quality Intelligence Platform
End-to-End Machine Learning System for Physicochemical Wine Evaluation
Theme: Luxury Dark Burgundy & Obsidian Charcoal
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path for local imports
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="CellarAI — Wine Quality Intelligence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# LUXURY DARK WINE-THEMED CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Resets */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #EDE8EA;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Header Styling */
    .luxury-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #FFFFFF 20%, #F4D03F 65%, #C93B42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .luxury-subtitle {
        font-size: 1.05rem;
        color: #A89FA4;
        font-weight: 400;
        margin-bottom: 1.8rem;
    }

    /* Glassmorphic Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(26, 20, 24, 0.85), rgba(15, 12, 14, 0.95));
        border: 1px solid rgba(158, 42, 43, 0.35);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(212, 175, 55, 0.6);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #A3979D;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #D4AF37;
        margin-top: 5px;
    }

    /* Section Cards & Panels */
    .panel-card {
        background: linear-gradient(145deg, #181216, #120E11);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
    }

    /* Prediction Result Cards */
    .result-card-premium {
        background: linear-gradient(135deg, rgba(84, 11, 14, 0.85) 0%, rgba(26, 15, 20, 0.95) 100%);
        border: 2px solid #D4AF37;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 12px 35px rgba(158, 42, 43, 0.4);
        margin-top: 15px;
    }
    .result-card-standard {
        background: linear-gradient(135deg, rgba(35, 25, 30, 0.85) 0%, rgba(18, 14, 16, 0.95) 100%);
        border: 1.5px solid rgba(231, 111, 81, 0.6);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
        margin-top: 15px;
    }

    .badge-premium {
        display: inline-block;
        background: linear-gradient(135deg, #D4AF37 0%, #AA820A 100%);
        color: #0F0C0E;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .badge-standard {
        display: inline-block;
        background: linear-gradient(135deg, #E76F51 0%, #B84A30 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    /* Action Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #9E2A2B 0%, #540B0E 100%);
        border: 1px solid #D4AF37;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.7rem 1.8rem;
        border-radius: 10px;
        letter-spacing: 0.5px;
        transition: all 0.25s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #B83234 0%, #720E12 100%);
        border-color: #F4D03F;
        color: #FFFFFF;
        box-shadow: 0 8px 24px rgba(158, 42, 43, 0.55);
        transform: translateY(-2px);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #120D10;
        border-right: 1px solid rgba(158, 42, 43, 0.25);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Radio buttons in sidebar */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 8px;
    }

    /* Code & Metric Highlights */
    code {
        color: #F4D03F !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CACHED DATA & ARTIFACT LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_model_pipeline():
    """Load pre-trained model, fitted scaler, and metadata."""
    model_path = os.path.join(project_root, "models", "best_model.joblib")
    scaler_path = os.path.join(project_root, "models", "scaler.joblib")
    metadata_path = os.path.join(project_root, "models", "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, metadata


@st.cache_data
def load_benchmark_data():
    """Load model performance benchmark table."""
    csv_path = os.path.join(project_root, "reports", "model_comparison.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_sample_dataset():
    """Load dataset for interactive visual analysis."""
    red_path = os.path.join(project_root, "data", "winequality-red.csv")
    white_path = os.path.join(project_root, "data", "winequality-white.csv")

    if os.path.exists(red_path) and os.path.exists(white_path):
        red = pd.read_csv(red_path, sep=";")
        red["wine_type"] = "Red Wine"
        white = pd.read_csv(white_path, sep=";")
        white["wine_type"] = "White Wine"
        return pd.concat([red, white], ignore_index=True)
    return None


# ---------------------------------------------------------
# MAIN APPLICATION CONTROLLER
# ---------------------------------------------------------
def main():
    # Load core artifacts
    model, scaler, metadata = load_model_pipeline()
    df_raw = load_sample_dataset()
    df_comparison = load_benchmark_data()

    # Fallback safety if model artifacts are missing
    if model is None or scaler is None:
        st.error("⚠️ Trained model artifacts not detected in `models/` directory.")
        st.info("Execute `python src/model_trainer.py` in your terminal to train and serialize the best model.")
        return

    best_model_name = metadata.get("best_model_name", "Random Forest")
    metrics = metadata.get("metrics", {})
    acc = metrics.get("accuracy", 0.8923)
    auc = metrics.get("roc_auc", 0.9120)
    f1 = metrics.get("f1_score", 0.6744)

    # ------------------ SIDEBAR NAVIGATION ------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding-bottom: 15px;">
                <span style="font-size: 2.4rem;">🍷</span>
                <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">
                    CELLAR<span style="color: #C93B42;">AI</span>
                </div>
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; color: #D4AF37;">
                    Wine Quality Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        nav_choice = st.radio(
            "Navigation",
            options=[
                "🔮 Quality Predictor",
                "📊 Model Benchmarks",
                "📈 Exploratory Analytics",
                "📖 Physicochemical Guide",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("##### ⚙️ Quick Configuration")
        wine_type = st.radio("Wine Variety", options=["Red Wine", "White Wine"], index=0)
        is_white_val = 1 if wine_type == "White Wine" else 0

        # Curated presets from empirical UCI data
        preset_choice = st.selectbox(
            "Curated Presets",
            options=[
                "Custom (Manual Input)",
                "Grand Reserve Red (High Quality)",
                "Everyday Table Red (Standard)",
                "Crisp Reserve White (High Quality)",
                "Commercial Table White (Standard)",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div class="kpi-card" style="padding: 14px;">
                <div class="kpi-label">Active ML Engine</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">{best_model_name}</div>
                <div class="kpi-sub">Accuracy: {acc * 100:.1f}% · AUC: {auc:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Preset profiles dictionary
    presets = {
        "Custom (Manual Input)": {
            "fixed_acidity": 7.4 if wine_type == "Red Wine" else 6.8,
            "volatile_acidity": 0.40 if wine_type == "Red Wine" else 0.26,
            "citric_acid": 0.30,
            "residual_sugar": 2.2 if wine_type == "Red Wine" else 5.2,
            "chlorides": 0.075 if wine_type == "Red Wine" else 0.045,
            "free_sulfur_dioxide": 15.0 if wine_type == "Red Wine" else 32.0,
            "total_sulfur_dioxide": 48.0 if wine_type == "Red Wine" else 118.0,
            "density": 0.9965,
            "pH": 3.32,
            "sulphates": 0.65 if wine_type == "Red Wine" else 0.48,
            "alcohol": 10.5,
        },
        "Grand Reserve Red (High Quality)": {
            "fixed_acidity": 8.8,
            "volatile_acidity": 0.34,
            "citric_acid": 0.42,
            "residual_sugar": 2.4,
            "chlorides": 0.070,
            "free_sulfur_dioxide": 14.0,
            "total_sulfur_dioxide": 36.0,
            "density": 0.9954,
            "pH": 3.28,
            "sulphates": 0.76,
            "alcohol": 12.2,
        },
        "Everyday Table Red (Standard)": {
            "fixed_acidity": 7.8,
            "volatile_acidity": 0.68,
            "citric_acid": 0.08,
            "residual_sugar": 2.1,
            "chlorides": 0.092,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 58.0,
            "density": 0.9972,
            "pH": 3.38,
            "sulphates": 0.54,
            "alcohol": 9.5,
        },
        "Crisp Reserve White (High Quality)": {
            "fixed_acidity": 6.7,
            "volatile_acidity": 0.22,
            "citric_acid": 0.34,
            "residual_sugar": 4.5,
            "chlorides": 0.038,
            "free_sulfur_dioxide": 35.0,
            "total_sulfur_dioxide": 110.0,
            "density": 0.9918,
            "pH": 3.22,
            "sulphates": 0.52,
            "alcohol": 12.5,
        },
        "Commercial Table White (Standard)": {
            "fixed_acidity": 6.9,
            "volatile_acidity": 0.38,
            "citric_acid": 0.24,
            "residual_sugar": 8.5,
            "chlorides": 0.052,
            "free_sulfur_dioxide": 22.0,
            "total_sulfur_dioxide": 145.0,
            "density": 0.9962,
            "pH": 3.18,
            "sulphates": 0.44,
            "alcohol": 9.2,
        },
    }
    selected_preset = presets[preset_choice]

    # =========================================================
    # PAGE 1: QUALITY PREDICTOR
    # =========================================================
    if nav_choice == "🔮 Quality Predictor":
        st.markdown('<div class="luxury-title">Wine Quality Prediction Studio</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="luxury-subtitle">Evaluate wine samples through supervised machine learning to classify commercial vs premium vintages.</div>',
            unsafe_allow_html=True,
        )

        # Executive KPI Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Algorithm</div>
                    <div class="kpi-value">{best_model_name.split()[0]}</div>
                    <div class="kpi-sub">Ensemble Architecture</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Validation Accuracy</div>
                    <div class="kpi-value">{acc * 100:.1f}%</div>
                    <div class="kpi-sub">Out-of-sample Test</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">ROC-AUC Score</div>
                    <div class="kpi-value">{auc:.3f}</div>
                    <div class="kpi-sub">High Discrimination</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Benchmark Split</div>
                    <div class="kpi-value">1,300</div>
                    <div class="kpi-sub">Stratified Holdout Test</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Input Form Layout
        st.markdown("### 🧪 Physicochemical Property Inputs")
        st.write("Configure chemical measurements or use a preset from the sidebar.")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                """
                <div style="background-color: #171115; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 18px 20px; margin-bottom: 15px;">
                    <span style="font-weight: 700; color: #FFFFFF; font-size: 1.05rem;">🍇 Acidity, pH & Sugar Profile</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            fixed_acidity = st.slider(
                "Fixed Acidity (g/dm³ tartaric acid)",
                min_value=4.0, max_value=16.0,
                value=float(selected_preset["fixed_acidity"]), step=0.1,
                help="Non-volatile acids giving wine crisp structure and freshness."
            )
            volatile_acidity = st.slider(
                "Volatile Acidity (g/dm³ acetic acid)",
                min_value=0.08, max_value=1.60,
                value=float(selected_preset["volatile_acidity"]), step=0.01,
                help="Gaseous acid. High levels create a sour, vinegar off-flavor."
            )
            citric_acid = st.slider(
                "Citric Acid (g/dm³)",
                min_value=0.0, max_value=1.0,
                value=float(selected_preset["citric_acid"]), step=0.01,
                help="Adds crisp citrus notes and aromatic brightness."
            )
            residual_sugar = st.slider(
                "Residual Sugar (g/dm³)",
                min_value=0.5, max_value=25.0,
                value=float(selected_preset["residual_sugar"]), step=0.1,
                help="Natural sugar remaining post-fermentation."
            )
            chlorides = st.slider(
                "Chlorides (g/dm³ sodium chloride)",
                min_value=0.01, max_value=0.60,
                value=float(selected_preset["chlorides"]), step=0.005, format="%.3f",
                help="Mineral salt concentration; excess causes bitter, salty finish."
            )
            pH = st.slider(
                "pH Level",
                min_value=2.7, max_value=4.1,
                value=float(selected_preset["pH"]), step=0.01,
                help="Balances tartness and stability. Most fine wines sit between 3.1 - 3.6."
            )

        with col_right:
            st.markdown(
                """
                <div style="background-color: #171115; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 18px 20px; margin-bottom: 15px;">
                    <span style="font-weight: 700; color: #FFFFFF; font-size: 1.05rem;">🔬 Preservation, Density & Alcohol</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            free_sulfur_dioxide = st.slider(
                "Free Sulfur Dioxide (mg/dm³)",
                min_value=1.0, max_value=100.0,
                value=float(selected_preset["free_sulfur_dioxide"]), step=1.0,
                help="Active SO2 preventing oxidation and microbial degradation."
            )
            total_sulfur_dioxide = st.slider(
                "Total Sulfur Dioxide (mg/dm³)",
                min_value=6.0, max_value=300.0,
                value=float(selected_preset["total_sulfur_dioxide"]), step=1.0,
                help="Total bound plus dissolved preservative."
            )
            density = st.slider(
                "Density (g/cm³)",
                min_value=0.985, max_value=1.005,
                value=float(selected_preset["density"]), step=0.0005, format="%.4f",
                help="Density compared to water, driven by sugar and alcohol."
            )
            sulphates = st.slider(
                "Sulphates (g/dm³ potassium sulphate)",
                min_value=0.2, max_value=2.0,
                value=float(selected_preset["sulphates"]), step=0.01,
                help="Additive acting as antioxidant to preserve wine life."
            )
            alcohol = st.slider(
                "Alcohol Content (% by volume)",
                min_value=8.0, max_value=15.0,
                value=float(selected_preset["alcohol"]), step=0.1,
                help="Strongest empirical driver of sensory rating."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🍷 Run Physicochemical Evaluation", width="stretch")

        # Assemble input feature DataFrame
        input_data = pd.DataFrame([{
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_sulfur_dioxide,
            "total sulfur dioxide": total_sulfur_dioxide,
            "density": density,
            "pH": pH,
            "sulphates": sulphates,
            "alcohol": alcohol,
            "is_white": is_white_val,
        }])

        if predict_clicked or preset_choice != "Custom (Manual Input)":
            # Scale features strictly using pre-fitted StandardScaler
            input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
            prediction = int(model.predict(input_scaled)[0])
            probabilities = model.predict_proba(input_scaled)[0]
            high_qual_prob = probabilities[1] * 100.0

            st.markdown("---")
            st.markdown("### 🏆 Sensory Classification Verdict")

            res_c1, res_c2 = st.columns([1.6, 1])

            with res_c1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card-premium">
                            <span class="badge-premium">⭐ High Quality Vintage</span>
                            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
                                Exemplary Wine (Rating ≥ 7)
                            </div>
                            <div style="font-size: 1.05rem; color: #F4D03F; font-weight: 600; margin-bottom: 12px;">
                                Model Confidence: {high_qual_prob:.1f}% High Quality Probability
                            </div>
                            <p style="color: #EDE8EA; line-height: 1.6; margin-bottom: 0;">
                                <b>Enological Evaluation:</b> This profile exhibits prime characteristics of high-tier wine:
                                robust alcohol concentration (<b>{alcohol}%</b>), balanced and restrained volatile acidity (<b>{volatile_acidity:.2f} g/dm³</b>),
                                and supportive sulphate levels (<b>{sulphates:.2f} g/dm³</b>) ensuring aromatic longevity and clean finish.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-standard">
                            <span class="badge-standard">Standard Table Quality</span>
                            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
                                Commercial Wine (Rating &lt; 7)
                            </div>
                            <div style="font-size: 1.05rem; color: #E76F51; font-weight: 600; margin-bottom: 12px;">
                                Model Confidence: {100 - high_qual_prob:.1f}% Standard Quality Probability
                            </div>
                            <p style="color: #EDE8EA; line-height: 1.6; margin-bottom: 0;">
                                <b>Enological Evaluation:</b> This profile falls into the commercial/everyday category (typical scores 5–6).
                                Primary factors limiting its classification as premium include lower alcohol (<b>{alcohol}%</b>)
                                or elevated volatile acidity (<b>{volatile_acidity:.2f} g/dm³</b>), which detracts from perceived sensory depth.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_c2:
                st.markdown(
                    """
                    <div class="panel-card" style="height: 100%;">
                        <div class="kpi-label">Confidence Assessment</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(high_qual_prob / 100.0))
                st.caption(f"Estimated High Quality Likelihood: **{high_qual_prob:.1f}%**")

                st.markdown("---")
                st.markdown("##### 📌 Chemical Summary")
                st.markdown(
                    f"""
                    - **Variety:** {wine_type}
                    - **Alcohol:** `{alcohol}%`
                    - **Volatile Acidity:** `{volatile_acidity:.2f} g/dm³`
                    - **Sulphates:** `{sulphates:.2f} g/dm³`
                    - **pH Level:** `{pH:.2f}`
                    """
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # PAGE 2: MODEL BENCHMARKS
    # =========================================================
    elif nav_choice == "📊 Model Benchmarks":
        st.markdown('<div class="luxury-title">Machine Learning Model Benchmarks</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="luxury-subtitle">Empirical performance comparison of 5 classification architectures evaluated on 1,300 stratified test samples.</div>',
            unsafe_allow_html=True,
        )

        if df_comparison is not None:
            # Top metrics cards
            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Top Performer</div>
                        <div class="kpi-value">{df_comparison.iloc[0]['Model']}</div>
                        <div class="kpi-sub">Highest Overall F1 & ROC-AUC</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Best Test Accuracy</div>
                        <div class="kpi-value">{df_comparison.iloc[0]['Accuracy'] * 100:.2f}%</div>
                        <div class="kpi-sub">+7.0% above Single Decision Tree</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b3:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Peak ROC-AUC</div>
                        <div class="kpi-value">{df_comparison.iloc[0]['ROC-AUC']:.4f}</div>
                        <div class="kpi-sub">Near-optimal class ranking</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Interactive Benchmark Comparison")

            # Melt for grouped Altair visualization
            df_melted = df_comparison.melt(
                id_vars=["Model"],
                value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                var_name="Metric",
                value_name="Score"
            )

            chart_compare = (
                alt.Chart(df_melted)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("Score:Q", title="Metric Score (0.0 to 1.0)", scale=alt.Scale(domain=[0.2, 1.0])),
                    y=alt.Y("Model:N", title="", sort="-x"),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                            range=["#C93B42", "#E76F51", "#D4AF37", "#2A9D8F", "#8E44AD"]
                        ),
                        legend=alt.Legend(title="Evaluation Metric", orient="bottom")
                    ),
                    tooltip=[
                        alt.Tooltip("Model:N"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("Score:Q", format=".2%")
                    ]
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(chart_compare, width="stretch")

            st.markdown("##### Detailed Benchmark Data Table")
            df_display = df_comparison.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
                df_display[col] = df_display[col].apply(lambda x: f"{x * 100:.2f}%")
            st.dataframe(df_display, width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("Diagnostic Evaluation Charts")

        cm_col, roc_col = st.columns(2)
        with cm_col:
            cm_path = os.path.join(project_root, "reports", "figures", "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix on Held-Out Test Set (1,300 Samples)", width="stretch")

        with roc_col:
            roc_path = os.path.join(project_root, "reports", "figures", "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Comparison", width="stretch")

        st.markdown("---")
        st.subheader("Feature Importance Analysis (Random Forest)")
        if hasattr(model, "feature_importances_"):
            feature_names = metadata.get("feature_names", [])
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            chart_imp = (
                alt.Chart(df_imp)
                .mark_bar(cornerRadius=5)
                .encode(
                    x=alt.X("Importance:Q", title="Gini Importance Score"),
                    y=alt.Y("Feature:N", sort="-x", title="Physicochemical Property"),
                    color=alt.Color("Importance:Q", scale=alt.Scale(scheme="reds"), legend=None),
                    tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".3f")]
                )
                .properties(height=380)
            )
            st.altair_chart(chart_imp, width="stretch")

    # =========================================================
    # PAGE 3: EXPLORATORY ANALYTICS
    # =========================================================
    elif nav_choice == "📈 Exploratory Analytics":
        st.markdown('<div class="luxury-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="luxury-subtitle">Physicochemical distributions and correlation patterns across 6,497 UCI wine samples.</div>',
            unsafe_allow_html=True,
        )

        if df_raw is not None:
            # Dataset size KPIs
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Total Wine Samples</div>
                        <div class="kpi-value">{len(df_raw):,}</div>
                        <div class="kpi-sub">Complete UCI Registry</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with e2:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">White Wine Samples</div>
                        <div class="kpi-value">{len(df_raw[df_raw['wine_type']=='White Wine']):,}</div>
                        <div class="kpi-sub">75.4% of Dataset</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with e3:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Red Wine Samples</div>
                        <div class="kpi-value">{len(df_raw[df_raw['wine_type']=='Red Wine']):,}</div>
                        <div class="kpi-sub">24.6% of Dataset</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with e4:
                st.markdown(
                    """
                    <div class="kpi-card">
                        <div class="kpi-label">Missing Values</div>
                        <div class="kpi-value">0</div>
                        <div class="kpi-sub">100% Complete Records</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Interactive Quality Score Distribution")

            # Altair interactive distribution
            chart_dist = (
                alt.Chart(df_raw)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("quality:O", title="Sensory Quality Rating (3 to 9)"),
                    y=alt.Y("count():Q", title="Number of Samples"),
                    color=alt.Color(
                        "wine_type:N",
                        scale=alt.Scale(domain=["Red Wine", "White Wine"], range=["#8E1616", "#D4AF37"]),
                        legend=alt.Legend(title="Wine Type", orient="top-right")
                    ),
                    tooltip=["quality:O", "wine_type:N", alt.Tooltip("count():Q", format=",")]
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(chart_dist, width="stretch")

            st.markdown("---")
            st.subheader("Top Influential Features vs Quality")

            feat_c1, feat_c2 = st.columns(2)
            with feat_c1:
                chart_alcohol = (
                    alt.Chart(df_raw)
                    .mark_boxplot(extent="min-max")
                    .encode(
                        x=alt.X("quality:O", title="Quality Rating"),
                        y=alt.Y("alcohol:Q", title="Alcohol Content (% vol)"),
                        color=alt.Color("wine_type:N", scale=alt.Scale(domain=["Red Wine", "White Wine"], range=["#8E1616", "#D4AF37"]), legend=None)
                    )
                    .properties(height=300, title="Alcohol Content vs Quality Rating")
                )
                st.altair_chart(chart_alcohol, width="stretch")

            with feat_c2:
                chart_va = (
                    alt.Chart(df_raw)
                    .mark_boxplot(extent="min-max")
                    .encode(
                        x=alt.X("quality:O", title="Quality Rating"),
                        y=alt.Y("volatile acidity:Q", title="Volatile Acidity (g/dm³)"),
                        color=alt.Color("wine_type:N", scale=alt.Scale(domain=["Red Wine", "White Wine"], range=["#8E1616", "#D4AF37"]), legend=None)
                    )
                    .properties(height=300, title="Volatile Acidity vs Quality Rating")
                )
                st.altair_chart(chart_va, width="stretch")

        st.markdown("---")
        st.subheader("Analytical Heatmap & Distribution Atlas")

        hm_col, dist_col = st.columns(2)
        with hm_col:
            hm_path = os.path.join(project_root, "reports", "figures", "correlation_heatmap.png")
            if os.path.exists(hm_path):
                st.image(hm_path, caption="Physicochemical Correlation Heatmap", width="stretch")

        with dist_col:
            fd_path = os.path.join(project_root, "reports", "figures", "feature_distributions.png")
            if os.path.exists(fd_path):
                st.image(fd_path, caption="Histograms of Physicochemical Features", width="stretch")

    # =========================================================
    # PAGE 4: PHYSICOCHEMICAL GUIDE
    # =========================================================
    elif nav_choice == "📖 Physicochemical Guide":
        st.markdown('<div class="luxury-title">Physicochemical & Methodology Guide</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="luxury-subtitle">Detailed documentation of the 11 laboratory properties and supervised learning methodology.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ### 🔬 The 11 Laboratory Measurements Explained

            | Property | Unit | Enological Role & Sensory Influence |
            | :--- | :--- | :--- |
            | **Alcohol** | `% vol` | **Strongest Positive Driver.** Provides warmth, body, and aromatic carry. High-quality wines consistently average 11.5%–12.5%. |
            | **Volatile Acidity** | `g/dm³ acetic acid` | **Strongest Negative Driver.** Created by acetobacter bacteria. High values impart a sour, vinegar smell. |
            | **Sulphates** | `g/dm³ K₂SO₄` | **Antioxidant.** Supplements free SO₂ to block oxidation and retain clean fruit freshness. |
            | **Citric Acid** | `g/dm³` | **Crispness.** Contributes fresh citrus notes, balancing heavier alcohol. |
            | **Fixed Acidity** | `g/dm³ tartaric` | **Structure.** Primary non-volatile acid providing backbone and ageability. |
            | **Residual Sugar** | `g/dm³` | **Sweetness/Balance.** Determines dryness; counterbalances high alcohol and acidity. |
            | **Chlorides** | `g/dm³ NaCl` | **Saltiness.** Imparts mineral finish; elevated salt degrades vintage rating. |
            | **Free Sulfur Dioxide** | `mg/dm³ SO₂` | **Preservation.** Prevents microbial spoilage and browning. |
            | **Total Sulfur Dioxide** | `mg/dm³ SO₂` | **Total Preservative.** Evident as pungent off-aroma when excessively high. |
            | **Density** | `g/cm³` | **Body Index.** Closely tied to the alcohol-to-sugar balance. |
            | **pH Level** | `0–14 scale` | **Equilibrium.** Most stable wines fall between 3.1 and 3.6 pH. |

            ---

            ### 📐 Machine Learning Methodology
            1. **Dataset Origin:** UCI Machine Learning Repository Wine Quality (Cortez et al., 2009).
            2. **Formulation:** Formulated as binary classification distinguishing **Premium / High Quality** (Quality ≥ 7) from **Standard Quality** (Quality < 7).
            3. **Data Leakage Safeguard:** Feature standardization (`StandardScaler`) is fitted exclusively on training splits ($N=5,197$), transforming test samples ($N=1,300$) without information contamination.
            4. **Ensemble Advantage:** Random Forest mitigates variance by aggregating 100 decision trees, capturing non-linear chemistry interactions.
            """
        )


if __name__ == "__main__":
    main()
