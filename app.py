"""
Heart Disease Predictor - Streamlit web app
Created by ROHIT with love
Run with:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "KNN_heart.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"

MODEL_NAME = "K-Nearest Neighbors (KNN)"
MODEL_ACCURACY = "86%"  # from your notebook result; edit if yours changes

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #b3122d 0%, #e63860 55%, #ff7a93 100%);
        color: #ffffff;
        padding: 2.2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(230, 56, 96, 0.35);
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #ffffff; margin: 0; font-size: 2.3rem; }
    .hero p  { color: #ffe9ee; margin: 0.5rem 0 0 0; font-size: 1.05rem; }
    .beat { display: inline-block; animation: beat 1.2s infinite; }
    @keyframes beat {
        0%, 100% { transform: scale(1); }
        50%      { transform: scale(1.25); }
    }

    .result-card {
        padding: 1.4rem 1.5rem;
        border-radius: 16px;
        margin-top: 1rem;
        border-left: 8px solid;
    }
    .result-high { background: rgba(230, 56, 96, 0.12); border-color: #e63860; }
    .result-low  { background: rgba(38, 166, 91, 0.12); border-color: #26a65b; }
    .result-card h2 { margin: 0 0 0.3rem 0; }

    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.2rem 0 0.5rem 0;
        border-top: 1px solid rgba(128, 128, 128, 0.3);
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model files (cached so they load only once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns


def build_features(raw: dict, columns: list) -> pd.DataFrame:
    """Turn the form answers into the exact one-hot row the model was trained on."""
    row = {}
    for col in columns:
        if col in raw:  # numeric / binary columns: Age, RestingBP, ...
            row[col] = raw[col]
        else:  # one-hot columns such as ChestPainType_ATA, ST_Slope_Flat
            prefix, _, value = col.rpartition("_")
            row[col] = 1 if raw.get(prefix) == value else 0
    return pd.DataFrame([row], columns=columns)


def scaler_looks_wrong(scaler, columns: list) -> bool:
    """True if the saved scaler was fitted on already-scaled data (means close to 0).

    Real means are roughly Age 53, RestingBP 132, Cholesterol 244, MaxHR 137.
    """
    if not hasattr(scaler, "mean_"):
        return False
    idx = [columns.index(c) for c in ("Age", "RestingBP", "Cholesterol", "MaxHR") if c in columns]
    return bool(idx) and all(abs(scaler.mean_[i]) < 5 for i in idx)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1><span class="beat">❤️</span> Heart Disease Predictor</h1>
        <p>Enter the patient details below and get an instant machine-learning prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler, columns = load_artifacts()
except FileNotFoundError as err:
    st.error(
        f"Model file not found: **{Path(err.filename).name}**. "
        "Put `KNN_heart.pkl`, `scaler.pkl` and `columns.pkl` in the same folder as `app.py`."
    )
    st.stop()

if scaler_looks_wrong(scaler, columns):
    st.error(
        "**Your saved `scaler.pkl` is wrong, so predictions would be unreliable.**\n\n"
        "It was fitted on data that was already scaled (the numeric columns get "
        "standardized twice in the notebook).\n\n"
        "1. In the notebook, delete the cell containing `numeric_cols=[...]`\n"
        "2. Click *Kernel → Restart & Run All*\n"
        "3. Copy the new `KNN_heart.pkl`, `scaler.pkl` and `columns.pkl` into this folder\n"
        "4. Stop this app (Ctrl + C) and start it again"
    )
    st.stop()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
CHEST_PAIN = {
    "Typical Angina (TA)": "TA",
    "Atypical Angina (ATA)": "ATA",
    "Non-Anginal Pain (NAP)": "NAP",
    "Asymptomatic (ASY)": "ASY",
}
RESTING_ECG = {
    "Normal": "Normal",
    "ST-T wave abnormality (ST)": "ST",
    "Left ventricular hypertrophy (LVH)": "LVH",
}
ST_SLOPE = {
    "Upsloping (Up)": "Up",
    "Flat": "Flat",
    "Downsloping (Down)": "Down",
}

with st.form("patient_form"):
    st.subheader("🩺 Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", list(CHEST_PAIN))
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dl)", min_value=100, max_value=600, value=200
        )
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with col2:
        resting_ecg = st.selectbox("Resting ECG", list(RESTING_ECG))
        max_hr = st.number_input(
            "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150
        )
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input(
            "Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1
        )
        st_slope = st.selectbox("ST Slope", list(ST_SLOPE))

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    raw = {
        # The notebook converted these columns with .astype(int), so we do the same
        "Age": int(age),
        "RestingBP": int(resting_bp),
        "Cholesterol": int(cholesterol),
        "MaxHR": int(max_hr),
        "Oldpeak": int(oldpeak),
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        # Categorical columns (one-hot encoded in build_features)
        "Sex": "M" if sex == "Male" else "F",
        "ChestPainType": CHEST_PAIN[chest_pain],
        "RestingECG": RESTING_ECG[resting_ecg],
        "ExerciseAngina": "Y" if exercise_angina == "Yes" else "N",
        "ST_Slope": ST_SLOPE[st_slope],
    }

    features = build_features(raw, columns)
    scaled = scaler.transform(features)
    prediction = int(model.predict(scaled)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
        confidence = float(proba[prediction]) * 100

    if prediction == 1:
        st.markdown(
            """
            <div class="result-card result-high">
                <h2>⚠️ High Risk of Heart Disease</h2>
                <p>The model predicts that this patient is <b>likely to have heart disease</b>.
                Please consult a cardiologist for a proper medical check-up.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card result-low">
                <h2>✅ Low Risk of Heart Disease</h2>
                <p>The model predicts that this patient is <b>unlikely to have heart disease</b>.
                Keep up a healthy lifestyle and go for regular check-ups.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if confidence is not None:
        st.write("")
        st.metric("Model confidence", f"{confidence:.0f}%")
        st.progress(int(confidence))

    st.caption(
        "This tool is for learning and demonstration only. It is not a medical "
        "diagnosis - always consult a qualified doctor."
    )

# ----------------------------------------------------------------------------
# About + footer
# ----------------------------------------------------------------------------
with st.expander("ℹ️ About this project"):
    st.write(
        f"- **Model:** {MODEL_NAME}\n"
        f"- **Test accuracy:** about {MODEL_ACCURACY}\n"
        "- **Data:** 918 patient records with 11 clinical features\n"
        "- **Built with:** Python, scikit-learn, pandas and Streamlit"
    )

st.markdown(
    """
    <div class="footer">
        Created by <b>ROHIT</b> with <span class="beat">❤️</span>
    </div>
    """,
    unsafe_allow_html=True,
)
