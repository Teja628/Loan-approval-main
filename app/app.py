import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- Stable paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

# ---------- Load artifacts ----------
model = joblib.load(os.path.join(MODELS_DIR, "loan_rf_model.joblib"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
num_cols = joblib.load(os.path.join(MODELS_DIR, "num_cols.joblib"))  # columns scaled during training

st.set_page_config(page_title="Loan Approval Predictor", page_icon="✅", layout="centered")
st.title("🏦 Loan Approval Prediction")
st.caption("Enter applicant details and click **Predict**.")

# ---------- Input Form ----------
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
        Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=100, value=5000)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, step=100, value=0)
        LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0, step=1, value=120)
        Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0, step=12, value=360)
        Credit_History = st.selectbox("Credit History", [1, 0])  # 1=good, 0=bad

    submitted = st.form_submit_button("Predict")

# ---------- Build row exactly like training ----------
def build_model_row():
    # Start with raw dict
    raw = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
    }])

    # Binary maps (must match notebook)
    bin_maps = {
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
    }
    for c, m in bin_maps.items():
        if c in raw.columns:
            raw[c] = raw[c].map(m)

    # Dependents: "3+" -> 3 (int)
    if "Dependents" in raw.columns:
        raw["Dependents"] = raw["Dependents"].replace("3+", "3").astype(int)

    # One-hot encode ONLY Property_Area (drop_first=True to match training)
    raw = pd.get_dummies(raw, columns=["Property_Area"], drop_first=True)

    # Ensure every expected column exists; missing -> 0
    for col in feature_columns:
        if col not in raw.columns:
            raw[col] = 0

    # Reorder to exact training feature order
    raw = raw[feature_columns]

    # Scale ONLY training numeric columns
    raw_scaled = raw.copy()
    if len(num_cols) > 0:
        raw_scaled[num_cols] = scaler.transform(raw_scaled[num_cols])

    return raw_scaled

# ---------- Predict ----------
if submitted:
    try:
        X_one = build_model_row()
        pred = model.predict(X_one)[0]
        proba = None
        try:
            proba = model.predict_proba(X_one)[0][1]
        except Exception:
            pass

        if pred == 1:
            st.success("✅ Prediction: **Loan Approved**")
        else:
            st.error("❌ Prediction: **Loan Rejected**")

        if proba is not None:
            st.caption(f"Approval probability: {proba:.2%}")

        # Debug toggle (uncomment if needed)
        # st.write("Feature columns used by model:", feature_columns)
        # st.write("Row columns sent to model:", list(X_one.columns))

    except Exception as e:
        st.exception(e)
        st.warning("Input preparation failed. Check that notebook encoding/scaling matches the app.")
