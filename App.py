# streamlit run "c:/Users/Kirellos Youssef/Desktop/SIC/SIC_P2/Streamlit_App.py"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and parameters
@st.cache_resource
def load_model():
    model = joblib.load("best_svm_model.pkl")
    with open("svm_rbf_params.json", "r") as f:
        params = json.load(f)
    return model, params

model, params = load_model()

# App title
st.title("🧠 Breast Cancer Diagnostic Classifier (SVM - RBF)")
st.write("Predict breast cancer diagnosis (Malignant or Benign) using an SVM model.")

# Mode selection
mode = st.radio("Choose input mode:", ["🧍 Single Input", "📁 CSV Upload"])

# --- Columns to use ---
feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# --- Single Input Mode ---
if mode == "🧍 Single Input":
    st.subheader("Enter the feature values:")

    user_input = {}
    for col in feature_columns:
        # Dynamic precision settings
        if "area" in col.lower():
            user_input[col] = st.number_input(col, value=0.0, step=0.1, format="%.3f")
        elif any(x in col.lower() for x in ["radius", "perimeter", "texture"]):
            user_input[col] = st.number_input(col, value=0.0, step=0.001, format="%.4f")
        else:
            user_input[col] = st.number_input(col, value=0.0, step=0.0001, format="%.4f")

    if st.button("Predict Diagnosis"):
        X = np.array([list(user_input.values())])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        diagnosis = "Malignant (M)" if pred == 1 or pred == "M" else "Benign (B)"

        st.success(f"Prediction: **{diagnosis}**")
        st.write("### Probability:")
        st.json({
            "Benign (B)": float(proba[0]),
            "Malignant (M)": float(proba[1])
        })

# --- CSV Upload Mode ---
else:
    st.subheader("Upload a CSV file")

    uploaded_file = st.file_uploader("Upload your file (CSV format)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Drop non-feature columns if present
        drop_cols = [col for col in ['id', 'diagnosis'] if col in data.columns]
        data = data.drop(columns=drop_cols, errors='ignore')

        # Ensure feature order
        X = data[feature_columns]

        st.write("### Sample Data:")
        st.dataframe(X.head())

        if st.button("Run Batch Prediction"):
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]  # Probability of Malignant

            result = pd.DataFrame({
                "Prediction": ["Malignant (M)" if p == 1 or p == "M" else "Benign (B)" for p in preds],
                "Probability_Malignant": probs
            })

            final = pd.concat([data, result], axis=1)

            st.write("### Results:")
            st.dataframe(final.head())

            csv = final.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
