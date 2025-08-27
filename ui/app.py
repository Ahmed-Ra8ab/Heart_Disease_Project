
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

st.title("Heart Disease Risk Predictor")
st.write("Enter patient information to get a probability prediction.")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(r"E:\py\project\models\final_model.pkl")

model = load_model()

# Typical UCI columns. Adjust if your pipeline expects different features.
# NOTE: The model built in notebooks uses the reduced/selected features from `selected_features.csv`.
# For production, train a pipeline that includes preprocessing from raw columns.
# Here we accept common raw inputs then assemble a dataframe with matching columns if needed.
st.subheader("Inputs")

age = st.number_input("Age", 18, 100, 54)
sex = st.selectbox("Sex (1=male, 0=female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 246)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=yes,0=no)", [0,1])
restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1=yes,0=no)", [0,1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (1=normal,2=fixed defect,3=reversible)", [1,2,3])

# Build input row. You may need to align these with the features used in training.
input_row = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}])

st.write("Raw input:", input_row)

if st.button("Predict"):
    try:
        # If the trained model expects scaled/encoded features, ensure it is a Pipeline.
        # The notebooks end with a model trained on selected (already encoded) features.
        # For a robust app, re-train and save a Pipeline on raw columns.
        proba = model.predict_proba(input_row)[:,1]
        pred = (proba >= 0.5).astype(int)
        st.success(f"Predicted probability of heart disease: {proba[0]:.3f}")
        st.write(f"Predicted class: {int(pred[0])}")
    except Exception as e:
        st.error("Model input mismatch. Make sure the saved model is a Pipeline that handles preprocessing from raw inputs.")
        st.exception(e)
