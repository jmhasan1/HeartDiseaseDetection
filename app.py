import streamlit as st
import numpy as np
import pickle

# Load the trained pipeline model
with open('heart_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.write("Enter the following patient details to assess the risk of heart disease.")

# Sidebar with input fields
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type", options=[1, 2, 3, 4])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
resting_ecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2])
max_hr = st.sidebar.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
st_slope = st.sidebar.selectbox("ST Slope", options=[1, 2, 3])

# Convert inputs into the correct format
# input_data = np.array([[age, sex, cp, resting_bp, cholesterol, fasting_bs,
#                         resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

import pandas as pd

# Define column names in the exact order used during training
columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
           'fasting blood sugar', 'resting ecg', 'max heart rate',
           'exercise angina', 'oldpeak', 'ST slope']

input_data = pd.DataFrame([[age, sex, cp, resting_bp, cholesterol, fasting_bs,
                            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
                          columns=columns)


# Predict button
if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"❗ High Risk of Heart Disease ({prediction_proba:.2%} probability)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({1 - prediction_proba:.2%} probability)")

st.markdown("---")
st.markdown("Made with ❤️ by Jahid Hasan • Powered by Streamlit + Random Forest Classifier")

