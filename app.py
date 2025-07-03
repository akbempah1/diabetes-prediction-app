import numpy as np
import streamlit as st
import joblib

model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Diabetes Prediction App")
st.write("Enter patient details below:")
# my very own app
# Input fields
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=10, max_value=100, value=33)

if st.button('Predict Diabetes'):
    # Collect all inputs into a single array (as expected by the model)
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    # Scale the input (using the same scaler as training!)
    user_input_scaled = scaler.transform(user_input)

    # Get model prediction
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]  # probability of being diabetic

    if prediction == 1:
        st.error(f"⚠️ The model predicts this patient is **likely to have diabetes** (probability: {probability:.2f})")
    else:
        st.success(
            f"✅ The model predicts this patient is **not likely** to have diabetes (probability: {probability:.2f})")
