import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

import shap

import joblib

import streamlit as st

# Load the model
model = joblib.load('rf.pkl')

# Define feature names
feature_names = ['CD4', 'CD4/CD8 ratio', 'CD8', 'HGB', 'WBC', 'TBIL', 
                 'Age at ART initial', 'PLT', "VL"]
# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
cd4 = st.number_input("CD4:")
# sex: categorical selection
cd4_cd8 = st.number_input("CD4/CD8 ratio:")
# cp: categorical selection
cd8 = st.number_input("CD8:")
# trestbps: numerical input
hgb = st.number_input("HGB:")
# chol: numerical input
wbc = st.number_input("WBC:")
# fbs: categorical selection
tbil = st.number_input("TBIL:")
# restecg: categorical selection
age = st.number_input("Age at ART initial:")
# thalach: numerical input
plt = st.number_input("PLT:")
# exang: categorical selection
vl = st.number_input("VL:")

# Process inputs and make predictions
feature_values = [cd4, cd4_cd8, cd8, hgb, wbc, tbil, age, plt, vl]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")