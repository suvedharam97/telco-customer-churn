import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.what_if_analysis import load_model, predict_what_if
import pandas as pd
from preprocess import preprocess_input
import seaborn as sns

# Load the pre-trained model
model_path = "models/best_logreg_model.pkl"
model = load_model(model_path)

st.title("Churn Prediction What-If Analysis")
st.sidebar.header("Modify Features")

# Example input features
tenure = st.sidebar.slider("tenure", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0, 200, 70)
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

print(SeniorCitizen)

# Create a DataFrame for the input data
input_df = pd.DataFrame({
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges]    
})

print(input_df.dtypes)


# Preprocess the input data
input_data = preprocess_input(input_df)


# Predict and show results
predictions = predict_what_if(model, input_data)
st.write(f"Predicted Churn: {predictions[0]}")
import matplotlib.pyplot as plt

# Predict probabilities
probabilities = model.predict_proba(input_data)[:, 1]

# Set the style of the visualization
sns.set(style="whitegrid")

# Plot the probability of churn
fig, ax = plt.subplots()
sns.barplot(x=["Churn Probability"], y=[probabilities[0]], palette="viridis", ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Probability of Churn")

# Add the probability value on top of the bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

st.pyplot(fig)