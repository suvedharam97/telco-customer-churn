import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.what_if_analysis import load_model, predict_what_if


def test_load_model():
    model = load_model("models/best_logreg_model.pkl")
    assert model is not None

def test_predict_what_if():
    model = load_model("models/best_logreg_model.pkl")
    input_data = np.array([[0,'Yes','No',1,'No phone service','DSL','No','Yes','No','No','No','No','Month-to-month','Yes','Electronic check',29.85]]) # Example input
    
    # Convert input_data to DataFrame
    input_data_df = pd.DataFrame(input_data, columns=['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges'])
    input_data_df['tenure'] = input_data_df['tenure'].astype(float)
    input_data_df['MonthlyCharges'] = input_data_df['MonthlyCharges'].astype(float)
    print(input_data_df.head())
    predictions = predict_what_if(model, input_data_df)
    assert len(predictions) == 1

if __name__ == "__main__":
    test_load_model()
    test_predict_what_if()