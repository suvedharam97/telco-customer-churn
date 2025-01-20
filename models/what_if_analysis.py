import pickle
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.preprocess import preprocess_input



def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_what_if(model, input_data):
    # Modify input_data for What-If scenarios
    predictions = model.predict(input_data)
    return predictions