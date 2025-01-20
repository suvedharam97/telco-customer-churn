import pandas as pd

def preprocess_input(input_data):
    # Print the columns of input_data for debugging
    #print(input_data['SeniorCitizen'])
    #print("Columns in input_data before preprocessing:", input_data.columns)

    # Ensure 'tenure' and 'MonthlyCharges' columns are present
    #if 'tenure' not in input_data.columns or 'MonthlyCharges' not in input_data.columns:
    #    raise KeyError("Input data must contain 'tenure' and 'MonthlyCharges' columns")
    input_data['SeniorCitizen'] = input_data['SeniorCitizen'].astype('object')
    input_data['tenure'] = input_data['tenure'].astype(float)
    input_data['MonthlyCharges'] = input_data['MonthlyCharges'].astype(float)

    print(input_data.dtypes)

    # Define the bins and labels
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-12', '12-24', '24-36', '36-48', '48-60', '60-72']
    
    # Create a new column 'tenure_bins' with the binned data
    input_data['tenure_bins'] = pd.cut(input_data['tenure'], bins=bins, labels=labels, right=False)
    input_data.drop('tenure', axis=1, inplace=True)
    print(input_data.dtypes)
    #print("Input Data after binning 'tenure':", input_data.columns)

    # Define the complete set of possible categories for each categorical variable
    categorical_columns = {
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'MultipleLines': ['No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'tenure_bins': ['0-12', '12-24', '24-36', '36-48', '48-60', '60-72']
    }

    # Ensure all categories are present in the input data
    for column, categories in categorical_columns.items():
        if column in input_data.columns:
            input_data[column] = input_data[column].astype(pd.CategoricalDtype(categories=categories))

    #Encode categorical variables
    encoded_data = pd.get_dummies(input_data, drop_first=True)
    print("Encoded Data:", encoded_data.columns)
    return encoded_data