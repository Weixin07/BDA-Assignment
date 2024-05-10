import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset
file_path = 'tsunamiinundation.xlsx'
data = pd.read_excel(file_path)
print("Data loaded successfully. Initial shape:", data.shape)

# Data Cleaning
data.dropna(axis=1, how='all', inplace=True)  # Drop fully empty columns
print("Columns with all missing values dropped. Shape:", data.shape)

data.ffill(inplace=True)  # Forward fill for temporal columns
print("Forward fill applied.")

# Handling missing values for numeric columns specifically
numeric_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
print("Missing numeric values imputed.")

# Normalize/Standardize numerical columns
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
print("Numerical data standardized.")

# Categorical data handling: Convert all categorical data to numeric
categorical_columns = data.select_dtypes(exclude=[np.number]).columns
for column in categorical_columns:
    data[column] = data[column].astype('category').cat.codes
print("Categorical data converted to numeric codes.")

# Check for presence and fill-status of essential date components
date_components = ['Year', 'Mo', 'Dy', 'Hr', 'Mn', 'Sec']
required_components = ['Year', 'Mo', 'Dy']  # These are absolutely necessary
missing_components = [comp for comp in required_components if comp not in data.columns or data[comp].isna().all()]

if not missing_components:
    # Ensure non-required components are filled with default values if missing
    default_values = {comp: 0 for comp in date_components if comp not in required_components}
    data.fillna(default_values, inplace=True)
    
    # Attempt datetime conversion
    try:
        data['Date'] = pd.to_datetime(data[date_components], errors='coerce')
        print("Date-Time conversion successful.")
    except Exception as e:
        print(f"Failed to convert to datetime: {str(e)}")
else:
    print(f"Missing essential date components: {missing_components}. Cannot create datetime object.")


# Save the cleaned and transformed data
cleaned_file_path = 'cleaned_tsunami_data.xlsx'
data.to_excel(cleaned_file_path, index=False)

print("Data preparation completed. Cleaned data saved to:", cleaned_file_path)
