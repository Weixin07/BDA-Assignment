import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline

# Load data
data_path = 'runups-2024-05-09_11-13-37_+0800.tsv'
df = pd.read_csv(data_path, sep='\t')

### Step 1: Select Data
# Include columns that are critical for the analysis and sorting data by year to respect temporal order
columns = ['Year', 'Latitude', 'Longitude', 'Earthquake Magnitude', 'Distance From Source (km)', 'Max Water Height (m)']
df_selected = df[columns].sort_values('Year')

### Step 2: Clean Data
# Setup for robustness checks with different imputation strategies
imputers = {
    'Iterative': IterativeImputer(random_state=42),
    'Median': SimpleImputer(strategy='median')
}

# Pipeline setup for preprocessing
scaler = RobustScaler()
power_transformer = PowerTransformer(method='yeo-johnson')

# Define a function to apply the transformations and return the transformed data and skewness/kurtosis
def preprocess_and_evaluate(data, imputer):
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('transformer', power_transformer)
    ])
    transformed_data = pipeline.fit_transform(data)
    df_transformed = pd.DataFrame(transformed_data, columns=data.columns)
    skewness = df_transformed.skew()
    kurtosis = df_transformed.kurtosis()
    return df_transformed, skewness, kurtosis

### Data Constructing is skipped as no additional columns are needed
### Step 3: Integrate Data and Format Data
# Perform preprocessing with each imputation strategy and evaluate skewness/kurtosis
for name, imputer in imputers.items():
    df_prepared, skewness, kurtosis = preprocess_and_evaluate(df_selected, imputer)
    print(f"Using {name} Imputation:")
    print("Skewness:\n", skewness)
    print("Kurtosis:\n", kurtosis)
    print()

# Choose the imputation strategy based on skewness/kurtosis results
# Assuming Iterative Imputation was chosen based on evaluation
final_imputer = imputers['Iterative']
df_final_prepared, _, _ = preprocess_and_evaluate(df_selected, final_imputer)

### Step 4: Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
X = df_final_prepared.drop('Max Water Height (m)', axis=1)
y = df_final_prepared['Max Water Height (m)']

# Performing time series cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Print training and testing indices to verify correct temporal splitting
    print("TRAIN:", train_index, "TEST:", test_index)
    print(X_train.head(), X_test.head())

# Print a summary of the cleaned and integrated dataset
print(X_train.head())
