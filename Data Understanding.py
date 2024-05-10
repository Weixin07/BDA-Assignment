import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## Step 1: Collect Initial Data
df = pd.read_csv('runups-2024-05-09_11-13-37_+0800.tsv', sep='\t')

## Step 2: Describe Data
print("First few rows of the dataframe:")
print(df.head())
print("\nData types of columns:")
print(df.dtypes)
print("\nSize of the dataset:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

## Step 3: Explore Data 
# Convert columns to numeric where applicable, setting errors='coerce' to handle non-convertible values
numeric_cols = ['Year', 'Mo', 'Dy', 'Hr', 'Mn', 'Sec', 'Earthquake Magnitude', 
                'Max Water Height (m)', 'Max Inundation Distance (m)', 'Deaths']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Descriptive statistics for numeric columns
print("\nDescriptive statistics for numeric columns:")
print(df.describe())

# Check for missing values in the dataframe
print("\nMissing values in each column:")
print(df.isnull().sum())

# Additional statistics like skewness and kurtosis
print("\nSkewness of the numeric data:")
print(df[numeric_cols].skew())
print("\nKurtosis of the numeric data:")
print(df[numeric_cols].kurtosis())

# Replot histograms for all numeric columns after cleaning data types
print("\nHistograms for numeric columns:")
df_numeric = df.select_dtypes(include=[np.number])  # Select only numeric columns
df_numeric.hist(bins=30, figsize=(20, 15))
plt.show()

# Correlation matrix heatmap of numeric columns
print("\nCorrelation matrix heatmap:")
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

## Temporal Analysis
# Trends over the years in key metrics
plt.figure(figsize=(15, 7))
df.groupby('Year')['Max Water Height (m)'].mean().plot(kind='line')
plt.title('Average Max Water Height by Year')
plt.ylabel('Max Water Height (m)')
plt.show()

plt.figure(figsize=(15, 7))
df.groupby('Year')['Deaths'].sum().plot(kind='line')
plt.title('Total Deaths by Year')
plt.ylabel('Deaths')
plt.show()

## Step 4: Verify Data Quality
# Box plots to identify outliers in a key column
print("\nBox plot for 'Max Water Height (m)' to identify outliers:")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Max Water Height (m)'])
plt.show()

## Additional Step: Handling Missing Data and Outliers
# Filling missing values with median for numerical columns only
df_numeric.fillna(df_numeric.median(), inplace=True)

# Removing outliers for 'Max Water Height (m)' using IQR method
Q1 = df_numeric['Max Water Height (m)'].quantile(0.25)
Q3 = df_numeric['Max Water Height (m)'].quantile(0.75)
IQR = Q3 - Q1
filter = (df_numeric['Max Water Height (m)'] >= Q1 - 1.5 * IQR) & (df_numeric['Max Water Height (m)'] <= Q3 + 1.5 * IQR)
df_filtered = df_numeric.loc[filter]

print("\nRevised box plot for 'Max Water Height (m)' after outlier removal:")
sns.boxplot(x=df_filtered['Max Water Height (m)'])
plt.show()
