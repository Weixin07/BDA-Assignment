# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the prepared data
data_path = 'preprocessed_runups.csv'
data = pd.read_csv(data_path)

# Define features and target variable
X = data[['Year', 'Latitude', 'Longitude', 'Earthquake Magnitude', 'Distance From Source (km)']]
y = data['Max Water Height (m)']

# Split data into training and testing sets using a hybrid approach
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index]
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

# Further split for final validation to ensure model robustness against unseen scenarios
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize base models for stacking
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbm', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
]

# Initialize the stacking regressor
stack_reg = StackingRegressor(estimators=estimators, final_estimator=GradientBoostingRegressor(random_state=42))

# Train the stacking model
stack_reg.fit(X_train, y_train)

# Predictions and Model Evaluation
y_pred_train = stack_reg.predict(X_train)
y_pred_validate = stack_reg.predict(X_validate)
y_pred_test = stack_reg.predict(X_test)

# Calculate MSE for the stacking model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_validate = mean_squared_error(y_validate, y_pred_validate)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f'Train MSE: {mse_train}, Validate MSE: {mse_validate}, Test MSE: {mse_test}')

# Calculate R² and Adjusted R²
r2 = r2_score(y_validate, y_pred_validate)
adj_r2 = 1 - (1-r2) * (len(y_validate)-1)/(len(y_validate)-X_train.shape[1]-1)
print(f'Validation R²: {r2}, Adjusted R²: {adj_r2}')

# Learning Curves
train_sizes, train_scores, test_scores = learning_curve(
    stack_reg, X, y, cv=tscv, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

# Plot learning curves
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("MSE")
plt.title("Learning Curves")
plt.legend(loc="best")
plt.show()

# Feature importance analysis (for RandomForest part of stack)
rf = stack_reg.named_estimators_['rf']
feature_importances = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

# Save the model
joblib.dump(stack_reg, 'tsunami_prediction_stacking_model.pkl')

# Function to simulate impact based on runup height
def simulate_impact(runup_heights, distances):
    # Placeholder function to calculate damage based on runup height and distance
    damage_ratio = np.exp(-distances/runup_heights)
    return damage_ratio

# Generate a range of distances from the coastline
distances = np.linspace(0, 1000, 100)  # From 0 to 1000 meters

# Simulate impact for a range of hypothetical tsunami heights
for height in [5, 10, 20]:  # Example heights in meters
    impacts = simulate_impact(height, distances)
    plt.plot(distances, impacts, label=f'Runup Height {height}m')

plt.xlabel('Distance from Coastline (m)')
plt.ylabel('Damage Ratio')
plt.title('Simulated Tsunami Impact by Distance from Coastline')
plt.legend()
plt.show()
