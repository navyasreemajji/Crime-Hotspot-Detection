import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Create folder for plots if it doesn't exist
os.makedirs("Crime-Prediction-Trends/plots", exist_ok=True)

# Load trained model
model = joblib.load('random_forest_classifier.pkl')

# Load dataset
data = pd.read_csv("grouped_crime_data.csv")

# Filter for 'Property' and 'Violence' categories only
data = data[data['crime_category'].isin(['Property', 'Violence'])]

# Convert latitude and longitude to numeric and fill missing with mean
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['latitude'] = data['latitude'].fillna(data['latitude'].mean())

data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data['longitude'] = data['longitude'].fillna(data['longitude'].mean())

# Select features (must be same as used in training)
X = data[['latitude', 'longitude', 'month', 'weekday', 'is_weekend']]

# Get feature importances from loaded model
importances = model.feature_importances_
features = X.columns

# Create a dataframe sorted by importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='mako')
plt.title('Feature Importance - Random Forest Classifier')
plt.tight_layout()

# Save the plot
plt.savefig('Crime-Prediction-Trends/plots/feature_importance.png')
plt.show()
