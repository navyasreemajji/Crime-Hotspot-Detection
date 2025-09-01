import pandas as pd

# Load the dataset you worked with
data = pd.read_csv(r'D:/Navya/modified_dataset.csv', encoding='ISO-8859-1')

# Feature Engineering: Extract time-based features
data['month'] = pd.to_datetime(data['year'], format='%Y').dt.month
data['weekday'] = pd.to_datetime(data['year'], format='%Y').dt.weekday
data['is_weekend'] = data['weekday'] >= 5  # Saturday and Sunday (weekend)

# Save the updated dataset with new features
data.to_csv('feature_engineered_data.csv', index=False)

# Optionally check the first few rows of the updated data
print(data.head())
