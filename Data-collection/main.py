import pandas as pd
import numpy as np

# Load CSV data into DataFrame
df = pd.read_csv('C:/Users/DELL/Downloads/modified csv.csv')

# Display the first few rows of the DataFrame
print(df.head())
df.replace(0, np.nan, inplace=True)
print(df.dtypes)
print(df[['forgery', 'negligent driving', 'criminal trespass', 'cruelty']].head(10))
# Convert timestamp column (replace 'timestamp' with your actual column name)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract hour, weekday, weekend
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday
df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)



