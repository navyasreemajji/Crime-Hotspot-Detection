import pandas as pd
import numpy as np

# Load CSV data into DataFrame
df = pd.read_csv('C:/Users/DELL/Downloads/modified csv.csv')

# Display the first few rows of the DataFrame
print(df.head())
df.replace(0, np.nan, inplace=True)
print(df.dtypes)
print(df[['forgery', 'negligent driving', 'criminal trespass', 'cruelty']])





