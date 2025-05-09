import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv('C:/Users/DELL/Downloads/modify.csv',encoding='latin1')

# Extract the unique districts
unique_districts = data['district_name'].unique()

# Create a dictionary to store latitude and longitude for each district
coords_dict = {}

# Loop through the unique districts to get coordinates
for district in unique_districts:
    # Filter the data for the current district
    district_data = data[data['district_name'] == district]
    
    # Get the first entry's latitude and longitude (or handle as needed)
    if not district_data.empty:
        latitude = district_data['latitude'].values[0]
        longitude = district_data['longitude'].values[0]
        coords_dict[district] = (latitude, longitude)

# Convert the dictionary to a DataFrame for better visualization
coords_df = pd.DataFrame.from_dict(coords_dict, orient='index', columns=['Latitude', 'Longitude'])

# Display the coordinates
print(coords_df)
# Count how many districts have missing latitude or longitude
missing_coords = coords_df[coords_df.isnull().any(axis=1)]
print(f"Number of districts with missing coordinates: {len(missing_coords)}")

# Optional: Show which districts are missing
print(missing_coords)
data.replace(0, np.nan, inplace=True)
print(data.dtypes)
print(data.head())





