import pandas as pd

# Load the feature engineered dataset
df = pd.read_csv('feature_engineered_data.csv')

# Step 1: List of columns representing individual crime types
crime_columns = [
    'murder', 'attempt murder', 'culp homicide', 'rape', 'attempt rape',
    'hurt by weapon', 'vlntrly hurt', 'simple hurt', 'dangerous weapon',
    'motor vehicle theft', 'other thefts', 'robbery', 'day time burglary', 'night burglary',
    'cheating', 'forgery', 'fake currency', 'blackmailing',
    'assault_on_women', 'sexual harassment', 'cruelty', 'stalking',
    'kidnp_abdctn_marrg', 'missing_child_kidnpd', 'other_kidnp_abduc'
]

# Check which columns are missing in the dataset
missing_columns = [col for col in crime_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")

# Remove missing columns from the crime_columns list
crime_columns = [col for col in crime_columns if col in df.columns]

# Step 2: Melt the crime columns to long format
crime_df = df.melt(
    id_vars=['latitude', 'longitude', 'month', 'weekday', 'is_weekend'],
    value_vars=crime_columns,
    var_name='crime_type',
    value_name='count'
)

# Step 3: Keep only rows where count > 0 (actual crimes)
crime_df = crime_df[crime_df['count'] > 0]

# Step 4: Map detailed crime types to broad categories
crime_group_map = {
    'murder': 'Violence', 'attempt murder': 'Violence', 'culp homicide': 'Violence',
    'rape': 'Violence', 'attempt rape': 'Violence', 'hurt by weapon': 'Violence',
    'vlntrly hurt': 'Violence', 'simple hurt': 'Violence', 'dangerous weapon': 'Violence',

    'motor vehicle theft': 'Property', 'other thefts': 'Property', 'robbery': 'Property',
    'day time burglary': 'Property', 'night burglary': 'Property', 'cheating': 'Property',
    'forgery': 'Property', 'fake currency': 'Property', 'blackmailing': 'Property',

    'assault_on_women': 'Women-related', 'sexual harassment': 'Women-related',
    'cruelty': 'Women-related', 'stalking': 'Women-related',

    'kidnp_abdctn_marrg': 'Kidnapping', 'missing_child_kidnpd': 'Kidnapping',
    'other_kidnp_abduc': 'Kidnapping',
}

crime_df['crime_category'] = crime_df['crime_type'].map(crime_group_map)

# Optional: Save this processed dataset
crime_df.to_csv('grouped_crime_data.csv', index=False)

print("Crime data grouped and saved successfully.")
