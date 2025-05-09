# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
import folium
from folium.plugins import HeatMap


# STEP 1: Load and clean the dataset
df = pd.read_csv("modify.csv", encoding='ISO-8859-1')

# Clean latitude and longitude
df['latitude'] = df['latitude'].astype(str).str.replace('°', '', regex=False).str.strip()
df['longitude'] = df['longitude'].astype(str).str.replace('°', '', regex=False).str.strip()

# Convert to float and handle non-convertible entries
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Drop rows with missing or out-of-bounds coordinates (India-specific filter)
df = df.dropna(subset=['latitude', 'longitude'])
df = df[(df['latitude'].between(6, 38)) & (df['longitude'].between(68, 98))]

# Remove duplicates
df.drop_duplicates(inplace=True)


# STEP 2: Select and process crime data
crime_columns = df.columns[9:]  # Adjust if your dataset has a different structure
crime_data = df[crime_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Normalize crime data
scaler = StandardScaler()
crime_scaled = scaler.fit_transform(crime_data)


# STEP 3: Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(crime_scaled)

# STEP 4: Apply DBSCAN for spatial clustering
coords = df[['latitude', 'longitude']].to_numpy()
kms_per_radian = 6371.0088
epsilon = 10 / kms_per_radian  # 10km radius

# Apply DBSCAN using haversine distance
db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
df['Cluster_DBSCAN'] = db.fit_predict(np.radians(coords))


# STEP 5: Save processed data
df['Total_Crime_Weight'] = crime_data.sum(axis=1)
df.to_csv("clustered_crime_data.csv", index=False)


# STEP 6: Plotly Mapbox Visualization (KMeans)
fig = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    color="Cluster_KMeans",
    hover_name="Cluster_KMeans",
    hover_data=crime_columns,
    zoom=5,
    height=600,
    width=1000,
    title="Crime Clusters on Map (KMeans)",
    mapbox_style="carto-positron",
    opacity=0.6
)
fig.show()


# STEP 7: Folium HeatMap (High-crime emphasis)
# Filter for top 25% high-crime areas
threshold = df['Total_Crime_Weight'].quantile(0.75)
heat_data = [
    [row['latitude'], row['longitude'], row['Total_Crime_Weight']]
    for index, row in df[df['Total_Crime_Weight'] >= threshold].iterrows()
]

# Create the map
map_center = [df["latitude"].mean(), df["longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=6)

# Add heatmap layer
HeatMap(heat_data).add_to(m)

# Save the map
m.save("crime_heatmap.html")
