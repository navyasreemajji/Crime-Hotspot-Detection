import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("Crime-Prediction-Trends/plots", exist_ok=True)

# Load data
data = pd.read_csv("grouped_crime_data.csv")

# Aggregate total count per crime type
crime_type_counts = data.groupby('crime_type')['count'].sum()

# Exclude specific low-frequency or unwanted crime types
excluded_types = [
    'blackmailing', 'fake currency', 'day time burglary', 'hurt by weapon',
    'missing_child_kidnpd', 'forgery', 'stalking', 'vlntrly hurt', 'other_kidnp_abduc'
]
filtered_crime_types = crime_type_counts[~crime_type_counts.index.isin(excluded_types)]


plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(9, 9), facecolor='black')

wedges, texts, autotexts = ax.pie(
    filtered_crime_types,
    labels=filtered_crime_types.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('Set2'),  # brighter color set
    textprops={'color': 'white', 'fontsize': 10}
)

# Make percentages bold and bigger
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

# Title with custom color
plt.title('Crime Type Distribution ', fontsize=14, color='white', weight='bold')
plt.axis('equal')

# Save & show
fig.savefig("Crime-Prediction-Trends/plots/crime_type_filtered_pie_chart.png", facecolor='black')
plt.show()
