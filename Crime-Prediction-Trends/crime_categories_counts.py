import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("Crime-Prediction-Trends/plots", exist_ok=True)

# Load data
data = pd.read_csv("grouped_crime_data.csv")

# Aggregate total count per crime category
category_counts = data.groupby('crime_category')['count'].sum().sort_values(ascending=False)

# Plot all categories 
plt.figure(figsize=(10, 5))
sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')

plt.title('Crime Counts by Crime Category')
plt.xlabel('Total Crime Count')
plt.ylabel('Crime Category')
plt.tight_layout()

# Save & show
plt.savefig("Crime-Prediction-Trends/plots/crime_categories_counts.png")
plt.show()
