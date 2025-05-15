import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings("ignore")

# Create folder to save plots if not exist
os.makedirs("Crime-Prediction-Trends/plots", exist_ok=True)

# Load data
data = pd.read_csv("grouped_crime_data.csv")

# Filter dataset for only 'Property' and 'Violence'
data = data[data['crime_category'].isin(['Property', 'Violence'])]

# Convert latitude and longitude to numeric and fill missing values with mean
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['latitude'] = data['latitude'].fillna(data['latitude'].mean())

data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data['longitude'] = data['longitude'].fillna(data['longitude'].mean())

# Encode the target labels
le = LabelEncoder()
data['crime_category_encoded'] = le.fit_transform(data['crime_category'])

# Features and target
X = data[['latitude', 'longitude', 'month', 'weekday', 'is_weekend']]
y = data['crime_category_encoded']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Random Forest with class_weight='balanced'
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
}

# Grid search with 3-fold cross-validation
grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# Best model from grid search
model = grid.best_estimator_

# Predict on test set
y_pred = model.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix with Class Weighting - Property vs Violence")
plt.tight_layout()
plt.savefig("Crime-Prediction-Trends/plots/confusion_matrix_weighted.png")
plt.show()
