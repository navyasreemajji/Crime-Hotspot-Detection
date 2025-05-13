import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("grouped_crime_data.csv")

# Convert latitude and longitude to numeric, force errors to NaN
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')

# Fill missing values
data['latitude'] = data['latitude'].fillna(data['latitude'].mean())
data['longitude'] = data['longitude'].fillna(data['longitude'].mean())

# Encode target variable
le = LabelEncoder()
data['crime_category_encoded'] = le.fit_transform(data['crime_category'])

# Features and labels
X = data[['latitude', 'longitude', 'month', 'weekday', 'is_weekend']]
y = data['crime_category_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Cross-validation score
cv_score = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation accuracy: {cv_score.mean():.4f}")
