# main.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
import folium
from folium.plugins import HeatMap
import tempfile
import webbrowser

warnings.filterwarnings("ignore")
os.makedirs("Crime-Prediction-Trends/plots", exist_ok=True)
os.makedirs("Data-collection", exist_ok=True)


def feature_engineering():
    df = pd.read_csv('Data-collection/modified_dataset.csv', encoding='ISO-8859-1')
    df['month'] = pd.to_datetime(df['year'], format='%Y').dt.month
    df['weekday'] = pd.to_datetime(df['year'], format='%Y').dt.weekday
    df['is_weekend'] = df['weekday'] >= 5
    df.to_csv('Data-collection/feature_engineered_data.csv', index=False)
    print("✅ Feature Engineering Complete")
    return df


def group_crimes():
    df = pd.read_csv('Data-collection/feature_engineered_data.csv')
    crime_columns = [
        'murder', 'attempt murder', 'culp homicide', 'rape', 'attempt rape',
        'hurt by weapon', 'vlntrly hurt', 'simple hurt', 'dangerous weapon',
        'motor vehicle theft', 'other thefts', 'robbery', 'day time burglary', 'night burglary',
        'cheating', 'forgery', 'fake currency', 'blackmailing',
        'assault_on_women', 'sexual harassment', 'cruelty', 'stalking',
        'kidnp_abdctn_marrg', 'missing_child_kidnpd', 'other_kidnp_abduc'
    ]
    crime_columns = [col for col in crime_columns if col in df.columns]
    crime_df = df.melt(
        id_vars=['latitude', 'longitude', 'month', 'weekday', 'is_weekend'],
        value_vars=crime_columns,
        var_name='crime_type',
        value_name='count'
    )
    crime_df = crime_df[crime_df['count'] > 0]
    crime_group_map = {
        **dict.fromkeys(['murder', 'attempt murder', 'culp homicide', 'rape', 'attempt rape',
                        'hurt by weapon', 'vlntrly hurt', 'simple hurt', 'dangerous weapon'], 'Violence'),
        **dict.fromkeys(['motor vehicle theft', 'other thefts', 'robbery', 'day time burglary', 'night burglary',
                        'cheating', 'forgery', 'fake currency', 'blackmailing'], 'Property'),
        **dict.fromkeys(['assault_on_women', 'sexual harassment', 'cruelty', 'stalking'], 'Women-related'),
        **dict.fromkeys(['kidnp_abdctn_marrg', 'missing_child_kidnpd', 'other_kidnp_abduc'], 'Kidnapping'),
    }
    crime_df['crime_category'] = crime_df['crime_type'].map(crime_group_map)
    crime_df.to_csv('Data-collection/grouped_crime_data.csv', index=False)
    print("✅ Crimes Grouped and Saved")
    return crime_df


def train_model():
    data = pd.read_csv("Data-collection/grouped_crime_data.csv")
    data = data[data['crime_category'].isin(['Property', 'Violence'])]

    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['latitude'] = data['latitude'].fillna(data['latitude'].mean())

    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data['longitude'] = data['longitude'].fillna(data['longitude'].mean())

    le = LabelEncoder()
    data['crime_category_encoded'] = le.fit_transform(data['crime_category'])

    X = data[['latitude', 'longitude', 'month', 'weekday', 'is_weekend']]
    y = data['crime_category_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10],
        'min_samples_split': [2],
    }
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    y_pred = model.predict(X_test)

    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, 'random_forest_classifier.pkl')
    print("✅ Model saved")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("Crime-Prediction-Trends/plots/confusion_matrix.png")
    plt.close()
    print("✅ Model Training & Confusion Matrix Complete")


def show_feature_importance():
    model = joblib.load('random_forest_classifier.pkl')
    data = pd.read_csv("Data-collection/grouped_crime_data.csv")
    data = data[data['crime_category'].isin(['Property', 'Violence'])]

    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['latitude'] = data['latitude'].fillna(data['latitude'].mean())

    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data['longitude'] = data['longitude'].fillna(data['longitude'].mean())

    X = data[['latitude', 'longitude', 'month', 'weekday', 'is_weekend']]

    importances = model.feature_importances_
    features = X.columns
    df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Importance', y='Feature', palette='mako')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig("Crime-Prediction-Trends/plots/feature_importance.png")
    plt.close()
    print("✅ Feature Importance Saved")


def cluster_and_visualize():
    df = pd.read_csv("Data-collection/modified_dataset.csv", encoding='ISO-8859-1')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'].between(6, 38)) & (df['longitude'].between(68, 98))]
    df.drop_duplicates(inplace=True)

    crime_columns = df.columns[9:]
    crime_data = df[crime_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    crime_scaled = scaler.fit_transform(crime_data)

    df['Cluster_KMeans'] = KMeans(n_clusters=4, random_state=42).fit_predict(crime_scaled)

    coords = df[['latitude', 'longitude']].to_numpy()
    epsilon = 10 / 6371.0088
    df['Cluster_DBSCAN'] = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit_predict(np.radians(coords))

    df['Total_Crime_Weight'] = crime_data.sum(axis=1)
    df.to_csv("clustered_crime_data.csv", index=False)

    fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude", color="Cluster_KMeans",
        zoom=5, mapbox_style="carto-positron", opacity=0.6,
        title="Crime Clusters (KMeans)", height=600
    )
    fig.show()

    # Folium Heatmap
    threshold = df['Total_Crime_Weight'].quantile(0.75)
    heat_data = [[row['latitude'], row['longitude'], row['Total_Crime_Weight']] for i, row in df[df['Total_Crime_Weight'] >= threshold].iterrows()]
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)
    HeatMap(heat_data).add_to(m)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    m.save(temp_file.name)
    webbrowser.open('file://' + os.path.realpath(temp_file.name))
    print("✅ Clustering and Heatmap Completed")


# --- Run Entire Pipeline ---
if __name__ == "__main__":
    feature_engineering()
    group_crimes()
    train_model()
    show_feature_importance()
    cluster_and_visualize()
