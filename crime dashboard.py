import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from flask import Flask, request, jsonify, send_from_directory

# Flask server
server = Flask(__name__, static_folder=".", static_url_path="")




# Serve predict.html directly
@server.route("/predict.html")
def serve_predict():
    return send_from_directory(".", "predict.html")

# Dash app attached to Flask
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.SANDSTONE],
    url_base_pathname="/dashboard/"
)

# Load dataset
data = pd.read_csv("feature_engineered_data.csv")

crime_columns = [
    'murder', 'culp homicide', 'attempt murder', 'dowry deaths',
    'abetment of suicide', 'assault_on_women', 'heart and life safety',
    'missing_child_kidnpd', 'attempt rape', 'robbery'
]

# Layout
app.layout = dbc.Container([
    html.H1("Crime Data Visualization", className="text-center my-4 text-primary"),

    dbc.Row([
        dbc.Col([
            html.Label("Select District", className="fw-bold"),
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': d, 'value': d} for d in data['district_name'].unique()],
                value=data['district_name'].unique()[0],
                clearable=False
            )
        ], md=6),
        dbc.Col([
            html.Label("Select Crime Type", className="fw-bold"),
            dcc.Dropdown(
                id='crime-type-dropdown',
                options=[{'label': c.replace('_', ' ').title(), 'value': c} for c in crime_columns],
                value='murder',
                clearable=False
            )
        ], md=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Reported Cases", className="card-title text-success"),
                    html.H2(id='kpi-output', className="card-text")
                ])
            ], className="shadow")
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=[dcc.Graph(id='crime-graph')]
            )
        ])
    ])
], fluid=True)

@app.callback(
    Output('crime-graph', 'figure'),
    Output('kpi-output', 'children'),
    Input('district-dropdown', 'value'),
    Input('crime-type-dropdown', 'value')
)
def update_dashboard(district, crime_type):
    filtered_data = data[data['district_name'] == district]
    total = filtered_data[crime_type].sum()

    fig = px.bar(
        filtered_data,
        x='year',
        y=crime_type,
        title=f"{crime_type.replace('_', ' ').title()} in {district} Over the Years",
        labels={'year': 'Year', crime_type: 'Number of Cases'},
        color_discrete_sequence=['steelblue']
    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Reported Cases',
        template='plotly_white',
        title_x=0.5
    )

    fig.update_traces(
        hoverlabel=dict(bgcolor="indianred", font_size=14, font_family="Arial", font_color="white")
    )

    return fig, f"{int(total):,}"

# Dummy prediction API
@server.route("/api/predict", methods=["POST"])
def predict():
    req_data = request.get_json()
    area = req_data.get("area", "")
    return jsonify({
        "result": {
            "message": f"Prediction successful for area: {area}",
            "cluster": "Cluster 2",
            "avg_crime_weight": 7.3
        }
    })

# Run Flask + Dash
if __name__ == "__main__":
    app.run(debug=True)
