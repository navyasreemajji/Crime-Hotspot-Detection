import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
data = pd.read_csv(r"D:\\Navya\\feature_engineered_data.csv")

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Crime Data Visualization"),
    
    # Dropdown for selecting districts
    dcc.Dropdown(
        id='district-dropdown',
        options=[{'label': district, 'value': district} for district in data['district_name'].unique()],
        value=data['district_name'].unique()[0],  # Default value
        clearable=False
    ),
    
    # Dropdown for selecting type of murder
    dcc.Dropdown(
        id='murder-type-dropdown',
        options=[
            {'label': 'Murder', 'value': 'murder'},
            {'label': 'Culp Homicide', 'value': 'culp homicide'},
            {'label': 'Attempt Murder', 'value': 'attempt murder'},
            {'label': 'Dowry Deaths', 'value': 'dowry deaths'},
            {'label': 'Abetment of Suicide', 'value': 'abetment of suicide'},
        ],
        value='murder',  # Default value
        clearable=False
    ),
    
    # Graph to display the results
    dcc.Graph(id='crime-graph')
])

# Callback to update graph based on dropdown selections
@app.callback(
    Output('crime-graph', 'figure'),
    Input('district-dropdown', 'value'),
    Input('murder-type-dropdown', 'value')
)
def update_graph(selected_district, selected_murder_type):
    # Filter the data based on selections
    filtered_data = data[data['district_name'] == selected_district]

    # Create a bar chart based on the selected type of murder
    if selected_murder_type == 'murder':
        fig = px.bar(filtered_data, x='year', y='murder', title=f'Murder Data in {selected_district}')
    elif selected_murder_type == 'culp homicide':
        fig = px.bar(filtered_data, x='year', y='culp homicide', title=f'Culp Homicide Data in {selected_district}')
    elif selected_murder_type == 'attempt murder':
        fig = px.bar(filtered_data, x='year', y='attempt murder', title=f'Attempt Murder Data in {selected_district}')
    elif selected_murder_type == 'dowry deaths':
        fig = px.bar(filtered_data, x='year', y='dowry deaths', title=f'Dowry Deaths Data in {selected_district}')
    elif selected_murder_type == 'abetment of suicide':
        fig = px.bar(filtered_data, x='year', y='abetment of suicide', title=f'Abetment of Suicide Data in {selected_district}')

    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
