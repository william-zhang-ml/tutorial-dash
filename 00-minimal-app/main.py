"""
Example from https://dash.plotly.com/minimal-app.
"""
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

# Underlying app data in tabular form
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

# The app itself
app = Dash()

# Define the front-end layout, requires Dash 2.17.0 or later
app.layout = [
    html.H1(
        children='Title of Dash App',
        style={'textAlign': 'center'}
    ),                             # header
    dcc.Dropdown(
        df.country.unique(),
        'Canada',
        id='dropdown-selection'
    ),                             # dropdown options, default, HTML ID
    dcc.Graph(id='graph-content')  # graph
]


# Dropdown menu callback
@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value: str) -> px.line:
    """Extract population data for a specific country.

    Args:
        value (str): country of interest

    Returns:
        px.line: population versus year data
    """
    dff = df[df.country == value]
    return px.line(dff, x='year', y='pop')


if __name__ == '__main__':
    app.run(debug=True)
