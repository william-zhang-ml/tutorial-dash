"""
Example from https://dash.plotly.com/datatable.
"""
from dash import Dash, Input, Output, dash_table
import pandas as pd
import dash_bootstrap_components as dbc


# Underlying app data in tabular form
df = pd.read_csv('https://git.io/Juf1t')

# The app itself
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the front-end layout
app.layout = dbc.Container([
    dbc.Label('Click a cell in the table:'),
    dash_table.DataTable(
        df.to_dict('records'),
        [{"name": i, "id": i} for i in df.columns],
        id='tbl'
    ),
    dbc.Alert(id='tbl_out'),
])


# Changes the alert message
@app.callback(
    Output('tbl_out', 'children'),
    Input('tbl', 'active_cell')
)
def update_graphs(active_cell):
    return str(active_cell) if active_cell else "Click the table"


if __name__ == "__main__":
    app.run(debug=True)
