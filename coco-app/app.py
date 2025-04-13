"""
Application for browsing object detection datasets.
"""
from dash import (
    Dash,
    dcc
)
import dash_bootstrap_components as dbc
import hydra
from omegaconf import DictConfig


STORE_ID = 'store'


app = Dash()


@hydra.main(version_base=None, config_path='.', config_name='default')
def launch_app(cfg: DictConfig) -> None:
    """Launch the Dash-Plotly app.

    Args:
        cfg (DictConfig): app configuration
    """
    app.layout = dbc.Row(
        [
            dbc.Col(
                class_name='col',
                id='left-col',
                style={
                    'border': '1px solid black'
                }
            ),
            dbc.Col(
                class_name='col',
                id='right-col',
                style={
                    'border': '1px solid black'
                }
            ),
            dcc.Store(id=STORE_ID)
        ],
        style={
            'min-height': '100vh',
            'margin': 0,
            'background': '#eeeeee'
        }
    )
    app.run(debug=True)


if __name__ == '__main__':
    launch_app()
