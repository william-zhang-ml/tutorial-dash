"""
Application for browsing object detection datasets.
"""
import random
from dash import (
    Dash,
    dcc,
    Input, Output, State
)
import dash_bootstrap_components as dbc
import hydra
from omegaconf import DictConfig
from torchvision.datasets import CocoDetection


IMG_STORE_ID = 'img-store'
OVERLAY_STORE_ID = 'overlay-store'
BUTTON_ID = 'randomizer'


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
dataset = None


@hydra.main(version_base=None, config_path='.', config_name='default')
def launch_app(cfg: DictConfig = None) -> None:
    """Launch the Dash-Plotly app.

    Args:
        cfg (DictConfig): app configuration
    """
    global dataset
    dataset = CocoDetection(root=cfg.image_path, annFile=cfg.annot_path)

    app.layout = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Button(
                        'Random',
                        color='primary',
                        id=BUTTON_ID,
                        style={'width': '8rem'}
                    ),
                ],
                class_name='col',
                id='left-col',
                style={
                    'border': '1px solid black',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'center',
                    'align-items': 'center'
                }
            ),
            dbc.Col(
                class_name='col',
                id='right-col',
                style={
                    'border': '1px solid black'
                }
            ),
            dcc.Store(
                data={
                    'num_images': len(dataset),
                    'idx': None,
                },
                id=IMG_STORE_ID
            )
        ],
        style={
            'min-height': '100vh',
            'margin': 0,
            'background': '#eeeeee'
        }
    )
    app.run(debug=True)


@app.callback(
    Output(IMG_STORE_ID, 'data'),
    [
        Input(BUTTON_ID, 'n_clicks'),
        State(IMG_STORE_ID, 'data')
    ]
)
def select_new_image(_, img_store) -> dict:
    """Determine if app needs to acquire a different image.

    Args:
        _ (_type_): _description_

    Returns:
        dict: _description_
    """
    new_store = img_store.copy()
    idx = random.randint(0, img_store['num_images'] - 1)
    new_store['idx'] = idx
    print(new_store)
    return new_store


if __name__ == '__main__':
    launch_app()
