"""
Application for browsing object detection datasets.
"""
import random
from typing import Dict, List, Tuple
from dash import (
    Dash,
    dcc,
    html,
    Input, Output, State, no_update
)
import dash_bootstrap_components as dbc
import hydra
from omegaconf import DictConfig
from PIL.Image import Image
from PIL.ImageDraw import Draw
from torchvision.datasets import CocoDetection


IMG_STORE_ID = 'img-store'
OVERLAY_STORE_ID = 'overlay-store'
IMAGE_ID = 'image'
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
                    html.Div(
                        html.Img(
                            src=None,
                            id=IMAGE_ID,
                            style={
                                'max-height': '100%',
                                'max-width': '100%',
                                'height': 'auto',
                                'width': 'auto',
                                'object-fit': 'contain'
                            }
                        ),
                        style={
                            'height': '648px',
                            'width': '648px',
                            'padding': '4px',
                            'border': '1px solid black',
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }
                    ),
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
    """Determine if app needs to get a different image.

    Args:
        _ (_type_): _description_

    Returns:
        dict: _description_
    """
    idx = random.randint(0, img_store['num_images'] - 1)
    if idx == img_store['idx']:
        return no_update
    new_store = img_store.copy()
    new_store['idx'] = idx
    return new_store


@app.callback(
    Output(IMAGE_ID, 'src'),
    Input(IMG_STORE_ID, 'data')
)
def display_new_image(img_store) -> Image:
    """_summary_

    Args:
        img_store (_type_): _description_

    Returns:
        Image: _description_
    """
    img, _ = get_image(img_store['idx'])
    return img


def get_image(idx: int) -> Tuple[Image, List[Dict]]:
    """Get the idx-th image.

    Args:
        idx (int): dataset index

    Returns:
        Tuple[Image, List[Dict]]: image, instance annotations
    """
    global dataset
    return dataset[idx]


if __name__ == '__main__':
    launch_app()
