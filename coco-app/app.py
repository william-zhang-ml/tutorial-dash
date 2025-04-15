"""
Application for browsing object detection datasets.
"""
from ast import literal_eval
from base64 import b64decode, b64encode
from io import BytesIO
from random import randint
from typing import Dict, List, Tuple
from dash import (
    Dash,
    ctx,
    dcc,
    html,
    Input, Output, State, no_update
)
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
import hydra
from omegaconf import DictConfig
from PIL import Image
from PIL.ImageDraw import Draw
from torchvision.datasets import CocoDetection


IMG_STORE_ID = 'img-store'
OVERLAY_STORE_ID = 'overlay-store'
IMAGE_ID = 'image'
DROPDOWN_ID = 'selector'
BUTTON_ID = 'randomizer'
TOGGLE_ID = 'toggle'
META_TABLE_ID = 'metadata-table'
ANNOT_TABLE_ID = 'annotation-table'


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
            dcc.Store(
                data={
                    'num_images': len(dataset),
                    'idx': None,
                    'img': None,
                    'annots': None
                },
                id=IMG_STORE_ID
            ),
            dcc.Store(
                data={
                    'show': False,
                    'highlight_idx': None,
                },
                id=OVERLAY_STORE_ID
            ),
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
                    dcc.Dropdown(
                        list(range(cfg.num_dropdown)),
                        None,
                        id=DROPDOWN_ID,
                        style={'width': '16rem'}
                    ),
                    dbc.Button(
                        'Random',
                        color='primary',
                        id=BUTTON_ID,
                        style={'width': '8rem'}
                    ),
                    dbc.Button(
                        'Toggle Boxes',
                        color='primary',
                        id=TOGGLE_ID,
                        style={'width': '8rem'}
                    )
                ],
                class_name='col',
                id='left-col',
                style={
                    'border': '1px solid black',
                    'padding-top': '4%',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'flex-start',
                    'align-items': 'center'
                }
            ),
            dbc.Col(
                [
                    DataTable(
                        [],
                        [{"name": col, "id": col} for col in ['Field', 'Value']],
                        cell_selectable=False,
                        id=META_TABLE_ID
                    ),
                    DataTable(
                        [],
                        [{"name": col, "id": col} for col in ['Box', 'Category']],
                        id=ANNOT_TABLE_ID,
                        style_table={'height': '480px', 'overflowY': 'auto'}
                    )
                ],
                class_name='col',
                id='right-col',
                style={
                    'border': '1px solid black',
                    'padding-top': '4%',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'flex-start',
                    'align-items': 'center'
                }
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
    [
        Output(IMAGE_ID, 'src'),
        Output(META_TABLE_ID, 'data'),
        Output(ANNOT_TABLE_ID, 'data'),
        Output(ANNOT_TABLE_ID, 'active_cell')
    ],
    State(IMG_STORE_ID, 'data'),
    Input(OVERLAY_STORE_ID, 'data'),
    State(ANNOT_TABLE_ID, 'active_cell'),
    prevent_initial_call=True
)
def update_display(data_store, overlay_store, active_cell):
    img = Image.open(
        BytesIO(
            b64decode(
                data_store['img'].encode()
            )
        )
    )
    metadata = [
        {'Field': 'image shape', 'Value': str(img.size)},
        {'Field': 'num boxes', 'Value': len(data_store['annots'])}
    ]
    annotations = data_store['annots']

    if overlay_store['show']:
        draw = Draw(img)
        for idx, instance in enumerate(annotations):
            box_x, box_y, box_w, box_h = literal_eval(instance['Box'])
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline='cyan' if idx == overlay_store['highlight_idx'] else 'magenta'
            )

    if overlay_store['highlight_idx'] is None:
        active_cell = None

    return img, metadata, annotations, active_cell


@app.callback(
    Output(OVERLAY_STORE_ID, 'data', allow_duplicate=True),
    Input(IMG_STORE_ID, 'data'),
    State(OVERLAY_STORE_ID, 'data'),
    prevent_initial_call=True
)
def update_overlay(_, overlay_store) -> dict:
    new_store = overlay_store.copy()
    new_store['highlight_idx'] = None
    return new_store


@app.callback(
    Output(IMG_STORE_ID, 'data'),
    [
        Input(BUTTON_ID, 'n_clicks'),
        State(IMG_STORE_ID, 'data')
    ],
    prevent_initial_call=True
)
def get_sample(_, store) -> dict:
    """Get a different sample from the dataset.

    Args:
        _ (_type_): _description_

    Returns:
        dict: _description_
    """
    idx = randint(0, store['num_images'] - 1)
    if idx == store['idx']:
        return no_update

    img, annots = get_image(idx)
    img_stream = BytesIO()
    img.save(img_stream, format='PNG')
    img_stream.seek(0)

    new_store = store.copy()
    new_store['idx'] = idx
    new_store['img'] = b64encode(img_stream.read()).decode()
    new_store['annots'] = [
        {
            'Box': str(instance['bbox']),
            'Category': str(instance['category_id'])
        }
        for instance in annots
    ]
    return new_store


@app.callback(
    [
        Output(OVERLAY_STORE_ID, 'data', allow_duplicate=True),
        Output(ANNOT_TABLE_ID, 'style_data_conditional'),
    ],
    [
        Input(TOGGLE_ID, 'n_clicks'),
        Input(ANNOT_TABLE_ID, 'active_cell')
    ],
    State(OVERLAY_STORE_ID, 'data'),
    prevent_initial_call=True
)
def update_overlay_settings(_, active_cell, store):
    if ctx.triggered_id == TOGGLE_ID:
        new_store = store.copy()
        new_store['show'] = not new_store['show']
        style = no_update

    if ctx.triggered_id == ANNOT_TABLE_ID:
        if active_cell is None:
            new_store = store.copy()
            new_store['highlight_idx'] = None
            return new_store, []

        new_store = store.copy()
        new_store['highlight_idx'] = active_cell['row']

        # first condition overrides default active cell highlight
        # second condition highlights cells in active row
        color = '#4682b4'
        style = [
            {
                "if": {"state": "active"},
                'backgroundColor': color,
                'border': f'1px solid {color}',
                'fontWeight': 'bold',
                'color': 'white',
            },
            {
                'if': {'row_index': active_cell['row']},
                'backgroundColor': color,
                'border': f'1px solid {color}',
                'fontWeight': 'bold',
                'color': 'white',
            }
        ]

    return new_store, style


@app.callback(
    Output(ANNOT_TABLE_ID, 'selected_cells'),
    Input(ANNOT_TABLE_ID, 'selected_cells')
)
def suppress_cell_highlight(_) -> List:
    """Unselect selected annotation cells. """
    return []


def get_image(idx: int) -> Tuple[Image.Image, List[Dict]]:
    """Get the idx-th image.

    Args:
        idx (int): dataset index

    Returns:
        Tuple[Image.Image, List[Dict]]: image, instance annotations
    """
    global dataset
    return dataset[idx]


if __name__ == '__main__':
    launch_app()
