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


SAMPLE_STORE_ID = 'img-store'
SETTINGS_STORE_ID = 'overlay-store'
IMAGE_ID = 'image'
SELECT_SAMP_DROPDOWN_ID = 'selector'
RANDOM_SAMP_BUTTON_ID = 'randomizer'
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
                    'samp_idx': None,
                    'img': None,
                    'annots': None
                },
                id=SAMPLE_STORE_ID
            ),
            dcc.Store(
                data={
                    'show': False,
                    'highlight_idx': None,
                },
                id=SETTINGS_STORE_ID
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
                        id=SELECT_SAMP_DROPDOWN_ID,
                        style={'width': '16rem'}
                    ),
                    dbc.Button(
                        'Random',
                        color='primary',
                        id=RANDOM_SAMP_BUTTON_ID,
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
                        [
                            {"name": 'col', "id": col}
                            for col in ['Field', 'Value']
                        ],
                        cell_selectable=False,
                        id=META_TABLE_ID
                    ),
                    DataTable(
                        [],
                        [
                            {"name": col, "id": col}
                            for col in ['Box', 'Category']
                        ],
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
    Output(SAMPLE_STORE_ID, 'data'),
    [
        Input(SELECT_SAMP_DROPDOWN_ID, 'value'),
        Input(RANDOM_SAMP_BUTTON_ID, 'n_clicks'),
        State(SAMPLE_STORE_ID, 'data')
    ],
    prevent_initial_call=True
)
def select_and_get_sample(selected: int, _: int, store: Dict) -> Dict:
    """Select and get new sample from dataset.

    Args:
        selected (int): dropdown menu selection
        _ (int): cumulative button clicks
        store (Dict): sample store state

    Returns:
        Dict: new sample store state
    """
    if ctx.triggered_id == SELECT_SAMP_DROPDOWN_ID:
        if selected is None:
            return no_update
        idx = selected
    elif ctx.triggered_id == RANDOM_SAMP_BUTTON_ID:
        idx = randint(0, store['num_images'] - 1)

    if idx == store['samp_idx']:
        return no_update

    img, annots = get_sample(idx)
    new_store = store.copy()
    new_store['samp_idx'] = idx
    new_store['img'] = img
    new_store['annots'] = annots

    return new_store


def get_sample(idx: int) -> Tuple[str, List[Dict]]:
    """Get idx-th sample.

    Args:
        idx (int): dataset index

    Returns:
        Tuple[str, List[Dict]]: image, instance annotations
    """
    global dataset
    img, annots = dataset[idx]

    img_stream = BytesIO()
    img.save(img_stream, format='PNG')
    img_stream.seek(0)
    img = b64encode(img_stream.read()).decode()
    annots = [
        {
            'Box': str(instance['bbox']),
            'Category': str(instance['category_id'])
        }
        for instance in annots
    ]
    annots.sort(key=lambda itm: int(itm['Category']))

    return img, annots


@app.callback(
    Output(SETTINGS_STORE_ID, 'data', allow_duplicate=True),
    Input(SAMPLE_STORE_ID, 'data'),
    State(SETTINGS_STORE_ID, 'data'),
    prevent_initial_call=True
)
def alert_new_sample(_, store: Dict) -> Dict:
    """Clear highlighted box setting and force display update.

    Args:
        _ (Dict): sample store state
        store: display settings store state

    Dict: new display settings store state
    """
    new_store = store.copy()
    new_store['highlight_idx'] = None
    return new_store


@app.callback(
    Output(SETTINGS_STORE_ID, 'data', allow_duplicate=True),
    [
        Input(TOGGLE_ID, 'n_clicks'),
        Input(ANNOT_TABLE_ID, 'active_cell')
    ],
    State(SETTINGS_STORE_ID, 'data'),
    prevent_initial_call=True
)
def update_display_settings(_: int, active_cell: Dict, store: Dict) -> Dict:
    """Update display settings based on UI interactions

    Args:
        _ (int): cumulative button clicks
        active_cell (Dict): active table cell
        store (Dict): display settings store state

    Returns:
        Dict: new display settings store state
    """
    new_store = store.copy()

    if ctx.triggered_id == TOGGLE_ID:
        new_store['show'] = not new_store['show']

    if ctx.triggered_id == ANNOT_TABLE_ID:
        if active_cell is None:
            new_store['highlight_idx'] = None
        else:
            new_store['highlight_idx'] = active_cell['row']

    return new_store


@app.callback(
    [
        Output(IMAGE_ID, 'src'),
        Output(META_TABLE_ID, 'data'),
        Output(ANNOT_TABLE_ID, 'data'),
        Output(ANNOT_TABLE_ID, 'active_cell')
    ],
    State(SAMPLE_STORE_ID, 'data'),
    Input(SETTINGS_STORE_ID, 'data'),
    State(ANNOT_TABLE_ID, 'active_cell'),
    prevent_initial_call=True
)
def update_display(
    sample_store: Dict,
    settings_store: Dict,
    active_cell: Dict
) -> Tuple[Image.Image, Dict, Dict, Dict]:
    """Update user display.

    Args:
        sample_store (Dict): sample store state
        settings_store (Dict): display settings store state
        active_cell (Dict): active cell in annotation table

    Returns:
        Tuple: new image, new table data, new table data, new active cell
    """
    img = Image.open(
        BytesIO(
            b64decode(
                sample_store['img'].encode()
            )
        )
    )
    metadata = [
        {'Field': 'image shape', 'Value': str(img.size)},
        {'Field': 'num boxes', 'Value': len(sample_store['annots'])}
    ]
    annotations = sample_store['annots']

    if settings_store['show']:
        draw = Draw(img)
        for idx, instance in enumerate(annotations):
            box_x, box_y, box_w, box_h = literal_eval(instance['Box'])
            if idx == settings_store['highlight_idx']:
                color = 'cyan'
            else:
                color = 'magenta'
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline=color
            )

    if settings_store['highlight_idx'] is None:
        active_cell = None

    return img, metadata, annotations, active_cell


@app.callback(
    [
        Output(ANNOT_TABLE_ID, 'selected_cells'),
        Output(ANNOT_TABLE_ID, 'style_data_conditional')
    ],
    [
        Input(ANNOT_TABLE_ID, 'active_cell'),
        Input(ANNOT_TABLE_ID, 'selected_cells')
    ]
)
def format_annotation_table(
    active_cell: Dict,
    _: List[Dict]
) -> Tuple[List, List[Dict]]:
    """Highlight active row (not active cell) and disallow multi-selection.

    Args:
        active_cell (Dict): active table cell
        _ (List[Dict]): selected table cells

    Returns:
        Tuple[List, List[Dict]]: no selected cells, row formatting
    """
    style = []
    if active_cell is not None:
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
    return [], style


if __name__ == '__main__':
    launch_app()
