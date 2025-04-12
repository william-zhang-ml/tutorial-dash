
"""
An application to teach me the following skills that are good to know.
- display an image
- annotate the image with bounding boxes and text
- switch the image from a dropdown menu
- switch the image randomly from a button press
- toggle bounding boxes on and off
"""
from ast import literal_eval
import logging
import random
from dash import Dash, html, dcc, Input, Output, State, no_update, ctx, dash_table
import dash_bootstrap_components as dbc
import hydra
from omegaconf import DictConfig
import pandas as pd
from PIL.Image import Image
from PIL.ImageDraw import Draw
from torchvision.datasets import CocoDetection


STORE_ID = 'store'
DROPDOWN_ID = 'selector'
BUTTON_ID = 'randomizer'
CHECK_ID = 'annot-check'
IMAGE_ID = 'img'
METATABLE_ID = 'meta-table'
ANNOTTABLE_ID = 'annot-table'


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
dataset = None
curr_idx, curr_img, curr_annots = None, None, None
highlight = None


@hydra.main(version_base=None, config_path='.', config_name='default')
def launch_app(cfg: DictConfig) -> None:
    """Launch the Dash-Plotly app.

    Args:
        cfg (DictConfig): app configuration
    """
    global dataset, curr_idx, curr_img, curr_annots
    dataset = CocoDetection(
        root=cfg.image_path,
        annFile=cfg.annot_path
    )
    curr_idx = 0
    curr_img, curr_annots = dataset[curr_idx]  # img is PIL
    meta_table, annot_table = pack_tables()

    # dashboard structure
    app.layout = dbc.Row(
        [
            html.Div(
                [
                    dbc.Row(
                        html.Img(
                            src=curr_img,
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
                            'width': '100%',
                            'padding': 0,
                            'border': '2px solid black',
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }
                    ),
                    dbc.Row(
                        [
                            dcc.Dropdown(
                                list(range(cfg.num_dropdown)),
                                curr_idx,
                                id=DROPDOWN_ID,
                                style={'width': '16rem'}
                            ),
                            dbc.Button(
                                'Random',
                                color='primary', 
                                id=BUTTON_ID,
                                style={'width': '8rem'}
                            ),
                            dcc.Checklist(
                                options=['boxes'],
                                id=CHECK_ID,
                                style={'width': 'auto'}
                            ),
                        ],
                        style={
                            'width': '100%',
                            'padding': '1rem',
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'space-evenly'
                        }
                    )
                ],
                id='left-col',
                style={
                    'height': '100%',
                    'width': '648px',
                    'padding-top': '4%',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'flex-start',
                    'align-items': 'center'
                }
            ),
            html.Div(
                [
                    dash_table.DataTable(
                        meta_table,
                        [{"name": col, "id": col} for col in ['Field', 'Value']],
                        cell_selectable=False,
                        id=METATABLE_ID
                    ),
                    dash_table.DataTable(
                        annot_table,
                        [{"name": col, "id": col} for col in ['Field', 'Value']],
                        id=ANNOTTABLE_ID
                    ),
                ],
                id='right-col',
                style={
                    'height': '100%',
                    'max-width': '30%',
                    'padding-top': '4%'
                }
            ),
            dcc.Store(STORE_ID)
        ],
        style={
            'background': '#eeeeee',
            'height': '100vh',
            'margin': 0,
            'padding': 0,
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        },
    )

    app.run(debug=True)


def pack_tables() -> pd.DataFrame:
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """
    global dataset, curr_idx, curr_img, curr_annots
    meta_table = [
        {'Field': 'image shape', 'Value': str(curr_img.size)},
        {'Field': 'num boxes', 'Value': len(curr_annots)}
    ]
    annot_table = [
        {
            'Field': 'box',
            'Value': f'{instance["bbox"]}, {instance["category_id"]}'
        }
        for instance in curr_annots
    ]
    return meta_table, annot_table


def get_image(show_boxes: bool = True) -> Image:
    """Load specific image from dataset.

    Args:
        idx (int): which image
        show_boxes (bool): whether to draw bounding boxes

    Returns:
        Image: idx-th image
    """
    global dataset, curr_idx, curr_img, curr_annots, highlight
    img = curr_img.copy()
    if show_boxes:
        draw = Draw(img)
        for instance in curr_annots:
            box_x, box_y, box_w, box_h = instance['bbox']
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline='magenta'
            )
        if highlight is not None:
            box_x, box_y, box_w, box_h = highlight
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline='cyan'
            )
    return img


# Callback to update image display
@app.callback(
    [
        Output(IMAGE_ID, 'src'),
        Output(DROPDOWN_ID, 'value'),
        Output(METATABLE_ID, 'data'),
        Output(ANNOTTABLE_ID, 'data')
    ],
    [
        Input(DROPDOWN_ID, 'value'),
        Input(BUTTON_ID, 'n_clicks'),
        Input(CHECK_ID, 'value'),
        Input(STORE_ID, 'data')
    ]
)
def update_datastate(idx: int, n_clicks: int, checked: str, store_data) -> Image:
    """Load random image from dataset.

    Returns:
        Image: random image
    """
    global dataset, curr_idx, curr_img, curr_annots
    if ctx.triggered_id == CHECK_ID:
        return get_image(checked is not None and 'boxes' in checked), no_update, no_update, no_update

    if ctx.triggered_id == DROPDOWN_ID:
        if idx is None:
            return no_update
    elif ctx.triggered_id == BUTTON_ID:
        idx = random.randint(0, len(dataset) - 1)

    if idx == curr_idx:
        return no_update, no_update, no_update, no_update

    # only runs when need to present user a different image
    curr_idx = idx
    curr_img, curr_annots = dataset[idx]
    return get_image(checked is not None and 'boxes' in checked), idx, *pack_tables()


@app.callback(
    [
        Output(ANNOTTABLE_ID, 'style_data_conditional'),
        Output(ANNOTTABLE_ID, 'selected_cells'),
        Output(STORE_ID, 'data')
    ],
    [
        Input(ANNOTTABLE_ID, 'active_cell'),
        Input(ANNOTTABLE_ID, 'selected_cells')
    ],
    State(ANNOTTABLE_ID, 'data')
)
def update_graphs(active_cell, selected_cells, table_data):
    if active_cell is None:
        return no_update  # Dash will run callbacks on launch

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

    global highlight
    highlight = table_data[active_cell['row']]['Value']
    highlight = literal_eval(highlight.rsplit(',', 1)[0])

    # setting 'selected_cells' to [] spoofs disabling shift-clicks
    return style, [], highlight


if __name__ == '__main__':
    launch_app()
