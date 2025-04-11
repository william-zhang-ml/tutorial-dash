
"""
An application to teach me the following skills that are good to know.
- display an image
- annotate the image with bounding boxes and text
- switch the image from a dropdown menu
- switch the image randomly from a button press
- toggle bounding boxes on and off
"""
import logging
import random
from dash import Dash, html, dcc, Input, Output, no_update, ctx, dash_table
import dash_bootstrap_components as dbc
import hydra
from omegaconf import DictConfig
import pandas as pd
from PIL.Image import Image
from PIL.ImageDraw import Draw
from torchvision.datasets import CocoDetection


DROPDOWN_ID = 'selector'
BUTTON_ID = 'randomizer'
CHECK_ID = 'annot-check'
IMAGE_ID = 'img'
TABLE_ID = 'table'


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
dataset = None
curr_idx, curr_img, curr_annots = None, None, None


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
    sample_data = pack_table()

    # dashboard structure
    app.layout = html.Div([
        dcc.Dropdown(
            list(range(cfg.num_dropdown)),
            curr_idx,
            id=DROPDOWN_ID
        ),
        dbc.Button('Random', color='primary', id=BUTTON_ID),
        dcc.Checklist(options=['boxes'], id=CHECK_ID),
        html.Img(src=curr_img, id=IMAGE_ID),
        dash_table.DataTable(
            sample_data.to_dict('records'),
            [{"name": col, "id": col} for col in sample_data.columns],
            id=TABLE_ID
        )
    ])

    app.run(debug=True)


def pack_table() -> pd.DataFrame:
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """
    global dataset, curr_idx, curr_img, curr_annots
    records = [
        {'Field': 'image shape', 'Value': str(curr_img.size)},
        {'Field': 'num boxes', 'Value': len(curr_annots)}
    ]
    for instance in curr_annots:
        records.append({
            'Field': 'box',
            'Value': f'{instance["bbox"]}, {instance["category_id"]}'
        })
    return pd.DataFrame.from_records(records)


def get_image(show_boxes: bool = True) -> Image:
    """Load specific image from dataset.

    Args:
        idx (int): which image
        show_boxes (bool): whether to draw bounding boxes

    Returns:
        Image: idx-th image
    """
    global dataset, curr_idx, curr_img, curr_annots
    img = curr_img.copy()
    if show_boxes:
        draw = Draw(img)
        for instance in curr_annots:
            box_x, box_y, box_w, box_h = instance['bbox']
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline='magenta',
            )
    return img


# Callback to update image display
@app.callback(
    [
        Output(IMAGE_ID, 'src'),
        Output(DROPDOWN_ID, 'value'),
        Output(TABLE_ID, 'data')
    ],
    [
        Input(DROPDOWN_ID, 'value'),
        Input(BUTTON_ID, 'n_clicks'),
        Input(CHECK_ID, 'value')
    ]
)
def update_datastate(idx: int, n_clicks: int, checked: str) -> Image:
    """Load random image from dataset.

    Returns:
        Image: random image
    """
    global dataset, curr_idx, curr_img, curr_annots
    if ctx.triggered_id == CHECK_ID:
        return get_image(checked is not None and 'boxes' in checked), no_update, no_update

    if ctx.triggered_id == DROPDOWN_ID:
        if idx is None:
            return no_update
    elif ctx.triggered_id == BUTTON_ID:
        idx = random.randint(0, len(dataset) - 1)

    if idx == curr_idx:
        return no_update, no_update, no_update

    # only runs when need to present user a different image
    curr_idx = idx
    curr_img, curr_annots = dataset[idx]
    return get_image(checked is not None and 'boxes' in checked), idx, pack_table().to_dict('records')


if __name__ == '__main__':
    launch_app()
