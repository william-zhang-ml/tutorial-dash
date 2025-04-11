
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
from dash import Dash, html, dcc, callback, Input, Output, no_update, ctx
import hydra
from omegaconf import DictConfig
from PIL.Image import Image
from PIL.ImageDraw import Draw
from torchvision.datasets import CocoDetection


DROPDOWN_ID = 'selector'
BUTTON_ID = 'randomizer'
CHECK_ID = 'annot-check'
IMAGE_ID = 'img'


app = Dash()
dataset = None


@hydra.main(version_base=None, config_path='.', config_name='default')
def launch_app(cfg: DictConfig) -> None:
    """Launch the Dash-Plotly app.

    Args:
        cfg (DictConfig): app configuration
    """
    global dataset
    dataset = CocoDetection(
        root=cfg.image_path,
        annFile=cfg.annot_path
    )
    img, _ = dataset[0]  # img is PIL

    # dashboard structure
    app.layout = html.Div([
        dcc.Dropdown(
            list(range(cfg.num_dropdown)),
            0,
            id=DROPDOWN_ID
        ),
        html.Button('Random', id=BUTTON_ID),
        dcc.Checklist(options=['boxes'], id=CHECK_ID),
        html.Img(src=img, id=IMAGE_ID)
    ])

    app.run(debug=True)


def get_image(idx: int, show_boxes: bool = True) -> Image:
    """Load specific image from dataset.

    Args:
        idx (int): which image
        show_boxes (bool): whether to draw bounding boxes

    Returns:
        Image: idx-th image
    """
    global dataset
    img, target = dataset[idx]
    if show_boxes:
        draw = Draw(img)
        for instance in target:
            box_x, box_y, box_w, box_h = instance['bbox']
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline='magenta',
            )
    return img


# Callback to update image display
@app.callback(
    Output('img', 'src'),
    [
        Input(DROPDOWN_ID, 'value'),
        Input(BUTTON_ID, 'n_clicks'),
        Input(CHECK_ID, 'value')
    ]
)
def randomize_image(idx: int, n_clicks: int, checked: str) -> Image:
    """Load random image from dataset.

    Returns:
        Image: random image
    """
    global dataset
    if ctx.triggered_id == DROPDOWN_ID:
        if idx is None:
            return no_update
    elif ctx.triggered_id == BUTTON_ID:
        idx = random.randint(0, len(dataset) - 1)
    elif ctx.triggered_id == CHECK_ID:
        pass

    return get_image(
        idx,
        checked is not None and 'boxes' in checked
    )


if __name__ == '__main__':
    launch_app()
