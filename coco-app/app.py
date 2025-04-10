
"""
An application to teach me the following skills that are good to know.
- display an image
- annotate the image with bounding boxes and text
- switch the image from a dropdown menu
- switch the image randomly from a button press
"""
import logging
import random
from dash import Dash, html, dcc, callback, Input, Output, no_update
import hydra
from omegaconf import DictConfig
from PIL.Image import Image
from PIL.ImageDraw import Draw
from torchvision.datasets import CocoDetection


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
            id='img-selection'
        ),
        html.Button('Random', id='randomizer'),
        html.Img(src=img, id='img')
    ])

    app.run(debug=True)


def get_image(idx: int) -> Image:
    """Load specific image from dataset (backend).

    Args:
        idx (int): which image

    Returns:
        Image: idx-th image
    """
    global dataset
    img, target = dataset[idx]
    draw = Draw(img)
    for instance in target:
        box_x, box_y, box_w, box_h = instance['bbox']
        draw.rectangle(
            [(box_x, box_y), (box_x + box_w, box_y + box_h)],
            outline='magenta',
        )
    return img


# Dropdown menu callback
@callback(
    Output('img', 'src'),
    Input('img-selection', 'value')
)
def update_image(idx: int) -> Image:
    """Load specific image from dataset.

    Args:
        idx (int): which image

    Returns:
        Image: idx-th image
    """
    if idx is None:
        return no_update
    return get_image(idx)


# Button callback
@app.callback(
    Output('img', 'src'),
    Input('randomizer', 'n_clicks')
)
def randomize_image(_) -> Image:
    """Load random image from dataset.

    Returns:
        Image: random image
    """
    global dataset
    idx = random.randint(0, len(dataset) - 1)
    return get_image(idx)


if __name__ == '__main__':
    launch_app()
