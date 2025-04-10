
"""
An application to teach me the following skills that are good to know.
- display an image
- annotate the image with bounding boxes and text
- switch the image from a dropdown menu
- switch the image randomly from a button press
"""
import base64
from io import BytesIO
import logging
from dash import Dash, html, dcc, callback, Input, Output, no_update
import hydra
from omegaconf import DictConfig
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
        html.Img(src=img, id='img')
    ])

    app.run(debug=True)


# Dropdown menu callback
@callback(
    Output('img', 'src'),
    Input('img-selection', 'value')
)
def update_graph(idx: int) -> None:
    """Load specific img from dataset.

    Args:
        idx (int): which image

    Returns:
        PIL.Image.Image: idx-th image
    """
    if idx is None:
        return no_update

    global dataset
    img, _ = dataset[int(idx)]
    return img


if __name__ == '__main__':
    launch_app()
