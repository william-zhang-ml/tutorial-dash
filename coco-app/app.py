"""
An application to teach me the following skills that are good to know.
- display an image
- annotate the image with bounding boxes and text
- switch the image from a dropdown menu
- switch the image randomly from a button press
"""
import logging
from dash import Dash, html
import hydra
from omegaconf import DictConfig
from torchvision.datasets import CocoDetection


app = Dash()


@hydra.main(version_base=None, config_path='.', config_name='default')
def launch_app(cfg: DictConfig) -> None:
    """Launch the Dash-Plotly app.

    Args:
        cfg (DictConfig): app configuration
    """
    dataset = CocoDetection(
        root=cfg.image_path,
        annFile=cfg.annot_path
    )
    img, _ = dataset[0]  # img is PIL

    app.layout = html.Img(src=img)
    app.run(debug=True)


if __name__ == '__main__':
    launch_app()
