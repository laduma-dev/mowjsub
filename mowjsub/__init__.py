from importlib import metadata
from omegaconf import OmegaConf

__version__ = metadata.version(__package__)

BIN = OmegaConf.create({
    "im_plane": "im-mowjsub",
    "vis_plane": "vis-mowjsub",
})