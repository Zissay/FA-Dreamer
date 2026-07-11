"""FA-AFDM position optimization environment and Dreamer-style trainer."""

from .channel import FAAFDMChannel, FAAFDMConfig, make_default_channel
from .env import FAAFDMEnv

__all__ = [
    "FAAFDMChannel",
    "FAAFDMConfig",
    "FAAFDMEnv",
    "make_default_channel",
]
