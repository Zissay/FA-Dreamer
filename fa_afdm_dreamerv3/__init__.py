"""FA-AFDM channel and environment shared by the Dreamer and PPO trainers."""

from .channel import FAAFDMChannel, FAAFDMConfig, make_default_channel

__all__ = [
    "FAAFDMChannel",
    "FAAFDMConfig",
    "make_default_channel",
]
