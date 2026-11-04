"""Position-aware PPO feature extractor for periodic fluid-antenna responses."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FourierPositionFeatures(BaseFeaturesExtractor):
    """Keep the original channel-plus-position observation and add sin/cos bases.

    The environment exposes the four normalised physical coordinates as its
    final four entries.  This extractor does not evaluate AFDM rate and does
    not introduce an optimum label; it only makes the same coordinates easier
    for an MLP to represent at millimetre-scale spatial frequencies.
    """

    def __init__(self, observation_space: spaces.Box, max_frequency: int = 64, include_time_feature: bool = False) -> None:
        if max_frequency < 1:
            raise ValueError("max_frequency must be positive")
        base_dim = int(observation_space.shape[0])
        self.max_frequency = int(max_frequency)
        self.include_time_feature = include_time_feature
        extra_dim = 2 * 4 * self.max_frequency
        super().__init__(observation_space, features_dim=base_dim + extra_dim)
        frequencies = torch.arange(1, self.max_frequency + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        positions = observations[..., -5:-1] if self.include_time_feature else observations[..., -4:]
        phase = torch.pi * positions.unsqueeze(-1) * self.frequencies
        encoded = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(start_dim=-2)
        return torch.cat([observations, encoded], dim=-1)


class ChannelPhaseFeatures(BaseFeaturesExtractor):
    """Encode spatial interference from CSI directions and present position.

    Uses only existing observations, never rates, derivatives, next positions
    or a reference optimizer. Actor/critic weights are learned only by PPO.
    Inputs must retain physical normalization (no running observation scaler).
    """
    def __init__(self, observation_space, n_paths=20, wavelength=.0107068735,
                 tx_region=.5, rx_region=.5, include_time_feature=True):
        super().__init__(observation_space, features_dim=int(observation_space.shape[0])+n_paths*(n_paths-1))
        self.n_paths=n_paths
        self.wavelength=wavelength
        self.include_time_feature=include_time_feature
        self.register_buffer('scales',torch.tensor([tx_region/2]*2+[rx_region/2]*2))
        ij=torch.triu_indices(n_paths,n_paths,offset=1)
        self.register_buffer('pair_i',ij[0])
        self.register_buffer('pair_j',ij[1])

    def forward(self, observations):
        p=self.n_paths
        positions=observations[..., -5:-1] if self.include_time_feature else observations[..., -4:]
        positions=positions*self.scales
        tx=observations[..., :2*p].reshape(-1,p,2)
        rx=observations[..., 2*p:4*p].reshape(-1,p,2)
        phase=(2*torch.pi/self.wavelength)*(
            (tx*positions[:,None,:2]).sum(-1)-(rx*positions[:,None,2:]).sum(-1))
        phase=phase+observations[...,5*p:6*p]
        delta=phase[...,self.pair_i]-phase[...,self.pair_j]
        return torch.cat([observations,torch.sin(delta),torch.cos(delta)],dim=-1)
