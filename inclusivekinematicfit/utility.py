from typing import NamedTuple

import numpy as np

__all__ = [
    "MassiveParticleKinematicInfo",
    "MassConstrainedParticleKinematicInfo",
    "MasslessParticleKinematicInfo",
    "rms",
]


class MassiveParticleKinematicInfo(NamedTuple):
    covariance_matrix: np.ndarray
    four_momentum: np.ndarray


class MassConstrainedParticleKinematicInfo(NamedTuple):
    covariance_matrix: np.ndarray
    three_momentum: np.ndarray
    mass: float


class MasslessParticleKinematicInfo(NamedTuple):
    three_momentum: np.ndarray


def rms(x: np.ndarray):
    """Calculates the root-mean-square error (RMSE)
    of an given sample `x`. The predicted values for
    x are assumed to be zero since its primarily used
    for resolution distributions.
    """
    return np.sqrt(np.mean(np.square(x)))
