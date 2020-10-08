from typing import NamedTuple

import numpy as np
import scipy.stats

__all__ = [
    "MassiveParticleKinematicInfo",
    "MassConstrainedParticleKinematicInfo",
    "MasslessParticleKinematicInfo",
    "rms",
    "calculate_chi2_prob",
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


def calculate_chi2_prob(chi2_values: np.ndarray, ndf: int) -> np.ndarray:
    """Calculates the p-value of the given chi square
    value assuming an chi square distribution with
    `ndf` degrees of freedom.

    :param chi2_values: Input chi square values
    :type chi2_values: np.ndarray
    :param ndf: Number of degrees of freedom for the
    chi square distribution
    :type ndf: int
    :return: Calculated p-values
    :rtype: np.ndarray
    """
    return 1 - scipy.stats.chi2.cdf(chi2_values, df=ndf)
