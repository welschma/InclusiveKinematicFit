"""This module contains functions for calculating the chi square
sum and constraints used to fit the different particle four momenta.

All functions expect as first argument the parameter array used for the minimization.
In total, there are 14 parameters (the parameter array is expected to follow this
convetion):
    - x[0:4] 4 four momenta componets of the tag side B meson, x[0:3] are the momenta while x[4] is the energy
    - x[4:8] 4 four momenta componets of the x system, x[0:3] are the momenta while x[4] is the energy
    - x[8:11] 3 three momenta componets of the lepton
    - x[11:14] 3 three momenta componets of the neutrino
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def _x_mom_function(x: np.ndarray, beam_momentum: np.ndarray) -> float:
    return x[0] + x[4] + x[8] + x[11] - beam_momentum[0]


@numba.jit(nopython=True)
def _y_mom_function(x: np.ndarray, beam_momentum: np.ndarray) -> float:
    return x[1] + x[5] + x[9] + x[12] - beam_momentum[1]


@numba.jit(nopython=True)
def _z_mom_function(x: np.ndarray, beam_momentum: np.ndarray) -> float:
    return x[2] + x[6] + x[10] + x[13] - beam_momentum[2]


@numba.jit(nopython=True)
def _lepton_energy(x: np.ndarray, lepton_mass: np.ndarray) -> float:
    return np.sqrt(np.sum(x[8:11] ** 2) + lepton_mass ** 2)


@numba.jit(nopython=True)
def _neutrino_energy(x: np.ndarray) -> float:
    return np.sqrt(np.sum(x[11:] ** 2))


@numba.jit(nopython=True)
def _E_function(x: np.ndarray, beam_momentum: np.ndarray, lepton_mass: float) -> float:
    return (
        x[3]
        + x[7]
        + _lepton_energy(x, lepton_mass)
        + _neutrino_energy(x)
        - beam_momentum[3]
    )


@numba.jit(nopython=True)
def _tag_mass_sq(x: np.ndarray) -> float:
    return x[3] ** 2 - np.sum(x[0:3] ** 2)


@numba.jit(nopython=True)
def _sig_x_mom(x: np.ndarray) -> float:
    return x[4] + x[8] + x[11]


@numba.jit(nopython=True)
def _sig_y_mom(x: np.ndarray) -> float:
    return x[5] + x[9] + x[12]


@numba.jit(nopython=True)
def _sig_z_mom(x: np.ndarray) -> float:
    return x[6] + x[10] + x[13]


@numba.jit(nopython=True)
def _sig_mass_sq(x: np.ndarray, lepton_mass: float) -> float:
    return (
        (x[7] + _lepton_energy(x, lepton_mass) + _neutrino_energy(x)) ** 2
        - _sig_x_mom(x) ** 2
        - _sig_y_mom(x) ** 2
        - _sig_z_mom(x) ** 2
    )


@numba.jit(nopython=True)
def _eq_mass_function(x: np.ndarray, lepton_mass: float) -> float:
    return np.sqrt(_tag_mass_sq(x)) - np.sqrt(_sig_mass_sq(x, lepton_mass))


@numba.jit(nopython=True)
def _tag_mass_function(x: np.ndarray) -> float:
    return np.sqrt(_tag_mass_sq(x)) - 5.279


@numba.jit(nopython=True)
def _sig_mass_function(x: np.ndarray, lepton_mass: float) -> float:
    return np.sqrt(_sig_mass_sq(x, lepton_mass)) - 5.279


@numba.jit(nopython=True)
def _objective_function(
    x: np.ndarray,
    tag_icov: np.ndarray,
    lep_icov: np.ndarray,
    x_icov: np.ndarray,
    tag_meas: np.ndarray,
    lep_meas: np.ndarray,
    x_meas: np.ndarray,
    lep_mass: float,
) -> float:
    tag_side_four_momentum = x[:4]
    x_system_four_momentum = x[4:8]
    lepton_three_momentum = x[8:11]
    neutrino_three_momentum = x[11:14]

    tag_side_residuals = tag_side_four_momentum - tag_meas
    tag_side_chi_square = tag_side_residuals @ tag_icov @ tag_side_residuals

    lep_residuals = lepton_three_momentum - lep_meas
    lepton_chi_square = lep_residuals @ lep_icov @ lep_residuals

    x_system_residuals = x_system_four_momentum - x_meas
    x_system_chi_square = x_system_residuals @ x_icov @ x_system_residuals

    return tag_side_chi_square + lepton_chi_square + x_system_chi_square
