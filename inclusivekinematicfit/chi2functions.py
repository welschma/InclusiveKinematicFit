from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from scipy.optimize import minimize, OptimizeResult

from inclusivekinematicfit import funclib
from inclusivekinematicfit.utility import (
    MassConstrainedParticleKinematicInfo,
    MassiveParticleKinematicInfo,
    MasslessParticleKinematicInfo,
)


class AbstractKinematicFitCostFunction(ABC):
    @property
    @abstractmethod
    def initial_params(self):
        pass

    @property
    @abstractmethod
    def constraints(self):
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray):
        pass


class KinematicFitCostFunction(AbstractKinematicFitCostFunction):
    n_free_params = 11
    n_constrained_params = 3
    n_fit_params = n_free_params + n_constrained_params

    def __init__(
        self,
        tag_side_info: MassiveParticleKinematicInfo,
        lepton_info: MassConstrainedParticleKinematicInfo,
        x_system_info: MassiveParticleKinematicInfo,
        missing_mom_info: MasslessParticleKinematicInfo,
        beam_four_momentum: np.ndarray,
    ):

        self.tag_side_measured_momentum = tag_side_info.four_momentum
        self.tag_side_cov = tag_side_info.covariance_matrix

        try:
            self.tag_side_inv_cov = np.linalg.inv(tag_side_info.covariance_matrix)
        except np.linalg.LinAlgError as exc:
            print(tag_side_info.covariance_matrix)
            print(np.linalg.eigvals(tag_side_info.covariance_matrix))
            raise ValueError(
                "Given Covariance not invertible"
            ) from np.linalg.LinAlgError

        self.lepton_measured_three_momentum = lepton_info.three_momentum
        self.lepton_mass = lepton_info.mass
        self.lepton_cov = lepton_info.covariance_matrix
        self.lepton_inv_cov = np.linalg.inv(lepton_info.covariance_matrix)

        self.x_system_measured_momentum = x_system_info.four_momentum
        self.x_system_cov = x_system_info.covariance_matrix
        self.x_system_inv_cov = np.linalg.inv(x_system_info.covariance_matrix)

        self.neutrino_measured_three_momentum = missing_mom_info.three_momentum

        self.beam_four_momentum = beam_four_momentum

    def x_mom_cons(self, x: np.ndarray):
        return funclib._x_mom_function(x, self.beam_four_momentum)

    def y_mom_cons(self, x: np.ndarray):
        return funclib._y_mom_function(x, self.beam_four_momentum)

    def z_mom_cons(self, x: np.ndarray):
        return funclib._z_mom_function(x, self.beam_four_momentum)

    def E_cons(self, x: np.ndarray):
        return funclib._E_function(x, self.beam_four_momentum, self.lepton_mass)

    def eq_mass_cons(self, x: np.ndarray):
        return funclib._eq_mass_function(x, self.lepton_mass)

    def tag_mass_cons(self, x: np.ndarray):
        return funclib._tag_mass_function(x)

    def sig_mass_cons(self, x: np.ndarray):
        return funclib._sig_mass_function(x, self.lepton_mass)

    @property
    def initial_params(self):
        initial_params = np.array(
            [
                *self.tag_side_measured_momentum,
                *self.x_system_measured_momentum,
                *self.lepton_measured_three_momentum,
                *self.neutrino_measured_three_momentum,
            ]
        )

        if initial_params.size != self.n_fit_params:
            raise ValueError(
                f"Number of initial parameters ({initial_params.size}) "
                f"not compatible with number of fit paramters ({self.n_fit_params})! "
                "Check initialization of the cost function!"
            )

        return initial_params

    @property
    def constraints(self):

        x_mom_cons = {"type": "eq", "fun": self.x_mom_cons}

        y_mom_cons = {"type": "eq", "fun": self.y_mom_cons}

        z_mom_cons = {"type": "eq", "fun": self.z_mom_cons}

        energy_cons = {"type": "eq", "fun": self.E_cons}

        equal_mass_cons = {"type": "eq", "fun": self.eq_mass_cons}

        tag_mass_cons = {"type": "eq", "fun": self.tag_mass_cons}

        sig_mass_cons = {"type": "eq", "fun": self.sig_mass_cons}

        return [x_mom_cons, y_mom_cons, z_mom_cons, energy_cons, equal_mass_cons]

    def __call__(self, x: np.ndarray):
        return funclib._objective_function(
            x,
            self.tag_side_inv_cov,
            self.lepton_inv_cov,
            self.x_system_inv_cov,
            self.tag_side_measured_momentum,
            self.lepton_measured_three_momentum,
            self.x_system_measured_momentum,
            self.lepton_mass,
        )


def minimize_with_slsqp(
    cost_function: AbstractKinematicFitCostFunction,
    slsqp_options: Dict[str, Any] = {"ftol": 1e-12, "disp": False, "maxiter": 1000},
) -> OptimizeResult:
    return minimize(
        fun=cost_function,
        x0=cost_function.initial_params,
        method="SLSQP",
        constraints=cost_function.constraints,
        options=slsqp_options,
    )
