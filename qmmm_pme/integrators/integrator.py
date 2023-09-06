#! /usr/bin/env python3
"""A module for defining the :class:`Integrator` base class.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from qmmm_pme.common import Core
from qmmm_pme.common import KB


class Integrator(Core):
    """The absract :class:`Integrator` base class, which also contains
    tools for generating velocities and calculating kinetic energies.

    :param timestep: |timestep|
    :param temperature: |temperature|
    """

    def __init__(
            self,
            timestep: int | float,
            temperature: int | float,
    ) -> None:
        super().__init__()
        self.timestep = timestep
        self.temperature = temperature

    @abstractmethod
    def integrate(self):
        """A placeholder method to integrate forces into new positions
        and velocities, which is implemented by the integrators that
        inherit from :class:`Integrator`.

        :return: The new positions and velocities of the
            :class:`System`, in Angstroms and ???, respectively.

        .. note:: Based on the integrator kernels from OpenMM.
        """

    def compute_velocities(self) -> NDArray[np.float64]:
        """Calculate velocities based on the Maxwell-Boltzmann
        distribution at a given temperature.

        :return: The sampled velocities, in ???.

        .. note:: Based on ase.md.velocitydistribution.py
        """
        avg_ke = self.temperature * KB
        masses = self._state.masses.reshape((-1, 1)) * (10**-3)
        np.random.seed(10101)
        z = np.random.standard_normal((len(masses), 3))
        momenta = z * np.sqrt(avg_ke * masses)
        velocities = (momenta / masses) * (10**-5)
        return velocities

    def compute_kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the :class:`System`.

        :return: The kinetic energy, in kJ/mol.
        """
        masses = self._state.masses.reshape(-1, 1)
        velocities = (
            self._state.velocities
            + (
                0.5*self.timestep
                * self._state.forces*(10**-4)/masses
            )
        )
        kinetic_energy = np.sum(0.5*masses*(velocities)**2) * (10**4)
        return kinetic_energy

    def update(self, attr, value):
        pass
