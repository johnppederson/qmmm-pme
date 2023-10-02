#! /usr/bin/env python3
"""A module for defining the :class:`Integrator` base class.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from qmmm_pme import System
    from qmmm_pme.plugins.plugin import IntegratorPlugin


KB = 8.31446261815324  # J / (mol * K)


class ModifiableIntegrator(ABC):
    """An abstract :class:`Integrator` base class for interfacing with
    plugins.
    """
    timestep: float | int
    temperature: float | int
    system: System
    _plugins: list[str] = []

    @abstractmethod
    def integrate(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate forces from the :class:`System` into new positions
        and velocities.

        :return: The new positions and velocities of the
            :class:`System`, in Angstroms and Angstroms per
            femtosecond, respectively.

        .. note:: Based on the implementation of the integrator
            kernels from OpenMM.
        """

    @abstractmethod
    def compute_velocities(self) -> NDArray[np.float64]:
        """Calculate initial velocities based on the Maxwell-Boltzmann
        distribution at a given temperature.

        :return: The sampled velocities, in Angstroms per femtosecond.

        .. note:: Based on the implementation in ASE.
        """

    @abstractmethod
    def compute_kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the :class:`System`.

        :return: The kinetic energy, in kJ/mol.

        .. note:: Based on the implementation in OpenMM
        """

    def register_plugin(self, plugin: IntegratorPlugin) -> None:
        """Register a :class:`Plugin` modifying an :class:`Integrator`
        routine.

        :param plugin: An :class:`IntegratorPlugin` object.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Get the current list of active plugins.

        :return: A list of the active plugins being employed by the
            :class:`Integrator`.
        """
        return self._plugins


@dataclass
class Integrator(ModifiableIntegrator):
    """An abstract :class:`Integrator` base class, which also contains
    implementations for generating velocities and calculating kinetic
    energies.

    :param timestep: |timestep|
    :param temperature: |temperature|
    :param system: |system| to integrate.
    """
    timestep: float | int
    temperature: float | int
    system: System

    def compute_velocities(self) -> NDArray[np.float64]:
        avg_ke = self.temperature * KB
        masses = self.system.state.masses().reshape((-1, 1)) * (10**-3)
        np.random.seed(10101)
        z = np.random.standard_normal((len(masses), 3))
        momenta = z * np.sqrt(avg_ke * masses)
        velocities = (momenta / masses) * (10**-5)
        return velocities

    def compute_kinetic_energy(self) -> float:
        masses = self.system.state.masses().reshape(-1, 1)
        velocities = (
            self.system.state.velocities()
            + (
                0.5*self.timestep
                * self.system.state.forces()*(10**-4)/masses
            )
        )
        kinetic_energy = np.sum(0.5*masses*(velocities)**2) * (10**4)
        return kinetic_energy
