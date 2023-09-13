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
    """The absract :class:`Integrator` base class, which also contains
    tools for generating velocities and calculating kinetic energies.
    """
    timestep: float | int
    temperature: float | int
    system: System
    _plugins: list[str] = []

    @abstractmethod
    def integrate(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """A placeholder method to integrate forces into new positions
        and velocities, which is implemented by the integrators that
        inherit from :class:`Integrator`.

        :return: The new positions and velocities of the
            :class:`System`, in Angstroms and ???, respectively.

        .. note:: Based on the integrator kernels from OpenMM.
        """

    @abstractmethod
    def compute_velocities(self) -> NDArray[np.float64]:
        """Calculate velocities based on the Maxwell-Boltzmann
        distribution at a given temperature.

        :return: The sampled velocities, in ???.
        """

    @abstractmethod
    def compute_kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the :class:`System`.

        :return: The kinetic energy, in kJ/mol.
        """

    def register_plugin(self, plugin: IntegratorPlugin) -> None:
        """Register any plugins applied to an instance of a class
        inheriting from :class:`Core`.

        :param plugin: An instance of a class inheriting from
            :class:`Plugin`.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Return the list of active plugins.

        :return: A list of the active plugins being employed by the
            class.
        """
        return self._plugins


@dataclass
class Integrator(ModifiableIntegrator):
    """The absract :class:`Integrator` base class, which also contains
    tools for generating velocities and calculating kinetic energies.

    :param timestep: |timestep|
    :param temperature: |temperature|
    """
    timestep: float | int
    temperature: float | int
    system: System

    def compute_velocities(self) -> NDArray[np.float64]:
        """Calculate velocities based on the Maxwell-Boltzmann
        distribution at a given temperature.

        :return: The sampled velocities, in ???.

        .. note:: Based on ase.md.velocitydistribution.py
        """
        avg_ke = self.temperature * KB
        masses = self.system.state.masses().reshape((-1, 1)) * (10**-3)
        np.random.seed(10101)
        z = np.random.standard_normal((len(masses), 3))
        momenta = z * np.sqrt(avg_ke * masses)
        velocities = (momenta / masses) * (10**-5)
        return velocities

    def compute_kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the :class:`System`.

        :return: The kinetic energy, in kJ/mol.
        """
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
