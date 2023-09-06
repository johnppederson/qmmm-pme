#! /usr/bin/env python3
"""A module defining the pluggable implementation of the SETTLE
algorithm for the QM/MM/PME repository.
"""
from __future__ import annotations
__author__ = "Jesse McDaniel, John Pederson"
__version__ = "1.0.0"

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from qmmm_pme.plugins.plugin import Plugin

from .settle_utils import settle_positions, settle_velocities

if TYPE_CHECKING:
    from qmmm_pme.integrators.integrator import Integrator


class SETTLE(Plugin):
    """A :class:`Plugin` which implements the SETTLE algorithm for
    positions and velocities.

    :param oh_distance: The distance between the oxygen and hydrogen, in
        Angstroms.
    :param hh_distance: The distance between the hydrogens, in
        Angstroms.
    :param hoh_residue: The name of the water residues in the
        :class:`System`.
    """

    def __init__(
            self,
            oh_distance: float | int | None = 1.,
            hh_distance: float | int | None = 1.632981,
            hoh_residue: str | None = "HOH",
    ) -> None:
        self.oh_distance = oh_distance
        self.hh_distance = hh_distance
        self.hoh_residue = hoh_residue
        self._keys = ["integrator"]

    def modify(
            self,
            integrator: Integrator,
    ) -> None:
        """Perform necessary modifications to the :class:`Integrator`
        object.

        :param integrator: The integrator to modify with the SETTLE
            functionality.
        """
        self._modifieds.append(integrator.__class__.__name__)
        self._state = integrator._state
        self._topology = integrator._topology
        self.timestep = integrator.timestep
        self.residues = [
            res for i, res
            in enumerate(self._topology.groups["mm_atom"])
            if self._topology.residues[i] == self.hoh_residue
        ]
        integrator.integrate = self._modify_integrate(integrator.integrate)
        integrator.compute_velocities = self._modify_compute_velocities(
            integrator.compute_velocities,
        )
        integrator.compute_kinetic_energy = self._modify_compute_kinetic_energy(
            integrator.compute_kinetic_energy,
        )

    def _modify_integrate(
            self,
            integrate:
                Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        """
        def inner():
            positions, velocities = integrate()
            positions = settle_positions(
                self.residues,
                self._state.positions,
                positions,
                self._state.masses,
                self.oh_distance,
                self.hh_distance,
            )
            velocities[self.residues, :] = (
                (
                    positions[self.residues, :]
                    - self._state.positions[self.residues, :]
                ) / self.timestep
            )
            return positions, velocities
        return inner

    def _modify_compute_velocities(
            self,
            compute_velocities: Callable[[], NDArray[np.float64]],
    ) -> Callable[[], NDArray[np.float64]]:
        """
        """
        def inner():
            velocities = compute_velocities()
            velocities = settle_velocities(
                self.residues,
                self._state.positions,
                velocities,
                self._state.masses,
            )
            return velocities
        return inner

    def _modify_compute_kinetic_energy(
            self,
            compute_kinetic_energy: Callable[[], float],
    ) -> Callable[[], float]:
        """
        """
        def inner():
            masses = self._state.masses.reshape(-1, 1)
            velocities = (
                self._state.velocities
                + (
                    0.5*self.timestep
                    * self._state.forces*(10**-4)/masses
                )
            )
            velocities = settle_velocities(
                self.residues,
                self._state.positions,
                velocities,
                self._state.masses,
            )
            kinetic_energy = (
                np.sum(0.5*masses*(velocities)**2)
                * (10**4)
            )
            return kinetic_energy
        return inner
