#! /usr/bin/env python3
"""A module defining the pluggable implementation of the SETTLE
algorithm for the QM/MM/PME repository.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .settle_utils import settle_positions
from .settle_utils import settle_velocities
from qmmm_pme.plugins.plugin import IntegratorPlugin

if TYPE_CHECKING:
    from qmmm_pme.integrators.integrator import ModifiableIntegrator


class SETTLE(IntegratorPlugin):
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
            oh_distance: float | int = 1.,
            hh_distance: float | int = 1.632981,
            hoh_residue: str = "HOH",
    ) -> None:
        self.oh_distance = oh_distance
        self.hh_distance = hh_distance
        self.hoh_residue = hoh_residue

    def modify(
            self,
            integrator: ModifiableIntegrator,
    ) -> None:
        """Perform necessary modifications to the :class:`Integrator`
        object.

        :param integrator: The integrator to modify with the SETTLE
            functionality.
        """
        self._modifieds.append(type(integrator).__name__)
        self.system = integrator.system
        self.timestep = integrator.timestep
        self.residues = [
            res for i, res
            in enumerate(self.system.topology.mm_atoms())
            if self.system.topology.residue_names()[i] == self.hoh_residue
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
            integrate: Callable[
                [], tuple[
                    NDArray[np.float64], NDArray[np.float64],
                ],
            ],
    ) -> Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        """
        def inner() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            positions, velocities = integrate()
            positions = settle_positions(
                self.residues,
                self.system.state.positions(),
                positions,
                self.system.state.masses(),
                self.oh_distance,
                self.hh_distance,
            )
            velocities[self.residues, :] = (
                (
                    positions[self.residues, :]
                    - self.system.state.positions()[self.residues, :]
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
        def inner() -> NDArray[np.float64]:
            velocities = compute_velocities()
            velocities = settle_velocities(
                self.residues,
                self.system.state.positions(),
                velocities,
                self.system.state.masses(),
            )
            return velocities
        return inner

    def _modify_compute_kinetic_energy(
            self,
            compute_kinetic_energy: Callable[[], float],
    ) -> Callable[[], float]:
        """
        """
        def inner() -> float:
            masses = self.system.state.masses().reshape(-1, 1)
            velocities = (
                self.system.state.velocities()
                + (
                    0.5*self.timestep
                    * self.system.state.forces()*(10**-4)/masses
                )
            )
            velocities = settle_velocities(
                self.residues,
                self.system.state.positions(),
                velocities,
                self.system.state.masses(),
            )
            kinetic_energy = (
                np.sum(0.5*masses*(velocities)**2)
                * (10**4)
            )
            return kinetic_energy
        return inner
