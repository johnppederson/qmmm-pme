#! /usr/bin/env python3
"""A module defining the pluggable implementation of the rigid bodies
algorithm for the |package| repository.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from qmmm_pme.plugins.plugin import IntegratorPlugin

if TYPE_CHECKING:
    from qmmm_pme.integrators.integrator import ModifiableIntegrator


class Stationary(IntegratorPlugin):
    """A :class:`Plugin` which implements stationary residues during
    simulation.

    :param stationary_residues: The names of residues to hold stationary
        in the :class:`System`.
    """

    def __init__(
            self,
            stationary_residues: list[str],
    ) -> None:
        self.stationary_residues = stationary_residues

    def modify(
            self,
            integrator: ModifiableIntegrator,
    ) -> None:
        self._modifieds.append(type(integrator).__name__)
        self.system = integrator.system
        self.residues = [
            res for i, res
            in enumerate(self.system.topology.mm_atoms())
            if self.system.topology.residue_names()[i]
            in self.stationary_residues
        ]
        integrator.integrate = self._modify_integrate(
            integrator.integrate,
        )
        integrator.compute_velocities = self._modify_compute_velocities(
            integrator.compute_velocities,
        )

    def _modify_integrate(
            self,
            integrate: Callable[
                [], tuple[
                    NDArray[np.float64], NDArray[np.float64],
                ],
            ],
    ) -> Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Modify the integrate call in the :class:`Integrator` to
        make the positions of a subset of residues constant and their
        velocities zero.

        :param integrate: The default integrate method of the
            :class:`Integrator`.
        :return: The modified integrate method.
        """
        def inner() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            positions, velocities = integrate()
            positions[self.residues, :] = self.system.state.positions(
            )[self.residues, :]
            velocities[self.residues, :] = 0
            return positions, velocities
        return inner

    def _modify_compute_velocities(
            self,
            compute_velocities: Callable[[], NDArray[np.float64]],
    ) -> Callable[[], NDArray[np.float64]]:
        """Modify the compute_velocities call in the :class:`Integrator`
        to make the velocities of a subset of residues zero.

        :param compute_velocities: The default compute_velocities method
            of the :class:`Integrator`.
        :return: The modified compute_velocities method.
        """
        def inner() -> NDArray[np.float64]:
            velocities = compute_velocities()
            velocities[self.residues, :] = 0
            return velocities
        return inner


class RigidBody(IntegratorPlugin):
    """A :class:`Plugin` which implements rigid body dynamics during
    simulation.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def modify(
            self,
            integrator: ModifiableIntegrator,
    ) -> None:
        pass
