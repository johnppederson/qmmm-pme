#! /usr/bin/env python3
"""A module for defining the :class:`Simulation` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import TYPE_CHECKING

import numpy as np

from .logger import NullLogger

if TYPE_CHECKING:
    from qmmm_pme.hamiltonians.hamiltonian import Hamiltonian
    from qmmm_pme.dynamics.dynamics import Dynamics
    from qmmm_pme.plugins.plugin import Plugin
    from .system import System
    EnergyDict = Dict[str, float]


@dataclass
class Simulation:
    """An object which manages and performs simulations.

    :param system: The :class:`System` to perform calculations on.
    :param hamiltonian: The :class:`Hamiltonian` to perform calculations
        with.
    :param integrator: The :class:`Integrator` to perform calculations
        with.
    :param logger: Not Implemented.
    :param num_threads: The number of threads to run calculations on.
    :param memory: The amount of memory to allocate to calculations.
    :param plugins: Any :class:`Plugin` objects to apply to the
        simulation.
    """
    system: System
    hamiltonian: Hamiltonian
    dynamics: Dynamics
    logger: Any = NullLogger
    num_threads: int = 1
    memory: str = "1 GB"
    plugins: list[Plugin] = field(default_factory=list)
    frame: int = 0
    energy: EnergyDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.calculator = self.hamiltonian.build_calculator(self.system)
        self.integrator = self.dynamics.build_integrator(self.system)
        self._register_plugins()
        if not self.system.state.velocities().size:
            self.system.state.velocities.update(
                self.integrator.compute_velocities(),
            )
        self.calculate_energy_forces()

    def run_dynamics(self, steps: int) -> None:
        """Run simulation using the :class:`System`.

        :param steps: The number of steps to take.
        """
        with self.logger as logger:
            for i in range(steps):
                new_positions, new_velocities = self.integrator.integrate()
                self.system.state.positions.update(new_positions)
                self.system.state.velocities.update(new_velocities)
                self.wrap_positions()
                logger.record(self)
                self.calculate_energy_forces()
                self.frame += 1

    def calculate_energy_forces(self) -> None:
        """Update the :class:`State` using calculations from the
        :class:`System`.
        """
        (
            potential_energy, forces, components,
        ) = self.calculator.calculate()
        self.system.state.forces.update(forces)
        kinetic_energy = self.integrator.compute_kinetic_energy()
        energy = {
            "Total Energy": kinetic_energy + potential_energy,
            ".": {
                "Kinetic Energy": kinetic_energy,
                "Potential Energy": potential_energy,
                ".": components,
            },
        }
        self.energy = energy

    def calculate_forces(self) -> None:
        """Update the :class:`State` using calculations from the
        :class:`System`.
        """
        (
            potential_energy, forces, components,
        ) = self.calculator.calculate()
        self.system.state.forces.update(forces)

    def wrap_positions(self) -> None:
        """Atoms are wrapped to stay inside of the periodic box.  This
        function ensures molecules are not broken up by a periodic
        boundary, since OpenMM electrostatics will be incorrect if atoms
        in a molecule are not on the same side of the periodic box.
        This method currently assumes an isotropic box.
        """
        box = self.system.state.box()
        inverse_box = np.linalg.inv(box)
        positions = self.system.state.positions()
        new_positions = np.zeros_like(positions)
        for residue in self.system.topology.atoms():
            residue_positions = positions[residue, :]
            residue_centroid = np.average(residue_positions, axis=1)
            inverse_centroid = residue_centroid @ inverse_box
            mask = np.floor(inverse_centroid)
            diff = (-mask @ box).reshape((-1, 3))
            temp_pos = residue_positions + diff[:, np.newaxis, :]
            new_positions[residue] = temp_pos.reshape((len(residue), 3))
        self.system.state.positions.update(new_positions)

    def _register_plugins(self) -> None:
        """Register dynamically loaded plugins.
        """
        for plugin in self.plugins:
            getattr(self, plugin._key).register_plugin(plugin)
