#! /usr/bin/env python3
"""A module for defining the :class:`Simulation` class.
"""
from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

import numpy as np

from .logger import NullLogger
from qmmm_pme.calculators import calculator_factory
from qmmm_pme.common import align_dict
from qmmm_pme.hamiltonians.hamiltonian import MMHamiltonianInterface
from qmmm_pme.hamiltonians.hamiltonian import PBCHamiltonianInterface
from qmmm_pme.hamiltonians.hamiltonian import QMHamiltonianInterface
from qmmm_pme.hamiltonians.hamiltonian import QMMMHamiltonianInterface
from qmmm_pme.hamiltonians.hamiltonian import QMMMPMEHamiltonianInterface

if TYPE_CHECKING:
    from qmmm_pme.hamiltonians.hamiltonian import Hamiltonian
    from qmmm_pme.integrators.integrator import Integrator
    from qmmm_pme.plugins.plugin import Plugin
    from .system import System


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
    energy = {}

    def __init__(
            self,
            system: System,
            hamiltonian: Hamiltonian,
            integrator: Integrator,
            logger: Any = NullLogger,
            num_threads: int | None = 1,
            memory: str | None = "1 GB",
            plugins: list[Plugin] | None = [],
    ) -> None:
        self.system = system
        self._hamiltonian = hamiltonian
        self.integrator = integrator
        self.num_threads = num_threads
        self.memory = memory
        self.logger = logger
        self.plugins = plugins
        self._generate_calculator()

    def run_dynamics(self, steps: int) -> None:
        """Run simulation using the :class:`System`.

        :param steps: The number of steps to take.
        """
        if not self.system.state.velocities.size:
            self.system.state.velocities = self.integrator.compute_velocities()
        with self.logger as logger:
            for i in range(steps):
                self.calculate_energy_forces()
                logger.record(self)
                (
                    self.system.state.positions,
                    self.system.state.velocities,
                ) = self.integrator.integrate()
                self.wrap_positions()
                self.system.state.frame += 1

    def calculate_energy_forces(self) -> None:
        """Update the :class:`State` using calculations from the
        :class:`System`.
        """
        (
            potential_energy, forces, components,
        ) = self.calculator.calculate()
        self.system.state.forces = forces
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

    def wrap_positions(self) -> None:
        """Atoms are wrapped to stay inside of the periodic box.  This
        function ensures molecules are not broken up by a periodic
        boundary, since OpenMM electrostatics will be incorrect if atoms
        in a molecule are not on the same side of the periodic box.
        This method currently assumes an isotropic box.
        """
        box = self.system.state.box
        inverse_box = np.linalg.inv(box)
        positions = self.system.state.positions
        new_positions = np.zeros_like(positions)
        for residue in self.system.topology.groups["all"]:
            residue_positions = positions[residue, :]
            residue_centroid = np.average(residue_positions, axis=1)
            inverse_centroid = residue_centroid @ inverse_box
            mask = np.floor(inverse_centroid)
            diff = (-mask @ box).reshape((-1, 3))
            temp_pos = residue_positions + diff[:, np.newaxis, :]
            new_positions[residue] = temp_pos.reshape((len(residue), 3))
        self.system.state.positions = new_positions

    def _generate_calculator(self) -> None:
        """Create a new calculator and update the :class:`Topology` of
        the :class:`System`.
        """
        all_atoms = self.system.topology.groups["all"]
        if isinstance(self.hamiltonian, MMHamiltonianInterface):
            mm_atoms = [
                x for residue in all_atoms if
                (
                    x := [
                        atom for atom in residue if atom in
                        self.hamiltonian.atoms
                    ]
                )
            ]
            self.system.topology.groups.update({"mm_atom": mm_atoms})
        elif isinstance(self.hamiltonian, QMHamiltonianInterface):
            qm_atoms = self.hamiltonian.atoms
            self.system.topology.groups.update({"qm_atom": qm_atoms})
        elif isinstance(
            self.hamiltonian,
                (QMMMHamiltonianInterface, QMMMPMEHamiltonianInterface),
        ):
            qm_atoms = self.hamiltonian.qm_hamiltonian.atoms
            self.system.topology.groups.update({"qm_atom": qm_atoms})
            mm_atoms = [
                x for residue in all_atoms if
                (
                    x := [
                        atom for atom in residue if atom in
                        self.hamiltonian.mm_hamiltonian.atoms
                    ]
                )
            ]
            self.system.topology.groups.update({"mm_atom": mm_atoms})
        elif isinstance(self.hamiltonian, PBCHamiltonianInterface):
            raise TypeError(
                (
                    "PBCHamiltonian is not a standalone Hamiltonian "
                    + "object which can be used for simulation."
                ),
            )
        else:
            raise TypeError(
                (
                    f"Hamiltonian '{self.hamiltonian}' is unrecognized "
                    + "Hamiltonian of type "
                    + f"'{self.hamiltonian.__class__.__name__}'."
                ),
            )
        self.calculator = calculator_factory(self.hamiltonian)
        if isinstance(self.hamiltonian, MMHamiltonianInterface):
            self.openmm = self.calculator.openmm
        elif isinstance(self.hamiltonian, QMHamiltonianInterface):
            self.psi4 = self.calculator.psi4
        elif isinstance(
            self.hamiltonian,
                (QMMMHamiltonianInterface, QMMMPMEHamiltonianInterface),
        ):
            self.openmm = self.calculator.mm_calculator.openmm
            self.psi4 = self.calculator.qm_calculator.psi4
        self._register_plugins()

    def _register_plugins(self):
        """Register dynamically loaded plugins.
        """
        for plugin in self.plugins:
            for key in plugin._keys:
                self.__dict__[key].register_plugin(plugin)

    @property
    def hamiltonian(self) -> Hamiltonian:
        """The :class:`Hamiltonian` belonging to the
        :class:`Simulation`.
        """
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian) -> None:
        self._hamiltonian = hamiltonian
        self._generate_calculator()
