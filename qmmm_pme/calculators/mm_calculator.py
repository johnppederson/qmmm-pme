#! /usr/bin/env python3
"""A module to define the :class:`MMCalculator` class.
"""
from __future__ import annotations

import re
from typing import Any
from typing import Optional

import numpy as np
import openmm.app
import simtk.unit
from numpy.typing import NDArray

from .calculator import Calculator
from qmmm_pme.common import NM_PER_ANGSTROM
from qmmm_pme.records import Namespace


class MMCalculator(Calculator):
    """A :class:`Calculator` class for performing molecular mechanics
    calculations for an entire system using OpenMM.

    :param nonbonded_method: |nonbonded_method|
    :param nonbonded_cutoff: |nonbonded_cutoff|
    :param pme_gridnumber: |pme_gridnumber|
    :param pme_alpha: |pme_alpha|
    """

    def __init__(
            self,
            nonbonded_method: str | None = "PME",
            nonbonded_cutoff: float | int | None = 14.,
            pme_gridnumber: int | None = 60,
            pme_alpha: float | int | None = 5.,
    ) -> None:
        super().__init__()
        # Create OpenMM namespace.
        self.openmm = Namespace()
        self.openmm.nonbonded_method = nonbonded_method
        self.openmm.nonbonded_cutoff = (
            nonbonded_cutoff
            * simtk.unit.angstrom
        )
        self.openmm.pme_gridnumber = pme_gridnumber
        self.openmm.pme_alpha = pme_alpha
        self.openmm.temperature = 300*simtk.unit.kelvin
        self.openmm.friction = 1/simtk.unit.picosecond
        self.openmm.timestep = 0.001*simtk.unit.picoseconds
        self.openmm.properties = {}
        # Create necessary OpenMM objects.
        self._generate_system()
        self._adjust_forces()
        self._generate_context()

    def _generate_system(self) -> None:
        """Generate the OpenMM System object.
        """
        for xml in self.files.topology_list:
            openmm.app.topology.Topology().loadBondDefinitions(xml)
        self.openmm.pdb = openmm.app.pdbfile.PDBFile(
            self.files.pdb_list[0],
        )
        self.openmm.modeller = openmm.app.modeller.Modeller(
            self.openmm.pdb.topology,
            self.openmm.pdb.positions,
        )
        self.openmm.forcefield = openmm.app.forcefield.ForceField(
            *self.files.forcefield_list,
        )
        self.openmm.modeller.addExtraParticles(self.openmm.forcefield)
        self.openmm.system = self.openmm.forcefield.createSystem(
            self.openmm.modeller.topology,
            nonbondedCutoff=self.openmm.nonbonded_cutoff,
            constraints=None,
            rigidWater=False,
        )
        # Set force groups.
        for i in range(self.openmm.system.getNumForces()):
            force = self.openmm.system.getForce(i)
            force.setForceGroup(i)

    def _adjust_forces(self) -> None:
        """Generate OpenMM non-bonded forces.
        """
        # Get force types and set method.
        self.nonbonded_force = [
            force for force in [
                self.openmm.system.getForce(i)
                for i in range(self.openmm.system.getNumForces())
            ] if isinstance(force, openmm.NonbondedForce)
        ][0]
        self.custom_nonbonded_force = [
            force for force in [
                self.openmm.system.getForce(i)
                for i in range(self.openmm.system.getNumForces())
            ] if isinstance(force, openmm.CustomNonbondedForce)
        ]
        # Set long-range interaction method.
        if self.openmm.nonbonded_method == "NoCutoff":
            self.nonbonded_force.setNonbondedMethod(
                openmm.NonbondedForce.NoCutoff,
            )
        elif self.openmm.nonbonded_method == "PME":
            self.nonbonded_force.setNonbondedMethod(
                openmm.NonbondedForce.PME,
            )
            self.nonbonded_force.setPMEParameters(
                self.openmm.pme_alpha,
                self.openmm.pme_gridnumber,
                self.openmm.pme_gridnumber,
                self.openmm.pme_gridnumber,
            )
        else:
            raise ValueError(
                (
                    f"OpenMM NonbondedMethod "
                    + f"'{self.openmm.nonbonded_method}' "
                    + "is not a recognized NonbondedMethod"
                ),
            )
        if self.custom_nonbonded_force:
            self.custom_nonbonded_force = self.custom_nonbonded_force[0]
            self.custom_nonbonded_force.setNonbondedMethod(
                min(
                    self.nonbonded_force.getNonbondedMethod(),
                    openmm.NonbondedForce.CutoffPeriodic,
                ),
            )

    def _generate_context(self) -> None:
        """Create the OpenMM Context.
        """
        self.openmm.integrator = openmm.LangevinIntegrator(
            self.openmm.temperature,
            self.openmm.friction,
            self.openmm.timestep,
        )
        self.openmm.platform = openmm.Platform.getPlatformByName("CPU")
        self.openmm.context = openmm.Context(
            self.openmm.system,
            self.openmm.integrator,
            self.openmm.platform,
            self.openmm.properties,
        )
        self.openmm.context.setPositions(self.openmm.modeller.positions)

    def _generate_state(self, **kwargs) -> None:
        """Create the OpenMM State which is used to compute energies
        and forces.
        """
        self.openmm.state = self.openmm.context.getState(
            getEnergy=True,
            getForces=True,
            getPositions=True,
            **kwargs,
        )

    def _compute_components(self) -> dict[str: float]:
        """Calculate the components of the energy.

        :return: The individual contributions to the energy.
        """
        components = {}
        for i in range(self.openmm.system.getNumForces()):
            force = self.openmm.system.getForce(i)
            key = force.__class__.__name__.replace("Force", "Energy")
            key = " ".join(re.findall("[A-Z][a-z]*", key))
            value = self.openmm.context.getState(
                getEnergy=True,
                groups=2**i,
            ).getPotentialEnergy()/simtk.unit.kilojoule_per_mole
            components[key] = value
        return components

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> tuple[Any, ...]:
        # Get energy and forces from the state.
        self._generate_state()
        openmm_results = [
            (
                self.openmm.state.getPotentialEnergy()
                / simtk.unit.kilojoule_per_mole
            ),
        ]
        if return_forces:
            openmm_results.append(
                self.openmm.state.getForces(asNumpy=True)
                / simtk.unit.kilojoule_per_mole
                * simtk.unit.nanometers
                * NM_PER_ANGSTROM,
            )
        if return_components:
            openmm_results.append(self._compute_components())
        return tuple(openmm_results)

    def update(self, attr: str, value: Any) -> None:
        if "charges" in attr:
            self._update_charges(value)
        elif "positions" in attr:
            self._update_positions(value)
        elif "box" in attr:
            self._update_box(value)
        # else:
        #    raise AttributeError(
        #        (
        #            f"Unknown attribute '{attr}' attempted to be updated "
        #            + f"for {self.__class__.__name__}."
        #        ),
        #    )

    def _update_charges(self, charges: NDArray[np.float64]) -> None:
        """Update the particle charges for OpenMM.

        :param charges: |charges|
        """
        for i, charge in enumerate(charges):
            (
                _, sigma, epsilon,
            ) = self.nonbonded_force.getParticleParameters(i)
            self.nonbonded_force.setParticleParameters(
                i, charge, sigma, epsilon,
            )
        self.nonbonded_force.updateParametersInContext(
            self.openmm.context,
        )

    def _update_positions(self, positions: NDArray[np.float64]) -> None:
        """Update the particle positions for OpenMM.

        :param positions: |positions|
        """
        positions_temp = []
        for i in range(len(positions)):
            positions_temp.append(
                openmm.Vec3(
                    self._state.positions[i][0]*NM_PER_ANGSTROM,
                    self._state.positions[i][1]*NM_PER_ANGSTROM,
                    self._state.positions[i][2]*NM_PER_ANGSTROM,
                ) * simtk.unit.nanometer,
            )
        self.openmm.context.setPositions(positions_temp)

    def _update_box(self, box: NDArray[np.float64]) -> None:
        """Update the box vectors for OpenMM.

        :param box: |box|

        .. warning:: This method is not currently implemented.
        """
