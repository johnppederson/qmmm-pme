#! /usr/bin/env python3
"""A module defining the :class:`MMQMMMCalculator` class.
"""
from __future__ import annotations

import copy
from typing import Any
from typing import Optional

import openmm
import simtk.unit

from .mm_calculator import MMCalculator
from qmmm_pme.common import NM_PER_ANGSTROM


class MMQMMMCalculator(MMCalculator):
    """A :class:`Calculator` class for performing molecular mechanics
    calculations for the MM subsystem of a QM/MM system using OpenMM.
    """

    def _generate_system(self) -> None:
        super()._generate_system()
        self.openmm.system_me = self.openmm.forcefield.createSystem(
            self.openmm.modeller.topology,
            nonbondedCutoff=self.openmm.nonbonded_cutoff,
            constraints=None,
            rigidWater=False,
        )
        # Set force groups.
        for i in range(self.openmm.system_me.getNumForces()):
            force = self.openmm.system_me.getForce(i)
            force.setForceGroup(i)

    def _adjust_forces(self) -> None:
        super()._adjust_forces()
        if not self.custom_nonbonded_force:
            sigmas = []
            epsilons = []
            for residue in self.openmm.pdb.topology.residues():
                for part in residue._atoms:
                    (
                        _, sig, eps,
                    ) = self.nonbonded_force.getParticleParameters(
                        part.index,
                    )
                    sigmas.append(sig._value)
                    epsilons.append(eps._value)
            self.sigmas = sigmas
            self.epsilons = epsilons
        self._generate_exclusions()

    def _generate_exclusions(self) -> None:
        """Generate OpenMM Force exclusions for the QM atoms.
        """
        harmonic_bond_force = [
            force for force in [
                self.openmm.system.getForce(i)
                for i in range(self.openmm.system.getNumForces())
            ]
            if isinstance(force, openmm.HarmonicBondForce)
        ][0]
        harmonic_angle_force = [
            force for force in [
                self.openmm.system.getForce(i)
                for i in range(self.openmm.system.getNumForces())
            ]
            if isinstance(force, openmm.HarmonicAngleForce)
        ][0]
        periodic_torsion_force = [
            force for force in [
                self.openmm.system.getForce(i)
                for i in range(self.openmm.system.getNumForces())
            ]
            if isinstance(force, openmm.PeriodicTorsionForce)
        ]
        rb_torsion_force = [
            force for force in [
                self.openmm.system.getForce(i)
                for i in range(self.openmm.system.getNumForces())
            ]
            if isinstance(force, openmm.RBTorsionForce)
        ]
        qm_atoms = self._topology.groups["qm_atom"]
        # Remove double-counted intramolecular interactions for QM atoms.
        for i in range(harmonic_bond_force.getNumBonds()):
            p1, p2, r0, k = harmonic_bond_force.getBondParameters(i)
            if p1 in qm_atoms or p2 in qm_atoms:
                k = simtk.unit.Quantity(0, unit=k.unit)
                harmonic_bond_force.setBondParameters(i, p1, p2, r0, k)
        for i in range(harmonic_angle_force.getNumAngles()):
            (
                p1, p2, p3, r0, k,
            ) = harmonic_angle_force.getAngleParameters(i)
            if p1 in qm_atoms or p2 in qm_atoms or p3 in qm_atoms:
                k = simtk.unit.Quantity(0, unit=k.unit)
                harmonic_angle_force.setAngleParameters(
                    i, p1, p2, p3, r0, k,
                )
        if periodic_torsion_force:
            periodic_torsion_force = periodic_torsion_force[0]
            for i in range(periodic_torsion_force.getNumTorsions()):
                (
                    p1, p2, p3, p4, n, t, k,
                ) = periodic_torsion_force.getTorsionParameters(i)
                if (
                    p1 in qm_atoms or p2 in qm_atoms
                    or p3 in qm_atoms or p4 in qm_atoms
                ):
                    k = simtk.unit.Quantity(0, unit=k.unit)
                    periodic_torsion_force.setTorsionParameters(
                        i, p1, p2, p3, p4, n, t, k,
                    )
        if rb_torsion_force:
            rb_torsion_force = rb_torsion_force[0]
            for i in range(rb_torsion_force.getNumTorsions()):
                (
                    p1, p2, p3, p4, c0, c1, c2, c3, c4, c5,
                ) = rb_torsion_force.getTorsionParameters(i)
                if (
                    p1 in qm_atoms or p2 in qm_atoms
                    or p3 in qm_atoms or p4 in qm_atoms
                ):
                    c0 = simtk.unit.Quantity(0, unit=c0.unit)
                    c1 = simtk.unit.Quantity(0, unit=c1.unit)
                    c2 = simtk.unit.Quantity(0, unit=c2.unit)
                    c3 = simtk.unit.Quantity(0, unit=c3.unit)
                    c4 = simtk.unit.Quantity(0, unit=c4.unit)
                    c5 = simtk.unit.Quantity(0, unit=c5.unit)
                    rb_torsion_force.setTorsionParameters(
                        i, p1, p2, p3, p4, c0, c1, c2, c3, c4, c5,
                    )

    def _generate_embedding_forces(self) -> None:
        """Create additional OpenMM System with mechanical embedding
        forces.
        """
        # Interaction groups require atom indices to be input as sets.
        qm_atom_set = set(self._topology.groups["qm_atom"])
        mm_atom_set = {
            part
            for residue in self._topology.groups["mm_atom"]
            for part in residue
        }
        # Create a new CustomNonbondedForce for mechanical embedding.
        if self.custom_nonbonded_force:
            custom_nonbonded_force = copy.deepcopy(
                self.custom_nonbonded_force[0],
            )
        else:
            custom_nonbonded_force = openmm.CustomNonbondedForce(
                """4*epsilon*((sigma/r)^12-(sigma/r)^6);
                 sigma=0.5*(sigma1+sigma2);
                 epsilon=sqrt(epsilon1*epsilon2)""",
            )
            custom_nonbonded_force.addPerParticleParameter("epsilon")
            custom_nonbonded_force.addPerParticleParameter("sigma")
            # add particles with LJ parameters
            for residue in self._topology.groups["all"]:
                for part in residue:
                    custom_nonbonded_force.addParticle(
                        [self.epsilons[part], self.sigmas[part]],
                    )
        custom_nonbonded_force.addInteractionGroup(
            qm_atom_set,
            mm_atom_set,
        )
        num_forces = self.openmm.system_me.getNumForces()
        for i in range(num_forces):
            self.openmm.system_me.removeForce(0)
        self.openmm.system_me.addForce(custom_nonbonded_force)

    def _generate_context(self) -> None:
        super()._generate_context()
        self._generate_embedding_forces()
        self.openmm.context_me = openmm.Context(
            self.openmm.system_me,
            copy.copy(self.openmm.integrator),
            self.openmm.platform,
            self.openmm.properties,
        )
        self.openmm.context_me.setPositions(
            self.openmm.modeller.positions,
        )

    def _generate_state(self, **kwargs) -> None:
        super()._generate_state(**kwargs)
        self.openmm.state_me = self.openmm.context_me.getState(
            getEnergy=True,
            getForces=True,
            getPositions=True,
        )

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> tuple[Any, ...]:
        # Get energy and forces from the state.
        self._generate_state()
        qm_atoms = self._topology.groups["qm_atom"]
        mm_energy = (
            self.openmm.state.getPotentialEnergy()
            / simtk.unit.kilojoule_per_mole
        )
        me_energy = (
            self.openmm.state_me.getPotentialEnergy()
            / simtk.unit.kilojoule_per_mole
        )
        openmm_results = [mm_energy]
        if return_forces:
            mm_forces = (
                self.openmm.state.getForces(asNumpy=True)
                / simtk.unit.kilojoule_per_mole
                * simtk.unit.nanometers
                * NM_PER_ANGSTROM
            )
            me_forces = (
                self.openmm.state_me.getForces(asNumpy=True)
                / simtk.unit.kilojoule_per_mole
                * simtk.unit.nanometers
                * NM_PER_ANGSTROM
            )
            mm_forces[qm_atoms, :] = me_forces[qm_atoms, :]
            openmm_results.append(mm_forces)
        if return_components:
            components = self._compute_components()
            components["Nonbonded Energy"] -= me_energy
            components["Mechanical Embedding Energy"] = me_energy
            openmm_results.append(components)
        return tuple(openmm_results)

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
        self.openmm.context_me.setPositions(positions_temp)
