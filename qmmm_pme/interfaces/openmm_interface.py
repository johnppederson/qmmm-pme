#! /usr/bin/env python3
"""A module to define the :class:`OpenMMInterface` class.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from openmm import Context
from openmm import CustomNonbondedForce
from openmm import HarmonicAngleForce
from openmm import HarmonicBondForce
from openmm import LangevinIntegrator
from openmm import NonbondedForce
from openmm import PeriodicTorsionForce
from openmm import Platform
from openmm import RBTorsionForce
from openmm import State
from openmm import System
from openmm import Vec3
from openmm.app import ForceField
from openmm.app import Modeller
from openmm.app import PDBFile
from simtk.unit import angstrom
from simtk.unit import femtosecond
from simtk.unit import kelvin
from simtk.unit import kilojoule_per_mole
from simtk.unit import nanometer

from .interface import MMSettings
from .interface import SoftwareInterface
from .interface import SoftwareTypes
from .interface import SystemTypes

if TYPE_CHECKING:
    from numpy.typing import NDArray


SOFTWARE_TYPE = SoftwareTypes.MM


@dataclass(frozen=True)
class OpenMMInterface(SoftwareInterface):
    """A :class:`SoftwareInterface` class which wraps the functional
    components of OpenMM.

    :param pdb: The OpenMM PDBFile object for the interface.
    :param modeller: The OpenMM Modeller object for the interface.
    :param forcefield: The OpenMM ForceField object for the interface.
    :param system: The OpenMM System object for the interface.
    :param context: The OpenMM Context object for the interface.
    """
    pdb: PDBFile
    modeller: Modeller
    forcefield: ForceField
    system: System
    context: Context

    def _generate_state(self, **kwargs: bool | set[int]) -> State:
        """Create the OpenMM State object which is used to compute
        energies and forces.

        :return: The current OpenMM State object.
        """
        state = self.context.getState(
            **kwargs,
        )
        return state

    def compute_energy(self, **kwargs: bool) -> float:
        # Get energy and forces from the state.
        state = self._generate_state(getEnergy=True, **kwargs)
        energy = state.getPotentialEnergy() / kilojoule_per_mole
        return energy

    def compute_forces(
            self,
            **kwargs: bool,
    ) -> NDArray[np.float64]:
        state = self._generate_state(getForces=True, **kwargs)
        forces = state.getForces(asNumpy=True) / kilojoule_per_mole * angstrom
        return forces

    def compute_components(
            self,
            **kwargs: bool,
    ) -> dict[str, float]:
        components = {}
        for force in self.system.getForces():
            key = type(force).__name__.replace("Force", "Energy")
            key = " ".join(re.findall("[A-Z][a-z]*", key))
            value = self._generate_state(
                getEnergy=True,
                groups={force.getForceGroup()},
                **kwargs,
            ).getPotentialEnergy() / kilojoule_per_mole
            components[key] = value
        return components

    def compute_pme_potential(self, **kwargs: bool) -> Any:
        state = self._generate_state(getVext_grids=True, **kwargs)
        pme_potential = np.array(state.getVext_grid())
        return pme_potential

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Update the atom charges for OpenMM.

        :param charges: |charges|
        """
        nonbonded_forces = [
            force for force in self.system.getForces()
            if isinstance(force, NonbondedForce)
        ]
        for force in nonbonded_forces:
            for i, charge in enumerate(charges):
                _, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, charge, sigma, epsilon)
            force.updateParametersInContext(self.context)

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        """Update the atom positions for OpenMM.

        :param positions: |positions|
        """
        positions_temp = []
        for i in range(len(positions)):
            positions_temp.append(
                Vec3(
                    positions[i][0],
                    positions[i][1],
                    positions[i][2],
                ) * angstrom,
            )
        self.context.setPositions(positions_temp)

    def update_box(self, box: NDArray[np.float64]) -> None:
        """Update the box vectors for OpenMM.

        :param box: |box|

        .. warning:: This method is not currently implemented.
        """

    def get_state_notifiers(
            self,
    ) -> dict[str, Callable[[NDArray[np.float64]], None]]:
        notifiers = {
            "charges": self.update_charges,
            "positions": self.update_positions,
            "box": self.update_box,
        }
        return notifiers

    def get_topology_notifiers(
            self,
    ) -> dict[str, Callable[..., None]]:
        notifiers: dict[str, Callable[..., None]] = {}
        return notifiers


def openmm_system_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface` for a
    standalone MM system.

    :param settings: The :class:`MMSettings` object to build the
        standalone MM system interface from.
    :return: The :class:`OpenMMInterface` for the standalone MM system.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def openmm_subsystem_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface` for an
    MM subsystem of a QM/MM supersystem.

    :param settings: The :class:`MMSettings` object to build the
        MM subsystem interface from.
    :return: The :class:`OpenMMInterface` for the MM subsystem.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_qm_atoms(settings, system)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def openmm_embedding_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface` for a
    mechanical embedding subsystem of a QM/MM supersystem.

    :param settings: The :class:`MMSettings` object to build the
        mechanical embedding subsystem interface from.
    :return: The :class:`OpenMMInterface` for the mechanical embedding
        subsystem.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_non_embedding(settings, pdb, system)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def _build_base(
        settings: MMSettings,
) -> tuple[PDBFile, Modeller, ForceField, System]:
    """Build the common OpenMM PDBFile, Modeller, ForceField, and System
    objects.

    :param settings: The :class:`MMSettings` object to build from.
    :return: The OpenMM PDBFile, Modeller, ForceField, and System
        objects built from the given settings.
    """
    pdb = _build_pdb(settings)
    modeller = _build_modeller(pdb)
    forcefield = _build_forcefield(settings, modeller)
    system = _build_system(settings, forcefield, modeller)
    _adjust_forces(settings, system)
    return pdb, modeller, forcefield, system


def _build_pdb(settings: MMSettings) -> PDBFile:
    """Build the OpenMM PDBFile object.

    :param settings: The :class:`MMSettings` object to build from.
    :return: The OpenMM PDBFile object built from the given settings.
    """
    # for xml in files.topology_list:
    #    Topology().loadBondDefinitions(xml)
    pdb = PDBFile(settings.system.files.pdb_list()[0])
    pdb.topology.loadBondDefinitions(
        settings.system.files.topology_list()[0],
    )
    return pdb


def _build_modeller(pdb: PDBFile) -> Modeller:
    """Build the OpenMM Modeller object.

    :param pdb: The OpenMM PDBFile object to build from.
    :return: The OpenMM Modeller object built from the given pdb.
    """
    modeller = Modeller(pdb.topology, pdb.positions)
    return modeller


def _build_forcefield(settings: MMSettings, modeller: Modeller) -> ForceField:
    """Build the OpenMM ForceField object.

    :param settings: The :class:`MMSettings` object to build from.
    :param modeller: The OpenMM Modeller object to build from.
    :return: The OpenMM ForceField object built from the given settings
        and modeller.
    """
    forcefield = ForceField(*settings.system.files.forcefield_list())
    modeller.addExtraParticles(forcefield)
    return forcefield


def _build_system(
        settings: MMSettings, forcefield: ForceField, modeller: Modeller,
) -> System:
    """Build the OpenMM System object.

    :param settings: The :class:`MMSettings` object to build from.
    :param forcefield: The OpenMM ForceField object to build from.
    :param modeller: The OpenMM Modeller object to build from.
    :return: The OpenMM System object built from the given settings,
        forcefield, and modeller.
    """
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedCutoff=settings.nonbonded_cutoff * angstrom,
        rigidWater=False,
    )
    return system


def _adjust_forces(settings: MMSettings, system: System) -> None:
    """Adjust the OpenMM Nonbonded forces.

    :param settings: The :class:`MMSettings` object to adjust with.
    :param system: The OpenMM System object to adjust.
    """
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)
    # Get force types and set method.
    nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, NonbondedForce)
    ]
    for force in nonbonded_forces:
        force.setNonbondedMethod(
            getattr(NonbondedForce, settings.nonbonded_method),
        )
        force.setPMEParameters(
            settings.pme_alpha,
            settings.pme_gridnumber,
            settings.pme_gridnumber,
            settings.pme_gridnumber,
        )


def _build_context(
        settings: MMSettings, system: System, modeller: Modeller,
) -> Context:
    """Build the OpenMM Context object.

    :param settings: The :class:`MMSettings` object to build from.
    :param system: The OpenMM System object to build from.
    :param modeller: The OpenMM Modeller object to build from.
    :return: The OpenMM System object built from the given settings,
        system, and modeller.
    """
    integrator = LangevinIntegrator(
        settings.temperature * kelvin,
        settings.friction / femtosecond,
        settings.timestep * femtosecond,
    )
    platform = Platform.getPlatformByName("CPU")
    context = Context(system, integrator, platform)
    context.setPositions(modeller.positions)
    return context


def _exclude_qm_atoms(settings: MMSettings, system: System) -> None:
    """Generate OpenMM Force exclusions for the QM atoms.

    :param settings: The :class:`MMSettings` object to make exclusions
        with.
    :param system: The OpenMM System object to make exclusions on.
    """
    # Remove double-counted intramolecular interactions for QM atoms.
    qm_atoms = {
        atom for residue in settings.system.topology.qm_atoms()
        for atom in residue
    }
    _exclude_harmonic_bond(system, qm_atoms)
    _exclude_harmonic_angle(system, qm_atoms)
    _exclude_periodic_torsion(system, qm_atoms)
    _exclude_rb_torsion(system, qm_atoms)


def _exclude_harmonic_bond(system: System, atoms: set[int]) -> None:
    """Generate OpenMM HarmonicBondForce exclusions for a given set of
    atoms.

    :param system: The OpenMM System object to make exclusions on.
    :param atoms: The atoms to exlcude from HarmonicBondForce
        calculations.
    """
    harmonic_bond_forces = [
        force for force in system.getForces()
        if isinstance(force, HarmonicBondForce)
    ]
    for force in harmonic_bond_forces:
        for i in range(force.getNumBonds()):
            *p, r0, k = force.getBondParameters(i)
            if not set(p).isdisjoint(atoms):
                k *= 0
                force.setBondParameters(i, *p, r0, k)


def _exclude_harmonic_angle(system: System, atoms: set[int]) -> None:
    """Generate OpenMM HarmonicAngleForce exclusions for a given set of
    atoms.

    :param system: The OpenMM System object to make exclusions on.
    :param atoms: The atoms to exlcude from HarmonicAngleForce
        calculations.
    """
    harmonic_angle_forces = [
        force for force in system.getForces()
        if isinstance(force, HarmonicAngleForce)
    ]
    for force in harmonic_angle_forces:
        for i in range(force.getNumAngles()):
            *p, r0, k = force.getAngleParameters(i)
            if not set(p).isdisjoint(atoms):
                k *= 0
                force.setAngleParameters(i, *p, r0, k)


def _exclude_periodic_torsion(system: System, atoms: set[int]) -> None:
    """Generate OpenMM PeriodicTorsionForce exclusions for a given set
    of atoms.

    :param system: The OpenMM System object to make exclusions on.
    :param atoms: The atoms to exlcude from PeriodicTorsionForce
        calculations.
    """
    periodic_torsion_forces = [
        force for force in system.getForces()
        if isinstance(force, PeriodicTorsionForce)
    ]
    for force in periodic_torsion_forces:
        for i in range(force.getNumTorsions()):
            *p, n, t, k = force.getTorsionParameters(i)
            if not set(p).isdisjoint(atoms):
                k *= 0
                force.setTorsionParameters(i, *p, n, t, k)


def _exclude_rb_torsion(system: System, atoms: set[int]) -> None:
    """Generate OpenMM RBTorsionForce exclusions for a given set of
    atoms.

    :param system: The OpenMM System object to make exclusions on.
    :param atoms: The atoms to exlcude from RBTorsionForce
        calculations.
    """
    rb_torsion_forces = [
        force for force in system.getForces()
        if isinstance(force, RBTorsionForce)
    ]
    for force in rb_torsion_forces:
        for i in range(force.getNumTorsions()):
            *p, c0, c1, c2, c3, c4, c5 = force.getTorsionParameters(i)
            if not set(p).isdisjoint(atoms):
                c0 *= 0
                c1 *= 0
                c2 *= 0
                c3 *= 0
                c4 *= 0
                c5 *= 0
                force.setTorsionParameters(i, *p, c0, c1, c2, c3, c4, c5)


def _exclude_non_embedding(
        settings: MMSettings, pdb: PDBFile, system: System,
) -> None:
    """Generate OpenMM exclusions for mechanical embedding.

    :param settings: The :class:`MMSettings` object to make exclusions
        with.
    :param pdb: The OpenMM PDBFile object to make exclusions with.
    :param system: The OpenMM System object to make exclusions on.
    """
    nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, NonbondedForce)
    ]
    custom_nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, CustomNonbondedForce)
    ]
    embedding_forces = []
    # Interaction groups require atom indices to be input as sets.
    qm_atom_set = {
        atom for residue in settings.system.topology.qm_atoms()
        for atom in residue
    }
    mm_atom_set = {
        atom for residue in settings.system.topology.mm_atoms()
        for atom in residue
    }
    if custom_nonbonded_forces:
        for force in custom_nonbonded_forces:
            embedding_forces.append(force.__copy__())
    else:
        for force in nonbonded_forces:
            sigmas = []
            epsilons = []
            for residue in pdb.topology.residues():
                for atom in residue.atoms():
                    _, sigma, epsilon = force.getParticleParameters(
                        atom.index,
                    )
                    sigmas.append(sigma / nanometer)
                    epsilons.append(epsilon / kilojoule_per_mole)
            force = CustomNonbondedForce(
                """4*epsilon*((sigma/r)^12-(sigma/r)^6);
                sigma=0.5*(sigma1+sigma2);
                epsilon=sqrt(epsilon1*epsilon2)""",
            )
            force.addPerParticleParameter("epsilon")
            force.addPerParticleParameter("sigma")
            # add particles with LJ parameters
            for residue in pdb.topology.residues():
                for atom in residue.atoms():
                    force.addParticle(
                        [epsilons[atom.index], sigmas[atom.index]],
                    )
            embedding_forces.append(force)
    for _ in system.getForces():
        system.removeForce(0)
    for force in embedding_forces:
        force.addInteractionGroup(
            qm_atom_set,
            mm_atom_set,
        )
        system.addForce(force)


FACTORIES = {
    SystemTypes.SYSTEM: openmm_system_factory,
    SystemTypes.SUBSYSTEM: openmm_subsystem_factory,
    SystemTypes.EMBEDDING: openmm_embedding_factory,
}
