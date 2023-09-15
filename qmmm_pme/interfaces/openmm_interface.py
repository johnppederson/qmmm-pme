#! /usr/bin/env python3
"""A module to define the :class:`OpenMMInterface` class.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
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
from simtk.unit import Quantity

from .interface import SoftwareInterface
from .interface import SoftwareSettings
from .interface import SystemTypes

if TYPE_CHECKING:
    import qmmm_pme
    from numpy.typing import NDArray


# def default_properties():
#    return {"ReferenceVextGrid": "true"}

@dataclass(frozen=True)
class OpenMMSettings(SoftwareSettings):
    """A class which holds the OpenMM settings.
    """
    system: qmmm_pme.System
    nonbonded_method: str = "PME"
    nonbonded_cutoff: float | int = 14.
    pme_gridnumber: int = 30
    pme_alpha: float | int = 5.
    temperature: float | int = 300.
    friction: float | int = 0.001
    timestep: float | int = 1.
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenMMInterface(SoftwareInterface):
    """A class which wraps the functional components of OpenMM.
    """
    pdb: PDBFile
    modeller: Modeller
    forcefield: ForceField
    system: System
    context: Context

    def _generate_state(self, **kwargs: bool | set[int]) -> State:
        """Create the OpenMM State which is used to compute energies
        and forces.
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
        """Calculate the components of the energy.

        :return: The individual contributions to the energy.
        """
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
        """Creates the PME potential grid.

        :return: The PME potential grid calculated by OpenMM.

        .. warning:: Requires investigation of return type.
        """
        state = self._generate_state(getVext_grids=True, **kwargs)
        pme_potential = np.array(state.getVext_grid())
        return pme_potential

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Update the particle charges for OpenMM.

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
        """Update the particle positions for OpenMM.

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
        """
        """
        notifiers = {
            "charges": self.update_charges,
            "positions": self.update_positions,
            "box": self.update_box,
        }
        return notifiers

    def get_topology_notifiers(
            self,
    ) -> dict[str, Callable[..., None]]:
        """
        """
        notifiers: dict[str, Callable[..., None]] = {}
        return notifiers


def openmm_system_factory(settings: OpenMMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def openmm_subsystem_factory(settings: OpenMMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_qm_atoms(settings, system)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def openmm_embedding_factory(settings: OpenMMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_non_embedding(settings, pdb, system)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def _build_base(
        settings: OpenMMSettings,
) -> tuple[PDBFile, Modeller, ForceField, System]:
    pdb = _build_pdb(settings)
    modeller = _build_modeller(pdb)
    forcefield = _build_forcefield(settings, modeller)
    system = _build_system(settings, forcefield, modeller)
    _adjust_forces(settings, system)
    return pdb, modeller, forcefield, system


def _build_pdb(settings: OpenMMSettings) -> PDBFile:
    """
    """
    # for xml in files.topology_list:
    #    Topology().loadBondDefinitions(xml)
    pdb = PDBFile(settings.system.files.pdb_list()[0])
    pdb.topology.loadBondDefinitions(
        settings.system.files.topology_list()[0],
    )
    return pdb


def _build_modeller(pdb: PDBFile) -> Modeller:
    """
    """
    modeller = Modeller(pdb.topology, pdb.positions)
    return modeller


def _build_forcefield(settings: OpenMMSettings, modeller: Modeller) -> Modeller:
    """
    """
    forcefield = ForceField(*settings.system.files.forcefield_list())
    modeller.addExtraParticles(forcefield)
    return forcefield


def _build_system(
        settings: OpenMMSettings, forcefield: ForceField, modeller: Modeller,
) -> System:
    """
    """
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedCutoff=settings.nonbonded_cutoff * angstrom,
        rigidWater=False,
    )
    return system


def _adjust_forces(settings: OpenMMSettings, system: System) -> None:
    """Generate OpenMM non-bonded forces.
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
        settings: OpenMMSettings, system: System, modeller: Modeller,
) -> Context:
    """
    """
    integrator = LangevinIntegrator(
        settings.temperature * kelvin,
        settings.friction / femtosecond,
        settings.timestep * femtosecond,
    )
    platform = Platform.getPlatformByName("CPU")
    context = Context(system, integrator, platform, settings.properties)
    context.setPositions(modeller.positions)
    return context


def _exclude_qm_atoms(settings: OpenMMSettings, system: System) -> None:
    """Generate OpenMM Force exclusions for the QM atoms.
    """
    harmonic_bond_forces = [
        force for force in system.getForces()
        if isinstance(force, HarmonicBondForce)
    ]
    harmonic_angle_forces = [
        force for force in system.getForces()
        if isinstance(force, HarmonicAngleForce)
    ]
    periodic_torsion_forces = [
        force for force in system.getForces()
        if isinstance(force, PeriodicTorsionForce)
    ]
    rb_torsion_forces = [
        force for force in system.getForces()
        if isinstance(force, RBTorsionForce)
    ]
    qm_atoms = [
        atom for residue in settings.system.topology.qm_atoms()
        for atom in residue
    ]
    # Remove double-counted intramolecular interactions for QM atoms.
    for force in harmonic_bond_forces:
        for i in range(force.getNumBonds()):
            p1, p2, r0, k = force.getBondParameters(i)
            if p1 in qm_atoms and p2 in qm_atoms:
                k = Quantity(0, unit=k.unit)
                force.setBondParameters(i, p1, p2, r0, k)
    for force in harmonic_angle_forces:
        for i in range(force.getNumAngles()):
            p1, p2, p3, r0, k = force.getAngleParameters(i)
            if p1 in qm_atoms and p2 in qm_atoms and p3 in qm_atoms:
                k = Quantity(0, unit=k.unit)
                force.setAngleParameters(
                    i, p1, p2, p3, r0, k,
                )
    for force in periodic_torsion_forces:
        for i in range(force.getNumTorsions()):
            (
                p1, p2, p3, p4, n, t, k,
            ) = force.getTorsionParameters(i)
            if (
                p1 in qm_atoms and p2 in qm_atoms
                and p3 in qm_atoms and p4 in qm_atoms
            ):
                k = Quantity(0, unit=k.unit)
                force.setTorsionParameters(
                    i, p1, p2, p3, p4, n, t, k,
                )
    for force in rb_torsion_forces:
        for i in range(force.getNumTorsions()):
            (
                p1, p2, p3, p4, c0, c1, c2, c3, c4, c5,
            ) = force.getTorsionParameters(i)
            if (
                p1 in qm_atoms and p2 in qm_atoms
                and p3 in qm_atoms and p4 in qm_atoms
            ):
                c0 = Quantity(0, unit=c0.unit)
                c1 = Quantity(0, unit=c1.unit)
                c2 = Quantity(0, unit=c2.unit)
                c3 = Quantity(0, unit=c3.unit)
                c4 = Quantity(0, unit=c4.unit)
                c5 = Quantity(0, unit=c5.unit)
                force.setTorsionParameters(
                    i, p1, p2, p3, p4, c0, c1, c2, c3, c4, c5,
                )


def _exclude_non_embedding(
        settings: OpenMMSettings, pdb: PDBFile, system: System,
) -> None:
    """Create additional OpenMM System with mechanical embedding
    forces.
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
