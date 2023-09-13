#! /usr/bin/env python3
"""A module to define the :class:`Psi4Interface` class.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import psi4.core
from numpy.typing import NDArray

from .interface import SoftwareInterface
from .interface import SoftwareSettings
from .interface import SystemTypes
from qmmm_pme.common.units import BOHR_PER_ANGSTROM
from qmmm_pme.common.units import KJMOL_PER_EH

if TYPE_CHECKING:
    from qmmm_pme import System
    ComputationOptions = float


@dataclass(frozen=True)
class Psi4Settings(SoftwareSettings):
    """A class which holds the Psi4 settings.
    """
    system: System
    basis_set: str
    functional: str
    charge: int
    spin: int
    quadrature_spherical: int = 302
    quadrature_radial: int = 75
    scf_type: str = "df"
    read_guess: bool = True
    reference_energy: float | int | None = None


@dataclass
class Psi4Context:
    atoms: list[list[int]]
    embedding: list[list[int]]
    elements: list[str]
    positions: NDArray[np.float64]
    charge: int
    spin: int

    @lru_cache
    def generate_molecule(self) -> psi4.core.Molecule:
        """Create the Molecule object for the Psi4 calculation.
        """
        geometrystring = """\n"""
        for residue in self.atoms:
            for atom in residue:
                position = self.positions[atom]
                element = self.elements[atom]
                geometrystring = (
                    geometrystring
                    + str(element) + " "
                    + str(position[0]) + " "
                    + str(position[1]) + " "
                    + str(position[2]) + "\n"
                )
        geometrystring += str(self.charge) + " "
        geometrystring += str(self.spin) + "\n"
        geometrystring += "symmetry c1\n"
        geometrystring += "noreorient\nnocom\n"
        return psi4.geometry(geometrystring)

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        self.positions = positions
        self.generate_molecule.cache_clear()


@dataclass(frozen=True)
class Psi4Options:
    """
    """
    basis: str
    dft_spherical_points: int
    dft_radial_points: int
    scf_type: str
    scf__reference: str
    scf__guess: str


@dataclass(frozen=True)
class Psi4Interface(SoftwareInterface):
    """A class which wraps the functional components of Psi4.
    """
    options: Psi4Options
    functional: str
    context: Psi4Context
    reference_energy: float | int

    @lru_cache
    def _generate_wavefunction(
            self,
            **kwargs: ComputationOptions,
    ) -> psi4.core.Wavefunction:
        molecule = self.context.generate_molecule()
        psi4.set_options(asdict(self.options))
        _, wfn = psi4.energy(
            self.functional,
            return_wfn=True,
            molecule=molecule,
            **kwargs,
        )
        wfn.to_file(
            wfn.get_scratch_filename(180),
        )
        return wfn

    def compute_energy(self, **kwargs: ComputationOptions) -> float:
        wfn = self._generate_wavefunction(**kwargs)
        energy = wfn.energy()
        energy = (energy-self.reference_energy) * KJMOL_PER_EH
        return energy

    def compute_forces(
            self,
            **kwargs: ComputationOptions,
    ) -> NDArray[np.float64]:
        wfn = self._generate_wavefunction(**kwargs)
        psi4.set_options(asdict(self.options))
        forces = psi4.gradient(
            self.functional,
            ref_wfn=wfn,
            **kwargs,
        )
        forces = forces.to_array() * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
        forces_temp = np.zeros_like(self.context.positions)
        forces_temp[self.context.atoms, :] = forces[
            :len(
                self.context.atoms,
            ), :
        ]
        forces_temp[self.context.embedding, :] = forces[
            len(
                self.context.atoms,
            ):, :
        ]
        forces = forces_temp
        return forces

    def compute_components(
            self,
            **kwargs: ComputationOptions,
    ) -> dict[str, float]:
        """Calculate the components of the energy.

        :return: The individual contributions to the energy.
        """
        wfn = self._generate_wavefunction(**kwargs)
        T = wfn.mintshelper().ao_kinetic()
        V = wfn.mintshelper().ao_potential()
        Da = wfn.Da()
        Db = wfn.Db()
        one_e_components = {
            "Electronic Kinetic Energy":
                Da.vector_dot(T) + Db.vector_dot(T),
            "Electronic Potential Energy":
                Da.vector_dot(V) + Db.vector_dot(V),
        }
        components = {
            "Nuclear Repulsion Energy":
                wfn.variable("NUCLEAR REPULSION ENERGY"),
            "One-Electron Energy":
                wfn.variable("ONE-ELECTRON ENERGY"),
            ".": one_e_components,
            "Two-Electron Energy":
                wfn.variable("TWO-ELECTRON ENERGY"),
            "Exchange-Correlation Energy":
                wfn.variable("DFT XC ENERGY"),
        }
        return components

    def compute_quadrature(self) -> NDArray[np.float64]:
        """Build a reference quadrature to interpolate into the PME
        potential grid.

        :return: A reference quadrature constructed from the geometry of
            the QM subsystem.
        """
        molecule = self.context.generate_molecule()
        sup_func = psi4.driver.dft.build_superfunctional(
            self.functional,
            True,
        )[0]
        basis = psi4.core.BasisSet.build(
            molecule,
            "ORBITAL",
            self.options.basis,
        )
        vbase = psi4.core.VBase.build(basis, sup_func, "RV")
        vbase.initialize()
        quadrature = (vbase.get_np_xyzw().T)[:, 0:3]
        return quadrature

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        """Update the particle positions for Psi4.

        :param positions: |positions|
        """
        self.context.update_positions(positions)
        self._generate_wavefunction.cache_clear()

    def update_num_threads(self, num_threads: int) -> None:
        """Update the particle positions for Psi4.

        :param positions: |positions|
        """
        psi4.set_num_threads(num_threads)

    def update_memory(self, memory: str) -> None:
        """Update the particle positions for Psi4.

        :param positions: |positions|
        """
        psi4.set_memory(memory)


def psi4_system_factory(settings: Psi4Settings) -> Psi4Interface:
    """A function which constructs the :class:`Psi4Interface`.
    """
    options = _build_options(settings)
    functional = settings.functional
    context = _build_context(settings)
    reference_energy = _build_reference_energy(
        settings, options, functional, context,
    )
    wrapper = Psi4Interface(options, functional, context, reference_energy)
    return wrapper


def _build_options(settings: Psi4Settings) -> Psi4Options:
    options = Psi4Options(
        settings.basis_set,
        settings.quadrature_spherical,
        settings.quadrature_radial,
        settings.scf_type,
        "uks" if settings.spin > 1 else "rks",
        "read" if settings.read_guess else "auto",
    )
    return options


def _build_context(settings: Psi4Settings) -> Psi4Context:
    context = Psi4Context(
        settings.system.topology.qm_atoms(),
        [],
        settings.system.topology.elements(),
        settings.system.state.positions(),
        settings.charge,
        settings.spin,
    )
    return context


def _build_reference_energy(
        settings: Psi4Settings, options: Psi4Options,
        functional: str, context: Psi4Context,
) -> float | int:
    """Calculate the ground state energy of the QM atoms.
    """
    if isinstance(settings.reference_energy, (int, float)):
        reference_energy = settings.reference_energy
    else:
        psi4.set_options(options)
        molecule = context.generate_molecule()
        reference_energy = psi4.optimize(
            functional,
            molecule=molecule,
        )
    return reference_energy


FACTORIES = {
    SystemTypes.SYSTEM: psi4_system_factory,
    SystemTypes.SUBSYSTEM: psi4_system_factory,
}
