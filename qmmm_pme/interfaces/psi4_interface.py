#! /usr/bin/env python3
"""A module to define the :class:`Psi4Interface` class.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
import psi4.core
from numpy.typing import NDArray

from .interface import QMSettings
from .interface import SoftwareInterface
from .interface import SoftwareTypes
from .interface import SystemTypes
from qmmm_pme.common.units import BOHR_PER_ANGSTROM
from qmmm_pme.common.units import KJMOL_PER_EH

if TYPE_CHECKING:
    ComputationOptions = float


SOFTWARE_TYPE = SoftwareTypes.QM


psi4.core.be_quiet()


class Psi4Context:
    """A wrapper class for managing Psi4 Geometry object generation.

    :param atoms: |qm_atoms|
    :param embedding: |ae_atoms|
    :param elements: |elements|
    :param positions: |positions|
    :param charge: |charge|
    :param spin: |spin|
    """

    def __init__(
            self,
            atoms: list[list[int]],
            embedding: list[list[int]],
            elements: list[str],
            positions: NDArray[np.float64],
            charge: int,
            spin: int,
    ) -> None:
        self.atoms = atoms
        self.embedding = embedding
        self.elements = elements
        self.positions = positions
        self.charge = charge
        self.spin = spin

    @lru_cache
    def generate_molecule(self) -> psi4.core.Molecule:
        """Create the Geometry object for Psi4 calculations.

        :return: The Psi4 Geometry object.
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
        """Update the atom positions for Psi4.

        :param positions: |positions|
        """
        self.positions = positions
        self.generate_molecule.cache_clear()

    def update_embedding(self, embedding: list[list[int]]) -> None:
        """Update the analytic embedding atoms for Psi4.

        :param embedding: |ae_atoms|
        """
        self.embedding = embedding


@dataclass(frozen=True)
class Psi4Options:
    """An immutable wrapper class for storing Psi4 global options.

    :param basis: |basis_set|
    :param dft_spherical_points: |quadrature_spherical|
    :param dft_radial_points: |quadrature_radial|
    :param scf_type: |scf_type|
    :param scf__reference: The restricted or unrestricted Kohn-Sham SCF.
    :param scf__guess: The type of guess to use for the Psi4.
        calculation.
    """
    basis: str
    dft_spherical_points: int
    dft_radial_points: int
    scf_type: str
    scf__reference: str
    scf__guess: str


@dataclass(frozen=True)
class Psi4Interface(SoftwareInterface):
    """A :class:`SoftwareInterface` class which wraps the functional
    components of Psi4.

    :param options: The :class:`Psi4Options` object for the interface.
    :param functional: |functional|
    :param context: The :class:`Psi4Context` object for the interface.
    """
    options: Psi4Options
    functional: str
    context: Psi4Context

    @lru_cache
    def _generate_wavefunction(
            self,
            **kwargs: ComputationOptions,
    ) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object for use in Psi4
        calculations.

        :return: The Psi4 Wavefunction object.
        """
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
        energy = energy * KJMOL_PER_EH
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
        qm_indices = [
            atom for residue in self.context.atoms
            for atom in residue
        ]
        forces_temp[qm_indices, :] = forces[:len(qm_indices), :]
        ae_indices = [
            atom for residue in self.context.embedding
            for atom in residue
        ]
        if ae_indices:
            forces_temp[ae_indices, :] = forces[len(qm_indices):, :]
        forces = forces_temp
        return forces

    def compute_components(
            self,
            **kwargs: ComputationOptions,
    ) -> dict[str, float]:
        components: dict[str, float] = {}
        return components

    def compute_quadrature(self) -> NDArray[np.float64]:
        """Build a reference quadrature.

        :return: A reference quadrature constructed from the Psi4
            Geometry object.
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
        quadrature = np.concatenate(
            tuple([coord.reshape(-1, 1) for coord in vbase.get_np_xyzw()[0:3]]),
            axis=1,
        )
        return quadrature

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        """Update the atom positions for Psi4.

        :param positions: |positions|
        """
        self.context.update_positions(positions)
        self._generate_wavefunction.cache_clear()

    def update_embedding(self, embedding: list[list[int]]) -> None:
        """Update the analytic embedding atoms for Psi4.

        :param embedding: |ae_atoms|
        """
        self.context.update_embedding(embedding)
        self._generate_wavefunction.cache_clear()

    def update_num_threads(self, num_threads: int) -> None:
        """Update the number of threads for Psi4 to use.

        :param num_threads: The number of threads for Psi4 to use.
        """
        psi4.set_num_threads(num_threads)

    def update_memory(self, memory: str) -> None:
        """Update the amount of memory for Psi4 to use.

        :param memory: The amount of memory for Psi4 to use.
        """
        psi4.set_memory(memory)

    def get_state_notifiers(
            self,
    ) -> dict[str, Callable[..., None]]:
        notifiers = {
            "positions": self.update_positions,
        }
        return notifiers

    def get_topology_notifiers(
            self,
    ) -> dict[str, Callable[..., None]]:
        notifiers = {
            "ae_atoms": self.update_embedding,
        }
        return notifiers


def psi4_system_factory(settings: QMSettings) -> Psi4Interface:
    """A function which constructs the :class:`Psi4Interface` for a QM
    system.

    :param settings: The :class:`QMSettings` object to build the
        QM system interface from.
    :return: The :class:`Psi4Interface` for the QM system.
    """
    options = _build_options(settings)
    functional = settings.functional
    context = _build_context(settings)
    wrapper = Psi4Interface(
        options, functional, context,
    )
    return wrapper


def _build_options(settings: QMSettings) -> Psi4Options:
    """Build the :class:`Psi4Options` object.

    :param settings: The :class:`QMSettings` object to build from.
    :return: The :class:`Psi4Options` object built from the given
        settings.
    """
    options = Psi4Options(
        settings.basis_set,
        settings.quadrature_spherical,
        settings.quadrature_radial,
        settings.scf_type,
        "uks" if settings.spin > 1 else "rks",
        "read" if settings.read_guess else "auto",
    )
    return options


def _build_context(settings: QMSettings) -> Psi4Context:
    """Build the :class:`Psi4Context` object.

    :param settings: The :class:`QMSettings` object to build from.
    :return: The :class:`Psi4Context` object built from the given
        settings.
    """
    context = Psi4Context(
        settings.system.topology.qm_atoms(),
        [],
        settings.system.topology.elements(),
        settings.system.state.positions(),
        settings.charge,
        settings.spin,
    )
    return context


FACTORIES = {
    SystemTypes.SYSTEM: psi4_system_factory,
    SystemTypes.SUBSYSTEM: psi4_system_factory,
}
