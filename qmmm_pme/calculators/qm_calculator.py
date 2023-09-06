#! /usr/bin/env python3
"""A module to define the :class:`QMCalculator` class.
"""
from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np
import psi4.core

from .calculator import Calculator
from qmmm_pme.common import BOHR_PER_ANGSTROM
from qmmm_pme.common import KJMOL_PER_EH
from qmmm_pme.records import Namespace


class QMCalculator(Calculator):
    """A :class:`Calculator` class for performing quantum mechanics
    calculations for an entire system using Psi4.

    :param basis_set: |basis_set|
    :param functional: |functional|
    :param charge: |charge|
    :param spin: |spin|
    :param quadrature_spherical: |quadrature_spherical|
    :param quadrature_radial: |quadrature_radial|
    :param scf_type: |scf_type|
    :param read_guess: |read_guess|
    :param reference_energy: |reference_energy|
    """

    def __init__(
            self,
            basis_set: str,
            functional: str,
            charge: int,
            spin: int,
            quadrature_spherical: int | None = 302,
            quadrature_radial: int | None = 75,
            scf_type: str | None = "df",
            read_guess: bool | None = True,
            reference_energy: float | int | None = None,
    ) -> None:
        super().__init__()
        # Set options for Psi4.
        self.basis_set = basis_set
        self.functional = functional
        self.charge = charge
        self.spin = spin
        self.quadrature_spherical = quadrature_spherical
        self.quadrature_radial = quadrature_radial
        self.scf_type = scf_type
        self.read_guess = read_guess
        self.reference_energy = reference_energy
        psi4.core.be_quiet()
        self.psi4 = Namespace()
        self.options = {
            "basis": self.basis_set,
            "dft_spherical_points": self.quadrature_spherical,
            "dft_radial_points": self.quadrature_radial,
            "scf_type": self.scf_type,
        }
        psi4.set_options(self.options)
        # Create Namespace.
        self.psi4.wfn = None
        # Generate reference energy if none is given.
        if self.reference_energy is None:
            self._generate_reference_energy()

    def _generate_geometry(self) -> None:
        """Create the Molecule object for the Psi4 calculation.
        """
        qm_atoms = self._topology.groups["qm_atom"]
        if self.spin > 1:
            psi4.core.set_local_option("SCF", "REFERENCE", "UKS")
        # Construct geometry string.
        geometrystring = """\n"""
        for atom in qm_atoms:
            position = self._state.positions[atom]
            element = self._topology.elements[atom]
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
        self.psi4.molecule = psi4.geometry(geometrystring)

    def _generate_reference_energy(self) -> None:
        """Calculate the ground state energy of the QM atoms.
        """
        self._generate_geometry()
        self.reference_energy = psi4.optimize(
            self.functional,
            molecule=self.psi4.molecule,
        )

    def _compute_components(
            self,
    ) -> dict[str: float | dict[str: float]]:
        """Calculate the components of the energy.

        :return: The individual contributions to the energy.
        """
        T = self.psi4.wfn.mintshelper().ao_kinetic()
        V = self.psi4.wfn.mintshelper().ao_potential()
        Da = self.psi4.wfn.Da()
        Db = self.psi4.wfn.Db()
        components = {
            "Nuclear Repulsion Energy":
                self.psi4.wfn.variable("NUCLEAR REPULSION ENERGY"),
            "One-Electron Energy":
                self.psi4.wfn.variable("ONE-ELECTRON ENERGY"),
            ".": {
                "Electronic Kinetic Energy":
                    Da.vector_dot(T) + Db.vector_dot(T),
                "Electronic Potential Energy":
                    Da.vector_dot(V) + Db.vector_dot(V),
            },
            "Two-Electron Energy":
                self.psi4.wfn.variable("TWO-ELECTRON ENERGY"),
            "Exchange-Correlation Energy":
                self.psi4.wfn.variable("DFT XC ENERGY"),
        }
        return components

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
            **kwargs,
    ) -> tuple[Any, ...]:
        self._generate_geometry()
        self._update_options()
        (psi4_energy, wfn) = psi4.energy(
            self.functional,
            return_wfn=True,
            molecule=self.psi4.molecule,
            **kwargs,
        )
        self.psi4.wfn = wfn
        psi4_results = [
            (psi4_energy-self.reference_energy) * KJMOL_PER_EH,
        ]
        if return_forces:
            psi4_forces = psi4.gradient(
                self.functional,
                ref_wfn=wfn,
                **kwargs,
            )
            psi4_results.append(
                -np.asarray(psi4_forces)
                * KJMOL_PER_EH
                * BOHR_PER_ANGSTROM,
            )
        if return_components:
            psi4_results.append(self._compute_components())
        return tuple(psi4_results)

    def update(self, attr: str, value: Any) -> None:
        if "memory" in attr:
            psi4.set_memory(memory)
        elif "num_threads" in attr:
            psi4.set_num_threads(num_threads)
        elif "namespace" in attr:
            self.psi4[attr.split(".")[1]] = value
        # else:
        #    raise AttributeError(
        #        (
        #            f"Unknown attribute '{attr}' attempted to be updated "
        #            + f"for {self.__class__.__name__}."
        #        ),
        #    )

    def _update_options(self) -> None:
        """Update the Psi4 options.
        """
        self.options.update({
            "basis": self.basis_set,
            "dft_spherical_points": self.quadrature_spherical,
            "dft_radial_points": self.quadrature_radial,
            "scf_type": self.scf_type,
        })
        psi4.set_options(self.options)
        # Check for wavefunction if read_guess is True.
        if self.psi4.wfn and self.read_guess:
            self.psi4.wfn.to_file(
                self.psi4.wfn.get_scratch_filename(180),
            )
            psi4.core.set_local_option("SCF", "GUESS", "READ")
