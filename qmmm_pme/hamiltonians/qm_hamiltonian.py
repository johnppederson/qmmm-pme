#! /usr/bin/env python3
"""
ASE Calculator to combine QM and MM forces and energies.
"""
from __future__ import annotations

from .hamiltonian import MMHamiltonianInterface
from .hamiltonian import QMHamiltonianInterface
from .qmmm_hamiltonian import QMMMHamiltonian


class QMHamiltonian(QMHamiltonianInterface):
    """A wrapper for the QM.
    """

    def __init__(
            self,
            system,
            basis_set,
            functional,
            charge,
            spin,
            quadrature_spherical=302,
            quadrature_radial=75,
            scf_type="df",
            read_guess=True,
            reference_energy=None,
    ):
        super().__init__(system)
        self.basis_set = basis_set
        self.functional = functional
        self.quadrature_spherical = quadrature_spherical
        self.quadrature_radial = quadrature_radial
        self.charge = charge
        self.spin = spin
        self.scf_type = scf_type
        self.read_guess = read_guess
        self.reference_energy = reference_energy

    def __add__(self, other):
        if not isinstance(other, MMHamiltonianInterface):
            raise TypeError("...")
        return QMMMHamiltonian(self, other)

    def __str__(self):
        return "H^{QM}" + super().__str__()
