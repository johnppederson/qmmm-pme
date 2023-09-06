#! /usr/bin/env python3
"""
ASE Calculator to combine QM and MM forces and energies.
"""
from __future__ import annotations

from .hamiltonian import MMHamiltonianInterface
from .hamiltonian import QMHamiltonianInterface
from .qmmm_hamiltonian import QMMMHamiltonian


class MMHamiltonian(MMHamiltonianInterface):
    """A wrapper for the MM.
    """

    def __init__(
            self,
            system,
            nonbonded_method="PME",
            nonbonded_cutoff=14.,
            pme_gridnumber=60,
            pme_alpha=5.0,
    ):
        super().__init__(system)
        self.nonbonded_method = nonbonded_method
        self.nonbonded_cutoff = nonbonded_cutoff
        self.pme_gridnumber = pme_gridnumber
        self.pme_alpha = pme_alpha

    def __add__(self, other):
        if not isinstance(other, QMHamiltonianInterface):
            raise TypeError("...")
        return QMMMHamiltonian(other, self)

    def __str__(self):
        return "H^{MM}" + super().__str__()
