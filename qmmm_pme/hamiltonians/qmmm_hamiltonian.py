#! /usr/bin/env python3
"""
ASE Calculator to combine QM and MM forces and energies.
"""
from __future__ import annotations

import warnings

from .hamiltonian import QMMMHamiltonianBase
from .hamiltonian import QMMMHamiltonianInterface


class QMMMHamiltonian(QMMMHamiltonianBase, QMMMHamiltonianInterface):
    """A wrapper for the QMMM.
    """

    def __init__(
            self,
            qm_hamiltonian,
            mm_hamiltonian,
            embedding_cutoff=14.,
    ):
        if qm_hamiltonian.system == mm_hamiltonian.system:
            super().__init__(qm_hamiltonian.system)
        else:
            raise ValueError
        if not set(qm_hamiltonian.atoms).isdisjoint(set(mm_hamiltonian.atoms)):
            warnings.warn("", RuntimeWarning)
        self.qm_hamiltonian = qm_hamiltonian
        self.mm_hamiltonian = mm_hamiltonian
        self.embedding_cutoff = embedding_cutoff

    def __add__(self, other):
        return other + self

    def __str__(self):
        string = (
            "H^{QM/MM} = "
            + str(self.qm_hamiltonian) + " + "
            + str(self.mm_hamiltonian)
        )
        return string
