#! /usr/bin/env python3
"""
ASE Calculator to combine QM and MM forces and energies.
"""
from __future__ import annotations

from .hamiltonian import QMMMHamiltonianBase
from .hamiltonian import QMMMPMEHamiltonianInterface


class QMMMPMEHamiltonian(QMMMHamiltonianBase, QMMMPMEHamiltonianInterface):
    """A wrapper for the QMMM.
    """

    def __init__(
            self,
            qmmm_hamiltonian,
            pbc_hamiltonian,
            embedding_cutoff=14.,
    ):
        if qmmm_hamiltonian.system == pbc_hamiltonian.system:
            super().__init__(qmmm_hamiltonian.system)
        else:
            raise ValueError
        self.qm_hamiltonian = qmmm_hamiltonian.qm_hamiltonian
        self.mm_hamiltonian = qmmm_hamiltonian.mm_hamiltonian
        self.pbc_hamiltonian = pbc_hamiltonian
        self.embedding_cutoff = embedding_cutoff

    def __add__(self, other):
        pass
