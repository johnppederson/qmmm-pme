#! /usr/bin/env python3
"""
ASE Calculator to combine QM and MM forces and energies.
"""
from __future__ import annotations

from .hamiltonian import PBCHamiltonianInterface
from .hamiltonian import QMMMHamiltonianInterface
from .qmmm_pme_hamiltonian import QMMMPMEHamiltonian


class PBCHamiltonian(PBCHamiltonianInterface):
    """A wrapper for the QM.
    """

    def __add__(self, other):
        if not isinstance(other, QMMMHamiltonianInterface):
            raise TypeError("...")
        return QMMMPMEHamiltonian(other, self)

    def __str__(self):
        return "H^{PME}"
