#! /usr/bin/env python3
"""
ASE Calculator to combine QM and MM forces and energies.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .hamiltonian import MMHamiltonianInterface
from .hamiltonian import QMHamiltonianInterface
from .qmmm_hamiltonian import QMMMHamiltonian
from qmmm_pme.calculators import StandaloneCalculator
from qmmm_pme.interfaces import mm_factories
from qmmm_pme.interfaces import MMSettings

if TYPE_CHECKING:
    from qmmm_pme import System
    from .qm_hamiltonian import QMHamiltonian


@dataclass
class MMHamiltonian(MMHamiltonianInterface):
    """A wrapper for the MM.

    :param nonbonded_method:
    """
    nonbonded_method: str = "PME"
    nonbonded_cutoff: float | int = 14.
    pme_gridnumber: int = 60
    pme_alpha: float | int = 5.0

    def build_calculator(self, system: System) -> StandaloneCalculator:
        settings = MMSettings(system=system, **asdict(self))
        interface = mm_factories[self.system_type](settings)
        calculator = StandaloneCalculator(system=system, interface=interface)
        return calculator

    def __add__(self, other: QMHamiltonian) -> QMMMHamiltonian:
        if not isinstance(other, QMHamiltonianInterface):
            raise TypeError("...")
        return QMMMHamiltonian(other, self)

    def __str__(self) -> str:
        return "H^{MM}" + super().__str__()
