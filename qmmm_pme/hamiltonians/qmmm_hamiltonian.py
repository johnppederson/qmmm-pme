#! /usr/bin/env python3
"""
ASE Calculator to combine QM and MM forces and energies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

from qmmm.interfaces import SystemTypes

from .hamiltonian import QMMMHamiltonianInterface
from qmmm_pme.calculators import QMMMCalculator
from qmmm_pme.calculators.calculator import CalculatorType

if TYPE_CHECKING:
    from qmmm_pme import System
    from .hamiltonian import Hamiltonian
    from .qm_hamiltonian import QMHamiltonian
    from .mm_hamiltonian import MMHamiltonian


@dataclass
class QMMMHamiltonian(QMMMHamiltonianInterface):
    """A wrapper for the QMMM.
    """
    qm_hamiltonian: QMHamiltonian
    mm_hamiltonian: MMHamiltonian
    embedding_cutoff: float | int = 14.

    def __post_init__(self) -> None:
        self.qm_hamiltonian.system_type = SystemTypes.SUBSYSTEM
        self.mm_hamiltonian.system_type = SystemTypes.SUBSYSTEM
        self.me_hamiltonian = deepcopy(self.mm_hamiltonian)
        self.me_hamiltonian.system_type = SystemTypes.EMBEDDING

    def __or__(self, other: Any) -> Hamiltonian:
        if not isinstance(other, (int, float)):
            raise TypeError("...")
        self.embedding_cutoff = other
        return self

    def build_calculator(self, system: System) -> QMMMCalculator:
        qm_calculator = self.qm_hamiltonian.build_calculator(system)
        mm_calculator = self.mm_hamiltonian.build_calculator(system)
        me_calculator = self.me_hamiltonian.build_calculator(system)
        calculators = {
            CalculatorType.QM: qm_calculator,
            CalculatorType.MM: mm_calculator,
            CalculatorType.ME: me_calculator,
        }
        calculator = QMMMCalculator(
            system=system,
            calculators=calculators,
            embedding_cutoff=self.embedding_cutoff,
        )
        return calculator

    def __add__(self, other: Any) -> Any:
        return other + self

    def __str__(self) -> str:
        string = (
            "H^{QM/MM} = "
            + str(self.qm_hamiltonian) + " + "
            + str(self.mm_hamiltonian)
        )
        return string
