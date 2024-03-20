#! /usr/bin/env python3
"""A module defining the :class:`MMHamiltonian` class.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .hamiltonian import CalculatorHamiltonian
from qmmm_pme.calculators import InterfaceCalculator
from qmmm_pme.common import Subsystem
from qmmm_pme.common import TheoryLevel
from qmmm_pme.interfaces import mm_factory
from qmmm_pme.interfaces import MMSettings

if TYPE_CHECKING:
    from qmmm_pme import System


@dataclass
class MMHamiltonian(CalculatorHamiltonian):
    """A wrapper class to store settings for MM calculations.

    :param nonbonded_method: |nonbonded_method|
    :param nonbonded_cutoff: |nonbonded_cutoff|
    :param pme_gridnumber: |pme_gridnumber|
    :param pme_alpha: |pme_alpha|
    """
    forcefield_file: str | list[str]
    topology_file: str | list[str] | None = None
    nonbonded_method: str = "PME"
    nonbonded_cutoff: float | int = 14.
    pme_gridnumber: int | None = None
    pme_alpha: float | int | None = None

    def __post_init__(self) -> None:
        self.theory_level = TheoryLevel.MM

    def build_calculator(self, system: System) -> InterfaceCalculator:
        mm_atoms = self._parse_atoms(system)
        system.subsystems[mm_atoms] = Subsystem.III
        settings = MMSettings(system=system, **asdict(self))
        interface = mm_factory(settings)
        calculator = InterfaceCalculator(system=system, interface=interface)
        return calculator

    def __str__(self) -> str:
        return "H^{MM}" + super().__str__()
