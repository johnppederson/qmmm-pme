#! /usr/bin/env python3
"""A module defining the :class:`QMHamiltonian` class.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .hamiltonian import MMHamiltonianInterface
from .hamiltonian import QMHamiltonianInterface
from .qmmm_hamiltonian import QMMMHamiltonian
from qmmm_pme.calculators import StandaloneCalculator
from qmmm_pme.interfaces import qm_factories
from qmmm_pme.interfaces import QMSettings

if TYPE_CHECKING:
    from qmmm_pme import System
    from .mm_hamiltonian import MMHamiltonian


@dataclass
class QMHamiltonian(QMHamiltonianInterface):
    """A wrapper class to store settings for QM calculations.

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
    basis_set: str
    functional: str
    charge: int
    spin: int
    quadrature_spherical: int = 302
    quadrature_radial: int = 75
    scf_type: str = "df"
    read_guess: bool = True
    reference_energy: float | int | None = None

    def build_calculator(self, system: System) -> StandaloneCalculator:
        qm_atoms = self.parse_atoms(system)
        system.topology.qm_atoms.update(qm_atoms)
        settings = QMSettings(system=system, **asdict(self))
        interface = qm_factories[self.system_type](settings)
        calculator = StandaloneCalculator(system=system, interface=interface)
        return calculator

    def __add__(self, other: MMHamiltonian) -> QMMMHamiltonian:
        if not isinstance(other, MMHamiltonianInterface):
            raise TypeError("...")
        return QMMMHamiltonian(self, other)

    def __str__(self) -> str:
        return "H^{QM}" + super().__str__()
