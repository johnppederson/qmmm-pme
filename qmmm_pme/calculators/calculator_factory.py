#! /usr/bin/env python3
"""A module containing methods to create :class:`Calculator` objects
given a :class:`Hamiltonian` object.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .mm_calculator import MMCalculator
from .mm_pme_calculator import MMPMECalculator
from .mm_qmmm_calculator import MMQMMMCalculator
from .pbc_calculator import PBCCalculator
from .qm_calculator import QMCalculator
from .qm_pme_calculator import QMPMECalculator
from .qm_qmmm_calculator import QMQMMMCalculator
from .qmmm_calculator import QMMMCalculator
from .qmmm_pme_calculator import QMMMPMECalculator
from qmmm_pme.hamiltonians.hamiltonian import MMHamiltonianInterface
from qmmm_pme.hamiltonians.hamiltonian import QMHamiltonianInterface
from qmmm_pme.hamiltonians.hamiltonian import QMMMHamiltonianInterface
from qmmm_pme.hamiltonians.hamiltonian import QMMMPMEHamiltonianInterface

if TYPE_CHECKING:
    from .calculator import Calculator
    from qmmm_pme.hamiltonians.hamiltonian import Hamiltonian


def _create_mm_calculator(
        cls: type,
        hamiltonian: Hamiltonian,
) -> Calculator:
    """Creates an instance of an MM :class:`Calculator` class.

    :param cls: The specific class for the MM :class:`Calculator`.
    :param hamiltonian: A :class:`Hamiltonian` object.
    :return: A :class:`Calculator` object corresponding to the given
        :class:`Hamiltonian`.
    """
    calculator = cls(
        nonbonded_method=hamiltonian.nonbonded_method,
        nonbonded_cutoff=hamiltonian.nonbonded_cutoff,
        pme_gridnumber=hamiltonian.pme_gridnumber,
        pme_alpha=hamiltonian.pme_alpha,
    )
    return calculator


def _create_qm_calculator(
        cls: type,
        hamiltonian: Hamiltonian,
) -> Calculator:
    """Creates an instance of an QM :class:`Calculator` class.

    :param cls: The specific class for the QM :class:`Calculator`.
    :param hamiltonian: A :class:`Hamiltonian` object.
    :return: A :class:`Calculator` object corresponding to the given
        :class:`Hamiltonian`.
    """
    calculator = cls(
        basis_set=hamiltonian.basis_set,
        functional=hamiltonian.functional,
        charge=hamiltonian.charge,
        spin=hamiltonian.spin,
        quadrature_spherical=hamiltonian.quadrature_spherical,
        quadrature_radial=hamiltonian.quadrature_radial,
        scf_type=hamiltonian.scf_type,
        read_guess=hamiltonian.read_guess,
        reference_energy=hamiltonian.reference_energy,
    )
    return calculator


SUBSYSTEMS = {
    QMMMHamiltonianInterface:
        {
            MMHamiltonianInterface: MMQMMMCalculator,
            QMHamiltonianInterface: QMQMMMCalculator,
        },
    QMMMPMEHamiltonianInterface:
        {
            MMHamiltonianInterface: MMPMECalculator,
            QMHamiltonianInterface: QMPMECalculator,
        },
}


def _create_qmmm_calculator(
        cls: type,
        hamiltonian: Hamiltonian,
) -> Calculator:
    """Creates an instance of an QM/MM :class:`Calculator` class.

    :param cls: The specific class for the QM/MM :class:`Calculator`.
    :param hamiltonian: A :class:`Hamiltonian` object.
    :return: A :class:`Calculator` object corresponding to the given
        :class:`Hamiltonian`.
    """
    h_mm = hamiltonian.mm_hamiltonian
    h_qm = hamiltonian.qm_hamiltonian
    calcs = [
        _create_qm_calculator(
            SUBSYSTEMS[
                type(hamiltonian).__bases__[-1]
            ][type(h_qm).__bases__[-1]],
            h_qm,
        ),
        _create_mm_calculator(
            SUBSYSTEMS[
                type(hamiltonian).__bases__[-1]
            ][type(h_mm).__bases__[-1]],
            h_mm,
        ),
    ]
    if type(hamiltonian).__bases__[-1] is QMMMPMEHamiltonianInterface:
        calcs.append(PBCCalculator(h_mm.pme_gridnumber, h_mm.pme_alpha))
    calculator = cls(
        *calcs,
        embedding_cutoff=hamiltonian.embedding_cutoff,
    )
    return calculator


FACTORIES = {
    MMHamiltonianInterface:
        (_create_mm_calculator, MMCalculator),
    QMHamiltonianInterface:
        (_create_qm_calculator, QMCalculator),
    QMMMHamiltonianInterface:
        (_create_qmmm_calculator, QMMMCalculator),
    QMMMPMEHamiltonianInterface:
        (_create_qmmm_calculator, QMMMPMECalculator),
}


def calculator_factory(hamiltonian: Hamiltonian) -> Calculator:
    """A Factory for the various :class:`Calculator` classes.

    :param hamiltonian: A :class:`Hamiltonian` object.
    :return: A :class:`Calculator` object corresponding to the given
        :class:`Hamiltonian`.
    """
    try:
        factory, cls = FACTORIES[type(hamiltonian).__bases__[-1]]
    except KeyError:
        raise TypeError(
            (
                f"Unrecognized Hamiltonian '{hamiltonian}'.  Unable to "
                + "create a Calculator."
            ),
        )
    calculator = factory(cls, hamiltonian)
    return calculator
