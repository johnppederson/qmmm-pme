#! /usr/bin/env python3
"""A module to define the :class:`SoftwareInterface` base class and the
various :class:`SoftwareSettings` classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qmmm_pme import System
    from numpy.typing import NDArray
    import numpy as np


class SoftwareTypes(Enum):
    """Enumerations of the different types of software to interface
    with.
    """
    QM = "A software for performing QM calculations."
    MM = "A software for performing MM calculations."


class SystemTypes(Enum):
    """Enumerations of the types of subsystems for a QM/MM calculation.
    """
    SYSTEM = "A System."
    SUBSYSTEM = "A Subsystem of a QM/MM System."
    EMBEDDING = "A Subsystem for Mechanical Embedding."


class SoftwareSettings(ABC):
    """An abstract :class:`SoftwareSettings` base class.

    .. note:: This currently doesn't do anything.
    """


@dataclass(frozen=True)
class MMSettings(SoftwareSettings):
    """An immutable wrapper class which holds settings for an MM
    software interface.

    :param system: |system| to perform MM calculations on.
    :param nonbonded_method: |nonbonded_method|
    :param nonbonded_cutoff: |nonbonded_cutoff|
    :param pme_gridnumber: |pme_gridnumber|
    :param pme_alpha: |pme_alpha|
    :param temperature: |temperature|
    :param friction: |friction|
    :param timestep: |timestep|
    """
    system: System
    nonbonded_method: str = "PME"
    nonbonded_cutoff: float | int = 14.
    pme_gridnumber: int = 30
    pme_alpha: float | int = 5.
    temperature: float | int = 300.
    friction: float | int = 0.001
    timestep: float | int = 1.


@dataclass(frozen=True)
class QMSettings(SoftwareSettings):
    """An immutable wrapper class which holds settings for a QM software
    interface.

    :param system: |system| to perform QM calculations on.
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
    system: System
    basis_set: str
    functional: str
    charge: int
    spin: int
    quadrature_spherical: int = 302
    quadrature_radial: int = 75
    scf_type: str = "df"
    read_guess: bool = True
    reference_energy: float | int | None = None


class SoftwareInterface(ABC):
    """The abstract :class:`SoftwareInterface` base class.
    """

    @abstractmethod
    def compute_energy(self) -> float:
        """Compute the energy for the :class:`System` with the
        :class:`SoftwareInterface`.

        :return: The calculated energy, in kJ/mol.
        """

    @abstractmethod
    def compute_forces(self) -> NDArray[np.float64]:
        """Compute the forces for the :class:`System` with the
        :class:`SoftwareInterface`.

        :return: The calculated forces, in kJ/mol/Angstrom.
        """

    @abstractmethod
    def compute_components(self) -> dict[str, float]:
        """Compute the components of the potential energy for the
        :class:`System` with the :class:`SoftwareInterface`.

        :return: The individual contributions to the energy, in kJ/mol.
        """

    @abstractmethod
    def get_state_notifiers(
            self,
    ) -> dict[str, Callable[[NDArray[np.float64]], None]]:
        """Get the methods which should be called when a given
        :class:`StateVariable` is updated.

        :return: A dictionary of :class:`StateVariable` names and their
            respective notifier methods.
        """

    @abstractmethod
    def get_topology_notifiers(
            self,
    ) -> dict[str, Callable[..., None]]:
        """Get the methods which should be called when a given
        :class:`TopologyVariable` is updated.

        :return: A dictionary of :class:`TopologyVariable` names and
            their respective notifier methods.
        """
