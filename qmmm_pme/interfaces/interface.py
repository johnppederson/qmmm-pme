#! /usr/bin/env python3
"""A module to define the :class:`SoftwareInterface` class.
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
    QM = "A software for performing QM calculations."
    MM = "A software for performing MM calculations."


class SystemTypes(Enum):
    SYSTEM = "A System."
    SUBSYSTEM = "A Subsystem of a QM/MM System."
    EMBEDDING = "A Subsystem for Mechanical Embedding."


class SoftwareSettings(ABC):
    """
    """


@dataclass(frozen=True)
class MMSettings(SoftwareSettings):
    """A class which holds MM settings.
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
    """A class which holds the Psi4 settings.
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
    """
    """

    @abstractmethod
    def compute_energy(self) -> float:
        """
        """

    @abstractmethod
    def compute_forces(self) -> NDArray[np.float64]:
        """
        """

    @abstractmethod
    def compute_components(self) -> dict[str, float]:
        """Calculate the components of the energy.

        :return: The individual contributions to the energy.
        """

    @abstractmethod
    def get_state_notifiers(
            self,
    ) -> dict[str, Callable[[NDArray[np.float64]], None]]:
        """
        """

    @abstractmethod
    def get_topology_notifiers(
            self,
    ) -> dict[str, Callable[..., None]]:
        """
        """
