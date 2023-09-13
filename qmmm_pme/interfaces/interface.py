#! /usr/bin/env python3
"""A module to define the :class:`SoftwareInterface` class.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np


class SoftwareTypes(Enum):
    QM = "A quantum chemistry software."
    MM = "A molecular dynamics software."


class SystemTypes(Enum):
    SYSTEM = "A System."
    SUBSYSTEM = "A Subsystem of a QM/MM System."
    EMBEDDING = "A Subsystem for Mechanical Embedding."


class SoftwareSettings(ABC):
    """
    """


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
