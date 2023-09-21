#! /usr/bin/env python3
"""A module defining the :class:`Plugin` base class and derived classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qmmm_pme.calculators.calculator import ModifiableCalculator
    from qmmm_pme.calculators import QMMMCalculator
    from qmmm_pme.integrators.integrator import ModifiableIntegrator


class Plugin(ABC):
    """The base class for creating |package| plugins.
    """
    _modifieds: list[str] = []
    _key: str = ""


class CalculatorPlugin(Plugin):
    """The base class for creating a :class:`Plugin` which modifies any
    :class:`Calculator` class.
    """
    _key: str = "calculator"

    @abstractmethod
    def modify(self, calculator: ModifiableCalculator) -> None:
        """A placeholder method for modifying the functionality of a
        :class:`Calculator`.
        """


class QMMMCalculatorPlugin(Plugin):
    """The base class for creating a :class:`Plugin` which modifies a
    :class:`QMMMCalculator` class.
    """
    _key: str = "calculator"

    @abstractmethod
    def modify(self, calculator: QMMMCalculator) -> None:
        """A placeholder method for modifying the functionality of a
        :class:`QMMMCalculator`.
        """


class IntegratorPlugin(Plugin):
    """The base class for creating a :class:`Plugin` which modifies any
    :class:`Integrator` class.
    """
    _key: str = "integrator"

    @abstractmethod
    def modify(self, integrator: ModifiableIntegrator) -> None:
        """A placeholder method for modifying the functionality of an
        :class:`Integrator`.
        """
