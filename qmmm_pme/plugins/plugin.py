#! /usr/bin/env python3
"""A module defining the :class:`Plugin` base class.
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
    """The base class for creating QM/MM/PME plugins.
    """
    _modifieds: list[str] = []
    _key: str = ""


class CalculatorPlugin(Plugin):
    """
    """
    _key: str = "Calculator"

    @abstractmethod
    def modify(self, calculator: ModifiableCalculator) -> None:
        """
        """


class QMMMCalculatorPlugin(Plugin):
    """
    """
    _key: str = "Calculator"

    @abstractmethod
    def modify(self, calculator: QMMMCalculator) -> None:
        """
        """


class IntegratorPlugin(Plugin):
    """
    """
    _key: str = "Integrator"

    @abstractmethod
    def modify(self, integrator: ModifiableIntegrator) -> None:
        """
        """
