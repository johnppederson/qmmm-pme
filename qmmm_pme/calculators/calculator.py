#! /usr/bin/env python3
"""A module defining the :class:`Calculator` base class.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import astuple
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qmmm_pme.interfaces.interface import SoftwareInterface
    from qmmm_pme.plugins.plugin import CalculatorPlugin
    from qmmm_pme import System
    from numpy.typing import NDArray


class CalculatorType(Enum):
    QM = "A QM Calculator."
    MM = "An MM Calculator."
    ME = "An ME Calculator."


@dataclass
class Results:
    """
    """
    energy: float = 0
    forces: NDArray[np.float64] = np.empty(0)
    components: dict[str, float] = field(default_factory=dict)


class ModifiableCalculator(ABC):
    """The base class for defining calculators.
    """
    _plugins: list[str] = []

    @abstractmethod
    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> tuple[Any, ...]:
        """Calculate energies and forces for the :class:`System` with
        the :class:`Calculator`.

        :param return_forces: Determine whether or not to return forces.
        :param return_components: Determine whether or not to return
            the components of the energy.
        """

    def register_plugin(self, plugin: CalculatorPlugin) -> None:
        """Register any plugins applied to an instance of a class
        inheriting from :class:`Core`.

        :param plugin: An instance of a class inheriting from
            :class:`Plugin`.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Return the list of active plugins.

        :return: A list of the active plugins being employed by the
            class.
        """
        return self._plugins


@dataclass
class StandaloneCalculator(ModifiableCalculator):
    """The base class for defining standalone calculators.
    """
    system: System
    interface: SoftwareInterface
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state_generator = self.interface.get_state_notifiers().items()
        for state_key, state_value in state_generator:
            getattr(self.system.state, state_key).register_notifier(state_value)
        topology_generator = self.interface.get_topology_notifiers().items()
        for topology_key, topology_value in topology_generator:
            getattr(self.system.topology, topology_key).register_notifier(
                topology_value,
            )

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> tuple[Any, ...]:
        energy = self.interface.compute_energy(**self.options)
        results = Results(energy)
        if return_forces:
            forces = self.interface.compute_forces(**self.options)
            results.forces = forces
        if return_components:
            components = self.interface.compute_components(**self.options)
            results.components = components
        return astuple(results)
