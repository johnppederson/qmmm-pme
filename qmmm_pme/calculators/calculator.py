#! /usr/bin/env python3
"""A module defining the :class:`Calculator` base class and derived
non-multiscale classes.
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
    """Enumeration of types of non-multiscale calculators.
    """
    QM = "A QM Calculator."
    MM = "An MM Calculator."
    ME = "An ME Calculator."


@dataclass
class Results:
    """A wrapper class for storing the results of a calculation.

    :param energy: The energy calculated for the system.
    :param forces: The forces calculated for the system.
    :components: The components of the energy calculated for the system.
    """
    energy: float = 0
    forces: NDArray[np.float64] = np.empty(0)
    components: dict[str, float] = field(default_factory=dict)


class ModifiableCalculator(ABC):
    """An abstract :class:`Calculator` base class for interfacing with
    plugins.
    """
    system: System
    _plugins: list[str] = []

    @abstractmethod
    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> tuple[Any, ...]:
        """Calculate energies and forces for the :class:`System` with
        the :class:`Calculator`.

        :param return_forces: Whether or not to return forces.
        :param return_components: Whether or not to return
            the components of the energy.
        :return: The energy, forces, and energy components of the
            calculation.
        """

    def register_plugin(self, plugin: CalculatorPlugin) -> None:
        """Register a :class:`Plugin` modifying a :class:`Calculator`
        routine.

        :param plugin: An :class:`CalculatorPlugin` object.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Get the current list of active plugins.

        :return: A list of the active plugins being employed by the
            :class:`Calculator`.
        """
        return self._plugins


@dataclass
class StandaloneCalculator(ModifiableCalculator):
    """A :class:`Calculator` class, defining the procedure for
    standalone QM or MM calculations.

    :param system: |system| to perform calculations on.
    :param interface: |interface| to perform calculations with.
    :param options: Options to provide to the
        :class:`SoftwareInterface`.
    """
    system: System
    interface: SoftwareInterface
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Send notifier functions from the interface to the respective
        state or topology variable for monitoring, immediately after
        initialization.
        """
        state_generator = self.interface.get_state_notifiers().items()
        for state_key, state_value in state_generator:
            getattr(self.system.state, state_key).register_notifier(
                state_value,
            )
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
