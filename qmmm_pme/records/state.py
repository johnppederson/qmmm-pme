#! /usr/bin/env python3
"""A module defining the :class:`State` data container.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .record import Record
from .record import Variable


class StateVariable(Variable):
    """A wrapper class for variables belonging to the :class:`State`
    record.
    """
    _value: NDArray[np.float64] = np.empty(0)

    def update(self, value: NDArray[np.float64]) -> None:
        """Update the value of the :class:`StateVariable`.

        :param value: The updated value to set the
            :class:`StateVariable` value to.
        """
        self._value = value
        for notify in self._notifiers:
            notify(value)

    def __call__(self) -> NDArray[np.float64]:
        """Get the value of the :class:`StateVariable`.

        :return: The value of the :class:`StateVariable`.
        """
        return self._value


@dataclass(frozen=True)
class State(Record):
    """A data container for state information about the the
    :class:`System`, comprising atomic masses, charges, positions,
    velocities, momenta, and forces; and the box vectors defining the
    periodic boundary condition.
    """
    masses: StateVariable = StateVariable()
    charges: StateVariable = StateVariable()
    positions: StateVariable = StateVariable()
    velocities: StateVariable = StateVariable()
    momenta: StateVariable = StateVariable()
    forces: StateVariable = StateVariable()
    box: StateVariable = StateVariable()
