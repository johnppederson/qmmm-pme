#! /usr/bin/env python3
"""A module defining the :class:`Calculator` base class.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from qmmm_pme.common import Core
from qmmm_pme.records import Files


class Calculator(Core):
    """The base class for defining calculators.
    """

    def __init__(self) -> None:
        super().__init__()
        self.files = Files(self)

    @abstractmethod
    def calculate(
            self,
            return_forces: bool = True,
            return_components: bool = True,
    ) -> tuple[Any, ...]:
        """Calculate energies and forces for the :class:`System` with
        the :class:`Calculator`.

        :param return_forces: Determine whether or not to return forces.
        :param return_components: Determine whether or not to return
            the components of the energy.
        """

    @abstractmethod
    def update(self, attr: str, value: Any) -> None:
        """A method to update the :class:`Calculator`.

        :param attr: The attribute to update.
        :param value: The new value of the attribute.
        """
