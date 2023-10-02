#! /usr/bin/env python3
"""A module defining the :class:`Topology` data container.
"""
from __future__ import annotations

from dataclasses import dataclass

from .record import Record
from .record import Variable


class NameVariable(Variable):
    """A wrapper class for name variables belonging to the
    :class:`Topology` record.
    """
    _value: list[str] = []

    def update(self, value: list[str]) -> None:
        """Update the value of the :class:`NameVariable`.

        :param value: The updated value to set the
            :class:`NameVariable` value to.
        """
        self._value = value
        for notify in self._notifiers:
            notify(value)

    def __call__(self) -> list[str]:
        """Get the value of the :class:`NameVariable`.

        :return: The value of the :class:`NameVariable`.
        """
        return self._value


class ResidueVariable(Variable):
    """A wrapper class for residue group variables belonging to the
    :class:`Topology` record.
    """
    _value: list[list[int]] = []

    def update(self, value: list[list[int]]) -> None:
        """Update the value of the :class:`ResidueVariable`.

        :param value: The updated value to set the
            :class:`ResidueVariable` value to.
        """
        self._value = value
        for notify in self._notifiers:
            notify(value)

    def __call__(self) -> list[list[int]]:
        """Get the value of the :class:`ResidueVariable`.

        :return: The value of the :class:`ResidueVariable`.
        """
        return self._value


@dataclass(frozen=True)
class Topology(Record):
    """A data container for information about the topology of the
    :class:`System`, comprising atom groups, residue names, element
    symbols, and atom names.
    """
    atoms: ResidueVariable = ResidueVariable()
    qm_atoms: ResidueVariable = ResidueVariable()
    mm_atoms: ResidueVariable = ResidueVariable()
    ae_atoms: ResidueVariable = ResidueVariable()
    elements: NameVariable = NameVariable()
    residue_names: NameVariable = NameVariable()
    atom_names: NameVariable = NameVariable()
