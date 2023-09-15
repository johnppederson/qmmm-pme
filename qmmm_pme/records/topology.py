#! /usr/bin/env python3
"""A module defining the :class:`Topology` record container.
"""
from __future__ import annotations

from dataclasses import dataclass

from .record import Record
from .record import Variable


class NameVariable(Variable):
    """A class wrapping a variable belonging to the :class:`Topology`.
    """
    _value: list[str] = []

    def update(self, value: list[str]) -> None:
        """Update the value of the :class:`TopologyVariable`.

        :param value: The updated value to set the
            :class:`TopologyVariable` to.
        """
        self._value = value
        for notify in self._notifiers:
            notify(value)

    def __call__(self) -> list[str]:
        """Return the value of the :class:`TopologyVariable`.

        :return: The value of the :class:`TopologyVariable`.
        """
        return self._value


class ResidueVariable(Variable):
    """A class wrapping a variable belonging to the :class:`Topology`.
    """
    _value: list[list[int]] = []

    def update(self, value: list[list[int]]) -> None:
        """Update the value of the :class:`TopologyVariable`.

        :param value: The updated value to set the
            :class:`TopologyVariable` to.
        """
        self._value = value
        for notify in self._notifiers:
            notify(value)

    def __call__(self) -> list[list[int]]:
        """Return the value of the :class:`TopologyVariable`.

        :return: The value of the :class:`TopologyVariable`.
        """
        return self._value


@dataclass(frozen=True)
class Topology(Record):
    """A :class:`Record` class for maintaining data about the topology
    of the :class:`System`, comprising atom groups, residue names,
    element symbols, and atom names.
    """
    atoms: ResidueVariable = ResidueVariable()
    qm_atoms: ResidueVariable = ResidueVariable()
    mm_atoms: ResidueVariable = ResidueVariable()
    ae_atoms: ResidueVariable = ResidueVariable()
    elements: NameVariable = NameVariable()
    residue_names: NameVariable = NameVariable()
    atom_names: NameVariable = NameVariable()
