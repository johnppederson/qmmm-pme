#! /usr/bin/env python3
""" Files data container.
"""
from __future__ import annotations

from dataclasses import dataclass

from .record import Record
from .record import Variable


class FilesVariable(Variable):
    """A class wrapping a variable belonging to the :class:`Files`.
    """
    __slots__ = Variable.__slots__
    _value: list[str] = []

    def update(self, value: list[str]) -> None:
        """Update the value of the :class:`FilesVariable`.

        :param value: The updated value to set the
            :class:`FilesVariable` to.
        """
        self._value = value
        for notify in self._notifiers:
            notify(value)

    def __call__(self) -> list[str]:
        """Return the value of the :class:`FilesVariable`.

        :return: The value of the :class:`FilesVariable`.
        """
        return self._value


@dataclass(frozen=True)
class Files(Record):
    """Class for maintaining records about input and output files.
    """
    pdb_list: FilesVariable = FilesVariable()
    topology_list: FilesVariable = FilesVariable()
    forcefield_list: FilesVariable = FilesVariable()
    input_json: FilesVariable = FilesVariable()
