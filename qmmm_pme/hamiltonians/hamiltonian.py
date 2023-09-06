#! /usr/bin/env python3
"""Base class for constructing a Hamiltonian.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class Hamiltonian(ABC):
    """The Hamiltonian base class.
    """

    def __init__(self, system) -> None:
        self.system = system
        self._atoms = list(range(len(self.system)))

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        atoms = []
        for i in indices:
            if isinstance(i, int):
                atoms.append(i)
            elif isinstance(i, slice):
                atoms.extend(
                    list(
                        range(
                            i.start if i.start else 0,
                            i.stop if i.stop else len(self.system),
                            i.step if i.step else 1,
                        ),
                    ),
                )
            else:
                raise TypeError
        self.atoms = atoms
        return self

    @abstractmethod
    def __add__(self, other):
        pass

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        current = self.atoms[0]
        string = "_{" + f"{current}:"
        for atom in self.atoms:
            if atom - current > 1:
                string += f"{current+1}, {atom}:"
            current = atom
        string += f"{current}" + "}"
        return string

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        for atom in atoms:
            if atom >= len(self.system):
                raise IndexError
            if atom < 0:
                raise IndexError
        self._atoms = atoms
        self._atoms.sort()


class QMMMHamiltonianBase(Hamiltonian):
    """A base class for QM/MM Hamiltonian functionality
    """

    def __or__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError
        self.embedding_cutoff = other
        return self


class MMHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """


class QMHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """


class PBCHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """


class QMMMHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """


class QMMMPMEHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """
