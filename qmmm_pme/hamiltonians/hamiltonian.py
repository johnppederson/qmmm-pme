#! /usr/bin/env python3
"""A module defining the base :class:`Hamiltonian` class and derived
interface classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import TYPE_CHECKING

from qmmm_pme.interfaces import SystemTypes

if TYPE_CHECKING:
    from qmmm_pme import System
    from qmmm_pme.calculators.calculator import ModifiableCalculator


class Hamiltonian(ABC):
    """An abstract :class:`Hamiltonian` base class for creating the
    Hamiltonian API.
    """
    atoms: list[int | slice] = []
    system_type: SystemTypes = SystemTypes.SYSTEM

    def __getitem__(
            self,
            indices: int | slice | tuple[int | slice, ...],
    ) -> Hamiltonian:
        """Sets the indices for atoms that are treated with this
        :class:`Hamiltonian`.

        :return: |Hamiltonian|.
        """
        indices = indices if isinstance(indices, tuple) else (indices,)
        atoms = []
        for i in indices:
            if isinstance(i, (int, slice)):
                atoms.append(i)
            else:
                raise TypeError("...")
        self.atoms = atoms
        return self

    @abstractmethod
    def build_calculator(self, system: System) -> ModifiableCalculator:
        """Build the :class:`Calculator` corresponding to the
        :class:`Hamiltonian` object.

        :param system: |system| to calculate energy and forces for.
        :return: |calculator|.
        """

    def parse_atoms(self, system: System) -> list[list[int]]:
        """Parse the indices provided to the :class:`Hamiltonian` object
        to create the list of residue-grouped atom indices.

        :param system: |system| to calculate energy and forces for.
        :return: |atoms|
        """
        indices = []
        for i in self.atoms:
            if isinstance(i, int):
                indices.append(i)
            else:
                indices.extend(
                    list(
                        range(
                            i.start if i.start else 0,
                            i.stop if i.stop else len(system),
                            i.step if i.step else 1,
                        ),
                    ),
                )
        if not self.atoms:
            indices = [i for i in range(len(system))]
        residues = [
            x for residue in system.topology.atoms()
            if (x := [i for i in residue if i in indices])
        ]
        return residues

    @abstractmethod
    def __add__(self, other: Any) -> Hamiltonian:
        """Add :class:`Hamiltonian` objects together.

        :param other: The object being added to the
            :class:`Hamiltonian`.
        :return: A new :class:`Hamiltonian` object.
        """
        pass

    def __or__(self, other: Any) -> Hamiltonian:
        """Set the embedding distance for a :class:`QMMMHamiltonian`.

        :param other: The embedding distance, in Angstroms.
        :return: |hamiltonian|.
        """
        return self

    def __radd__(self, other: Any) -> Any:
        """Add :class:`Hamiltonian` objects together.

        :param other: The object being added to the
            :class:`Hamiltonian`.
        :return: A new :class:`Hamiltonian` object.
        """
        return self.__add__(other)

    def __str__(self) -> str:
        """Create a LATEX string representation of the
        :class:`Hamiltonian` object.

        :return: The string representation of the :class:`Hamiltonian`
            object.
        """
        string = "_{"
        for atom in self.atoms:
            string += f"{atom}, "
        string += "}"
        return string


class MMHamiltonianInterface(Hamiltonian):
    """An interface for the :class:`MMHamiltonian`.
    """


class QMHamiltonianInterface(Hamiltonian):
    """An interface for the :class:`QMHamiltonian`.
    """


class QMMMHamiltonianInterface(Hamiltonian):
    """An interface for the :class:`QMMMHamiltonian`.
    """
