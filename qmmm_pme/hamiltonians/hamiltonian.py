#! /usr/bin/env python3
"""Base class for constructing a Hamiltonian.
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
    """The Hamiltonian base class.
    """
    atoms: list[int | slice] = []
    system_type: SystemTypes = SystemTypes.SYSTEM

    def __getitem__(
            self,
            indices: int | slice | tuple[int | slice, ...],
    ) -> Hamiltonian:
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
        """
        """

    def parse_atoms(self, system: System) -> list[list[int]]:
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
        pass

    def __or__(self, other: Any) -> Hamiltonian:
        return self

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __str__(self) -> str:
        string = "_{"
        for atom in self.atoms:
            string += f"{atom}, "
        string += "}"
        return string


class MMHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """


class QMHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """


class QMMMHamiltonianInterface(Hamiltonian):
    """An interface for the MM Hamiltonian
    """
