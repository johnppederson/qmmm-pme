#! /usr/bin/env python3
"""A module defining the pluggable implementation of the atom-wise
embedding algorithm for the |package| repository.
"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from qmmm_pme.plugins.plugin import QMMMCalculatorPlugin

if TYPE_CHECKING:
    from qmmm_pme.calculators import QMMMCalculator


class AtomEmbedding(QMMMCalculatorPlugin):
    """A :class:`Plugin` which implements atom-by-atom embedding for
    a given set of molecules in a :class:`System`.

    :param atom_embedding_residues: The names of residues on which to
        apply atom-by-atom electrostatic embedding.
    """

    def __init__(
            self,
            atom_embedding_residues: list[str],
    ) -> None:
        self.atom_embedding_residues = atom_embedding_residues

    def modify(
            self,
            calculator: QMMMCalculator,
    ) -> None:
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        self.residues = [
            res for i, res
            in enumerate(self.system.topology.mm_atoms())
            if self.system.topology.residue_names()[i]
            in self.atom_embedding_residues
        ]
        atoms = self.system.topology.atoms()
        self.atoms = [[atom] for residue in self.residues for atom in residue]
        self.atoms.extend(
            [residue for residue in atoms if residue not in self.residues],
        )
        calculator.compute_embedding = self._modify_compute_embedding(
            calculator.compute_embedding,
        )

    def _modify_compute_embedding(
            self,
            compute_embedding: Callable[[], tuple[Any, ...]],
    ) -> Callable[[], tuple[Any, ...]]:
        """Modfify the compute_embedding call in the
        :class:`QMMMCalculator` to consider a subset of atoms to be
        complete residues.

        :param compute_embedding: The default compute_embedding method
            of the :class:`QMMMCalculator`.
        :return: The modified compute_embedding method.
        """
        def inner() -> tuple[Any, ...]:
            temp = self.system.topology.atoms()
            self.system.topology.atoms.update(self.atoms)
            (
                correction_energy, correction_forces,
                charge_field,
            ) = compute_embedding()
            self.system.topology.atoms.update(temp)
            return correction_energy, correction_forces, charge_field
        return inner
