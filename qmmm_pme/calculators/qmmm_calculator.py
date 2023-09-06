#! /usr/bin/env python3
"""A module to define the :class:`QMMMCalculator` class.
"""
from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

import numpy as np

from .calculator import Calculator
from qmmm_pme.common import BOHR_PER_ANGSTROM
from qmmm_pme.common import compute_least_mirror
from qmmm_pme.common import KJMOL_PER_EH

if TYPE_CHECKING:
    from .qm_qmmm_calculator import QMQMMMCalculator
    from .mm_qmmm_calculator import MMQMMMCalculator


class QMMMCalculator(Calculator):
    """A :class:`Calculator` class for performing QM/MM calculations for
    an entire system.

    :param mm_calculator: A :class:`Calculator` for the MM Subsystem.
    :param qm_calculator: A :class:`Calculator` for the QM Subsystem.
    :param embedding_cutoff: The QM/MM analytic embedding cutoff, in
        Angstroms.
    """

    def __init__(
            self,
            qm_calculator: QMQMMMCalculator,
            mm_calculator: MMQMMMCalculator,
            embedding_cutoff: float | int,
    ):
        super().__init__()
        self.qm_calculator = qm_calculator
        self.mm_calculator = mm_calculator
        self.embedding_cutoff = embedding_cutoff

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
            **kwargs,
    ) -> tuple[Any, ...]:
        self._generate_embedding()
        (
            mm_energy, mm_forces, mm_components,
        ) = self.mm_calculator.calculate()
        (
            qm_energy, qm_forces, qm_components,
        ) = self.qm_calculator.calculate(**kwargs)
        (
            correction_energy, correction_forces,
        ) = self.compute_correction_energy()
        qmmm_energy = mm_energy + qm_energy + correction_energy
        qmmm_results = [qmmm_energy]
        if return_forces:
            qmmm_forces = mm_forces + qm_forces + correction_forces
            qmmm_results.append(qmmm_forces)
        if return_components:
            qmmm_components = {
                "OpenMM Energy": mm_energy,
                ".": mm_components,
                "Psi4 Energy": qm_energy,
                "..": qm_components,
                "Real-Space Correction Energy": correction_energy,
            }
            qmmm_results.append(qmmm_components)
        return tuple(qmmm_results)

    def _generate_embedding(self) -> None:
        """Create the embedding list for the current :class:`System`.

        The distances from the QM atoms are computed using the centroid
        of the non-QM molecule from the centroid of the QM atoms.  The
        legacy method involves computing distances using the first atom
        position from the non-QM molecule instead.
        """
        qm_atoms = self._topology.groups["qm_atom"]
        qm_centroid = np.average(
            self._state.positions[qm_atoms, :],
            axis=0,
        )
        embedding_list = []
        self.displacements = []
        for residue in self._topology.groups["all"]:
            nth_centroid = np.average(
                self._state.positions[residue, :],
                axis=0,
            )
            r_vector = compute_least_mirror(
                nth_centroid,
                qm_centroid,
                self._state.box,
            )
            distance = np.sum(r_vector**2)**0.5
            is_qm = any(atom in residue for atom in qm_atoms)
            if distance < self.embedding_cutoff and not is_qm:
                embedding_list.append(residue)
                self.displacements.append(
                    r_vector + qm_centroid - nth_centroid,
                )
        self._topology.groups["analytic"] = embedding_list

    def compute_correction_energy(self, return_forces=True):
        """Calculate the correction energy for analytic embedding atoms.

        :param return_forces: Determine whether or not to calculate
            forces, defaults to True.
        :param return_components: Determine whether or not to calculate
            energy components, defaults to True.
        :return: The correction energy or the energy and forces, defaults
            to True.
        """
        correction_forces = np.zeros_like(self._state.positions)
        correction_energy = 0
        for residue, displacement in zip(self._topology.groups["analytic"], self.displacements):
            for atom in residue:
                correction_force = np.zeros((3,))
                for qm_atom in self._topology.groups["qm_atom"]:
                    r_atom = (
                        displacement
                        + self._state.positions[atom, :]
                        - self._state.positions[qm_atom, :]
                    ) * BOHR_PER_ANGSTROM
                    dr = np.sum(r_atom**2)**0.5
                    q_prod = (
                        self._state.charges[qm_atom]
                        * self._state.charges[atom]
                    )
                    correction_force += (
                        r_atom * q_prod * dr**-3
                    ) * BOHR_PER_ANGSTROM * KJMOL_PER_EH
                    correction_energy -= KJMOL_PER_EH * q_prod * dr**-1
                correction_forces[atom, :] -= correction_force
        correction_results = [correction_energy]
        if return_forces:
            correction_results.append(correction_forces)
        return correction_results

    def update(self, attr, value):
        """Update the settings for OpenMM.

        :param settings: The updated settings.
        :type settings: dict
        """
        pass
