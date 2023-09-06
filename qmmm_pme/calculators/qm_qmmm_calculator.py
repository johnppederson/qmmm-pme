#! /usr/bin/env python3
"""A module to define the :class:`QMQMMMCalculator` class.
"""
from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np

from .qm_calculator import QMCalculator
from qmmm_pme.common import BOHR_PER_ANGSTROM
from qmmm_pme.common import compute_least_mirror


class QMQMMMCalculator(QMCalculator):
    """A :class:`Calculator` class for performing molecular mechanics
    calculations for the QM subsystem of a QM/MM system using Psi4.
    """
    charge_field = None

    def _generate_charge_field(self) -> None:
        """Create the charge field for analytic embedding in the Psi4
        calculation.
        """
        qm_atoms = self._topology.groups["qm_atom"]
        qm_centroid = np.average(
            self._state.positions[qm_atoms, :],
            axis=0,
        )
        embedding_list = self._topology.groups["analytic"]
        charge_field = []
        for residue in embedding_list:
            nth_centroid = np.average(
                self._state.positions[residue, :],
                axis=0,
            )
            new_centroid = compute_least_mirror(
                nth_centroid,
                qm_centroid,
                self._state.box,
            )
            displacement = new_centroid - nth_centroid
            for atom in residue:
                position = (
                    self._state.positions[atom]
                    + displacement
                    + qm_centroid
                ) * BOHR_PER_ANGSTROM
                charge_field.append(
                    [
                        self._state.charges[atom],
                        [position[0], position[1], position[2]],
                    ],
                )
        self.charge_field = charge_field

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
            **kwargs,
    ) -> tuple[Any, ...]:
        self._generate_charge_field()
        (psi4_energy, psi4_temp, psi4_components) = super().calculate(
            external_potentials=self.charge_field,
            **kwargs,
        )
        psi4_results = [psi4_energy]
        if return_forces:
            qm_atoms = self._topology.groups["qm_atom"]
            ae_atoms = [
                atom for residue in self._topology.groups["analytic"]
                for atom in residue
            ]
            psi4_forces = np.zeros_like(self._state.positions)
            psi4_forces[qm_atoms, :] = psi4_temp[0:len(qm_atoms), :]
            psi4_forces[ae_atoms, :] = psi4_temp[len(qm_atoms):, :]
            psi4_results.append(psi4_forces)
        if return_components:
            psi4_results.append(psi4_components)
        return tuple(psi4_results)
