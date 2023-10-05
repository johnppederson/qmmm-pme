#! /usr/bin/env python3
"""A module to define the :class:`QMMMCalculator` class.
"""
from __future__ import annotations

from dataclasses import astuple
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from .calculator import CalculatorType
from .calculator import ModifiableCalculator
from .calculator import Results
from qmmm_pme.common import BOHR_PER_ANGSTROM
from qmmm_pme.common import compute_least_mirror
from qmmm_pme.common import KJMOL_PER_EH

if TYPE_CHECKING:
    from .calculators import StandaloneCalculator
    from qmmm_pme import System


@dataclass
class QMMMCalculator(ModifiableCalculator):
    """A :class:`Calculator` class for performing QM/MM calculations for
    an entire system.

    :param system: |system| to perform calculations on.
    :param calculators: The subsystem :class:`Calculators` to perform
        calculations with.
    :param embedding_cutoff: |embedding_cutoff|
    :param options: Options to provide to either of the
        :class:`SoftwareInterface` objects.
    """
    system: System
    calculators: dict[CalculatorType, StandaloneCalculator]
    embedding_cutoff: float | int
    options: dict[str, Any] = field(default_factory=dict)

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> tuple[Any, ...]:
        (
            correction_energy, correction_forces,
            charge_field,
        ) = self.compute_embedding()
        self.calculators[CalculatorType.QM].options.update(
            {"external_potentials": charge_field},
        )
        (
            mm_energy, mm_forces,
            mm_components,
        ) = self.calculators[CalculatorType.MM].calculate()
        (
            me_energy, me_forces,
            me_components,
        ) = self.calculators[CalculatorType.ME].calculate()
        (
            qm_energy, qm_forces,
            qm_components,
        ) = self.calculators[CalculatorType.QM].calculate()
        qmmm_energy = mm_energy + me_energy + qm_energy + correction_energy
        results = Results(qmmm_energy)
        if return_forces:
            qmmm_forces = mm_forces + me_forces + qm_forces + correction_forces
            results.forces = qmmm_forces
        if return_components:
            qmmm_components = {
                "MM Energy": mm_energy,
                ".": mm_components,
                "QM Energy": qm_energy,
                "..": qm_components,
                "Mechanical Embedding Energy": me_energy,
                "Electrostatic Embedding Energy": 0.,
                "Real-Space Correction Energy": correction_energy,
            }
            results.components = qmmm_components
        return astuple(results)

    def compute_embedding(self) -> tuple[Any, ...]:
        """Create the embedding list for the current :class:`System`,
        as well as the corrective Coulomb potential and forces.

        The distances from the QM atoms are computed using the centroid
        of the non-QM molecule from the centroid of the QM atoms.

        :return: The corrective Coulomb energy and forces for the
            embedded point charges, and the charge field for the QM
            calculation.
        """
        # Collect QM atom information.
        qm_atoms = [
            atom for residue in self.system.topology.qm_atoms()
            for atom in residue
        ]
        qm_centroid = np.average(
            self.system.state.positions()[qm_atoms, :],
            axis=0,
        )
        # Initialize the relevent containers for the data to be
        # calculated.
        ae_atoms = []
        charge_field = []
        correction_energy = 0
        correction_forces = np.zeros_like(self.system.state.positions())
        # Loop over all residues in the system.
        for residue in self.system.topology.atoms():
            # Get distance between the residue and QM atom centroids
            nth_centroid = np.average(
                self.system.state.positions()[residue, :],
                axis=0,
            )
            r_vector = compute_least_mirror(
                nth_centroid,
                qm_centroid,
                self.system.state.box(),
            )
            distance = np.sum(r_vector**2)**0.5
            is_qm = any(atom in residue for atom in qm_atoms)
            if distance < self.embedding_cutoff and not is_qm:
                ae_atoms.append(residue)
                displacement = r_vector + qm_centroid - nth_centroid
                # Loop through each atom in the residue to add to the
                # charge field that will be sent to the QM calculation.
                for atom in residue:
                    ae_position = (
                        self.system.state.positions()[atom]
                        + displacement
                    ) * BOHR_PER_ANGSTROM
                    ae_charge = self.system.state.charges()[atom]
                    charge_field.append(
                        (
                            ae_charge,
                            (ae_position[0], ae_position[1], ae_position[2]),
                        ),
                    )
                    # Loop through each QM atom to add onto real-space
                    # correction energy and forces.
                    correction_force = np.zeros((3,))
                    for qm_atom in qm_atoms:
                        qm_position = (
                            self.system.state.positions()[qm_atom, :]
                            * BOHR_PER_ANGSTROM
                        )
                        qm_charge = self.system.state.charges()[qm_atom]
                        r_atom = ae_position - qm_position
                        dr = np.sum(r_atom**2)**0.5
                        q_prod = qm_charge * ae_charge
                        correction_energy -= KJMOL_PER_EH * q_prod * dr**-1
                        correction_force += (
                            r_atom * q_prod * dr**-3
                        ) * BOHR_PER_ANGSTROM * KJMOL_PER_EH
                    correction_forces[atom, :] -= correction_force
        # Update the topology with the current embedding atoms.
        self.system.topology.ae_atoms.update(ae_atoms)
        return correction_energy, correction_forces, tuple(charge_field)
