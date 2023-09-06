#! /usr/bin/env python3
"""A module defining the :class:`QMMMPMECalculator` class.
"""
from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from .qmmm_calculator import QMMMCalculator

if TYPE_CHECKING:
    from .mm_pme_calculator import MMPMECalculator
    from .pbc_calculator import PBCCalculator
    from .qm_pme_calculator import QMPMECalculator


class QMMMPMECalculator(QMMMCalculator):
    """A :class:`Calculator` class for performing QM/MM/PME calculations
    for an entire system.

    :param mm_calculator: A :class:`Calculator` for the MM Subsystem.
    :param qm_calculator: A :class:`Calculator` for the QM Subsystem.
    :param pbc_calculator: A :class:`Calculator` for the PBC Subsystem.
    :param embedding_cutoff: The QM/MM analytic embedding cutoff, in
        Angstroms.
    """

    def __init__(
            self,
            qm_calculator: QMPMECalculator,
            mm_calculator: MMPMECalculator,
            pbc_calculator: PBCCalculator,
            embedding_cutoff: float | int,
    ) -> None:
        super().__init__(
            qm_calculator,
            mm_calculator,
            embedding_cutoff,
        )
        self.pbc_calculator = pbc_calculator

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
            **kwargs,
    ) -> tuple[Any, ...]:
        self.mm_calculator._generate_state()
        pme_potential = self.mm_calculator.compute_pme_potential()
        quadrature = self.qm_calculator.compute_quadrature()
        self.pbc_calculator.update("pme_potential", pme_potential)
        self.pbc_calculator.update("quadrature", quadrature)
        self._generate_embedding()
        (
            reciprocal_energy,
            quadrature_pme_potential,
            nuclei_pme_potential,
            nuclei_pme_gradient,
        ) = self.pbc_calculator.calculate()
        qmmm_energy, qmmm_forces, qmmm_components = super().calculate(
            quad_extd_pot=quadrature_pme_potential,
            nuc_extd_pot=nuclei_pme_potential,
            nuc_extd_grad=nuclei_pme_gradient,
            **kwargs,
        )
        qmmm_energy += reciprocal_energy
        qmmm_results = [qmmm_energy]
        if return_forces:
            qmmm_results.append(qmmm_forces)
        if return_components:
            qmmm_components["Potential Energy"] = qmmm_energy
            qmmm_components.update(
                {"Reciprocal-Space Correction Energy": reciprocal_energy},
            )
            qmmm_results.append(qmmm_components)
        return qmmm_results
