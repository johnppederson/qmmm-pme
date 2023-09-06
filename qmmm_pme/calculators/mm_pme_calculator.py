#! /usr/bin/env python3
"""A module defining the :class:`MMPMECalculator` class.
"""
from __future__ import annotations

from typing import Any

from .mm_qmmm_calculator import MMQMMMCalculator


class MMPMECalculator(MMQMMMCalculator):
    """A :class:`Calculator` class for performing molecular mechanics
    calculations for the MM subsystem of a QM/MM/PME system using
    OpenMM.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.openmm.properties.update({"ReferenceVextGrid": "true"})
        self._generate_system()
        self._adjust_forces()
        self._generate_context()

    def _generate_state(self, **kwargs) -> None:
        super()._generate_state(getVext_grids=True, **kwargs)

    def compute_pme_potential(self) -> Any:
        """Creates the PME potential grid.

        :return: The PME potential grid calculated by OpenMM.

        .. warning:: Requires investigation of return type.
        """
        return self.openmm.state.getVext_grid()
