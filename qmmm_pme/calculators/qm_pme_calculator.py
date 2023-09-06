#! /usr/bin/env python3
"""A module to define the :class:`QMPMECalculator` class.
"""
from __future__ import annotations

import numpy as np
import psi4.core
from numpy.typing import NDArray

from .qm_qmmm_calculator import QMQMMMCalculator


class QMPMECalculator(QMQMMMCalculator):
    """A :class:`Calculator` class for performing molecular mechanics
    calculations for the QM subsystem of a QM/MM/PME system using Psi4.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.options["pme"] = "true"
        psi4.set_options(self.options)

    def compute_quadrature(self) -> NDArray[np.float64]:
        """Build a reference quadrature to interpolate into the PME
        potential grid.

        :return: A reference quadrature constructed from the geometry of
            the QM subsystem.
        """
        self._generate_geometry()
        sup_func = psi4.driver.dft.build_superfunctional(
            self.functional,
            True,
        )[0]
        basis = psi4.core.BasisSet.build(
            self.psi4.molecule,
            "ORBITAL",
            self.basis_set,
        )
        vbase = psi4.core.VBase.build(basis, sup_func, "RV")
        vbase.initialize()
        quadrature = vbase.get_np_xyzw()
        return quadrature
