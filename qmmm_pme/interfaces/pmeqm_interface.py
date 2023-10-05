#! /usr/bin/env python3
"""A module to define the :class:`Psi4Interface` class.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import psi4.core
from numpy.typing import NDArray

from .interface import QMSettings
from .interface import SoftwareTypes
from .interface import SystemTypes
from .psi4_interface import _build_context
from .psi4_interface import Psi4Context
from .psi4_interface import Psi4Interface
from .psi4_interface import Psi4Options
from qmmm_pme.common.units import BOHR_PER_ANGSTROM
from qmmm_pme.common.units import KJMOL_PER_EH

if TYPE_CHECKING:
    ComputationOptions = float


SOFTWARE_TYPE = SoftwareTypes.QM


psi4.core.be_quiet()


@dataclass(frozen=True)
class PMEPsi4Options(Psi4Options):
    """An immutable wrapper class for storing Psi4 global options.

    :param pme: Whether or not to perform a Psi4 calculation with the
        PME potential interpolated into the DFT functional quadrature.
    """
    pme: str = "true"


@dataclass(frozen=True)
class PMEPsi4Interface(Psi4Interface):
    options: Psi4Options
    functional: str
    context: Psi4Context

    @lru_cache
    def _generate_wavefunction(
            self,
            **kwargs: ComputationOptions,
    ) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object for use in Psi4
        calculations.

        :return: The Psi4 Wavefunction object.
        """
        molecule = self.context.generate_molecule()
        psi4.set_options(asdict(self.options))
        if (x := "quad_extd_pot") in kwargs:
            kwargs[x] = np.array(kwargs[x])
        if (x := "nuc_extd_pot") in kwargs:
            kwargs[x] = np.array(kwargs[x])
        if (x := "nuc_extd_grad") in kwargs:
            kwargs[x] = np.array(kwargs[x])
        _, wfn = psi4.energy(
            self.functional,
            return_wfn=True,
            molecule=molecule,
            **kwargs,
        )
        wfn.to_file(
            wfn.get_scratch_filename(180),
        )
        return wfn

    def compute_forces(
            self,
            **kwargs: ComputationOptions,
    ) -> NDArray[np.float64]:
        wfn = self._generate_wavefunction(**kwargs)
        psi4.set_options(asdict(self.options))
        if (x := "quad_extd_pot") in kwargs:
            kwargs[x] = np.array(kwargs[x])
        if (x := "nuc_extd_pot") in kwargs:
            kwargs[x] = np.array(kwargs[x])
        if (x := "nuc_extd_grad") in kwargs:
            kwargs[x] = np.array(kwargs[x])
        forces = psi4.gradient(
            self.functional,
            ref_wfn=wfn,
            **kwargs,
        )
        forces = forces.to_array() * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
        forces_temp = np.zeros_like(self.context.positions)
        qm_indices = [
            atom for residue in self.context.atoms
            for atom in residue
        ]
        forces_temp[qm_indices, :] = forces[:len(qm_indices), :]
        ae_indices = [
            atom for residue in self.context.embedding
            for atom in residue
        ]
        if ae_indices:
            forces_temp[ae_indices, :] = forces[len(qm_indices):, :]
        forces = forces_temp
        return forces


def pme_psi4_system_factory(settings: QMSettings) -> Psi4Interface:
    """A function which constructs the :class:`Psi4Interface` for a QM
    system.

    :param settings: The :class:`QMSettings` object to build the
        QM system interface from.
    :return: The :class:`Psi4Interface` for the QM system.
    """
    options = _build_options(settings)
    functional = settings.functional
    context = _build_context(settings)
    wrapper = PMEPsi4Interface(
        options, functional, context,
    )
    return wrapper


def _build_options(settings: QMSettings) -> PMEPsi4Options:
    """Build the :class:`Psi4Options` object.

    :param settings: The :class:`QMSettings` object to build from.
    :return: The :class:`Psi4Options` object built from the given
        settings.
    """
    options = PMEPsi4Options(
        settings.basis_set,
        settings.quadrature_spherical,
        settings.quadrature_radial,
        settings.scf_type,
        "uks" if settings.spin > 1 else "rks",
        "read" if settings.read_guess else "auto",
        "true",
    )
    return options


FACTORIES = {
    SystemTypes.SYSTEM: pme_psi4_system_factory,
    SystemTypes.SUBSYSTEM: pme_psi4_system_factory,
}
