#! /usr/bin/env python3
"""A module to define the :class:`OpenMMInterface` class.
"""
from __future__ import annotations

from dataclasses import dataclass

from openmm import Context
from openmm import LangevinIntegrator
from openmm import Platform
from openmm import State
from openmm import System
from openmm.app import Modeller
from simtk.unit import femtosecond
from simtk.unit import kelvin

from .interface import MMSettings
from .interface import SoftwareTypes
from .interface import SystemTypes
from .openmm_interface import _build_base
from .openmm_interface import _exclude_non_embedding
from .openmm_interface import _exclude_qm_atoms
from .openmm_interface import OpenMMInterface


SOFTWARE_TYPE = SoftwareTypes.MM


@dataclass(frozen=True)
class PMEOpenMMInterface(OpenMMInterface):
    """A class which wraps the functional components of OpenMM.
    """

    def _generate_state(self, **kwargs: bool | set[int]) -> State:
        """Create the OpenMM State which is used to compute energies
        and forces.
        """
        state = self.context.getState(
            getVext_grids=True,
            **kwargs,
        )
        return state


def pme_openmm_system_factory(settings: MMSettings) -> PMEOpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    context = _build_context(settings, system, modeller)
    wrapper = PMEOpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def pme_openmm_subsystem_factory(settings: MMSettings) -> PMEOpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_qm_atoms(settings, system)
    context = _build_context(settings, system, modeller)
    wrapper = PMEOpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def pme_openmm_embedding_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_non_embedding(settings, pdb, system)
    context = _build_context(settings, system, modeller)
    wrapper = PMEOpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def _build_context(
        settings: MMSettings, system: System, modeller: Modeller,
) -> Context:
    """
    """
    integrator = LangevinIntegrator(
        settings.temperature * kelvin,
        settings.friction / femtosecond,
        settings.timestep * femtosecond,
    )
    platform = Platform.getPlatformByName("CPU")
    context = Context(
        system, integrator, platform, {"ReferenceVextGrid": "true"},
    )
    context.setPositions(modeller.positions)
    return context


FACTORIES = {
    SystemTypes.SYSTEM: pme_openmm_system_factory,
    SystemTypes.SUBSYSTEM: pme_openmm_subsystem_factory,
    SystemTypes.EMBEDDING: pme_openmm_embedding_factory,
}
