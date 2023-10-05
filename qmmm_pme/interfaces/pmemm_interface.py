#! /usr/bin/env python3
"""A module to define the :class:`OpenMMInterface` class.
"""
from __future__ import annotations

from openmm import Context
from openmm import LangevinIntegrator
from openmm import Platform
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


def pme_openmm_system_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def pme_openmm_subsystem_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_qm_atoms(settings, system)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
    return wrapper


def pme_openmm_embedding_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface`.
    """
    pdb, modeller, forcefield, system = _build_base(settings)
    _exclude_non_embedding(settings, pdb, system)
    context = _build_context(settings, system, modeller)
    wrapper = OpenMMInterface(pdb, modeller, forcefield, system, context)
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
