#! /usr/bin/env python3
"""A module for defining the :class:`Dynamics` base and derived classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qmmm_pme.integrators import LangevinIntegrator
from qmmm_pme.integrators import VelocityVerletIntegrator
from qmmm_pme.integrators import VerletIntegrator

if TYPE_CHECKING:
    from qmmm_pme import System
    from qmmm_pme.integrators.integrator import Integrator


@dataclass
class Dynamics(ABC):
    """A base class for defining dynamics.

    :param timestep: |timestep|
    """
    timestep: float | int

    @abstractmethod
    def build_integrator(self, system: System) -> Integrator:
        """Build the :class:`Integrator` corresponding to the
        :class:`Dynamics` object.

        :param system: |system| to integrate forces for.
        :return: |Integrator|.
        """


@dataclass
class VelocityVerlet(Dynamics):
    """A :class:`Dynamics` object storing parameters necessary for
    creating the Velocity Verlet integrator.

    :param temperature: |temperature|
    """
    temperature: float | int

    def build_integrator(self, system: System) -> Integrator:
        integrator = VelocityVerletIntegrator(system=system, **asdict(self))
        return integrator


@dataclass
class Verlet(Dynamics):
    """A :class:`Dynamics` object storing parameters necessary for
    creating the Verlet integrator.

    :param temperature: |temperature|
    """
    temperature: float | int

    def build_integrator(self, system: System) -> Integrator:
        integrator = VerletIntegrator(system=system, **asdict(self))
        return integrator


@dataclass
class Langevin(Dynamics):
    """A :class:`Dynamics` object storing parameters necessary for
    creating the Langevin integrator.

    :param temperature: |temperature|
    :param friction: |friction|
    """
    temperature: float | int
    friction: float | int

    def build_integrator(self, system: System) -> Integrator:
        integrator = LangevinIntegrator(system=system, **asdict(self))
        return integrator
