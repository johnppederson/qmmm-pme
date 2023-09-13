#! /usr/bin/env python3
"""A module defining the :class:`Langevin` class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .integrator import Integrator
from .integrator import KB


@dataclass
class LangevinIntegrator(Integrator):
    """An integrator based on Langevin dynamics.

    :param timestep: |timestep|
    :param temperature: |temperature|
    :param friction: |friction|
    """
    friction: int | float

    def integrate(
            self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        masses = self.system.state.masses().reshape((-1, 1))
        vel_scale = np.exp(-self.timestep*self.friction)
        frc_scale = (
            self.timestep
            if self.friction == 0
            else (1 - vel_scale)/self.friction
        )
        noi_scale = (KB*self.temperature*(1 - vel_scale**2)*1000)**0.5
        z = np.random.standard_normal((len(masses), 3))
        momenta = self.system.state.velocities()*masses
        momenta = (
            vel_scale*momenta
            + frc_scale*self.system.state.forces()*(10**-4)
            + noi_scale*(10**-5)*z*masses**0.5
        )
        final_positions = (
            self.system.state.positions()
            + self.timestep*momenta/masses
        )
        final_velocities = momenta/masses
        return final_positions, final_velocities
