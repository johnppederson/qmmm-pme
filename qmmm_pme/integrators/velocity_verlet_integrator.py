#! /usr/bin/env python3
"""A module defining the :class:`VelocityVerlet` class.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .integrator import Integrator


class VelocityVerletIntegrator(Integrator):
    """An integrator based on the Velocity Verlet algorithm.
    """

    def integrate(
            self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        masses = self.system.state.masses().reshape((-1, 1))
        momenta = self.system.state.velocities()*masses
        momenta = momenta + self.timestep*self.system.state.forces()*(10**-4)
        final_positions = (
            self.system.state.positions()
            + self.timestep*momenta/masses
        )
        final_velocities = momenta/masses
        return final_positions, final_velocities
