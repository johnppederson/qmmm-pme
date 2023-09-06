#! /usr/bin/env python3
"""A module defining the :class:`VelocityVerlet` class.
"""
from __future__ import annotations

from .integrator import Integrator


class VelocityVerlet(Integrator):
    """An integrator based on the Velocity Verlet algorithm.
    """

    def integrate(
            self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        masses = self._state.masses.reshape((-1, 1))
        momenta = self._state.velocities*masses
        momenta = momenta + self.timestep*self._state.forces*(10**-4)
        final_positions = self._state.positions + self.timestep*momenta/masses
        final_velocities = momenta/masses
        return final_positions, final_velocities
