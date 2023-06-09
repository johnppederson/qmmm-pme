#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import time
import numpy as np

from .settle import *
from .units import *


class Langevin:
    """
    """
    def __init__(self, timestep, temperature, friction, residues=None, settle_dists=None):
        self.timestep = timestep
        self.temperature = temperature
        self.friction = friction
        self.residues = residues
        self.dists = settle_dists

    def integrate(self, masses, positions, velocities, forces):
        """
        """
        # Based on code in openmm and ase
        positions_0 = positions
        masses = masses.reshape((-1,1))
        vel_scale = np.exp(-self.timestep*self.friction)
        frc_scale = self.timestep if self.friction == 0 else (1 - vel_scale)/self.friction
        noi_scale = (kB*self.temperature*(1 - vel_scale**2)*1000)**0.5
        z = np.random.standard_normal((len(masses), 3))
        momenta = velocities * masses
        momenta = vel_scale * momenta + frc_scale * forces * (10**-4) + noi_scale * (10**-5) * z * masses**0.5
        positions_1 = positions + self.timestep * momenta / masses
        velocities_1 = momenta / masses
        if self.residues:
            positions_1 = settle(positions_0, positions_1, self.residues, masses, dists=self.dists)
            velocities_1[self.residues,:] = (positions_1[self.residues,:] - positions_0[self.residues,:]) / self.timestep
        return positions_1, velocities_1
