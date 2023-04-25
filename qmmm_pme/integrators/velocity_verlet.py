#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import time
import numpy as np

from .settle import *
from .units import *


class VelocityVerlet:
    """
    """
    def __init__(self, timestep, residues=None, settle_dists=None):
        self.timestep = timestep
        self.residues = residues
        self.dists = settle_dists

    def integrate(self, masses, positions, velocities, forces):
        """
        """
        # Based on code in ase.md.verlet
        positions_0 = positions
        masses = masses.reshape((-1,1))
        momenta = velocities * masses
        momenta = momenta + self.timestep * forces * (10**-4)
        positions_1 = positions + self.timestep * momenta / masses
        if self.residues:
            positions_1 = settle(positions_0, positions_1, self.residues, masses, dists=self.dists)
            velocities_1 = (positions_1 - positions_0) / self.timestep
        else:
            velocities_1 = momenta / masses
        return positions_1, velocities_1
