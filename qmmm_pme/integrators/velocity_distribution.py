#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import sys
import numpy as np

from .units import *


def MaxwellBoltzmann(temperature, masses):
    """
    """
    # Comparable to ase.md.velocitydistribution.py
    avg_ke = temperature * kB
    masses = masses.reshape((-1,1)) * (10**-3)
    np.random.seed(10101)
    z = np.random.standard_normal((len(masses), 3))
    momenta = z * np.sqrt(avg_ke * masses)
    velocities = (momenta / masses) * (10**-5)
    return velocities
