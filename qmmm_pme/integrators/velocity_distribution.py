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
    #if residues:
    #    res_masses = masses[residues,:].reshape((len(residues),3))
    #    res_positions = positions[residues, :]
    #    com_masses = np.sum(res_masses, axis=1).reshape((-1,1))
    #    coms = np.sum(res_masses[:,:,np.newaxis] * res_positions, axis=1) / com_masses
    #    msds = np.sum(res_masses[:,:,np.newaxis] * (res_positions - coms[:,np.newaxis,:])**2, axis=2)
         
    #    moments = np.zeros_like(coms)
    #    moments[:,0] = np.sum(msds[:,[1,2]],axis=1)
    #    moments[:,1] = np.sum(msds[:,[0,2]],axis=1)
    #    moments[:,2] = np.sum(msds[:,[0,1]],axis=1)
        
    #    z = np.random.standard_normal((len(residues), 3))
    #    com_momenta_t = z * np.sqrt(0.5 * avg_ke * com_masses)
    #    z = np.random.standard_normal((len(residues), 3))
    #    com_momenta_r = z * np.sqrt(0.5 * avg_ke * moments)

    #    velocities = (com_momenta_t / com_masses) * (10**-5)
    #    velocities = np.stack((velocities, velocities, velocities), axis=2)
    #    angular_vel = np.stack((com_momenta_r/moments, com_momenta_r/moments, com_momenta_r/moments), axis = 2)
    #    disps = res_positions - coms[:,np.newaxis,:]
    #    velocities += np.cross(angular_vel, disps) * (10**-5)
    #    velocities = velocities.reshape((len(masses), 3))
    #    #print(np.sum(((velocities*(10**5))**2)*masses/2))
    #    #sys.exit()
    #else:
    np.random.seed(10101)
    z = np.random.standard_normal((len(masses), 3))
    momenta = z * np.sqrt(avg_ke * masses)
    velocities = (momenta / masses) * (10**-5)
    return velocities
