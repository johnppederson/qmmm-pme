#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions accessed by multiple classes.
"""
import numpy as np


def least_mirror_vector(i_vector, j_vector, box):
    """Calculates a lease mirror vector.

    Returns the least mirror coordinates of i_vector with respect to
    j_vector given a set of box vectors from a periodic triclinic system.

    Parameters
    ----------
    i_vector : numpy.ndarray
        Position vector, in Angstroms.
    j_vector : numpy.ndarray
        Reference vector, in Angstroms.
    box : numpy.ndarray
        Periodic box vectors of the system, in Angstroms.

    Returns
    -------
    r_vector : numpy.ndarray
        Least mirror vector of the position vector with respect to the
        reference vector.
    """
    r_vector = [i_vector[k] - j_vector[k] for k in range(3)]
    r_vector -= box[2] * np.floor(r_vector[2]/box[2][2] + 0.5)
    r_vector -= box[1] * np.floor(r_vector[1]/box[1][1] + 0.5)
    r_vector -= box[0] * np.floor(r_vector[0]/box[0][0] + 0.5)
    return r_vector


def lattice_constants(box):
    """Calculates length and angle lattice constants.

    Returns the lattice constants a, b, c, alpha, beta, and gamma using
    a set of box vectors for a periodic triclinic system.

    Parameters
    ----------
    box : numpy.ndarray
        Periodic box vectors of the system, in Angstroms.

    Returns
    -------
    a : float
        One of the characteristic lengths of a triclinic box, in Angstroms.
    b : float
        One of the characteristic lengths of a triclinic box, in Angstroms.
    c : float
        One of the characteristic lengths of a triclinic box, in Angstroms.
    alpha : float
        One of the characteristic angles of a triclinic box, in radians.
    beta : float
        One of the characteristic angles of a triclinic box, in radians.
    gamma : float
        One of the characteristic angles of a triclinic box, in radians.
    """
    A = box[:,0]
    B = box[:,1]
    C = box[:,2]
    a = np.linalg.norm(A)
    b = np.linalg.norm(B)
    c = np.linalg.norm(C)
    alpha = np.arccos(np.dot(B,C)/np.linalg.norm(B)/np.linalg.norm(C))
    beta = np.arccos(np.dot(A,C)/np.linalg.norm(A)/np.linalg.norm(C))
    gamma = np.arccos(np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B))
    return a, b, c, alpha, beta, gamma
