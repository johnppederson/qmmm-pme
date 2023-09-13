#! /usr/bin/env python3
"""A module to define the Hamiltonian API.
"""
from __future__ import annotations

from .mm_hamiltonian import MMHamiltonian
from .qm_hamiltonian import QMHamiltonian
from .qmmm_hamiltonian import QMMMHamiltonian
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions
