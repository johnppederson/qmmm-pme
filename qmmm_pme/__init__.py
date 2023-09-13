#! /usr/bin/env python3
"""The qmmm-pme package, a simulation engine employing the QM/MM/PME
method described in `The Journal of Chemical Physics`_.

.. _The Journal of Chemical Physics: https://doi.org/10.1063/5.0087386
"""
from __future__ import annotations

from . import _version
from .dynamics import Langevin
from .dynamics import VelocityVerlet
from .hamiltonians import MMHamiltonian
from .hamiltonians import QMHamiltonian
from .hamiltonians import QMMMHamiltonian
from .wrappers import Logger
from .wrappers import Simulation
from .wrappers import System
__author__ = "Jesse McDaniel, John Pederson"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


__version__ = _version.get_versions()['version']
