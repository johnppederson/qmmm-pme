#! /usr/bin/env python3
"""A sub-package to define records kept by the simulation, which include
data pertaining to input files and the state and topology of atoms and
residues.
"""
from __future__ import annotations
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .files import Files
from .namespace import Namespace
from .state import State
from .topology import Topology
