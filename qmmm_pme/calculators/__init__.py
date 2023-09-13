#! /usr/bin/env python3
"""A sub-package containing the low-level calculators of the simulation
engine.
"""
from __future__ import annotations

from .calculator import StandaloneCalculator
from .qmmm_calculator import QMMMCalculator
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions
