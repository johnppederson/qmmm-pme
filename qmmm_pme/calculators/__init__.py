#! /usr/bin/env python3
"""A sub-package containing the low-level calculators of the simulation
engine.
"""
from __future__ import annotations
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .calculator_factory import calculator_factory
