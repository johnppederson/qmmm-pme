#! /usr/bin/env python3
"""A sub-package containing wrapper classes for the end user.
"""
from __future__ import annotations
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .logger import Logger
from .simulation import Simulation
from .system import System
