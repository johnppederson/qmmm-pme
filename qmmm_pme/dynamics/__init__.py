#! /usr/bin/env python3
"""A sub-package to define dynamics.
"""
from __future__ import annotations

from .dynamics import Langevin
from .dynamics import VelocityVerlet
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions
