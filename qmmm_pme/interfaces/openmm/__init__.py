#! /usr/bin/env python3
"""A sub-package for interfacing to OpenMM.
"""
from __future__ import annotations
__author__ = "John Pederson"

from qmmm_pme.common import TheoryLevel

from .openmm_factory import openmm_interface_factory


FACTORY = openmm_interface_factory
THEORY_LEVEL = TheoryLevel.MM
