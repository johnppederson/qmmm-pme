#! /usr/bin/env python3
"""A sub-package containing core functionalities, classes, utilities,
and constants.
"""
from __future__ import annotations
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .core import Core
from .file_manager import FileManager
from .units import (
    KB,
    BOHR_PER_NM,
    BOHR_PER_ANGSTROM,
    KJMOL_PER_EH,
    NM_PER_ANGSTROM,
)
from .utils import (
    align_dict,
    compute_least_mirror,
    compute_lattice_constants,
)
