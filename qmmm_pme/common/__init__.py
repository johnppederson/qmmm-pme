#! /usr/bin/env python3
"""A sub-package containing core functionalities, classes, utilities,
and constants.
"""
from __future__ import annotations

from .core import Core
from .file_manager import FileManager
from .units import BOHR_PER_ANGSTROM
from .units import BOHR_PER_NM
from .units import KB
from .units import KJMOL_PER_EH
from .units import NM_PER_ANGSTROM
from .utils import align_dict
from .utils import compute_lattice_constants
from .utils import compute_least_mirror
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions
