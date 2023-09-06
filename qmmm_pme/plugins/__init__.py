#! /usr/bin/env python3
"""A sub-package for defining plugin functionality and dynamically
loading plugins.
"""
from __future__ import annotations
__author__ = "Jesse McDaniel, John Pederson"

from .._version import get_versions
__version__ = get_versions()['version']
del get_versions

import importlib

from .settle import SETTLE
