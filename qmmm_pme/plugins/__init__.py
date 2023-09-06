#! /usr/bin/env python3
"""A sub-package for defining plugin functionality and dynamically
loading plugins.
"""
from __future__ import annotations
__author__ = "Jesse McDaniel, John Pederson"
__version__ = "1.0.0"

import importlib

from .settle import SETTLE
