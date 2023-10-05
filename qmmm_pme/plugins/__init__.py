#! /usr/bin/env python3
"""A sub-package for defining plugin functionality and dynamically
loading plugins.
"""
from __future__ import annotations

import importlib

from .atom_embedding import AtomEmbedding
from .pme import PME
from .rigid import RigidBody
from .rigid import Stationary
from .settle import SETTLE
# from .plumed import Plumed
__author__ = "John Pederson"
