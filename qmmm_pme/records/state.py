#! /usr/bin/env python3
"""A module defining the :class:`State` record container.
"""
from __future__ import annotations

import numpy as np

from .record import Record
from .record import recordclass


class State(Record, metaclass=recordclass):
    """A :class:`Record` class for maintaining data about the state
    of the :class:`System`, comprising atomic masses, charges,
    positions, velocities, momenta, and forces; the box vectors defining
    the periodic boundary condition; and the frame number.
    """
    _masses = np.empty(0)
    _charges = np.empty(0)
    _positions = np.empty(0)
    _velocities = np.empty(0)
    _momenta = np.empty(0)
    _forces = np.empty(0)
    _box = np.empty(0)
    _frame = 0
