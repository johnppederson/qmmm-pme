#! /usr/bin/env python3
"""A module defining the :class:`Topology` record container.
"""
from __future__ import annotations

from .record import Record
from .record import recordclass


class Topology(Record, metaclass=recordclass):
    """A :class:`Record` class for maintaining data about the topology
    of the :class:`System`, comprising atom groups, residue names,
    element symbols, and atom names.
    """
    _groups = {}
    _residues = []
    _elements = []
    _atoms = []
