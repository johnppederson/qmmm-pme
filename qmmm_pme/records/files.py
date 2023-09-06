#! /usr/bin/env python3
""" Files data container.
"""
from __future__ import annotations

from .record import Record
from .record import recordclass


class Files(Record, metaclass=recordclass):
    """Class for maintaining records about input and output files.
    """
    _pdb_list = []
    _topology_list = []
    _forcefield_list = []
    _input_json = ""
    _output_log = ""
    _output_csv = ""
    _output_dcd = ""
    _output_pdb = ""
