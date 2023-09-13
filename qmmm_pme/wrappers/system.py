#! /usr/bin/env python3
"""A module defining the :class:`System` class.
"""
from __future__ import annotations

from qmmm_pme.common import FileManager
from qmmm_pme.records import Files
from qmmm_pme.records import State
from qmmm_pme.records import Topology


class System:
    """An object designed to hold :class:`State` and :class:`Topology`
    record objects.

    :param pdb_list: |pdb_list|
    :param topology_list: |topology_list|
    :param forcefield_list: |forcefield_list|
    """

    def __init__(
            self,
            pdb_list: list[str],
            topology_list: list[str],
            forcefield_list: list[str],
    ) -> None:
        self.state = State()
        self.topology = Topology()
        self.files = Files()
        file_manager = FileManager()
        state_data, name_data, residue_data = file_manager.load(
            pdb_list,
            topology_list,
            forcefield_list,
        )
        for state_key, state_value in state_data.items():
            getattr(self.state, state_key).update(state_value)
        for name_key, name_value in name_data.items():
            getattr(self.topology, name_key).update(name_value)
        for residue_key, residue_value in residue_data.items():
            getattr(self.topology, residue_key).update(residue_value)

    def __len__(self) -> int:
        return len(self.topology.atom_names())
