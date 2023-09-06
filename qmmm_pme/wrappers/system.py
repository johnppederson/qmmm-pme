#! /usr/bin/env python3
"""A module defining the :class:`System` class.
"""
from __future__ import annotations

from qmmm_pme.common import FileManager
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
        file_manager = FileManager()
        state_data, topology_data = file_manager.load(
            pdb_list,
            topology_list,
            forcefield_list,
        )
        for key, value in state_data.items():
            setattr(self.state, key, value)
        for key, value in topology_data.items():
            setattr(self.topology, key, value)

    def __len__(self) -> int:
        return len(self.topology.atoms)
