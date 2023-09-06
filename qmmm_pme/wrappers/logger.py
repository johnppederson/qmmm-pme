#! /usr/bin/env python3
"""A module defining the :class:`Logger` class.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from qmmm_pme.common import align_dict
from qmmm_pme.common import FileManager

if TYPE_CHECKING:
    from .system import System
    from .simulation import Simulation


@dataclass
class Logger:
    """A logger for writing :class:`Simulation` and :class:`System`
    data.

    :param output_dir: The output file directory into which
    """
    output_directory: str
    system: System
    write_to_log: bool | None = True
    decimal_places: int | None = 6
    log_write_interval: int | None = 1
    write_to_csv: bool | None = True
    csv_write_interval: int | None = 1
    write_to_dcd: bool | None = True
    dcd_write_interval: int | None = 50
    write_to_pdb: bool | None = True

    def __enter__(self) -> None:
        """Create output files which will house output data from the
        :class:`Simulation`.
        """
        self.file_manager = FileManager(self.output_directory)
        if self.write_to_log:
            self.log = "output.log"
            self.file_manager.start_log(self.log)
        if self.write_to_csv:
            self.csv = "output.csv"
            self.file_manager.start_csv(self.csv, "")
        if self.write_to_dcd:
            self.dcd = "output.dcd"
            self.file_manager.start_dcd(
                self.dcd, self.dcd_write_interval, len(self.system), 1,
            )
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        """Perform any termination steps required at the end of the
        :class:`Simulation`.
        """
        if self.write_to_log:
            self.file_manager.end_log(self.log)
        if self.write_to_pdb:
            self.file_manager.write_to_pdb(
                "output.pdb",
                self.system.state.positions,
                self.system.state.box,
                self.system.topology.groups["all"],
                self.system.topology.residues,
                self.system.topology.elements,
                self.system.topology.atoms,
            )

    def record(self, simulation: Simulation) -> None:
        """Log the current state of the :class:`Simulation` to any
        relevant output files.

        :param simulation: The :class:`Simulation` to document.
        """
        if self.write_to_log:
            self.file_manager.write_to_log(
                self.log,
                self._unwrap_energy(simulation.energy),
                simulation.system.state.frame,
            )
        if self.write_to_csv:
            self.file_manager.write_to_csv(
                self.csv,
                ",".join(
                    f"{val}" for val
                    in align_dict(simulation.energy).values()
                ),
            )
        if self.write_to_dcd:
            self.file_manager.write_to_dcd(
                self.dcd,
                self.dcd_write_interval,
                len(self.system),
                simulation.system.state.positions,
                simulation.system.state.box,
                simulation.system.state.frame,
            )

    def _unwrap_energy(
            self,
            energy: dict[str: Any],
            spaces: int | None = 0,
            cont: list[Any, ...] | None = [],
    ) -> str:
        """Generate the Log string given the :class:`Simulation` energy
        dictionary.

        :param energy: The energy calculated by the :class:`Simulation`.
        :param spaces: The number of spaces by which to indent.
        :parma cont: A list to keep track of sub-component continuation.
        :return: The string to write to the Log file.
        """
        string = ""
        for i, (key, val) in enumerate(energy.items()):
            if isinstance(val, dict):
                string += self._unwrap_energy(
                    val, spaces + 1, (
                        cont+[spaces-1] if i != len(energy)-1 else cont
                    ),
                )
            else:
                value = f"{val:.{self.decimal_places}f} kJ/mol\n"
                if spaces:
                    key = "".join(
                        "| " if i in cont else "  "
                        for i in range(spaces - 1)
                    )+"|_"+key
                string += f"{key}:{value: >{72-len(key)}}"
        return string


class NullLogger(nullcontext):
    def record(*args, **kwargs):
        pass
