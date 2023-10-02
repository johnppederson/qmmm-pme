#! /usr/bin/env python3
"""A module defining the class and functions needed to load and write
files.
"""
from __future__ import annotations

import array
import os
import struct
from typing import TYPE_CHECKING

import numpy as np
import openmm.app
import simtk.unit

from .utils import compute_lattice_constants

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FileManager:
    """A class to load and generate inputs and outputs.

    :param working_directory: The current working directory.
    """

    def __init__(self, working_directory: str = "./") -> None:
        self._working_directory = working_directory
        if not os.path.isdir(working_directory):
            os.makedirs(working_directory)

    def load(
            self,
            pdb_list: list[str],
            topology_list: list[str],
            forcefield_list: list[str],
    ) -> tuple[
        dict[str, NDArray[np.float64]],
        dict[str, list[str]],
        dict[str, list[list[int]]],
    ]:
        """Load files necessary to generate a system.

        :param pdb_list: |pdb_list|
        :param topology_list: |topology_list|
        :param forcefield_list: |forcefield_list|
        :return: Data for the :class:`State` and :class:`Topology`
            record classes.
        """
        # Check the file extensions and add them to the :class:`Files`
        # record.
        for fh in pdb_list:
            _check_ext(fh, "pdb")
        for fh in topology_list:
            _check_ext(fh, "xml")
        for fh in forcefield_list:
            _check_ext(fh, "xml")
        # Generate and return :class:`System` data using OpenMM.
        state_data, name_data, residue_data = _get_system_data(
            pdb_list,
            topology_list,
            forcefield_list,
        )
        return state_data, name_data, residue_data

    def write_to_pdb(
            self,
            name: str,
            positions: NDArray[np.float64],
            box: NDArray[np.float64],
            all_groups: list[list[int]],
            residues: list[str],
            elements: list[str],
            atoms: list[str],
    ) -> None:
        """Utility to write PDB files with :class:`State` current
        coordinates.

        :param name: The directory/name of PDB file to be written.
        :param positions: |positions|
        :param box: |box|
        :param all_groups: The indices of all atoms in the
            :class:`System` grouped by residue.
        :param residues: |residues|
        :param elements: |elements|
        :param atoms: |atoms|

        .. note:: Based on PDB writer from OpenMM.
        """
        filename = self._parse_name(name, ext="pdb")
        (
            len_a, len_b, len_c,
            alpha, beta, gamma,
        ) = compute_lattice_constants(box)
        with open(filename, "w") as fh:
            fh.write(
                (
                    f"CRYST1{len_a:9.3f}{len_b:9.3f}{len_c:9.3f}"
                    + f"{alpha:7.2f}{beta:7.2f}"
                    + f"{gamma:7.2f} P 1           1 \n"
                ),
            )
            for i, residue in enumerate(all_groups):
                for atom in residue:
                    line = "HETATM"
                    line += f"{atom+1:5d}  "
                    line += f"{atoms[atom]:4s}"
                    line += f"{residues[i]:4s}"
                    line += f"A{i+1:4d}    "
                    line += f"{positions[atom,0]:8.3f}"
                    line += f"{positions[atom,1]:8.3f}"
                    line += f"{positions[atom,2]:8.3f}"
                    line += "  1.00  0.00           "
                    line += f"{elements[atom]:2s}  \n"
                    fh.write(line)
            fh.write("END")

    def start_dcd(
            self,
            name: str,
            write_interval: int,
            num_particles: int,
            timestep: int | float,
    ) -> None:
        """Utility to start writing a DCD file.

        :param name: The directory/name of DCD file to be written.
        :param write_interval: The interval between successive writes
            to the DCD file.
        :param num_particles: The number of particles in the
            :class:`System`.
        :param timestep: |timestep|

        .. note:: Based on DCD writer from OpenMM.
        """
        filename = self._parse_name(name, ext="dcd")
        with open(filename, "wb") as fh:
            header = struct.pack(
                "<i4c9if", 84, b"C", b"O", b"R", b"D",
                0, 0, write_interval, 0, 0, 0, 0, 0, 0, timestep,
            )
            header += struct.pack(
                "<13i", 1, 0, 0, 0, 0, 0, 0, 0, 0, 24,
                84, 164, 2,
            )
            header += struct.pack("<80s", b"Created by QM/MM/PME")
            header += struct.pack("<80s", b"Created now")
            header += struct.pack("<4i", 164, 4, num_particles, 4)
            fh.write(header)

    def write_to_dcd(
            self,
            name: str,
            write_interval: int,
            num_particles: int,
            positions: NDArray[np.float64],
            box: NDArray[np.float64],
            frame: int,
    ) -> None:
        """Write data to an existing DCD file.

        :param name: The directory/name of DCD file to be written.
        :param write_interval: The interval between successive writes
            to the DCD file.
        :param num_particles: The number of particles in the
            :class:`System`.
        :param positions: |positions|
        :param box: |box|
        :param frame: |frame|

        .. note:: Based on DCD writer from OpenMM.
        """
        filename = self._parse_name(name, ext="dcd")
        (
            len_a, len_b, len_c,
            alpha, beta, gamma,
        ) = compute_lattice_constants(box)
        with open(filename, "r+b") as fh:
            fh.seek(8, os.SEEK_SET)
            fh.write(struct.pack("<i", frame//write_interval))
            fh.seek(20, os.SEEK_SET)
            fh.write(struct.pack("<i", frame))
            fh.seek(0, os.SEEK_END)
            fh.write(
                struct.pack(
                    "<i6di", 48, len_a, gamma, len_b, beta,
                    alpha, len_c, 48,
                ),
            )
            num = struct.pack("<i", 4*num_particles)
            for i in range(3):
                fh.write(num)
                coordinate = array.array(
                    "f", (position[i] for position in positions),
                )
                coordinate.tofile(fh)
                fh.write(num)
            fh.flush()

    def start_log(
            self,
            name: str,
    ) -> None:
        """Utility to start writing a log file.

        :param name: The directory/name of log file to be written.
        """
        filename = self._parse_name(name, ext="log")
        with open(filename, "w") as fh:
            fh.write(f"{' QM/MM/PME Logger ':=^72}\n")

    def write_to_log(
            self,
            name: str,
            lines: str,
            frame: int,
    ) -> None:
        """Write data to an existing log file.

        :param name: The directory/name of log file to be written.
        :param lines: The lines to be written to the log file.
        :param frame: |frame|
        """
        filename = self._parse_name(name, ext="log")
        with open(filename, "a") as fh:
            fh.write(f"{' Frame ' + f'{frame:0>6}' + ' ':-^72}\n")
            fh.write(lines + "\n")
            fh.flush()

    def end_log(
            self,
            name: str,
    ) -> None:
        """Terminate an existing log file.

        :param name: The directory/name of log file to be terminated.
        """
        filename = self._parse_name(name, ext="log")
        with open(filename, "a") as fh:
            fh.write(f"{' End of Log ':=^72}")

    def start_csv(
            self,
            name: str,
            header: str,
    ) -> None:
        """Utility to start writing a CSV file.

        :param name: The directory/name of CSV file to be written.
        :param header: The header for the CSV file.
        """
        filename = self._parse_name(name, ext="csv")
        with open(filename, "w") as fh:
            fh.write(header + "\n")

    def write_to_csv(
            self,
            name: str,
            line: str,
            header: str | None = None,
    ) -> None:
        """Write data to an existing CSV file.

        :param name: The directory/name of CSV file to be written.
        :param lines: The lines to be written to the CSV file.
        """
        filename = self._parse_name(name, ext="csv")
        if header:
            with open(filename) as fh:
                lines = fh.readlines()
            with open(filename, "w") as fh:
                lines[0] = header + "\n"
                fh.writelines(lines)
        with open(filename, "a") as fh:
            fh.write(line + "\n")
            fh.flush()

    def _parse_name(self, name: str, ext: str = "") -> str:
        """Ensure that the given name has the correct directory path
        and extension.

        :param name: The directory/name of the file to be written.
        :param ext: The desired extension of the file to be written.
        """
        filename = ""
        if not name.startswith(self._working_directory):
            filename = self._working_directory
        filename += name
        if not name.endswith(ext):
            filename += "." + ext
        return filename


def _check_ext(filename: str, ext: str) -> None:
    """Ensure that the given filename has the correct extension.

    :param filename: The name of the file.
    :param ext: The desired extension.
    """
    if not filename.endswith(ext):
        bad_ext = filename.split(".")[-1]
        raise ValueError(
            (
                f"Got extension'{bad_ext}' from file '{filename}'.  "
                + f"Expected extension '{ext}'"
            ),
        )


def _get_system_data(
        pdb_list: list[str],
        topology_list: list[str],
        forcefield_list: list[str],
) -> tuple[
        dict[str, NDArray[np.float64]],
        dict[str, list[str]],
        dict[str, list[list[int]]],
]:
    """Extract :class:`State` and :class:`Topology` data from PDB and
    XML files using OpenMM.

    :param pdb_list: |pdb_list|
    :param topology_list: |topology_list|
    :param forcefield_list: |forcefield_list|
    :return: Data for the :class:`State` and :class:`Topology`
        record classes.
    """
    # Create the OpenMM Modeller object and extract relevant topology
    # data from the Atoms and Residues.
    for fh in topology_list:
        openmm.app.topology.Topology().loadBondDefinitions(fh)
    openmm_pdb = openmm.app.pdbfile.PDBFile(pdb_list[0])
    openmm_modeller = openmm.app.modeller.Modeller(
        openmm_pdb.topology,
        openmm_pdb.positions,
    )
    openmm_residues = list(openmm_pdb.topology.residues())
    openmm_atoms = list(openmm_pdb.topology.atoms())
    atoms = [
        [atom.index for atom in residue.atoms()]
        for residue in openmm_residues
    ]
    elements = [atom.element.symbol for atom in openmm_atoms]
    residue_names = [residue.name for residue in openmm_residues]
    atom_names = [atom.name for atom in openmm_atoms]
    name_data = {
        "elements": elements,
        "residue_names": residue_names,
        "atom_names": atom_names,
    }
    residue_data = {
        "atoms": atoms,
    }
    # Load the OpenMM ForceField and create a System in order to get
    # charge and mass data.
    openmm_forcefield = openmm.app.forcefield.ForceField(
        *forcefield_list,
    )
    openmm_modeller.addExtraParticles(openmm_forcefield)
    openmm_system = openmm_forcefield.createSystem(
        openmm_modeller.topology,
    )
    nonbonded_force = [
        force for force in [
            openmm_system.getForce(i) for i
            in range(openmm_system.getNumForces())
        ] if isinstance(force, openmm.NonbondedForce)
    ][0]
    masses = np.array(
        [
            openmm_system.getParticleMass(
                atom.index,
            )/simtk.unit.dalton
            for atom in openmm_atoms
        ],
    )
    charges = np.array(
        [
            nonbonded_force.getParticleParameters(
                atom.index,
            )[0]/simtk.unit.elementary_charges
            for atom in openmm_atoms
        ],
    )
    positions = np.array(
        [
            vector/simtk.unit.angstrom for vector in
            openmm_modeller.getPositions()
        ],
    )
    box = np.array(
        [
            vector/simtk.unit.angstrom for vector in
            openmm_modeller.topology.getPeriodicBoxVectors()
        ],
    )
    state_data = {
        "masses": masses,
        "charges": charges,
        "positions": positions,
        "box": box,
    }
    return state_data, name_data, residue_data
