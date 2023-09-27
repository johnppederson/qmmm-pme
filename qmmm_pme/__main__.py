#! /usr/bin/env python3
"""
"""
from __future__ import annotations

import re
from argparse import ArgumentParser
from configparser import ConfigParser

import qmmm_pme.dynamics
from qmmm_pme import Logger
from qmmm_pme import MMHamiltonian
from qmmm_pme import QMHamiltonian
from qmmm_pme import Simulation
from qmmm_pme import System


def _parse_input(input_value: str) -> int | float | slice | str:
    if re.findall("[a-z, A-Z]", input_value):
        return input_value
    elif re.findall("[.]", input_value):
        return float(input_value)
    elif re.findall("[:]", input_value):
        elements = input_value.split(":")
        start = None
        end = None
        step = None
        if len(elements) > 2:
            step = int(elements[2])
        if elements[0]:
            start = int(elements[0])
        if elements[1]:
            end = int(elements[1])
        return slice(start, end, step)
    else:
        return int(input_value)


def main() -> int:

    # Collect the input file directory.
    parser = ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    # Read input file.
    input_file = ConfigParser()
    input_file.read(args.input_file)

    # Create System object.
    system_args = {
        key: value.strip().split("\n") for key, value
        in input_file["System"].items()
    }
    system = System(**system_args)
    simulation_args = {"system": system}

    # Create Dynamics object.
    dynamics = input_file["Dynamics"]["dynamics"]
    dynamics_args = {
        key: _parse_input(value) for key, value
        in input_file["Dynamics"].items()
        if value != dynamics
    }
    Dynamics = getattr(qmmm_pme.dynamics, dynamics)
    dynamics = Dynamics(**dynamics_args)
    simulation_args["dynamics"] = dynamics

    # Create Hamiltonian objects.
    if "QMMM" in input_file:
        qm_args = {
            key: _parse_input(value) for key, value
            in input_file["QM"].items()
        }
        qm = QMHamiltonian(**qm_args)
        mm_args = {
            key: _parse_input(value) for key, value
            in input_file["MM"].items()
        }
        mm = MMHamiltonian(**mm_args)
        qm_atoms = [
            _parse_input(index) for index
            in input_file["QMMM"]["qm_atoms"].strip().split("\n")
        ]
        mm_atoms = [
            _parse_input(index) for index
            in input_file["QMMM"]["mm_atoms"].strip().split("\n")
        ]
        embedding_cutoff = _parse_input(
            input_file["QMMM"]["embedding_cutoff"],
        )
        hamiltonian = (
            qm.__getitem__(*qm_atoms)
            + mm.__getitem__(*mm_atoms) | embedding_cutoff
        )
    elif "QM" in input_file:
        qm_args = {
            key: _parse_input(value) for key, value
            in input_file["QM"].items()
        }
        hamiltonian = QMHamiltonian(**qm_args)
    elif "MM" in input_file:
        mm_args = {
            key: _parse_input(value) for key, value
            in input_file["MM"].items()
        }
        hamiltonian = MMHamiltonian(**mm_args)
    else:
        raise Exception
    simulation_args["hamiltonian"] = hamiltonian

    # Create Logger object.
    if "Logger" in input_file:
        logger_args = {
            key: _parse_input(value) for key, value
            in input_file["Logger"].items()
        }
        logger = Logger(system=system, **logger_args)
        simulation_args["logger"] = logger

    # Create and run the simulation.
    simulation = Simulation(**simulation_args)
    steps = _parse_input(input_file["Simulation"]["steps"])
    simulation.run_dynamics(steps)
    return 0
