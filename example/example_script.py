#! /usr/bin/env python3
"""
QM/MM, a method to perform single-point QM/MM calculations using the
QM/MM/PME direct electrostatic QM/MM embedding method.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from qmmm_pme import *
from qmmm_pme.plugins import SETTLE


def main() -> int:

    # Load system first.
    system = System(
        pdb_list=["./data/spce.pdb"],
        topology_list=["./data/spce_residues.xml"],
        forcefield_list=["./data/spce.xml"],
    )

    # Define QM Hamiltonian.
    qm = QMHamiltonian(
        basis_set="def2-SVP",
        functional="PBE",
        charge=0,
        spin=1,
    )

    # Define MM Hamiltonian.
    mm = MMHamiltonian(
        pme_gridnumber=30,
    )

    # Define QM/MM Hamiltonian
    qmmm = qm[0:3] + mm[3:]
    qmmm | 14.0

    # Define the integrator to use.
    dynamics = VelocityVerlet(1, 300)

    # Define the logger.
    logger = Logger("./output/", system, dcd_write_interval=1)
    settle = SETTLE()

    # Define simulation.
    simulation = Simulation(
        system=system,
        hamiltonian=mm,
        dynamics=dynamics,
        logger=logger,
        plugins=[settle],
    )

    # Run simulation
    simulation.run_dynamics(10)
    return 0


if __name__ == "__main__":
    exit(main())
