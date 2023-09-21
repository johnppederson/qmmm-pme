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
from qmmm_pme.plugins import Stationary


def main() -> int:

    # Load system first.
    system = System(
        pdb_list=["./data/bbh.pdb"],
        topology_list=["./data/bmim_residues.xml"],
        forcefield_list=["./data/bmim.xml"],
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
    qmmm = qm[2580:] + mm[:2580] | 14.0

    # Define the integrator to use.
    dynamics = VelocityVerlet(1, 300)

    # Define the logger.
    logger = Logger("./output/", system, dcd_write_interval=1, decimal_places=12)

    # Define plugin objects.
    #settle = SETTLE()
    #stationary = Stationary(["BF4"])

    # Define simulation.
    simulation = Simulation(
        system=system,
        hamiltonian=qmmm,
        dynamics=dynamics,
        logger=logger,
        #plugins=[stationary],
    )

    simulation.run_dynamics(10)
    return 0


if __name__ == "__main__":
    exit(main())
