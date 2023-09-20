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
        pdb_list=["./data/hoh_dimer.pdb"],
        topology_list=["./data/spce_residues.xml"],
        forcefield_list=["./data/spcfw.xml"],
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
    #qmmm = qm[0:3] + mm[3:] | 14.0

    # Define the integrator to use.
    dynamics = VelocityVerlet(1, 300)

    # Define the logger.
    logger = Logger("./output/", system, dcd_write_interval=1, decimal_places=12)

    # Define plugin objects.
    #settle = SETTLE()

    # Define simulation.
    simulation = Simulation(
        system=system,
        hamiltonian=mm,
        dynamics=dynamics,
        logger=logger,
    #    plugins=[settle],
    )

    v0 = simulation.integrator.compute_velocities()*100
    print("\nFORCES BEFORE FIRST STEP")
    print(simulation.system.state.forces())
    print("\nVELOCITIES BEFORE FIRST STEP")
    print(v0)
    p0 = simulation.system.state.positions()
    #sys.exit()

    simulation.run_dynamics(100)
    #for i in range(1):
        #simulation.run_dynamics(1)
        #print("\nFORCES AFTER FIRST STEP")
        #print(simulation.system.state.forces()*10)
        #print("\nVELOCITIES AFTER FIRST STEP")
        #print(simulation.system.state.velocities()*100)
        #print("\nPOSITIONS->VELOCITIES AFTER FIRST STEP")
        #print((simulation.system.state.positions()-p0)*100)
        #simulation.calculate_energy_forces()
        #print("\nFORCES AFTER FIRST STEP")
        #print(simulation.system.state.forces()*10)
    return 0


if __name__ == "__main__":
    exit(main())
