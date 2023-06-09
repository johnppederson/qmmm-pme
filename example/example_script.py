#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM/MM, a method to perform single-point QM/MM calculations using the 
QM/MM/PME direct electrostatic QM/MM embedding method.
"""
import os
import sys

sys.setrecursionlimit(2000)
sys.path.append("../")
sys.path.append("../../")

from qmmm_pme import *


def main():
    """
    """
    # Define QM subsystem inputs.
    basis_set = "def2-SVP"
    functional = "PBE"
    quadrature_spherical = 302
    quadrature_radial = 75
    qm_charge = 0
    qm_spin = 1
    charge_type = "none"
    charge_method = "none"
    charge_weights = "none"
    n_threads = 8
    memory = "8 GB"
    read_guess = True
    reference_energy = None
    # Define MM subsystem inputs.
    pdb_list = ["./data/spce_box.pdb"]
    residue_xml_list = ["./data/spce_residues.xml"]
    ff_xml_list = ["./data/spce.xml"]
    platform = "CPU"
    # Define QM/MM system inputs.
    name = "example"
    group_part_dict = {"qm_atom": [0,1,2]}
    embedding_cutoff = 14.0
    embedding_method = "analytic"
    ensemble = "nvt"
    qmmm_pme = False
    qmmm_pme_gridnumber = 60
    qmmm_pme_alpha = 5.0
    particle_write_freq = 50 # how many steps between dcd writes
    decimal_places = 6
    mm_only = False
    settle_dists = [1,1.632981]
    # Instantiate QM subsystem.
    qm_subsystem = QMSubsystem(
        basis_set,
        functional,
        quadrature_spherical,
        quadrature_radial,
        qm_charge,
        qm_spin,
        charge_type=charge_type,
        charge_method=charge_method,
        charge_weights=charge_weights,
        qmmm_pme=qmmm_pme,
        n_threads=n_threads,
        memory=memory,
        reference_energy=reference_energy,
        read_guess=read_guess,
    )
    # Instantiate MM subsystem.
    mm_subsystem = MMSubsystem(
        pdb_list,
        residue_xml_list,
        ff_xml_list,
        platform,
        qmmm_pme = qmmm_pme,
        qmmm_pme_gridnumber = qmmm_pme_gridnumber,
        qmmm_pme_alpha = qmmm_pme_alpha,
    )
    # Instantiate QM/MM system.
    qmmm_system = QMMMSystem(
        name,
        qm_subsystem,
        mm_subsystem,
        group_part_dict,
        embedding_cutoff=embedding_cutoff,
        embedding_method=embedding_method,
        ensemble=ensemble,
        qmmm_pme=qmmm_pme,
        qmmm_pme_gridnumber=mm_subsystem.qmmm_pme_gridnumber,
        qmmm_pme_alpha=mm_subsystem.qmmm_pme_alpha,
        decimal_places=decimal_places,
        particle_write_freq=particle_write_freq,
        mm_only=mm_only,
        settle_dists=settle_dists,
    )
    qmmm_system.run_dynamics(1000)

if __name__ == "__main__":
    main()
