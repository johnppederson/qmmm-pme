#! /usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

from qmmm_pme import *

sys.setrecursionlimit(2000)
sys.path.append("../")
sys.path.append("../../")


def main():
    """
    """
    params = {
        # Define QM subsystem inputs.
        "basis_set": (basis_set := "def2-SVP"),
        "functional": (functional := "PBE"),
        "quadrature_spherical": (quadrature_spherical := 302),
        "quadrature_radial": (quadrature_radial := 75),
        "qm_charge": (qm_charge := 0),
        "qm_spin": (qm_spin := 1),
        "charge_type": (charge_type := "mbis"),
        "charge_weights": (charge_weights := "none"),
        "n_threads": (n_threads := 8),
        "memory": (memory := "16 GB"),
        "read_guess": (read_guess := True),
        "reference_energy": (reference_energy := None),
        # Define MM subsystem inputs.
        "pdb_list": (pdb_list := ["./data/spce.pdb"]),
        "residue_xml_list": (residue_xml_list := ["./data/spce_residues.xml"]),
        "ff_xml_list": (ff_xml_list := ["./data/spce.xml"]),
        "platform": (platform := "CPU"),
        # Define QM/MM system inputs.
        "group_part_dict": (group_part_dict := {"qm_atom": [0, 1, 2]}),
        "embedding_cutoff": (embedding_cutoff := 14.0),
        "embedding_method": (embedding_method := "analytic"),
        "qmmm_pme": (qmmm_pme := True),
        "qmmm_pme_gridnumber": (qmmm_pme_gridnumber := 30),
        "qmmm_pme_alpha": (qmmm_pme_alpha := 5.0),
        "decimal_places": (decimal_places := 6),
        "mm_only": (mm_only := False),
        "settle_dists": (settle_dists := [1, 1.632981]),
        # settle_dists = None
        "ensemble": (ensemble := "nvt"),
        "num_steps": (num_steps := 50000),
    }
    system = pdb_list[0].split("/")[2].split(".")[0]
    method = "mm" if mm_only else "qmmmpme" if qmmm_pme else "qmmm"
    name = "_".join([system, method, ensemble, basis_set, charge_type])
    # Create new path.
    if os.path.exists("./" + name + "_sim_output"):
        name += "_"
        i = 1
        while os.path.exists("./" + name + f"{i}" + "_sim_output"):
            i += 1
        name += f"{i}"
    # Instantiate QM subsystem.
    qm_subsystem = QMSubsystem(
        basis_set,
        functional,
        quadrature_spherical,
        quadrature_radial,
        qm_charge,
        qm_spin,
        charge_type=charge_type,
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
        qmmm_pme=qmmm_pme,
        qmmm_pme_gridnumber=qmmm_pme_gridnumber,
        qmmm_pme_alpha=qmmm_pme_alpha,
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
        particle_write_freq=50,
        mm_only=mm_only,
        settle_dists=settle_dists,
    )
    with open(os.path.join(f"{name}_sim_output", f"{name}.inp"), "w") as fh:
        fh.write(json.dumps(params, indent=4))
    qmmm_system.run_dynamics(num_steps)


if __name__ == "__main__":
    main()
