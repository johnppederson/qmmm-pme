#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASE Calculator to combine QM and MM forces and energies.
"""
import sys
import time
import numpy as np
from .subsystems import *
from .integrators.units import *
from .integrators.settle import *
from .utils import *


class QMMMHamiltonian:
    """ 
    ASE Calculator.

    Modeled after SchNetPack calculator.

    Parameters
    ----------
    openmm_interface: OpenMMInterface object
        OpenMMInterface object containing all the needed info for
        getting forces from OpenMM.
    """

    def __init__(
            self,
            qmmm_system,
        ):
        self.qmmm_system = qmmm_system
        self.frame = qmmm_system.frame
        self.timestep = qmmm_system.integrator.timestep

    def calculate(self):
        """
        Obtains the total energy and forces using the above interfaces.
        """
        qm_atom_list = self.qmmm_system.particle_groups["qm_atom"]
        new_positions = self.wrap(self.qmmm_system.positions)
        self.qmmm_system.positions = new_positions
        # Calculate MM contributions.
        openmm_energy, openmm_forces, openmm_components = self.qmmm_system.mm_subsystem.compute_energy()
        # Report Kinetic Energy.
        energy = {"Total Energy": None, "-":{"Kinetic Energy": None, "Potential Energy": None}}
        psi4_energy = 0.
        recip_energy = 0.
        real_energy = 0.
        if not self.qmmm_system.mm_only:
            ljones_energy, ljones_forces = self.qmmm_system.lj_subsystem.compute_energy(return_components=False)
            # Perform embedding.
            if self.qmmm_system.embedding_method == "hybrid":
                if len(self.qmmm_system.embedding_cutoff) != 2:
                    print(
                        """Hybrid embedding requires two embedding cutoffs!
                        Please pass a tuple or list of two embedding cutoffs
                        when instantiating the QMMMSystem object with hybrid
                        embedding."""
                    )
                    sys.exit()
                embedding_list_low = self.generate_embedding_list(
                    threshold=min(self.qmmm_system.embedding_cutoff),
                )
                self.qmmm_system.particle_groups["analytic"] = embedding_list_low
                embedding_list_high = self.generate_embedding_list(
                    threshold=max(self.qmmm_system.embedding_cutoff),
                )
                embedding_list_real = list(
                    set(embedding_list_high)-set(embedding_list_low)
                )
                self.qmmm_system.particle_groups["realspace"] = embedding_list_real
            elif self.qmmm_system.embedding_method != "none":
                embedding_list = self.generate_embedding_list()
                self.qmmm_system.particle_groups[self.qmmm_system.embedding_method] = embedding_list
            # Generate Psi4 geometry string.
            self.qmmm_system.qm_subsystem.generate_geometry()
            # Evaluate extended potential if necessary.
            arguments = {}
            if self.qmmm_system.qmmm_pme:
                potential_grid = self.qmmm_system.mm_subsystem.build_potential_grid()
                ref_quadrature = self.qmmm_system.qm_subsystem.build_ref_quadrature()
                (quad_extd_pot, nuc_extd_pot, nuc_extd_grad, recip_energy) = self.qmmm_system.pbc_subsystem.build_extd_pot(
                    ref_quadrature,
                    potential_grid,
                    return_grad=True,
                )
                arguments["nuc_extd_grad"] = nuc_extd_grad
                arguments["quad_extd_pot"] = quad_extd_pot
                arguments["nuc_extd_pot"] = nuc_extd_pot
            psi4_energy, psi4_forces, psi4_charges = self.qmmm_system.qm_subsystem.compute_energy(**arguments)
            if self.qmmm_system.qm_subsystem.charge_type != "none" and self.qmmm_system.qm_subsystem.charge_method == "scf":
                energy_tol = 10e-3
                charge_tol = 10e-5
                energy_err = 1
                charge_err = 1
                while energy_err > energy_tol and charge_err > charge_tol:
                    self.qmmm_system.mm_subsystem.update_charges(psi4_charges)
                    for i, q in zip(qm_atom_list, psi4_charges):
                        self.qmmm_system.charges[i] = q
                    openmm_temp, openmm_forces, openmm_components = self.qmmm_system.mm_subsystem.compute_energy()
                    if self.qmmm_system.qmmm_pme:
                        potential_grid = self.qmmm_system.mm_subsystem.build_potential_grid()
                        ref_quadrature = self.qmmm_system.qm_subsystem.build_ref_quadrature()
                        (quad_extd_pot, nuc_extd_pot, nuc_extd_grad, recip_energy) = self.qmmm_system.pbc_subsystem.build_extd_pot(
                            ref_quadrature,
                            potential_grid,
                            return_grad=True,
                        )
                        arguments["nuc_extd_grad"] = nuc_extd_grad
                        arguments["quad_extd_pot"] = quad_extd_pot
                        arguments["nuc_extd_pot"] = nuc_extd_pot
                    psi4_temp, psi4_forces, charges_temp = self.qmmm_system.qm_subsystem.compute_energy(**arguments)
                    energy_err = (((psi4_temp + openmm_temp) - (psi4_energy + openmm_energy))**2)**0.5
                    charge_err = (sum([(q-c)**2 for q, c in zip(psi4_charges, charges_temp)]))**0.5
                    psi4_energy = psi4_temp
                    openmm_energy = openmm_temp
                    psi4_charges = charges_temp
                self.qmmm_system.mm_subsystem.update_charges(psi4_charges)
                for i, q in zip(qm_atom_list, psi4_charges):
                    self.qmmm_system.charges[i] = q
            if self.qmmm_system.qm_subsystem.charge_type != "none" and self.qmmm_system.qm_subsystem.charge_method == "post":
                self.qmmm_system.mm_subsystem.update_charges(psi4_charges)
                for i, q in zip(qm_atom_list, psi4_charges):
                    self.qmmm_system.charges[i] = q
                openmm_energy, openmm_forces, openmm_components = self.qmmm_system.mm_subsystem.compute_energy()
            # Separate Psi4 forces into QM atom and embedding atom contributions.
            qm_forces = psi4_forces[0:len(qm_atom_list),:]
            ae_forces = psi4_forces[len(qm_atom_list):,:]
            # Add Psi4 electrostatic forces and energy onto OpenMM forces
            # and energy for QM atoms.
            for i, qm_force in zip(qm_atom_list, qm_forces):
                for j in range(3):
                    openmm_forces[i,j] = ljones_forces[i,j] + qm_force[j]
            # Remove double-counting from embedding forces and energy.
            j = 0
            qm_centroid = [sum([self.qmmm_system.positions[i][j] for i in qm_atom_list])
                           / len(qm_atom_list) for j in range(3)]
            for residue in self.qmmm_system.particle_groups["analytic"]:
                nth_centroid = [sum([self.qmmm_system.positions[atom][k] for atom in residue])
                                / len(residue) for k in range(3)]
                r_vector = least_mirror_vector(
                    nth_centroid,
                    qm_centroid,
                    self.qmmm_system.box,
                )
                displacement = np.array(r_vector) + np.array(qm_centroid) - np.array(nth_centroid)
                for atom in residue:
                    co_force = [0,0,0]
                    for i in qm_atom_list:
                        x = (displacement[0] + self.qmmm_system.positions[atom][0] - self.qmmm_system.positions[i][0]
                            ) * bohr_per_angstrom
                        y = (displacement[1] + self.qmmm_system.positions[atom][1] - self.qmmm_system.positions[i][1]
                            ) * bohr_per_angstrom
                        z = (displacement[2] + self.qmmm_system.positions[atom][2] - self.qmmm_system.positions[i][2]
                            ) * bohr_per_angstrom
                        dr = (x**2 + y**2 + z**2)**0.5
                        q_prod = self.qmmm_system.charges[i] * self.qmmm_system.charges[atom]
                        co_force[0] += (angstrom_per_bohr * kJmol_per_eh
                                        * x * q_prod * dr**-3)
                        co_force[1] += (angstrom_per_bohr * kJmol_per_eh
                                        * y * q_prod * dr**-3)
                        co_force[2] += (angstrom_per_bohr * kJmol_per_eh
                                        * z * q_prod * dr**-3)
                        real_energy -= kJmol_per_eh * q_prod * dr**-1
                    for i in range(3):
                        openmm_forces[atom,i] += ae_forces[j][i]
                        openmm_forces[atom,i] -= co_force[i]
                    j += 1
        # Calculate total energy and forces.
        total_energy = openmm_energy + psi4_energy + real_energy + recip_energy
        total_forces = openmm_forces
        self.qmmm_system.forces = total_forces
        if self.frame == 0:
            with open("positions.csv","w") as fh:
                for force in self.qmmm_system.positions:
                    fh.write(f"{force[0]},{force[1]},{force[2]}\n")
            sys.exit()
        energy["-"]["Kinetic Energy"] = self.compute_kinetic_energy()
        # Log energy.
        energy["Total Energy"] = total_energy + energy["-"]["Kinetic Energy"]
        energy["-"]["Potential Energy"] = total_energy
        potentials = {
            "OpenMM Energy": openmm_energy,
            "-": openmm_components,
            "Psi4 Energy": psi4_energy,
            "Real-space Correction Energy": real_energy,
            "Reciprocal-space Correction Energy": recip_energy,
        }
        energy["-"]["-"] = potentials
        self.qmmm_system.energy = energy
        self.qmmm_system.logger.report()
        # Store results and advance frame.
        self.frame += 1
        self.qmmm_system.frame += 1

    def wrap(self, position_array):
        """
        Atoms are wrapped to stay inside of the periodic box. 

        This function ensures molecules are not broken up by a periodic
        boundary, as OpenMM electrostatics will be incorrect if all
        atoms in a molecule are not on the same side of the periodic
        box.  Assumes isotropic box.

        Parameters
        ----------
        position_array: NumPy array
            Array containing 3*N coordinates, where N is the number of
            atoms.

        Returns
        -------
        shift_array: NumPy array
            array containing the shifted positions
        """
        box = self.qmmm_system.box
        inv_box = np.linalg.inv(box)
        # Loop through the molecules in the residue list, which is a
        # list containing atom indices.  Molecules are wrapped according
        # to the position of the first atom listed for the residue.
        residues = np.array(self.qmmm_system.particle_groups["all"])
        res_pos = position_array[residues,:]
        res_cen = np.average(res_pos, axis=1)
        inv_cen = res_cen @ inv_box
        mask = np.floor(inv_cen)
        diff = -mask @ box
        temp_pos = res_pos + diff[:,np.newaxis,:]
        new_positions = temp_pos.reshape(position_array.shape)
        #for residue in self.qmmm_system.particle_groups["all"]:
        #    residue_positions = position_array[residue]
        #    residue_centroid = [
        #        sum([position[k] for position in residue_positions])
        #        / len(residue) for k in range(3)
        #    ]
        #    inv_centroid = residue_centroid @ inv_box
        #    mask = np.floor(inv_centroid)
        #    diff = -mask @ box
        #    for atom in residue:
        #        new_positions[atom] = position_array[atom] + diff
        return new_positions

    def generate_embedding_list(self, threshold=None):
        """
        Create the embedding list for the current state of the system.

        The distances from the QM atoms are computed using the centroid
        of the non-QM molecule from the centroid of the QM atoms.  The
        legacy method involves computing distances using the first atom
        position from the non-QM molecule instead.

        Parameters
        ----------
        threshold: float, Optional, default=None
            The threshold distance within which to include particles in
            the embedding list.  If no threshold is specified, the
            embedding_cutoff attribute is taken to be the threshold.

        Returns
        -------
        embedding_list: list of tuple of int
            The list of embedded particles, arranged by residue in
            tuples.
        """
        if not threshold:
            threshold = self.qmmm_system.embedding_cutoff
        positions = self.qmmm_system.positions
        qm_atom_list = self.qmmm_system.particle_groups["qm_atom"]
        qm_centroid = [sum([positions[atom][k] for atom in qm_atom_list])
                       / len(qm_atom_list) for k in range(3)]
        embedding_list = []
        for residue in self.qmmm_system.particle_groups["all"]:
            nth_centroid = [sum([positions[atom][k] for atom in residue])
                            / len(residue) for k in range(3)]
            r_vector = least_mirror_vector(
                nth_centroid,
                qm_centroid,
                self.qmmm_system.box,
            )
            distance = sum([x**2 for x in r_vector])**0.5
            is_qm = any(atom in residue for atom in qm_atom_list)
            if distance < threshold and not is_qm:
                    embedding_list.append(tuple(residue))
        return embedding_list

    def compute_kinetic_energy(self):
        """
        """
        mass = self.qmmm_system.masses.reshape(-1,1)
        vels = self.qmmm_system.velocities + 0.5*self.timestep*self.qmmm_system.forces*(10**-4)/mass
        if self.qmmm_system.settle:
            vels = settle_vel(self.qmmm_system.positions, vels, self.qmmm_system.integrator.residues, mass)
        kinetic_energy = np.sum((vels)**2 * mass / 2) * (10**4)
        return kinetic_energy

