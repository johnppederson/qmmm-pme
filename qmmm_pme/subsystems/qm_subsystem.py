#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Psi4 interface to model the QM subsystem of the QM/MM system.
"""
import copy
import sys
import time

import scipy.optimize
import psi4
import psi4.core

sys.path.append("../")

from ..system import System
from ..utils import *
from ..integrators.units import *

class QMSubsystem(System):
    """Psi4 interface for the QM subsystem.

    This class handles all interactions with Psi4 and provides relevant
    information to the QMMMSystem.

    Parameters
    ----------
    basis_set : str
        Name of desired Psi4 basis set.
    functional : str
        Name of desired Psi4 density functional.
    quadrature_spherical : int
        Number of spherical (angular and azimuthal) points for the Psi4
        exchange-correlation functional quadrature.
    quadrature_radial : int
        Number of radial points for the Psi4exchange-correlation
        functional quadrature.
    qm_charge : int
        Charge of the QM system, in proton charge units.
    qm_spin : int
        Spin of the QM system.
    scf_type : str, default="df"
        The SCF type for the Psi4 calculation.
    qmmm_pme : bool, default=False
        Determine whether or not to use the QM/MM/PME method.
    n_threads : int, default=1
        Number of threads across which to parallelize the QM calculation.
    read_guess : bool, default=False
        Determine whether to base calculations on previous wavefunction
        objects or on the Psi4 default SAD.
    particle_groups : dict of list of list of int, optional
        Indices of particles in the system grouped by residue, or
        molecule, and further grouped by a given key.  Example keys
        include "qm_atom", "qm_drude", and "analytic".
    element_symbols : list of str, Optional, default=None
        Element symbols of all particles in the system.
    charges : NumPy Array object, Optional, default=None
        The charges of all particles in the system, in proton charge
        units.
    positions : NumPy Array object, Optional, default=None
        The positions of all particles in the system, in Angstroms.
    box : NumPy Array object, Optional, default=None
        The box vectors defining the periodic system, in Angstroms.
    """

    def __init__(
            self,
            basis_set,
            functional,
            quadrature_spherical,
            quadrature_radial,
            qm_charge,
            qm_spin,
            scf_type="df",
            qmmm_pme=False,
            charge_type="none",
            charge_method="scf",
            charge_weights="none",
            q0 = None,
            n_threads=1,
            memory="500 MB",
            read_guess=False,
            reference_energy=None,
            particle_groups=None,
            element_symbols=None,
            charges=None,
            positions=None,
            box=None,
        ):
        System.__init__(self)
        self.basis_set = basis_set
        self.functional = functional
        self.quadrature_spherical = quadrature_spherical
        self.quadrature_radial = quadrature_radial
        self.qm_charge = qm_charge
        self.qm_spin = qm_spin
        self.scf_type = scf_type
        self.charge_type = charge_type
        self.charge_method = charge_method
        self.charge_weights = charge_weights
        self.q0 = q0
        self.qmmm_pme = qmmm_pme
        self.n_threads = n_threads
        psi4.set_num_threads(n_threads)
        psi4.set_memory(memory)
        psi4.core.set_global_option("PRINT", 0)
        self.options = {
            "basis": self.basis_set,
            "dft_spherical_points": self.quadrature_spherical,
            "dft_radial_points": self.quadrature_radial,
            "scf_type": self.scf_type,
            "pme": str(self.qmmm_pme).lower(),
        }
        psi4.set_options(self.options)
        self.read_guess = read_guess
        self.wfn = None
        self.ground_state_energy = reference_energy
        self.charge_field = None
        self.particle_groups = particle_groups
        self.element_symbols = element_symbols
        self.charges = charges
        self.positions = positions
        if box:
            self.box = box

    def generate_geometry(self):
        """
        Create the geometry string to feed into the Psi4 calculation.
        """
        qm_atom_list = self._particle_groups["qm_atom"]
        qm_centroid = np.array([sum([self._positions[atom][k] 
                                for atom in qm_atom_list])
                                / len(qm_atom_list) for k in range(3)])
        if self.qm_spin > 1:
            psi4.core.set_local_option("SCF", "REFERENCE", "UKS")
        # Add MM charges in QMregion for analytic embedding.
        if "analytic" in self._particle_groups:
            embedding_list = self._particle_groups["analytic"]
            charge_field = []
            for residue in embedding_list:
                nth_centroid = [sum([self._positions[atom][k] for atom in residue])
                                / len(residue) for k in range(3)]
                new_centroid = least_mirror_vector(
                    nth_centroid,
                    qm_centroid,
                    self._box,
                )
                displacement = np.array(new_centroid) - np.array(nth_centroid)
                for atom in residue:
                    position = (self._positions[atom]
                                + displacement
                                + qm_centroid) * bohr_per_angstrom
                    charge_field.append(
                        [
                            self._charges[atom],
                            [position[0],position[1],position[2]],
                        ]
                    )
            self.charge_field = charge_field
        # Construct geometry string.
        geometrystring = """\n"""
        for atom in qm_atom_list:
            position = self._positions[atom]
            geometrystring = (geometrystring
                              + str(self._element_symbols[atom]) + " " 
                              + str(position[0]) + " " 
                              + str(position[1]) + " " 
                              + str(position[2]) + "\n")
        geometrystring = geometrystring + str(self.qm_charge) + " "
        geometrystring = geometrystring + str(self.qm_spin) + "\n"
        # Do not reorient molecule.
        geometrystring = geometrystring + "noreorient\nnocom\n"
        self.geometry = psi4.geometry(geometrystring)
    
    def generate_ground_state_energy(self, given=None):
        """
        Calculate the ground state energy of the QM subsystem.

        Parameters
        ----------
        """
        if given:
            self.ground_state_energy = given
        elif self.ground_state_energy is None:
            qm_atom_list = self._particle_groups["qm_atom"]
            if self.qm_spin > 1:
                psi4.core.set_local_option("SCF", "REFERENCE", "UKS")
            # Construct geometry string.
            geometrystring = """\n"""
            for atom in qm_atom_list:
                position = self._positions[atom]
                geometrystring = (geometrystring
                                  + str(self._element_symbols[atom]) + " "
                                  + str(position[0]) + " "
                                  + str(position[1]) + " "
                                  + str(position[2]) + "\n")
            geometrystring = geometrystring + str(self.qm_charge) + " "
            geometrystring = geometrystring + str(self.qm_spin) + "\n"
            # Do not reorient molecule.
            geometrystring = geometrystring + "noreorient\nnocom\n"
            self.geometry = psi4.geometry(geometrystring)
            opt_options = copy.deepcopy(self.options)
            opt_options["pme"] = "false"
            psi4.set_options(opt_options)
            psi4_energy = psi4.optimize(
                self.functional, 
                molecule=self.geometry,
            )
            self.ground_state_energy = psi4_energy

    def build_ref_quadrature(self):
        """
        Build a reference quadrature for the external potential grid.

        Returns
        -------
        ref_quadrature: NumPy Array object
            A reference quadrature constructed from the current geometry
            of the QM subsystem.
        """
        sup_func = psi4.driver.dft.build_superfunctional(
            self.functional,
            True,
        )[0]
        basis = psi4.core.BasisSet.build(
            self.geometry,
            "ORBITAL",
            self.basis_set,
        )
        ref_v = psi4.core.VBase.build(basis, sup_func, "RV")
        ref_v.initialize()
        ref_quadrature = ref_v.get_np_xyzw()
        return ref_quadrature

    def compute_energy(self, return_forces=True, quad_extd_pot=None, nuc_extd_pot=None, nuc_extd_grad=None):
        """
        Calculates the energy and forces for the QM atoms.

        These forces are the intramolecular forces acting on amongst the
        QM atoms and the the electrostatic forces acting on the QM atoms
        from the extended environment.

        Parameters
        ----------
        return_forces: bool, Optional, default=True
        quad_extd_pot: NumPy Array object, Optional, default=None
            The extended potential evaluated on the Psi4 quadrature grid.
        nuc_extd_pot: NumPy Array object, Optional, default=None
            The extended potential evaluated at the nuclear coordinates.
        nuc_extd_grad: NumPy Array object, Optional, default=None
            The gradient of the extended potential evaluated at the
            nuclear coordinates.

        Returns
        -------
        psi4_energy: Numpy array
            QM subsystem energy.
        psi4_forces: Numpy array
            Forces acting on QM atoms.
        """
        psi4.set_options(self.options)
        # Check for wavefunction if read_guess is True.
        if self.wfn and self.read_guess:
            self.wfn.to_file(self.wfn.get_scratch_filename(180))
            psi4.core.set_local_option("SCF", "GUESS", "READ")
        kwargs = {}
        if self.qmmm_pme:
            psi4.core.set_local_option("SCF","PME", self.qmmm_pme)
            kwargs["quad_extd_pot"] = quad_extd_pot
            kwargs["nuc_extd_pot"] = nuc_extd_pot
            kwargs["nuc_extd_grad"] = nuc_extd_grad
        if len(self.charge_field) != 0:
            kwargs["external_potentials"] = self.charge_field
        (psi4_energy, psi4_wfn) = psi4.energy(
            self.functional,
            return_wfn=True,
            molecule=self.geometry,
            **kwargs,
        )
        psi4_charges = None
        if self.charge_type == "mulliken":
            psi4.oeprop(psi4_wfn, "MULLIKEN_CHARGES")
            psi4_charges = psi4_wfn.variable(f"MULLIKEN CHARGES")
        elif self.charge_type == "mbis":
            psi4.oeprop(psi4_wfn, "MBIS_CHARGES")
            psi4_charges = np.asarray(psi4_wfn.variable(f"MBIS CHARGES")).reshape((-1))
        elif self.charge_type == "resp-p" or self.charge_type == "resp-h":
            positions = self._positions[[x for y in self._particle_groups["mm_atom"] for x in y]]
            with open("grid.dat", "w") as fh:
                for position in positions:
                    fh.write("%16.10f%16.10f%16.10f\n" % (position[0], position[1], position[2]))
            psi4.oeprop(psi4_wfn, "GRID_ESP")
            with open("grid_esp.dat", "r") as fh:
                lines = fh.readlines()
            V = []
            for line in lines:
                V.append(float(line.strip()))
            mm_positions = positions * bohr_per_angstrom
            qm_positions = self._positions[[x for x in self._particle_groups["qm_atom"]]] * bohr_per_angstrom
            D = np.zeros((len(qm_positions),len(mm_positions)))
            for i, r_i in enumerate(qm_positions):
                for j, r_j in enumerate(mm_positions):
                    D[i,j] = np.linalg.norm(r_j - r_i)**-1
            q0 = self.q0
            if q0 is None:
                q0 = np.array(self._charges)[
                    [x for x in self._particle_groups["qm_atom"]]
                ].reshape((-1,1))
            a = 0.005
            b = 0.1
            m = len(qm_positions)
            n = len(mm_positions)
            if self.charge_weights == "distance":
                w = np.sum(D**-1, axis=0).reshape((-1,1))
                print(w)
            else:
                w = np.ones((n,1))
            nth_row = np.ones((1,m))
            nth_col = -np.ones((m+1,1))
            nth_col[-1,0] = 0
            V = np.array(V).reshape((-1,1))
            # Hyperbolic
            if self.charge_type == "resp-h":
                C = np.linalg.inv((np.eye(m)*b**2)**0.5)
                A = 2*(D @ (w*D.T)) + a*C
                A = np.concatenate((A, nth_row), axis=0)
                A = np.concatenate((A, nth_col), axis=1)
                B = 2*(D @ (w*V)) + a*(C @ q0)
                B = np.concatenate((B, np.array([[0]])), axis=0)
                q = np.linalg.inv(A) @ B
                qt = q + 1
                while np.linalg.norm(q-qt) > 1e-7:
                    qt = q[0:m,:]
                    C = np.linalg.inv(((((q0-qt)*(q0-qt))@np.ones((1,m)))*np.eye(m) + np.eye(m)*b**2)**0.5)
                    A = 2*(D @ (w*D.T)) + a*C
                    A = np.concatenate((A, nth_row), axis=0)
                    A = np.concatenate((A, nth_col), axis=1)
                    B = 2*(D @ (w*V)) + a*(C @ q0)
                    B = np.concatenate((B, np.array([[0]])), axis=0)
                    q = np.linalg.inv(A) @ B
                    q = q[0:m,:]
            # Parabolic
            if self.charge_type == "resp-p":
                A = 2*(D @ (w*D.T)) + 2*a*np.eye(m)
                A = np.concatenate((A, nth_row), axis=0)
                A = np.concatenate((A, nth_col), axis=1)
                b = 2*(D @ (w*V)) + 2*a*q0
                b = np.concatenate((b, np.array([[0]])), axis=0)
                q = np.linalg.inv(A) @ b
                q = q[0:m,:]
            psi4_charges = q.flatten()
        self.wfn = psi4_wfn
        psi4.core.clean()
        psi4_energy = (psi4_energy - self.ground_state_energy) * kJmol_per_eh
        if return_forces:
            psi4_forces = psi4.gradient(
                self.functional,
                ref_wfn=psi4_wfn,
                **kwargs
            )
            psi4_forces = (-np.asarray(psi4_forces)
                           * kJmol_per_eh
                           * bohr_per_angstrom)
            return psi4_energy, psi4_forces, psi4_charges
        else:
            return psi4_energy, psi4_charges
    
    def optimize_geometry(self):
        """
        Calculates the energy and forces for the QM atoms.

        These forces are the intramolecular forces acting on amongst the
        QM atoms and the the electrostatic forces acting on the QM atoms
        from the extended environment.

        Returns
        -------
        psi4_energy: Numpy array
            QM subsystem energy.
        psi4_forces: Numpy array
            Forces acting on QM atoms.
        """
        psi4.set_options(self.options)
        kwargs = {}
        if len(self.charge_field) != 0:
            kwargs["external_potentials"] = self.charge_field
        (psi4_energy, psi4_wfn) = psi4.optimize(
            self.functional,
            return_wfn=True,
            molecule=self.geometry,
            **kwargs,
        )
        psi4_coords = np.asarray(psi4_wfn.molecule().geometry()) / bohr_per_angstrom
        return psi4_coords

    @property
    def box(self):
        """
        The box vectors defining the periodic system.
        """
        return self._box
    
    @box.setter
    def box(self, box):
        self._box = box
        self._bohr_box = [[k*bohr_per_angstrom for k in vector]
                          for vector in box]
