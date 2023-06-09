#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM/MM system to propogate dynamic simulations.

Manages the interactions between the QM and MM subsytems through their
respective interface objects.
"""
import os
import sys
import time

import numpy as np
import simtk.unit

from .pbc_subsystem import *
from .mm_subsystem import *
from .qm_subsystem import *
from .logger import *
from ..qmmm_hamiltonian import *
from ..system import *
from ..integrators.units import *
from ..integrators.velocity_distribution import *
from ..integrators.velocity_verlet import *
from ..integrators.langevin import *
from ..integrators.settle import *


class QMMMSystem(System):
    """
    Sets up and runs the QM/MM/MD simulation.  
    
    Serves as an interface to OpenMM, Psi4 and ASE.  Based off of the
    ASE_MD class from SchNetPack.

    Parameters
    ----------
    atoms: str
        Location of input structure to create an ASE Atoms object.
    tmp: str
        Location for tmp directory.
    openmm_interface: OpenMMInterface object
        A pre-existing OpenMM interface with which the QM/MM system may
        communicate.
    psi4_interface: Psi4Interface object
        A pre-existing Psi4 interface with which the QM/MM system may
        communicate.
    qm_atoms_list: list of int
        List of atom indices representing the QM subsystem under
        investigation.
    embedding_cutoff: float
        Cutoff for analytic charge embedding, in Angstroms.
    rewrite_log: bool, Optional, default=True
        Determine whether or not an existing log file gets overwritten.
    """

    for attr in dir(System):
        if type(eval("System." + attr)) == property:
            exec(attr + " = System." + attr)

    def __init__(
            self,
            name,
            qm_subsystem,
            mm_subsystem,
            particle_groups,
            embedding_cutoff=0,
            embedding_method="none",
            ensemble="nve",
            qmmm_pme=False,
            qmmm_pme_gridnumber=100,
            qmmm_pme_alpha=5.0,
            rewrite_log=True,
            decimal_places=1,
            energy_log=True,
            report_charges=True,
            report_temperature=False,
            energy_verbose=3,
            energy_write_freq=1,
            particle_log=True,
            particle_write_freq=10,
            report_positions=True,
            report_velocities=False,
            report_forces=False,
            mm_only=False,
            settle_dists=None,
        ):
        System.__init__(self)
        # Set working directory.
        self.name = name
        if not os.path.isdir(f"{name}_sim_output"):
            os.makedirs(f"{name}_sim_output")
        # Set up output generation.
        self.logger = Logger(
            self.name,
            sum([len(res) for res in mm_subsystem.particle_groups["all"]]),
            mm_subsystem.box,
            rewrite_log=rewrite_log,
            decimal_places=decimal_places,
            energy_log=energy_log,
            energy_verbose=energy_verbose,
            energy_write_freq=energy_write_freq,
            particle_log=particle_log,
            particle_write_freq=particle_write_freq,
            include_positions=report_positions,
            include_velocities=report_velocities,
            include_forces=report_forces,
            timestep=mm_subsystem.timestep._value*1000,
        )
        # Collect subsystems.
        self.qm_subsystem = qm_subsystem
        self.mm_subsystem = mm_subsystem
        self.subsystems = [
            self.qm_subsystem,
            self.mm_subsystem,
            self.logger,
        ]
        # Collect essential information and share with subsystems.
        residue_list = self.mm_subsystem.particle_groups["all"]
        particle_groups["all"] = residue_list
        particle_groups["mm_atom"] = []
        for residue in residue_list:
            if all(atom in particle_groups["qm_atom"] for atom in residue):
                particle_groups["qm_drude"] = list(
                    set(residue)-set(particle_groups["qm_atom"]),
                )
            else:
                particle_groups["mm_atom"].append(residue)
        self.particle_groups = particle_groups
        self.element_symbols = self.mm_subsystem.element_symbols
        self.masses = self.mm_subsystem.masses
        self.charges = self.mm_subsystem.charges
        self.positions = self.mm_subsystem.positions
        self.forces = np.zeros(self.positions.shape)
        self.box = self.mm_subsystem.box
        # Set up remaining parts of subsystems.
        if mm_only:
            self.mm_subsystem.build_no_exclusions()
        else:
            self.mm_subsystem.build_qm_exclusions()
            self.qm_subsystem.generate_ground_state_energy()
            self.lj_subsystem = MMSubsystem(
                *self.mm_subsystem.args,
                "CPU",
                **self.mm_subsystem.kwargs,
            )
            self.subsystems.append(self.lj_subsystem)
            self.lj_subsystem.particle_groups = particle_groups
            self.lj_subsystem.build_lj_exclusions()
        self.mm_only = mm_only
        # Default embedding settings.
        self.embedding_cutoff = embedding_cutoff
        self.embedding_method = embedding_method
        self.qmmm_pme = qmmm_pme
        self.qmmm_pme_gridnumber = qmmm_pme_gridnumber
        self.qmmm_pme_alpha = qmmm_pme_alpha
        # Set up PME-related subsystems.
        if self.qmmm_pme:
            supported_platforms = ["Reference", "CPU"]
            # Ensure that there is an MM subsystem which can provide the
            # external potential grid.
            if self.mm_subsystem.platform.getName() not in supported_platforms:
                print("""Given OpenMM Platform is not ocmpatible with
                      QM/MM/PME.""")
                sys.exit()
            self.pbc_subsystem = PBCSubsystem(
                qmmm_pme_gridnumber = self.qmmm_pme_gridnumber,
                qmmm_pme_alpha = self.qmmm_pme_alpha,
                particle_groups = self._particle_groups,
                charges = self._charges,
                positions = self._positions,
                box = self._box,
                n_threads = self.qm_subsystem.n_threads,
            )
            self.subsystems.append(self.pbc_subsystem)
        #self.velocities = np.zeros_like(self._positions)
        residues = None
        self.settle = False
        self.velocities = MaxwellBoltzmann(self.mm_subsystem.temperature._value, self.masses)
        if settle_dists:
            residues = self.particle_groups["all"] if mm_only else self.particle_groups["mm_atom"]
            self.settle = True
            self.positions = settle(self._positions, self._positions, residues, self.masses, dists=settle_dists)
            self.velocities = settle_vel(self._positions, self._velocities, residues, self.masses)
        if ensemble.lower() == "nve":
            self.integrator = VelocityVerlet(
                self.mm_subsystem.timestep._value*1000,
                residues=residues,
                settle_dists=settle_dists
            )
        elif ensemble.lower() == "nvt":
            self.integrator = Langevin(
                self.mm_subsystem.timestep._value*1000,
                self.mm_subsystem.temperature._value,
                self.mm_subsystem.friction._value/1000,
                residues=residues,
                settle_dists=settle_dists
            )
        #
        #with open("./velocities.csv","w") as fh:
        #    for vel in self.velocities:
        #        fh.write(f"{vel[0]},{vel[1]},{vel[2]}\n")
        #
        self.calculator = QMMMHamiltonian(self)

    def write_atoms(self, name, ftype="xyz", append=False):
        """
        Write out current system structure.

        Parameters
        ----------
        name: str
            Name of the output file.
        ftype: str, Optional, defalt="xyz"
            Determines output file format.
        append: bool, Optional, default=False
            Determine whether to append to existing output file or not.
        """
        
        #
        path = os.path.join(self.name, f"{name}.{ftype}")
        ase.io.write(path, self.atoms, format=ftype, append=append)
        #

    def run_dynamics(self, steps):
        """
        Run simulation.

        Parameters
        ----------
        steps : int
            Number of MD steps.
        """
        for i in range(steps):
            self.calculator.calculate()
            self._positions, self.velocities = self.integrator.integrate(
                self.masses,
                self.positions,
                self.velocities,
                self.forces,
            )
        self.logger.terminate()

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer.

        Parameters
        ----------
        fmax: float, Optional, default=1.0e-2
            Maximum residual force change.
        steps: int
            Maximum number of steps.
        """
        name = "optimization"
        
        #
        optimize_file = os.path.join(self.name, name)
        optimizer = ase.optimize.QuasiNewton(self.atoms,
                                             trajectory="%s.traj" % optimize_file,
                                             restart="%s.pkl" % optimize_file,)
        optimizer.run(fmax, steps)
        self.write_atoms(name)
        #

    @particle_groups.setter
    def particle_groups(self, particle_groups):
        self._particle_groups = particle_groups
        for subsystem in self.subsystems:
            subsystem.particle_groups = particle_groups

    @element_symbols.setter
    def element_symbols(self, element_symbols):
        self._element_symbols = element_symbols
        for subsystem in self.subsystems:
            subsystem.element_symbols = element_symbols

    @masses.setter
    def masses(self, masses):
        self._masses = masses
        for subsystem in self.subsystems:
            subsystem.masses = masses

    @charges.setter
    def charges(self, charges):
        self._charges = charges
        for subsystem in self.subsystems:
            subsystem.charges = charges

    @positions.setter
    def positions(self, positions):
        self._positions = positions
        for subsystem in self.subsystems:
            subsystem.positions = positions
    
    @velocities.setter
    def velocities(self, velocities):
        self._velocities = velocities
        for subsystem in self.subsystems:
            subsystem.velocities = velocities
    
    @forces.setter
    def forces(self, forces):
        self._forces = forces
        for subsystem in self.subsystems:
            subsystem.forces = forces
    
    @energy.setter
    def energy(self, energy):
        self._energy = energy
        for subsystem in self.subsystems:
            subsystem.energy = energy

    @box.setter
    def box(self, box):
        self._box = box
        self._bohr_box = [[k*bohr_per_angstrom for k in vector]
                          for vector in box]
        for subsystem in self.subsystems:
            subsystem.box = box

    @frame.setter
    def frame(self, frame):
        self._frame = frame
        for subsystem in self.subsystems:
            subsystem.frame = frame
