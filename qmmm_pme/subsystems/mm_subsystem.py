#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMM interface to model the MM subsystem of the QM/MM system.
"""
import copy
import sys
import time

sys.path.append("../")

import numpy as np
import openmm.app
import openmm
import simtk.unit

from ..shared import *
from ..system import System
from ..utils import *
from ..integrators.units import *


class MMSubsystem(MM_base, System):
    """
    Child class of the MM base and System classes for QM/MM simulations.

    Must use compiled version of custom OpenMM to allow for QM/MM/PME.

    Parameters
    ----------
    pdb_list: list of str
        The directories containing the PDB files which define the system
        geometry.
    residue_xml_list: list of str
        The directories containing the XML files which define the system
        topology.
    ff_xml_list: list of str
        The directories containing the XML files which define the system
        interactions in the MM Hamiltonian.
    platform: str
        The platform for OpenMM.  These include "Reference", "CPU",
        "CUDA", and "OpenCL".
    qmmm_pme: bool, Optional, default=False
        Determine whether or not to include extended electrostatics
        using the QM/MM/PME method.
    qmmm_pme_gridnumber: int, Optional, default=100
        The number of grid points to use along each of the box vectors
        of the principal cell during the PME procedure.
    qmmm_pme_alpha: float, Optional, default=5.0
        The Gaussian width of the smeared point charges in the Ewald
        summation scheme, in inverse nanometers.  See OpenMM's
        documentation for further discussion.
    kwargs: dict
        Additional arguments to send to the MM base class.
    """

    def __init__(
            self, 
            pdb_list,
            residue_xml_list,
            ff_xml_list,
            platform,
            qmmm_pme = False,
            qmmm_pme_gridnumber = 60,
            qmmm_pme_alpha = 5.0,
            **kwargs,
        ):
        MM_base.__init__(
            self,
            pdb_list,
            residue_xml_list,
            ff_xml_list,
            **kwargs,
        )
        System.__init__(self)
        self.args = [pdb_list, residue_xml_list, ff_xml_list]
        self.kwargs = kwargs
        self.qmmm_pme = qmmm_pme
        self.qmmm_pme_gridnumber = qmmm_pme_gridnumber
        self.qmmm_pme_alpha = qmmm_pme_alpha
        # Sets the PME parameters in OpenMM.  The grid size is important
        # for the accuracy of the external potental in the DFT 
        # quadrature, since this is interpolated from the PME grid.
        #
        #if self.qmmm_pme:
        self.nbondedForce.setPMEParameters(
            self.qmmm_pme_alpha,
            self.qmmm_pme_gridnumber,
            self.qmmm_pme_gridnumber,
            self.qmmm_pme_gridnumber,
        )
        properties = {}
        if self.qmmm_pme:
            properties["ReferenceVextGrid"] = "true"
        if platform == "Reference":
            self.platform = openmm.Platform.getPlatformByName("Reference")
        elif platform == "CPU":
            self.platform = openmm.Platform.getPlatformByName("CPU")
        elif platform == "OpenCL":
            self.platform = openmm.Platform.getPlatformByName("OpenCL")
            if self.qmmm_pme:
                print(
                    """Only Reference and CPU OpenMM platforms are
                    currently supported for the QM/MM/PME
                    implementation."""
                )
                sys.exit()
        elif platform == "CUDA":
            self.platform = openmm.Platform.getPlatformByName("CUDA")
            properties["Precision"] = "mixed"
            if self.qmmm_pme:
                print(
                    """Only Reference and CPU OpenMM platforms are
                    currently supported for the QM/MM/PME
                    implementation."""
                )
                sys.exit()
        else:
            print("Platform '{}' is unrecognized.".format(platform))
            sys.exit()
        self.properties = properties
        particle_groups = {"all": []}
        element_symbols = []
        charges = []
        masses = []
        sigmas = []
        epsilons = []
        for residue in self.pdb.topology.residues():
            part_list = []
            for part in residue._atoms:
                masses.append(
                    self.system.getParticleMass(part.index) / simtk.unit.dalton,
                )
                part_list.append(part.index)
                element_symbols.append(part.element.symbol)
                (q, sig, eps) = self.nbondedForce.getParticleParameters(
                    part.index
                )
                charges.append(q._value)
                sigmas.append(sig._value)
                epsilons.append(eps._value)
            particle_groups["all"].append(part_list)
        self._particle_groups = particle_groups
        self._element_symbols = element_symbols
        self._masses = np.array(masses)
        self._charges = np.array(charges)
        self.sigmas = sigmas
        self.epsilons = epsilons
        box_temp = []
        for vector in self.modeller.topology.getPeriodicBoxVectors():
            box_temp.append(vector / simtk.unit.angstrom)
        self._box = np.array(box_temp)
        pos_temp = []
        for vector in self.modeller.getPositions():
            pos_temp.append(vector / simtk.unit.angstrom)
        self._positions = np.array(pos_temp)
        self.lj = False
        self.simulation = None

    def build_no_exclusions(self):
        """
        """
        # only save force objects internally for system_mm , we don't need to store force objects for system_lj
        # which will only have a customnonbondedforce after modification
        self.nonbonded_force = [f for f in [self.system.getForce(i)
                                for i in range(self.system.getNumForces())]
                                if type(f) == openmm.NonbondedForce][0]
        self.custom_nonbonded_force = [f for f in [self.system.getForce(i)
                                       for i in range(self.system.getNumForces())]
                                       if type(f) == openmm.CustomNonbondedForce]
        if len(self.custom_nonbonded_force) != 0:
           # Crash if CustomNonbondedForce exists, because
           # setup_system_lj_forces cannot handle this.
           print("""CustomNonbondedForce not currently supported.""")
           sys.exit()
        # Set long-range interaction method. It may be fine to hard code
        # these settings in as these should be used in anything but test cases.
        self.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        #self.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        if len(self.custom_nonbonded_force) != 0:
            self.custom_nonbonded_force.setNonbondedMethod(
                min(
                    self.nonbonded_force.getNonbondedMethod(),
                    openmm.NonbondedForce.CutoffPeriodic,
                )
            )
        # Set force groups for system_mm
        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i)
        self.simulation = simtk.openmm.app.Simulation(
            self.modeller.topology,
            self.system,
            copy.deepcopy(self.integrator),
            self.platform,
            self.properties,
        )
        self.simulation.context.setPositions(self.modeller.positions)
    
    def build_qm_exclusions(self):
        """
        """
        # only save force objects internally for system_mm , we don't need to store force objects for system_lj
        # which will only have a customnonbondedforce after modification
        self.nonbonded_force = [f for f in [self.system.getForce(i)
                                for i in range(self.system.getNumForces())]
                                if type(f) == openmm.NonbondedForce][0]
        self.custom_nonbonded_force = [f for f in [self.system.getForce(i)
                                       for i in range(self.system.getNumForces())]
                                       if type(f) == openmm.CustomNonbondedForce]
        self.harmonic_bond_force = [f for f in [self.system.getForce(i)
                                    for i in range(self.system.getNumForces())]
                                    if type(f) == openmm.HarmonicBondForce][0]
        self.harmonic_angle_force = [f for f in [self.system.getForce(i)
                                     for i in range(self.system.getNumForces())]
                                     if type(f) == openmm.HarmonicAngleForce][0]
        if len(self.custom_nonbonded_force) != 0:
           # Crash if CustomNonbondedForce exists, because
           # setup_system_lj_forces cannot handle this.
           print("""CustomNonbondedForce not currently supported.""")
           sys.exit()
        qm_atom_list = self._particle_groups["qm_atom"]
        # Remove double-counted intramolecular interactions for QM subsystem.
        for i in range(self.harmonic_bond_force.getNumBonds()):
            p1, p2, r0, k = self.harmonic_bond_force.getBondParameters(i)
            if p1 in qm_atom_list or p2 in qm_atom_list:
                k = simtk.unit.Quantity(0, unit=k.unit)
                self.harmonic_bond_force.setBondParameters(i, p1, p2, r0, k)
        for i in range(self.harmonic_angle_force.getNumAngles()):
            p1, p2, p3, r0, k = self.harmonic_angle_force.getAngleParameters(i)
            if p1 in qm_atom_list or p2 in qm_atom_list or p3 in qm_atom_list:
                k = simtk.unit.Quantity(0, unit=k.unit)
                self.harmonic_angle_force.setAngleParameters(i, p1, p2, p3, r0, k)
        # Set long-range interaction method. It may be fine to hard code
        # these settings in as these should be used in anything but test cases.
        self.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        #self.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        if len(self.custom_nonbonded_force) != 0:
            self.custom_nonbonded_force.setNonbondedMethod(
                min(
                    self.nonbonded_force.getNonbondedMethod(),
                    openmm.NonbondedForce.CutoffPeriodic,
                )
            )
        # Set force groups for system_mm
        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i)
        self.simulation = simtk.openmm.app.Simulation(
            self.modeller.topology,
            self.system,
            copy.deepcopy(self.integrator),
            self.platform,
            self.properties,
        )
        self.simulation.context.setPositions(self.modeller.positions)

    def build_lj_exclusions(self):
        """
        this sets up the force classes for QM/MM mechanical embedding,
        which is dowithin system_lj

        essentially, all irrelevant force classes are removed from the system
        object, and the mechanical embedding is done by creating a
        customnonbonded force with interaction groups between QM/MM
        """
        self.lj = True
        self.nonbonded_force = [f for f in [self.system.getForce(i)
                                for i in range(self.system.getNumForces())]
                                if type(f) == openmm.NonbondedForce][0]
        # Interaction groups require atom indices to be input as sets.
        qm_atom_set = set(self._particle_groups["qm_atom"])
        mm_atom_set = set([part for residue in self.particle_groups["mm_atom"] for part in residue])
        # Create a new CustomNonbondedForce for the mechanical embedding.
        custom_nonbonded_force = openmm.CustomNonbondedForce(
            """4*epsilon*((sigma/r)^12-(sigma/r)^6);
             sigma=0.5*(sigma1+sigma2);
             epsilon=sqrt(epsilon1*epsilon2)""")
        custom_nonbonded_force.addPerParticleParameter('epsilon')
        custom_nonbonded_force.addPerParticleParameter('sigma')
        custom_nonbonded_force.setNonbondedMethod(
            min(
                self.nonbonded_force.getNonbondedMethod(),
                openmm.NonbondedForce.CutoffPeriodic,
            )
        )
        # add particles with LJ parameters
        for residue in self.particle_groups["all"]:
            for part in residue:
                custom_nonbonded_force.addParticle(
                    [self.epsilons[part], self.sigmas[part]]
                )
        # Mechanical embedding:  Interaction only between QM and MM atoms
        custom_nonbonded_force.addInteractionGroup(qm_atom_set, mm_atom_set)
        # Remove all forces from system.
        n_forces = self.system.getNumForces()
        for i in range(n_forces):
            self.system.removeForce(0)
        # Add force to system.
        self.system.addForce(custom_nonbonded_force)
        # Set force groups for system_lj
        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i) 
        self.simulation = simtk.openmm.app.Simulation(
            self.modeller.topology,
            self.system,
            copy.deepcopy(self.integrator),
            self.platform,
            self.properties,
        )
        self.simulation.context.setPositions(self.modeller.positions)

    def build_potential_grid(self):
        """
        Return the PME external potential grid.

        Returns
        -------
        potential_grid: List of float
            The PME external potential grid which OpenMM uses to
            calculate electrostatic interactions, in 
            kJ/mol/proton charge.
        """
        return self.potential_grid

    def write_pdb(self, name):
        """
        Write the current state of the subsystem to a PDB file.

        Parameters
        ----------
        name: str
            Name of output file.
        """
        state = self.simulation.context.getState(
            getEnergy=False,
            getForces=False,
            getVelocities=False,
            getPositions=True,
            enforcePeriodicBox=True,
        )
        positions = state.getPositions()
        self.simulation.topology.setPeriodicBoxVectors(
            state.getPeriodicBoxVectors(),
        )
        simtk.openmm.app.PDBFile.writeFile(
            self.simulation.topology,
            positions,
            open(name + '.pdb','w'),
        )
    
    def update_charges(self, charges):
        """
        Write the current state of the subsystem to a PDB file.

        Parameters
        ----------
        charges: str
            Name of output file.
        """
        for i, q in zip(self.particle_groups["qm_atom"], charges):
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(i)
            self.nonbonded_force.setParticleParameters(i, q, sigma, epsilon)
        self.nonbonded_force.updateParametersInContext(self.simulation.context)

    def compute_energy(self, return_forces=True, return_components=True):
        """
        Calculates the MM subsubsystem energy and forces.

        Parameters
        ----------
        return_forces: bool, Optional, default=True
            Whether or not to calculate forces for the subsystem.

        Returns
        -------
        openmm_energy: Numpy array
            Subsubsystem energy.
        openmm_forces: Numpy array
            Forces acting on atoms in the subsystem.
        """
        # Get energy and forces from the state.
        if self.qmmm_pme:
            state = self.simulation.context.getState(
                getEnergy=True,
                getForces=True,
                getPositions=True,
                getVext_grids=True,
            )
            potential_grid = state.getVext_grid()
            self.potential_grid = potential_grid
        else:
            state = self.simulation.context.getState(
                getEnergy=True,
                getForces=True,
                getPositions=True,
            )
        openmm_energy = state.getPotentialEnergy()/simtk.unit.kilojoule_per_mole
        if return_components:
            components = {}
            for i in range(self.system.getNumForces()):
                f = self.system.getForce(i)
                key = str(type(f)).split("openmm.openmm.")[1].split("\'")[0]
                value = self.simulation.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()._value
                components[key] = value
            if return_forces:
                openmm_forces = (state.getForces(asNumpy=True)
                                 / simtk.unit.kilojoule_per_mole
                                 * simtk.unit.nanometers
                                 * nm_per_angstrom)
                return openmm_energy, openmm_forces, components
            else:
                return openmm_energy, components
        else:
            if return_forces:
                openmm_forces = (state.getForces(asNumpy=True)
                                 / simtk.unit.kilojoule_per_mole
                                 * simtk.unit.nanometers
                                 * nm_per_angstrom)
                return openmm_energy, openmm_forces
            else:
                return openmm_energy

    @property
    def positions(self):
        """
        The positions of all particles in the system.
        """
        return self._positions

    @positions.setter
    def positions(self, positions):
        self._positions = positions
        positions_vec = []
        for i in range(len(self._positions)):
            positions_vec.append(openmm.Vec3(
                self._positions[i][0]*nm_per_angstrom,
                self._positions[i][1]*nm_per_angstrom,
                self._positions[i][2]*nm_per_angstrom,
            ) * simtk.unit.nanometer)
        if self.simulation:
            self.simulation.context.setPositions(positions_vec)
