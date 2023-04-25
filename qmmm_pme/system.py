#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for defining systems and subsystems.
"""


class System:
    """The System base class for defining systems and subsystems.

    This class primarily serves to hold relevant properties of the
    system, such as masses, positions, and forces, as property objects.

    Parameters
    ----------
    particle_groups : dict of list of list of int, optional
        Indices of particles in the system grouped by residue, or
        molecule, and further grouped by a given key.  Example keys
        include "qm_atom", "qm_drude", and "analytic".
    element_symbols : list of str, optional
        Element symbols of all particles in the system.
    masses : numpy.ndarray, optional
        Masses of all particles in the system, in Daltons.
    charges : numpy.ndarray, optional
        Charges of all particles in the system, in proton charge
        units.
    positions : numpy.ndarray, optional
        Positions of all particles in the system, in Angstroms.
    velocities : numpy.ndarray, optional
        Velocities of all particles in the system, in Angstroms per
        femptosecond.
    forces : numpy.ndarray, optional
        Forces acting on all particles in the system, in kilojoules
        per mole per Angstrom.
    energy : numpy.ndarray, optional
        Energy and energy contributions of the system, in kilojoules
        per mole.
    box : numpy.ndarray, optional
        Periodic box vectors of the system, in Angstroms.
    frame : int, default=0
        Current frame of the simulation has taken.
    """

    def __init__(
            self,
            particle_groups=None,
            element_symbols=None,
            masses=None,
            charges=None,
            positions=None,
            velocities=None,
            forces=None,
            energy=None,
            box=None,
            frame=0,
        ):
        self._particle_groups = particle_groups
        self._element_symbols = element_symbols
        self._masses = masses
        self._charges = charges
        self._positions = positions
        self._velocities = velocities
        self._forces = forces
        self._box = box
        self._frame = frame

    @property
    def particle_groups(self):
        """
        The indices of particles in the system grouped by a given key.
        """
        return self._particle_groups

    @particle_groups.setter
    def particle_groups(self, particle_groups):
        self._particle_groups = particle_groups

    @property
    def element_symbols(self):
        """
        The element symbols of all particles in the system.
        """
        return self._element_symbols

    @element_symbols.setter
    def element_symbols(self, element_symbols):
        self._element_symbols = element_symbols

    @property
    def masses(self):
        """
        The masses of all particles in the system.
        """
        return self._masses

    @masses.setter
    def masses(self, masses):
        self._masses = masses

    @property
    def charges(self):
        """
        The charges of all particles in the system.
        """
        return self._charges

    @charges.setter
    def charges(self, charges):
        self._charges = charges

    @property
    def positions(self):
        """
        The positions of all particles in the system.
        """
        return self._positions

    @positions.setter
    def positions(self, positions):
        self._positions = positions

    @property
    def velocities(self):
        """
        The velocities of all particles in the system.
        """
        return self._velocities

    @velocities.setter
    def velocities(self, velocities):
        self._velocities = velocities
    
    @property
    def forces(self):
        """
        The forces of all particles in the system.
        """
        return self._forces

    @forces.setter
    def forces(self, forces):
        self._forces = forces
    
    @property
    def energy(self):
        """
        The energy and energy contributions of the system.
        """
        return self._energy

    @energy.setter
    def energy(self, energy):
        self._energy = energy
    
    @property
    def box(self):
        """
        The box vectors defining the periodic system.
        """
        return self._box

    @box.setter
    def box(self, box):
        self._box = box
    
    @property
    def frame(self):
        """
        The current frame of the simulation.
        """
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame
