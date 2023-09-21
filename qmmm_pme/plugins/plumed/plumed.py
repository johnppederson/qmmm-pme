#! /usr/bin/env python3
"""A module defining the pluggable implementation of the Plumed
for enhanced sampling in the QM/MM/PME repository.
"""
from __future__ import annotations

from dataclasses import astuple
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
import plumed

from qmmm_pme.calculators.calculator import Results
from qmmm_pme.plugins.plugin import CalculatorPlugin

if TYPE_CHECKING:
    from qmmm_pme.calculators.calculator import ModifiableCalculator


class Plumed(CalculatorPlugin):
    """A :class:`Plugin` which implements the SETTLE algorithm for
    positions and velocities.

    :param oh_distance: The distance between the oxygen and hydrogen, in
        Angstroms.
    :param hh_distance: The distance between the hydrogens, in
        Angstroms.
    :param hoh_residue: The name of the water residues in the
        :class:`System`.
    """

    def __init__(
            self,
            input_commands: str,
            log_file: str,
    ) -> None:
        self.input_commands = input_commands
        self.log_file = log_file
        self.plumed = plumed.Plumed()
        self.plumed.cmd("setMDEngine", "python")
        self.frame = 0

    def modify(
            self,
            calculator: ModifiableCalculator,
    ) -> None:
        """Perform necessary modifications to the :class:`Calculator`
        object.

        :param calculator: The calculator to modify with enhanced sampling
            biases.
        """
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        self.plumed.cmd("setNatoms", len(self.system))
        self.plumed.cmd("setMDLengthUnits", 1/10)
        self.plumed.cmd("setMDTimeUnits", 1/1000)
        self.plumed.cmd("setMDMassUnits", 1.)
        self.plumed.cmd("setTimestep", 1.)
        self.plumed.cmd("setKbT", 1.)
        self.plumed.cmd("setLogFile", self.log_file)
        self.plumed.cmd("init")
        for line in self.input_commands.split("\n"):
            self.plumed.cmd("readInputLine", line)
        calculator.calculate = self._modify_calculate(calculator.calculate)

    def _modify_calculate(
            self,
            calculate: Callable[..., tuple[Any, ...]],
    ) -> Callable[..., tuple[Any, ...]]:
        """
        """
        def inner(**kwargs: bool) -> tuple[Any, ...]:
            energy, forces, components = calculate(**kwargs)
            self.plumed.cmd("setStep", self.frame)
            self.frame += 1
            self.plumed.cmd("setBox", self.system.state.box())
            self.plumed.cmd("setPositions", self.system.state.positions())
            self.plumed.cmd("setEnergy", energy)
            self.plumed.cmd("setMasses", self.system.state.masses())
            biased_forces = np.zeros(self.system.state.positions().shape)
            self.plumed.cmd("setForces", biased_forces)
            virial = np.zeros((3, 3))
            self.plumed.cmd("setVirial", virial)
            self.plumed.cmd("prepareCalc")
            self.plumed.cmd("performCalc")
            biased_energy = np.zeros((1,))
            self.plumed.cmd("getBias", biased_energy)
            energy += biased_energy
            forces += biased_forces
            results = Results(energy)
            if kwargs["return_forces"]:
                results.forces = forces
            if kwargs["return_components"]:
                components.update(
                    {"Plumed Bias Energy": biased_energy},
                )
                results.components = components
            return astuple(results)
        return inner
