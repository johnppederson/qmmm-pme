#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger to record simulation data.
"""
import array
import copy
import os
import struct
import sys

sys.path.append("../")

from ..integrators.units import *
from ..system import System
from ..utils import *

class Logger(System):
    """A logger for QM/MM or QM/MM/PME simulations.

    Parameters
    ----------
    
    """
    
    def __init__(self, 
            name,
            num_particles,
            box,
            rewrite_log=True,
            decimal_places=1,
            energy_log=True,
            energy_verbose=3,
            energy_write_freq=1,
            particle_log=True,
            particle_write_freq=10,
            include_positions=True,
            include_velocities=False,
            include_forces=False,
            timestep=1,
        ):
        System.__init__(self)
        self.name = name
        self.num_particles = num_particles
        self.decimal_places = decimal_places
        self.energy_log = energy_log
        if energy_log:
            self.rewrite_log = rewrite_log
            self.log_file = os.path.join(f"{name}_sim_output", f"{name}.log")
            if rewrite_log and os.path.isfile(self.log_file):
                os.remove(self.log_file)
                with open(self.log_file, "w") as fh:
                    fh.write("="*27 + " QM/MM/PME Logger " + "="*27 + "\n\n")
        self.energy_verbose = energy_verbose
        self.energy_write_freq = energy_write_freq
        self.particle_log = particle_log
        if particle_log:
            self.traj_file = {}
            self.traj_file["positions"] = (
                os.path.join(f"{name}_sim_output", f"{name}_positions.dcd")
                if include_positions else None
            )
            self.traj_file["velocities"] = (
                os.path.join(f"{name}_sim_output", f"{name}_velocities.dcd")
                if include_velocities else None
            )
            self.traj_file["forces"] = (
                os.path.join(f"{name}_sim_output", f"{name}_forces.dcd")
                if include_forces else None
            )
            for key, value in self.traj_file.items():
                if value:
                    if rewrite_log and os.path.isfile(value):
                        os.remove(value)
                    if not os.path.isfile(value):
                        with open(value, "wb") as fh:
                            header = struct.pack("<i4c9if", 84, b"C", b"O", b"R", b"D", 0, 0, particle_write_freq, 0, 0, 0, 0, 0, 0, timestep)
                            header += struct.pack("<13i", 1, 0, 0, 0, 0, 0, 0, 0, 0, 24, 84, 164, 2)
                            header += struct.pack("<80s", b"Created by QM/MM/PME")
                            header += struct.pack("<80s", b"Created now")
                            header += struct.pack("<4i", 164, 4, num_particles, 4)
                            fh.write(header)

        self.particle_write_freq = particle_write_freq
        self.timestep = timestep
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = lattice_constants(box)

    def report(self):
        if self.particle_log and self.frame % self.particle_write_freq == 0:
            for key, value in self.traj_file.items():
                if value:
                    with open(value, "r+b") as fh:
                        fh.seek(8, os.SEEK_SET)
                        fh.write(struct.pack("<i", self.frame//self.particle_write_freq))
                        fh.seek(20, os.SEEK_SET)
                        fh.write(struct.pack("<i", self.frame))
                        fh.seek(0, os.SEEK_END)
                        # This code prints box information.
                        fh.write(struct.pack("<i6di", 48, self.a, self.gamma, self.b, self.beta, self.alpha, self.c, 48))
                        num = struct.pack("<i", 4*self.num_particles)
                        for i in range(3):
                            fh.write(num)
                            positions = array.array("f", (x[i] for x in eval(f"self.{key}")))
                            positions.tofile(fh)
                            fh.write(num)
                        fh.flush()
        if self.energy_log and self.frame % self.energy_write_freq == 0:
            with open(self.log_file, "a") as fh:
                fh.write("-"*29+" Frame "+"0"*(6-len(str(self.frame)))+str(self.frame)+" "+"-"*29+"\n")
                lines = self.unwrap_energy(self.energy)
                fh.write(lines + "\n")

    def unwrap_energy(self, energy, spaces=0, cont=[]):
        string = ""
        for i, (key, val) in enumerate(energy.items()):
            if type(val) is dict:
                if i != len(energy) - 1: cont.append(spaces - 1)
                string += self.unwrap_energy(val, spaces=spaces+1, cont=cont)
                if i != len(energy) - 1: cont.remove(spaces - 1)
            else:
                value = str(val)
                left, right = value.split(".")
                right = right[0:min(len(right),self.decimal_places)]
                right += "0"*(self.decimal_places-len(right))
                if spaces > 0:
                    string_temp = ["| " if i in cont else "  " for i in range(spaces - 1)]
                    string_temp = "".join(string_temp)
                    string += string_temp + "|_"
                string += (key + ":" + " "*(72-9-len(right)-2*spaces-len(left)-len(key))
                           + left + "." + right + " kJ/mol\n")
        return string

    def terminate(self, write_csv=True):
        if self.energy_log:
            with open(self.log_file, "a+") as fh:
                fh.write("="*30 + " End of Log " + "="*30)
                fh.seek(0)
                lines = fh.readlines()
            if write_csv:
                energy_output = {}
                for line in lines:
                    line = line.replace("|","")
                    line = line.replace("_","")
                    if ":" in line:
                        key, value = line.split(":")
                        if key.strip() not in energy_output.keys():
                            energy_output[key.strip()] = []
                        energy_output[key.strip()].append(value.strip().split()[0])
                with open(self.log_file[0:-4] + ".csv", "w") as fh:
                    fh.write("Frame," + ",".join(energy_output.keys()) + "\n")
                    energy_array = list(energy_output.values())
                    for i in range(len(energy_array[0])):
                        energy_list = [str(i)]
                        for j in range(len(energy_array)):
                            energy_list.append(energy_array[j][i])
                        fh.write(",".join(energy_list) + "\n")
