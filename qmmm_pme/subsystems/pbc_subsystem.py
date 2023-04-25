#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended subsystem for QM/MM/PME.
"""
import itertools
import sys
import time

import math
import numba
import numpy as np
import scipy.special
import scipy.interpolate

sys.path.append("../")

from ..system import System
from ..utils import *
from ..integrators.units import *


class PBCSubsystem(System):
    """
    The extended environment subsystem for the QM/MM/PME method.

    Parameters
    ----------
    qmmm_pme_gridnumber: int
        The number of grid points to use along each of the box vectors
        of the principal cell during the PME procedure.
    qmmm_pme_alpha: float
        The Gaussian width of the smeared point charges in the Ewald
        summation scheme, in inverse nanometers.  See OpenMM's
        documentation for further discussion.
    group_part_dict: dict of list of list of int
        The indices of particles in the system grouped by a given key.
        Example keys include "qm_atom", "qm_drude", and "analytic".
    charges: NumPy Array object
        The charges of all particles in the system, in proton charge
        units.
    positions: NumPy Array object
        The positions of all particles in the system, in Angstroms.
    box: NumPy Array object
        The box vectors defining the periodic system, in Angstroms.
    n_threads: int, Optional, default=1
        Number of threads across which to parallelize the QM calculation.
    """
    
    def __init__(self, 
            qmmm_pme_gridnumber,
            qmmm_pme_alpha,
            particle_groups,
            charges,
            positions,
            box,
            n_threads=1,
        ):
        System.__init__(self)
        self.qmmm_pme_gridnumber = qmmm_pme_gridnumber
        self.qmmm_pme_alpha = qmmm_pme_alpha
        self.particle_groups = particle_groups
        self.charges = charges
        self.positions = positions
        self.box = box
        self.n_threads = n_threads
        
    def build_extd_pot(self, ref_quadrature, extd_pot, return_grad=False):
        """
        Build extended potential grids to pass to Psi4.

        Parameters
        ----------
        ref_quadrature: NumPy Array object
            A reference quadrature grid from Psi4 with the appropriate
            geometry of the system.
        extd_pot: NumPy Array object
            The extended potential provided by OpenMM.
        return_grad: bool, Optional, default=False
            Determine whether or not to return the gradient of the
            extended potential at the nuclear coordinates.

        Returns
        -------
        quad_extd_pot: NumPy Array object
            The extended potential evaluated on the Psi4 quadrature grid.
        nuc_extd_pot: NumPy Array object
            The extended potential evaluated at the nuclear coordinates.
        nuc_extd_grad: NumPy Array object, Optional
            The gradient of the extended potential evaluated at the
            nuclear coordinates.
        """
        inverse_box = np.linalg.inv(self._bohr_box)
        xdim = np.array(
            [i for i in range(-1,self.qmmm_pme_gridnumber+1)]
        )
        grid = (xdim, xdim, xdim)
        quadrature_grid = np.transpose(np.array(ref_quadrature))[:,0:3]
        positions = []
        for atom in self._particle_groups["qm_atom"]:
            positions.append(self._positions[atom])
        nuclei_grid = np.array(positions) * bohr_per_angstrom
        
        #
        #dists = []
        #for i in quadrature_grid:
        #    r = nuclei_grid[0] - i
        #    r = np.linalg.norm(r)
        #    dists.append(r)
        #print(max(dists)/self.angstrom_to_bohr)
        #

        # Determine exlcusions.
        self.build_pme_exclusions(quadrature_grid, inverse_box)
        # Apply exclusions to extd_pot.
        extd_pot = self.apply_pme_exclusions(extd_pot)
        # Prepare arrays for interpolation.
        extd_pot_3d = np.reshape(
            extd_pot,
            (self.qmmm_pme_gridnumber, self.qmmm_pme_gridnumber, self.qmmm_pme_gridnumber),
        )
        extd_pot_3d = np.pad(extd_pot_3d, 1, mode="wrap")
        # Perform interpolations.
        quad_extd_pot = self.interp_extd_pot(
            quadrature_grid,
            grid,
            extd_pot_3d,
            inverse_box,
        )
        nuc_extd_pot = self.interp_extd_pot(
            nuclei_grid,
            grid,
            extd_pot_3d,
            inverse_box,
        )

        #
        #nuclei_grid = self.project_to_pme_grid(nuclei_grid, inverse_box)
        #for nucleus in nuclei_grid:
        #    ind = nucleus.astype(int)
        #    print(nucleus[0])
        #    print(extd_pot_3d[ind[0],ind[1],ind[2]])
        #sys.exit()
        #

        # Calculate reciprocal-space correction energy.
        corr_energy = -sum(
            [
                v*q for v,q in
                zip(
                    nuc_extd_pot, 
                    [
                        self._charges[i] for i in self._particle_groups["qm_atom"]
                    ],
                )
            ],
        ) * kJmol_per_eh
        if return_grad:
            nuc_extd_grad = self.interp_extd_grad(
                nuclei_grid,
                grid,
                extd_pot_3d,
                inverse_box,
            )
            return quad_extd_pot, nuc_extd_pot, nuc_extd_grad, corr_energy
        else:
            return quad_extd_pot, nuc_extd_pot, corr_energy

    def build_pme_exclusions(self, points, inverse_box):
        """
        Collect the PME points to which exclusions will be applied.

        The points include the region containing the quadrature grid.

        Parameters
        ----------
        points: NumPy Array object
            The grid points to be excluded.
        inverse_box: NumPy Array object
            The inverse box vectors, in inverse Bohr.
        """
        # Create real-space coordinates of the PME grid in Bohr.
        x = np.linspace(0, sum([self._bohr_box[0][i]**2 for i in range(3)])**0.5,
                        self.qmmm_pme_gridnumber, endpoint=False)
        y = np.linspace(0, sum([self._bohr_box[1][i]**2 for i in range(3)])**0.5,
                        self.qmmm_pme_gridnumber, endpoint=False)
        z = np.linspace(0, sum([self._bohr_box[2][i]**2 for i in range(3)])**0.5,
                        self.qmmm_pme_gridnumber, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        x = X.flatten()[:,np.newaxis]
        y = Y.flatten()[:,np.newaxis]
        z = Z.flatten()[:,np.newaxis]
        self.pme_xyz = np.concatenate((x,y,z), axis=1)
        # Project quadrature grid to reciprocal space.
        points_project = self.project_to_pme_grid(
            points,
            inverse_box,
        )
        xi = points_project
        indices = np.unique(np.floor(xi).T,axis=1)
        edges = list(itertools.product(*[[i, i + 1] for i in indices]))
        edges = [np.stack(x,axis=-1) for x in edges]
        xf = np.unique(np.concatenate(tuple(edges),axis=0),axis=0)
        xf[xf==self.qmmm_pme_gridnumber] = 0
        self.pme_exclusion_list = np.unique(xf,axis=0)

    def apply_pme_exclusions(self, external_grid):
        """
        Apply exlcusions to relevant external potential grid points.

        Parameters
        ----------
        external_grid: NumPy Array object
            The external potential grid calculated by OpenMM.

        Returns
        -------
        external_grid: NumPy Array object
            The external potential grid with exclusions applied for the
            potential associated with the QM atoms and analytic
            embedding particles in the principal box, as well as
            real-space embedding corrections if selected.
        """
        indices = (self.pme_exclusion_list[:,0]*self.qmmm_pme_gridnumber**2
                   + self.pme_exclusion_list[:,1]*self.qmmm_pme_gridnumber
                   + self.pme_exclusion_list[:,2]).astype(np.int)
        positions = []
        charges = []
        for atom in self._particle_groups["qm_atom"]:
            positions.append(self._positions[atom])
            charges.append(self._charges[atom])

        if "qm_drude" in self._particle_groups:
            for atom in self._particle_groups["qm_drude"]:
                positions.append(self._positions[atom])
                charges.append(self._charges[atom])
        if "analytic" in self._particle_groups:
            for residue in self._particle_groups["analytic"]:
                for atom in residue:
                    positions.append(self._positions[atom])
                    charges.append(self._charges[atom])
        positions = np.array(positions) * bohr_per_angstrom
        
        #
        #print(len(indices))
        #print(len(positions))
        #sys.exit()
        #
        
        charges = np.array(charges)
        external_grid = np.array(external_grid) / kJmol_per_eh
        exclusions = self.pme_xyz[indices,:]
        beta = self.qmmm_pme_alpha / bohr_per_nm
        external_grid = reciprocal_exclusions(external_grid, indices, positions, charges, exclusions, beta, self._bohr_box)
        return external_grid

    def interp_extd_pot(self, points, grid, extd_pot_3d, inverse_box):
        """
        Builds the extended potential to pass to the quadrature.
        
        Parameters
        ----------
        points: NumPy Array object
            The grid points to be interpolated into the external
            potential grid.
        grid: tuple of NumPy Array object
            Tuple of grid axis values.
        extd_pot_3d: NumPy Array object
            The 3d extended potential grid.
        inverse_box: NumPy Array object
            The inverse of the box vectors, in inverse Bohr.

        Returns
        -------
        interp_points: NumPy Array object
            The interpolated extended potential values at the points.
        """
        points_project = self.project_to_pme_grid(points, inverse_box)
        interp_points = scipy.interpolate.interpn(
            grid,
            extd_pot_3d,
            points_project,
            method='linear',
        )
        return interp_points

    def interp_extd_grad(self, points, grid, extd_pot_3d, inverse_box):
        """
        Create the chain rule for the extended potential on the nuclei.

        Parameters
        ----------
        points: NumPy Array object
            The grid points to be interpolated into the external
            potential grid.
        grid: tuple of NumPy Array object
            Tuple of grid axis values.
        extd_pot_3d: NumPy Array object
            The 3d extended potential grid.
        inverse_box: NumPy Array object
            The inverse of the box vectors, in inverse Bohr.

        Returns
        -------
        extd_grad: NumPy Array object
            The gradient of the interpolated extended potential values
            at the points.
        """
        points_project = self.project_to_pme_grid(
            points,
            inverse_box,
        )
        # This code is largely based on
        # scipy.interpolate.RegularGridInterpolator._evaluate linear.
        interp_function = scipy.interpolate.RegularGridInterpolator(
            grid,
            extd_pot_3d,
        )
        indices, norm_dist, out_of_bounds = interp_function._find_indices(
            points_project.T,
        )
        edges = list(itertools.product(*[[i, i + 1] for i in indices]))
        extd_x=0
        extd_y=0
        extd_z=0
        for edge_indices in edges:
            weight_x = 1
            weight_y = 1
            weight_z = 1
            for j, (ei, i, yi) in enumerate(zip(edge_indices, indices, norm_dist)):
                if j == 0:
                    weight_x *= np.where(ei == i, -1.0, 1.0)
                    weight_z *= np.where(ei == i, 1 - yi, yi)
                    weight_y *= np.where(ei == i, 1 - yi, yi)
                if j == 1:
                    weight_y *= np.where(ei == i, -1.0, 1.0)
                    weight_x *= np.where(ei == i, 1 - yi, yi)
                    weight_z *= np.where(ei == i, 1 - yi, yi)
                if j == 2:
                    weight_z *= np.where(ei == i, -1.0, 1.0)
                    weight_y *= np.where(ei == i, 1 - yi, yi)
                    weight_x *= np.where(ei == i, 1 - yi, yi)
            extd_x += np.array(interp_function.values[edge_indices]) * weight_x
            extd_y += np.array(interp_function.values[edge_indices]) * weight_y
            extd_z += np.array(interp_function.values[edge_indices]) * weight_z
        extd_du = np.concatenate(
            (extd_x.reshape((-1,1)), extd_y.reshape((-1,1)), extd_z.reshape((-1,1))),
            axis=1,
        )
        extd_grad = self.qmmm_pme_gridnumber * (extd_du @ inverse_box)
        return extd_grad

    def project_to_pme_grid(self, points, inverse_box):
        """
        Project points onto a PME grid in reciprocal space.

        This algorithm is identical to that used in method
        'pme_update_grid_index_and_fraction' in OpenMM source code,
        ReferencePME.cpp.

        Parameters
        ----------
        points: NumPy Array object
            The real-space points, in Bohr, to project onto the
            reciprocal-space PME grid.
        inverse_box: NumPy Array object
            The inverse of the box vector defining the principal cell of
            the system, in inverse Bohr.

        Returns
        -------
        scaled_grid_points: NumPy Array object
            The projected points in reciprocal-space, in inverse Bohr.
        """
        scaled_grid_points = np.matmul(points, inverse_box)
        scaled_grid_points = ((scaled_grid_points
                               - np.floor(scaled_grid_points))
                              * self.qmmm_pme_gridnumber)
        scaled_grid_points = np.mod(
            scaled_grid_points.astype(int),
            self.qmmm_pme_gridnumber,
        ) + (scaled_grid_points - scaled_grid_points.astype(int))
        return scaled_grid_points
    
    @property
    def box(self):
        """
        The box vectors defining the periodic system.
        """
        return self._box
    
    @box.setter
    def box(self, box):
        self._box = box
        self._bohr_box = np.array(
            [[k*bohr_per_angstrom for k in vector] for vector in box]
        )

@numba.jit(nopython=True, parallel=True, cache=True)
def reciprocal_exclusions(external_grid, indices, positions, charges, exclusions, beta, box):
    n = len(exclusions)
    m = len(positions)
    #rs = 0
    for i in numba.prange(n):
        for j in range(m):
            ssc = 0
            for k in range(3):
                r = exclusions[i,k] - positions[j,k]
                d = box[k,k] * math.floor(r/box[k,k] + 0.5)
                ssc += (r - d)**2
            dr = ssc**0.5
            #rs = rs + dr
            erf = math.erf(beta * dr)
            if erf <= 1*10**(-6):
                external_grid[indices[i]] -= beta * charges[j] * 2 * math.pi**(-0.5)
            else:
                external_grid[indices[i]] -= charges[j] * erf / dr
    #print(rs)
    return external_grid
