#! /usr/bin/env python3
"""A module to define the :class:`PBCCalculator` class.
"""
from __future__ import annotations

import itertools
import math
from typing import Any
from typing import Optional

import numba
import numpy as np
import scipy.interpolate
from numpy.typing import NDArray

from .calculator import Calculator
from qmmm_pme.common import BOHR_PER_ANGSTROM
from qmmm_pme.common import BOHR_PER_NM
from qmmm_pme.common import KJMOL_PER_EH


class PBCCalculator(Calculator):
    """A :class:`Calculator` class for performing operations on a PME
    potential grid produced by OpenMM with a quadrature supplied by
    Psi4 for the QM/MM/PME system.

    :param pme_gridnumber: |pme_gridnumber|
    :param pme_alpha: |pme_alpha|
    """
    pme_potential = None
    quadrature = None

    def __init__(
            self,
            pme_gridnumber: int | None = 30,
            pme_alpha: float | int | None = 5.0,
    ) -> None:
        super().__init__()
        self.pme_gridnumber = pme_gridnumber
        self.pme_alpha = pme_alpha
        indices = np.array(list(range(-1, self.pme_gridnumber+1)))
        self.grid = (indices,) * 3
        self.inverse_box = np.linalg.inv(
            self._state.box * BOHR_PER_ANGSTROM,
        )

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> tuple[Any, ...]:
        # Create necessary objects.
        qm_atoms = self._topology.groups["qm_atom"]
        self.nuclei = (
            self._state.positions[qm_atoms, :]
            * BOHR_PER_ANGSTROM
        )
        # Determine and apply exlcusions.
        pme_xyz, pme_exclusions = self._compute_pme_exclusions()
        self._apply_pme_exclusions(pme_xyz, pme_exclusions)
        # Prepare arrays for interpolation.
        self.pme_potential = np.reshape(
            self.pme_potential,
            (self.pme_gridnumber,) * 3,
        )
        self.pme_potential = np.pad(self.pme_potential, 1, mode="wrap")
        # Perform interpolations.
        pme_results = [
            self._interp_pme_potential(self.quadrature),
            self._interp_pme_potential(self.nuclei),
        ]
        if return_forces:
            pme_results.append(
                self._interp_pme_gradient(self.nuclei),
            )
        # Calculate reciprocal-space correction energy.
        pme_results.insert(
            0,
            sum(
                (
                    -v*q*KJMOL_PER_EH for v,q in
                    zip(pme_results[1], self._state.charges[qm_atoms])
                ),
            ),
        )
        return tuple(pme_results)

    def _compute_pme_exclusions(
            self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Create the PME points which will have exclusions applied.
        The points include the region containing the quadrature grid.

        :return: The x, y, and z, coordinates of PME gridpoints, and
            the indices to perform exclusions on.
        """
        # Create real-space coordinates of the PME grid in Bohr.
        norms = BOHR_PER_ANGSTROM*np.linalg.norm(self._state.box, axis=1)
        x = np.linspace(
            0, norms[0], self.pme_gridnumber, endpoint=False,
        )
        y = np.linspace(
            0, norms[1], self.pme_gridnumber, endpoint=False,
        )
        z = np.linspace(
            0, norms[2], self.pme_gridnumber, endpoint=False,
        )
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        x = X.flatten()[:,np.newaxis]
        y = Y.flatten()[:,np.newaxis]
        z = Z.flatten()[:,np.newaxis]
        pme_xyz = np.concatenate((x,y,z), axis=1)
        # Project quadrature grid to reciprocal space.
        points_project = self.project_to_grid(self.quadrature)
        x_i = points_project
        indices = np.unique(np.floor(x_i).T, axis=1)
        edges = list(itertools.product(*[[i, i + 1] for i in indices]))
        edges = [np.stack(x, axis=-1) for x in edges]
        x_f = np.unique(np.concatenate(tuple(edges), axis=0), axis=0)
        x_f[x_f==self.pme_gridnumber] = 0
        pme_exclusions = np.unique(x_f, axis=0)
        return pme_xyz, pme_exclusions

    def _apply_pme_exclusions(
            self,
            pme_xyz: NDArray[np.float64],
            pme_exclusions: NDArray[np.int32],
    ) -> None:
        """Apply exlcusions to relevant PME potential grid points.
        """
        indices = (
            pme_exclusions[:,0]*self.pme_gridnumber**2
            + pme_exclusions[:,1]*self.pme_gridnumber
            + pme_exclusions[:,2]
        ).astype(np.int)
        # Collect atoms to be excluded.
        qm_atoms = self._topology.groups["qm_atom"]
        #qm_drudes = self._topology.groups["qm_drude"]
        ae_atoms = [
            x for y in self._topology.groups["analytic"] for x in y
        ]
        atoms = qm_atoms + ae_atoms #+ qm_drudes
        # Gather relevant State data.
        positions = self._state.positions[atoms] * BOHR_PER_ANGSTROM
        charges = self._state.charges[atoms]
        # Perform exclusion calculation
        exclusions = pme_xyz[indices,:]
        beta = self.pme_alpha / BOHR_PER_NM
        self.pme_potential = _compute_reciprocal_exclusions(
            self.pme_potential,
            indices,
            positions,
            charges,
            exclusions,
            beta,
            self._state.box * BOHR_PER_ANGSTROM,
        )

    def _interp_pme_potential(
            self,
            points: NDArray[np.float64],
    ) -> NDArray[np.flaot64]:
        """Calculates the PME potential interpolated at the points.

        :param points: The points to be interpolated on the PME
            potential grid.
        :return: The interpolated PME potential at the points.
        """
        points_project = self.project_to_grid(points)
        interp_potential = scipy.interpolate.interpn(
            self.grid,
            self.pme_potential,
            points_project,
            method='linear',
        )
        return interp_potential

    def _interp_pme_gradient(
            self,
            points: NDArray[np.float64],
    ) -> NDArray[np.flaot64]:
        """Create the chain rule for the PME potential on the nuclei.

        :param points: The points to be interpolated on the PME
            potential gradient grid.
        :return: The interpolated PME gradient at the points.
        """
        points_project = self.project_to_grid(points)
        # This code is largely based on
        # scipy.interpolate.RegularGridInterpolator._evaluate linear.
        interp_function = scipy.interpolate.RegularGridInterpolator(
            self.grid,
            self.pme_potential,
        )
        indices, norm_dist, _ = interp_function._find_indices(
            points_project.T,
        )
        edges = list(itertools.product(*[[i, i + 1] for i in indices]))
        grad_x=0
        grad_y=0
        grad_z=0
        for edge_indices in edges:
            weight_x = 1
            weight_y = 1
            weight_z = 1
            for j, (e_i, i, y_i) in enumerate(
                    zip(edge_indices, indices, norm_dist),
            ):
                if j == 0:
                    weight_x *= np.where(e_i == i, -1.0, 1.0)
                    weight_z *= np.where(e_i == i, 1 - y_i, y_i)
                    weight_y *= np.where(e_i == i, 1 - y_i, y_i)
                if j == 1:
                    weight_y *= np.where(e_i == i, -1.0, 1.0)
                    weight_x *= np.where(e_i == i, 1 - y_i, y_i)
                    weight_z *= np.where(e_i == i, 1 - y_i, y_i)
                if j == 2:
                    weight_z *= np.where(e_i == i, -1.0, 1.0)
                    weight_y *= np.where(e_i == i, 1 - y_i, y_i)
                    weight_x *= np.where(e_i == i, 1 - y_i, y_i)
            grad_x += np.array(
                interp_function.values[edge_indices],
            ) * weight_x
            grad_y += np.array(
                interp_function.values[edge_indices],
            ) * weight_y
            grad_z += np.array(
                interp_function.values[edge_indices],
            ) * weight_z
        grad_du = np.concatenate(
            (
                grad_x.reshape((-1,1)),
                 grad_y.reshape((-1,1)),
                 grad_z.reshape((-1,1)),
            ),
            axis=1,
        )
        interp_gradient = (
            self.pme_gridnumber
            * (grad_du @ self.inverse_box)
        )
        return interp_gradient

    def project_to_grid(
            self,
            points: NDArray[np.float64],
    ) -> NDArray[np.flaot64]:
        """Project points onto a PME grid in reciprocal space.  This
        algorithm is identical to that used in method
        'pme_update_grid_index_and_fraction' in OpenMM source code,
        ReferencePME.cpp.

        :param points: The real-space points, in Bohr, to project onto
            the reciprocal-space PME grid.
        :return: The projected points in reciprocal-space, in inverse
            Bohr.
        """
        fractional_points = np.matmul(points, self.inverse_box)
        floor_points = np.floor(fractional_points)
        decimal_points = (
            (fractional_points - floor_points) * self.pme_gridnumber
        )
        integer_points = decimal_points.astype(int)
        scaled_grid_points = np.mod(
            integer_points,
            self.pme_gridnumber,
        ) + (decimal_points - integer_points)
        return scaled_grid_points

    def update(self, attr: str, value: Any) -> None:
        if "quadrature" in attr:
            self.quadrature = np.transpose(np.array(value))[:,0:3]
        elif "pme_potential" in attr:
            self.pme_potential = np.array(value) / KJMOL_PER_EH
        elif "box" in attr:
            self.inverse_box = np.linalg.inv(
                value * BOHR_PER_ANGSTROM,
            )
        elif "pme_gridnumber" in attr:
            indices = np.array(list(range(-1, self.pme_gridnumber+1)))
            self.grid = (indices,) * 3
        #else:
        #    raise AttributeError(
        #        (f"Unknown attribute '{attr}' attempted to be updated "
        #         + f"for {self.__class__.__name__}."),
        #    )


@numba.jit(nopython=True, parallel=True, cache=True)
def _compute_reciprocal_exclusions(
        external_grid,
        indices,
        positions,
        charges,
        exclusions,
        beta,
        box,
):
    n = len(exclusions)
    m = len(positions)
    for i in numba.prange(n):
        for j in range(m):
            ssc = 0
            for k in range(3):
                r = exclusions[i,k] - positions[j,k]
                d = box[k,k] * math.floor(r/box[k,k] + 0.5)
                ssc += (r - d)**2
            dr = ssc**0.5
            erf = math.erf(beta * dr)
            if erf <= 1*10**(-6):
                external_grid[indices[i]] -= beta * charges[j] * 2 * math.pi**(-0.5)
            else:
                external_grid[indices[i]] -= charges[j] * erf / dr
    return external_grid
