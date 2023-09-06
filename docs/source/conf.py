# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'QM/MM/PME'
copyright = '2023, John P. Pederson, Jesse G. McDaniel'
author = 'John P. Pederson, Jesse G. McDaniel'

# The full version, including alpha/beta/rc tags
release = '2.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Add support for function annotations.
autodoc_typehints = "description"

# Define substitutions for common parameters and returns.
rst_epilog = """
.. |masses| replace:: The current masses of all atoms in the :class:`System`, in Daltons.
.. |charges| replace:: The current charges of all atoms in the :class:`System`, in elementary charge units.
.. |positions| replace:: The current positions of all atoms in the :class:`System`, in Angstroms.
.. |velocities| replace:: The current velocities of all atoms in the :class:`System`, in ???.
.. |momenta| replace:: The current momenta of all atoms in the :class:`System`, in ???.
.. |forces| replace:: The current forces of all atoms in the :class:`System`, in kJ/mol/Angstrom.
.. |box| replace:: The current box vectors of the :class:`System`, in Angstroms.
.. |frame| replace:: The current frame of the :class:`System`.
.. |residues| replace:: The names of all residues in the :class:`System`.
.. |elements| replace:: The element symbols of all atoms in the :class:`System`.
.. |atoms| replace:: The names of all atoms in the :class:`System`.
.. |pdb_list| replace:: The directories containing the PDB files which define the system geometry.
.. |topology_list| replace:: The directories containing the XML files which define the system topology.
.. |forcefield_list| replace:: The directories containing the XML files which define the system interactions in the MM Hamiltonian.
.. |temperature| replace:: The temperature, in Kelvin.
.. |nonbonded_method| replace:: The OpenMM nonbonded method.
.. |nonbonded_cutoff| replace:: The OpenMM nonbonded cutoff.
.. |pme_gridnumber| replace:: The number of PME gridpoints to use along each box vector.
.. |pme_alpha| replace:: The PME alpha parameter for Gaussian widths.
.. |basis_set| replace:: The basis set to use for Psi4 calculations.
.. |functional| replace:: The DFT functional to use for the Psi4 calculations.
.. |charge| replace:: The net molecular charge of the QM atoms.
.. |spin| replace:: The net molecular spin of the QM atoms.
.. |quadrature_spherical| replace:: The number of spherical (angular and azimuthal) points to use in the Lebedev quadrature implemented in Psi4.
.. |quadrature_radial| replace:: The number of radial points to use in the Lebedev quadrature implemented in Psi4.
.. |scf_type| replace:: The type of SCF to perform in Psi4.
.. |read_guess| replace:: Determine whether or not read in wavefunction from previous calculation.
.. |reference_energy| replace:: The base potential energy for QM energies.
.. |friction| replace:: The friction felt by particles as dynamics propagate.
.. |timestep| replace:: The timestep with which to propagate the simulation.
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
