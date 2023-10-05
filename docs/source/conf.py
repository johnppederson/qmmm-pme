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

project: str = 'QM/MM/PME'
copyright: str = '2023, John P. Pederson, Jesse G. McDaniel'
author: str = 'John P. Pederson, Jesse G. McDaniel'

# The full version, including alpha/beta/rc tags
release: str = "0.1.1"


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
.. |package| replace:: QM/MM/PME

.. |masses| replace:: The current masses of all atoms in the :class:`System`, in Daltons.
.. |charges| replace:: The current charges of all atoms in the :class:`System`, in elementary charge units.
.. |positions| replace:: The current positions of all atoms in the :class:`System`, in Angstroms.
.. |velocities| replace:: The current velocities of all atoms in the :class:`System`, in ???.
.. |momenta| replace:: The current momenta of all atoms in the :class:`System`, in ???.
.. |forces| replace:: The current forces of all atoms in the :class:`System`, in kJ/mol/Angstrom.
.. |box| replace:: The current box vectors of the :class:`System`, in Angstroms.

.. |atoms| replace:: ...
.. |qm_atoms| replace:: ...
.. |mm_atoms| replace:: ...
.. |ae_atoms| replace:: ...

.. |elements| replace:: The element symbols of all atoms in the :class:`Topology` of the :class:`System` object.
.. |residue_names| replace:: The names of all residues in the :class:`Topology` of the :class:`System` object.
.. |atom_names| replace:: The names of all atoms in the :class:`Topology` of the :class:`System` object.

.. |pdb_list| replace:: The directories containing the PDB files which define the :class:`System` geometry.
.. |topology_list| replace:: The directories containing the XML files which define the :class:`System` topology.
.. |forcefield_list| replace:: The directories containing the XML files which define the :class:`System` interactions.

.. |nonbonded_method| replace:: The MM nonbonded method.
.. |nonbonded_cutoff| replace:: The MM nonbonded cutoff.
.. |pme_gridnumber| replace:: The number of PME gridpoints to use along each box vector.
.. |pme_alpha| replace:: The PME alpha parameter for Gaussian widths, in inverse nanometers.

.. |basis_set| replace:: The basis set to use for QM calculations.
.. |functional| replace:: The DFT functional to use for the QM calculations.
.. |charge| replace:: The net charge of the QM atoms.
.. |spin| replace:: The net spin of the QM atoms.
.. |quadrature_spherical| replace:: The number of spherical (angular and azimuthal) points to use in the Lebedev quadrature.
.. |quadrature_radial| replace:: The number of radial points to use in the Lebedev quadrature.
.. |scf_type| replace:: The type of SCF procedure to perform for QM calculations.
.. |read_guess| replace:: Whether or not read in wavefunction from previous a calculation.  This speeds up calculations.

.. |timestep| replace:: The timestep that the :class:`Simulation` propagates with, in femtoseconds.
.. |temperature| replace:: The temperature, in Kelvin.
.. |friction| replace:: The friction felt by particles as a result of a thermostat, in inverse femtoseconds.

.. |embedding_cutoff| replace:: The electrostatic QM/MM embedding cutoff, in Angstroms.

.. |frame| replace:: The current frame of the :class:`Simulation`.

.. |system| replace:: The :class:`System` object
.. |simulation| replace:: The :class:`Simulation` object
.. |logger| replace:: The :Class:`Logger` object
.. |hamiltonian| replace:: The :Class:`Hamiltonian` object
.. |integrator| replace:: The :Class:`Integrator` object
.. |calculator| replace:: The :Class:`Calculator` object
.. |interface| replace:: The :class:`SoftwareInterface` object
"""

# Add any paths that contain templates here, relative to this directory.
templates_path: list[str] = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme: str = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: list[str] = ['_static']
