=========
QM/MM/PME
=========

:Author: John Pederson
:Author Email: jpederson6@gatech.edu
:Project: qmmm-pme
:Date Written: August 27, 2022
:Last Date Modified: October 2, 2023

Summary
-------
This package implements dynamics for the both QM/MM and for the
QM/MM/PME method described by John Pederson and Professor Jesse
McDaniel:

DOI: `10.1063/5.0087386 <https://aip.scitation.org/doi/10.1063/5.0087386>`_

Installation
------------

This software depends on a `modified fork of openmm
<https://github.com/johnppederson/openmm>`_ and a `modified fork of
psi4 <https://github.com/johnppederson/psi4>`_.  These modified
repositories must be compiled from source.

The modified openmm requires the following dependencies:

- cython
- doxygen
- swig

The modified psi4 requires the following dependencies:

- gcc>=4.9
- gau2grid
- pint
- pydantic
- libxc
- numpy>=1.19.2

If you are performing calculations on large molecules, we also recommend
installing the excellent dftd3 repository developed by the Grimme lab.

Once the modified psi4 and openmm repositories are built, the qmmm-pme
repository may be cloned.::

    git clone https://github.com/johnppederson/qmmm-pme

The repository must then be pip installed.::

    cd qmmm-pme
    python -m pip install ./

Authors
-------

John Hymel

Shahriar Khan

Jesse McDaniel

John Pederson
