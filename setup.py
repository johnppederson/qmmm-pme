"""
setup.py: Builds repository and distribution information.
"""
import setuptools

__author__ = "Jesse McDaniel and John Pederson"
__version__ = "0.9.2"

with open("README.rst", "r") as fh:
    description = fh.read()

setuptools.setup(name="qmmm_pme",
                 version="0.9.2",
                 description="QM/MM with PME for long-range electrostatics",
                 author="Jesse McDaniel and John Pederson",
                 author_email="jpederson6@gatech.edu",
                 packages=["qmmm_pme"],
                 python_requires=">3.0")
