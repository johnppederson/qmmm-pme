"""
wfn = self._generate_wavefunction(**kwargs)
T = wfn.mintshelper().ao_kinetic()
V = wfn.mintshelper().ao_potential()
Da = wfn.Da()
Db = wfn.Db()
one_e_components = {
    "Electronic Kinetic Energy":
        (
            Da.vector_dot(T) + Db.vector_dot(T)
        ) * KJMOL_PER_EH,
    "Electronic Potential Energy":
        (
            Da.vector_dot(V) + Db.vector_dot(V)
        ) * KJMOL_PER_EH,
}
components = {
    "Nuclear Repulsion Energy":
        (
            wfn.variable("NUCLEAR REPULSION ENERGY")
        ) * KJMOL_PER_EH,
    "One-Electron Energy":
        (
            wfn.variable("ONE-ELECTRON ENERGY")
        ) * KJMOL_PER_EH,
    ".": one_e_components,
    "Two-Electron Energy":
        (
            wfn.variable("TWO-ELECTRON ENERGY")
        ) * KJMOL_PER_EH,
    "Exchange-Correlation Energy":
        (
            wfn.variable("DFT XC ENERGY")
        ) * KJMOL_PER_EH,
}
"""
from __future__ import annotations
