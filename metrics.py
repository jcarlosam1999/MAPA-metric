"""
metrics.py — the potential-domain metric and the exact-physics diagnostics.
==================================================================
DeltaI_bar (the article's central quantity): a defensible potential-error
metric must be (a) GAUGE-INVARIANT (inverted potentials carry an arbitrary
additive constant — the optimal L1 shift, a weighted median, removes it),
(b) DENSITY-WEIGHTED (errors matter where electrons are), and
(c) NORMALIZED (otherwise it grows with bond length just because the
integration volume does). Units: Eh (hartree), interpretable as the mean
per-electron potential error over the bond region.

Exact-condition diagnostics (by-parts forms verified to 1e-8; they need
only vxc, rho, grad rho — never grad vxc — and are gauge-safe because
Int (3 rho + r.grad rho) dr = 0):
    Levy-Perdew virial:  Exc + T_c = Int vxc (3 rho + r.grad rho) dr
    Zero-force theorem:  F_a = -Int vxc d_a rho dr = 0
They certify the inversion and are ready-made physics-
informed loss terms for functional development.
"""

import numpy as np

from config import CFG, BOHR


def weighted_median(values, weights):
    """L1-optimal constant — the gauge shift for DeltaI_bar."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, 0.5 * cw[-1])]


def bond_region_mask(coords_bohr, r_ang):
    """Cylinder between the nuclei. NOTE the unit conversion: coords are
    BOHR, r_ang is ANGSTROM — mixing these silently corrupted the metric
    in an early version"""
    r_bohr = r_ang / BOHR
    return ((np.abs(coords_bohr[:, 2]) <= r_bohr / 2.0)
            & (coords_bohr[:, 0]**2 + coords_bohr[:, 1]**2
               < CFG.cyl_radius_bohr**2))


def delta_metrics(v_exact, v_dfa, rho, coords_bohr, w, r_ang, weighting=None):
    """Returns (legacy raw integral [Eh*Bohr^3], DeltaI_bar [Eh], gauge c).

    `weighting` selects what DeltaI_bar averages |dv - c| over:

      'volume'  (default) — the bond VOLUME. This is the corrected metric.
      'density' — legacy: weight by rho. DO NOT USE for production; kept only
                  to reproduce old numbers for comparison.

    WHY THE CHANGE. The exact vxc builds a correlation BARRIER at the bond
    midpoint as the bond stretches (verified on the clean H2 sweep: the vxc
    maximum sits exactly at z=0 for R >= 1.258 A, with prominence growing
    2.77 -> 3.17 -> 3.30 Eh). That barrier is the physics this metric exists to
    detect — and it lives exactly where the density is LOWEST. Weighting by rho
    therefore suppresses it by construction and hands the metric to the nuclear
    region instead (at R=1.35: rho*|dv| = 2.63 at the nucleus vs 0.25 at the
    barrier). The consequence was a metric that peaked at EQUILIBRIUM:

        weighting     max of DeltaI_bar (clean H2)
        density       R = 0.740 A   <- equilibrium; unphysical
        volume        R = 1.258 A   <- Coulson-Fischer (D_CF = 1.305 A)

    The gauge c is still the L1-optimal constant under the SAME weighting, so
    DeltaI_bar remains invariant to the arbitrary constant in vxc.
    """
    from config import CFG
    weighting = weighting or getattr(CFG, 'dIbar_weighting', 'volume')
    m = bond_region_mask(coords_bohr, r_ang)
    dv, wm = v_exact[m] - v_dfa[m], w[m]
    dI_raw = float(np.sum(np.abs(dv) * wm))
    if weighting == 'density':
        rw = np.maximum(rho[m], 0.0) * wm
    elif weighting == 'volume':
        rw = wm
    else:
        raise ValueError(f"unknown dIbar weighting {weighting!r}")
    if rw.sum() <= 0:
        return dI_raw, np.nan, 0.0
    c = weighted_median(dv, rw)
    return dI_raw, float(np.sum(rw * np.abs(dv - c)) / rw.sum()), float(c)


def ref_exc(mol, e_total, D_ks, ref_dm, J_ref):
    """KS decomposition Exc = E - E_nn - Ts - Vext - E_H, with Ts from the
    inverted KS orbitals and the density-level pieces from ref_dm. Called
    TWICE per geometry: with e_consistent (the pair matching vxc) and with
    the NEVPT2 total (the best-energy label)."""
    T = mol.intor('int1e_kin')
    Ts = float(np.einsum('ij,ij->', D_ks, T))
    T_ref = float(np.einsum('ij,ij->', ref_dm, T))
    Ene = float(np.einsum('ij,ij->', ref_dm, mol.intor('int1e_nuc')))
    EH = 0.5 * float(np.einsum('ij,ij->', ref_dm, J_ref))
    Exc = e_total - mol.energy_nuc() - Ts - Ene - EH
    return Exc, Ts, T_ref


def exact_condition_diagnostics(vxc, rho, grad_rho, coords_bohr, w,
                                Exc_consistent, T_c):
    """Virial gap and zero-force residual (see module header). For open
    shell these use the spin-averaged vxc and total rho (approximate
    statements); for closed shell they are exact."""
    rdg = np.einsum('ga,ag->g', coords_bohr, grad_rho)
    virial = float(np.sum(vxc * (3.0 * rho + rdg) * w))
    virial_gap = virial - (Exc_consistent + T_c)
    zero_force = np.array([-float(np.sum(vxc * grad_rho[a] * w))
                           for a in range(3)])
    return virial_gap, zero_force