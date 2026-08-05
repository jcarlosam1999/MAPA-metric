"""
baselines.py — DFA potentials & energies evaluated on the reference density.
============================================================================
Evaluating the DFA on the REFERENCE density (not self-consistently) isolates
the functional-form error from the density-driven error — the comparison the
DeltaI metric is built on.

Potentials on the grid:
  * LDA/VWN and spin-polarized LSDA (exact expressions, robust).
  * PBE closed-shell:  v = vrho - 2 div(vsigma grad rho)   [EXPERIMENTAL]
  * PBE spin-polarized (NEW, closes the open-shell gap):    [EXPERIMENTAL]
      v^a = vrho_a - div( 2 f_{s_aa} grad(rho_a) + f_{s_ab} grad(rho_b) )
    expanded with the chain rule through fxc (v2rhosigma, v2sigma2) and the
    density Hessians. libxc sigma ordering assumed (aa, ab, bb) with
    v2rhosigma rows (a,aa)(a,ab)(a,bb)(b,aa)(b,ab)(b,bb) and v2sigma2 rows
    (aa,aa)(aa,ab)(aa,bb)(ab,ab)(ab,bb)(bb,bb).
  VALIDATION PROTOCOL for both PBE potentials before publication: check
  Int v(r) drho(r) dr == Tr[Vxc_mat dDM] for random density perturbations
  (pyscf numint provides the reference matrix elements).

Energies (scalar + per-grid eps_xc(r), the per-grid energy-density field):
  dfa_exc_energy (closed shell) / dfa_exc_energy_spin (open shell).
"""

import numpy as np
from pyscf import dft


# ---------------------------------------------------------------------------
# LDA family (exact, robust)
# ---------------------------------------------------------------------------
def lda_vxc_on_grid(rho):
    _, vxc, _, _ = dft.libxc.eval_xc('LDA,VWN', np.maximum(rho, 1e-15),
                                     deriv=1)
    return vxc[0]


def vwn_correlation_vc(rho):
    """Correlation-only potential — for isolating the exact v_c against it
    in 2e singlets (where v_x = -v_H/2 analytically)."""
    _, vxc, _, _ = dft.libxc.eval_xc(',VWN', np.maximum(rho, 1e-15), deriv=1)
    return vxc[0]


def lsda_vxc_spin(rho_a, rho_b, xc='LDA,VWN'):
    rho = np.stack([np.maximum(rho_a, 1e-15), np.maximum(rho_b, 1e-15)])
    _, vxc, _, _ = dft.libxc.eval_xc(xc, rho, spin=1, deriv=1)
    vr = np.asarray(vxc[0])
    assert vr.ndim == 2 and vr.shape[1] == 2, \
        "Unexpected LSDA vrho shape — check pyscf/libxc version."
    return vr[:, 0], vr[:, 1]


# ---------------------------------------------------------------------------
# DFA energies (scalar Exc + per-grid energy density)
# ---------------------------------------------------------------------------
def dfa_exc_energy(xc, rho_input, w):
    """rho_input: rho (LDA-type) or the (4,ngrid) rho+grad stack (GGA)."""
    exc, _, _, _ = dft.libxc.eval_xc(xc, rho_input, deriv=1)
    rho0 = rho_input[0] if np.asarray(rho_input).ndim == 2 else rho_input
    rho0 = np.maximum(rho0, 1e-15)
    return float(np.sum(exc * rho0 * w)), np.asarray(exc)


def dfa_exc_energy_spin(xc, rho_a, rho_b, w):
    """Spin-polarized version — MANDATORY for open shell (the unpolarized
    labels were a real bug for ~half the benchmark set)."""
    ra, rb = np.asarray(rho_a).copy(), np.asarray(rho_b).copy()
    if ra.ndim == 2:
        ra[0] = np.maximum(ra[0], 1e-15)
        rb[0] = np.maximum(rb[0], 1e-15)
        rho_tot = ra[0] + rb[0]
    else:
        ra = np.maximum(ra, 1e-15)
        rb = np.maximum(rb, 1e-15)
        rho_tot = ra + rb
    exc, _, _, _ = dft.libxc.eval_xc(xc, (ra, rb), spin=1, deriv=1)
    return float(np.sum(exc * rho_tot * w)), np.asarray(exc)


# ---------------------------------------------------------------------------
# Density derivative machinery (shared by both PBE potentials)
# ---------------------------------------------------------------------------
def density_derivs(mol, dm, coords):
    """rho, grad(rho) (3,N), Hessian (3,3,N), Laplacian — from GTO second
    derivatives: d_i d_j rho = 2 Sum D [ (d_i d_j phi) phi
                                         + (d_i phi)(d_j phi) ]."""
    ao2 = mol.eval_gto('GTOval_sph_deriv2', coords)   # (10, N, nao)
    ao0 = ao2[0]
    ii = {'xx': 4, 'xy': 5, 'xz': 6, 'yy': 7, 'yz': 8, 'zz': 9}
    rho = np.einsum('gi,ij,gj->g', ao0, dm, ao0)
    grho = 2.0 * np.einsum('dgi,ij,gj->dg', ao2[1:4], dm, ao0)

    def d2(a, b, key):
        return 2.0 * (np.einsum('gi,ij,gj->g', ao2[ii[key]], dm, ao0)
                      + np.einsum('gi,ij,gj->g', ao2[a], dm, ao2[b]))
    hess = np.empty((3, 3, rho.size))
    hess[0, 0] = d2(1, 1, 'xx')
    hess[1, 1] = d2(2, 2, 'yy')
    hess[2, 2] = d2(3, 3, 'zz')
    hess[0, 1] = hess[1, 0] = d2(1, 2, 'xy')
    hess[0, 2] = hess[2, 0] = d2(1, 3, 'xz')
    hess[1, 2] = hess[2, 1] = d2(2, 3, 'yz')
    return rho, grho, hess, hess[0, 0] + hess[1, 1] + hess[2, 2]


def pbe_vxc_on_grid(mol, dm, coords, xc='PBE,PBE'):
    """[EXPERIMENTAL] closed-shell GGA potential on the grid (any GGA xc)."""
    rho, grho, hess, lap = density_derivs(mol, dm, coords)
    rho_gga = np.vstack([np.maximum(rho, 1e-15), grho])
    _, vxc, fxc, _ = dft.libxc.eval_xc(xc, rho_gga, deriv=2)
    vrho, vsigma = vxc[0], vxc[1]
    v2rhosigma, v2sigma2 = fxc[1], fxc[2]
    grad_sigma = 2.0 * np.einsum('abg,bg->ag', hess, grho)
    grad_vsigma = v2rhosigma * grho + v2sigma2 * grad_sigma
    return vrho - 2.0 * (np.einsum('ag,ag->g', grad_vsigma, grho)
                         + vsigma * lap)


def pbe_vxc_spin_on_grid(mol, dma, dmb, coords, xc='PBE,PBE'):
    """[EXPERIMENTAL, NEW] spin-polarized PBE potential on the grid —
    closes the open-shell potential-target gap for ~half the benchmark set.
    Returns (v_alpha, v_beta). Validate per the module protocol before use
    in figures/training."""
    ra, ga, Eh, la = density_derivs(mol, dma, coords)
    rb, gb, Hb, lb = density_derivs(mol, dmb, coords)
    stack_a = np.vstack([np.maximum(ra, 1e-15), ga])
    stack_b = np.vstack([np.maximum(rb, 1e-15), gb])
    _, vxc, fxc, _ = dft.libxc.eval_xc(xc, (stack_a, stack_b),
                                       spin=1, deriv=2)
    vrho = np.asarray(vxc[0])          # (N,2)
    vs = np.asarray(vxc[1])            # (N,3): f_aa, f_ab, f_bb
    v2rs = np.asarray(fxc[1])          # (N,6): (a|b) x (aa,ab,bb)
    v2ss = np.asarray(fxc[2])          # (N,6): pair-symmetric sigma block

    # sigma gradients: s_aa = |ga|^2, s_ab = ga.gb, s_bb = |gb|^2
    gs_aa = 2.0 * np.einsum('abg,bg->ag', Eh, ga)
    gs_bb = 2.0 * np.einsum('abg,bg->ag', Hb, gb)
    gs_ab = (np.einsum('abg,bg->ag', Eh, gb)
             + np.einsum('abg,bg->ag', Hb, ga))

    def grad_f(col_ra, col_rb, c_aa, c_ab, c_bb):
        """grad of a vsigma component via the chain rule through fxc."""
        return (v2rs[:, col_ra] * ga + v2rs[:, col_rb] * gb
                + v2ss[:, c_aa] * gs_aa + v2ss[:, c_ab] * gs_ab
                + v2ss[:, c_bb] * gs_bb)

    g_faa = grad_f(0, 3, 0, 1, 2)      # d f_aa / d(ra,rb,s_aa,s_ab,s_bb)
    g_fab = grad_f(1, 4, 1, 3, 4)
    g_fbb = grad_f(2, 5, 2, 4, 5)

    # v^a = vrho_a - div(2 f_aa ga + f_ab gb); div(fu) = grad f . u + f div u
    div_a = (2.0 * (np.einsum('ag,ag->g', g_faa, ga) + vs[:, 0] * la)
             + np.einsum('ag,ag->g', g_fab, gb) + vs[:, 1] * lb)
    div_b = (2.0 * (np.einsum('ag,ag->g', g_fbb, gb) + vs[:, 2] * lb)
             + np.einsum('ag,ag->g', g_fab, ga) + vs[:, 1] * la)
    return vrho[:, 0] - div_a, vrho[:, 1] - div_b
