"""
response_metric.py — the regularization-invariant DeltaI.
=====================================================================
The kappa-invariant companion to metrics.delta_metrics. Instead of the raw
potential difference Dv = vxc_exact - vxc_DFA (whose bond-region average swings
5-13x per decade of the inversion regularizer kappa), this projects Dv through
the KS static response and measures the DENSITY it would move:

    drho = chi . Dv,   DeltaI_resp = int_Omega |drho| dr / int_Omega dr   [e/bohr^3]

WHY IT IS INVARIANT (both properties hold by construction, and were then
MEASURED — see the verdict below):
  * Null space. The component of Dv that the density fit cannot constrain —
    what kappa invents — has <phi_j|Dv|phi_i> = 0 for every pair, so chi
    annihilates it.
  * Gauge. <phi_j|(Dv + c)|phi_i> = <phi_j|Dv|phi_i> by orthogonality: the
    arbitrary additive constant drops out with no weighted-median needed.

VERDICT (diagnose_response_metric.py, H2 and N2, kappa 1e-7..1e-5):
  * invariant: DeltaI_resp varies 1.2-8.7% across kappa while the current
    metric's control swings 5.4-13.4x, WITH the real fractional ensemble
    occupations (n_frac up to 13), i.e. with chi's null space at its smallest.
  * chi validated: the finite-difference ladder converges O(s) in the regime
    above the round-off floor (fd_err ~ 1e-4 at kappa=1e-6).
  * NO hidden free parameter: sweeping deg_tol over 1e-4..1e-8 moves
    DeltaI_resp by 0.00% at the operating point — the dropped near-degenerate
    channels carry no weight. So this did NOT trade the kappa parameter for a
    deg_tol parameter.

THE ONE PHYSICAL CAVEAT: an EXACTLY
degenerate KS pair carrying UNEQUAL occupations is a genuine singularity of the
static response — linear response does not describe it, and chi drops it at
deg_tol. It appeared in H2 (2e-) and never in N2 (14e-), and where it appears
it carries ~0 weight (wskip ~ 1e-3). Reported per geometry as n_skipped so a
figure can be annotated, never silently swallowed.

This module is the SINGLE source of the response operator: diagnose_response_
metric.py imports the same functions, so the bench and the production column
can never diverge (a lesson paid for twice in this project).
"""

import numpy as np
import scipy.linalg


def ks_system(mol, coords, weights, ao, vxc_grid, T, V, J, S):
    """Rebuild the KS orbitals (occupied AND virtual) from an inverted vxc by
    projecting it onto the AO basis and diagonalizing T + V + J + Vxc against
    S. Verified on the real npz to reproduce the saved KS eigenvalues to
    ~1e-7 (float32 ao_value), so the KS system is fully recoverable from the
    saved exact side without re-inverting."""
    Vxc = np.einsum('gi,g,gj->ij', ao, vxc_grid * weights, ao)
    eps, C = scipy.linalg.eigh(T + V + J + Vxc, S)
    return eps, C


def ks_occupations(C, S, D_ks, tol=1e-8):
    """KS occupations from the inverted density matrix: C^T S D_ks S C =
    diag(n), exact with no ordering assumption. Recovers the fractional
    ensemble occupations (the integer-aufbau fallback that used to stand here
    was wrong by up to 0.74 e- on stretched N2)."""
    n = np.diag(C.T @ S @ D_ks @ S @ C)
    return np.where(np.abs(n) < tol, 0.0, n)


def drho_from_dv(eps, C, occ, ao, weights, dv, deg_tol=1e-6):
    """drho = chi . dv: the literal sum over all ordered pairs (i != j),

        chi = sum_{i!=j} (n_i - n_j)/(eps_i - eps_j) phi_i phi_j (x) phi_i phi_j

    The sum runs over all ordered pairs, with no factor of 2 and no restriction
    of i to the occupied set: with fractional occupations a form that halves the
    sum and doubles it double-counts occupied-occupied pairs, since those
    contribute through (n_i - n_j) in both orders. Vectorized as (K * M)
    contracted twice with phi,
    K_ij = (n_i-n_j)/(eps_i-eps_j), M_ij = <phi_i|dv|phi_j>. Degenerate pairs
    (|eps_i-eps_j| < deg_tol) are skipped."""
    phi = ao @ C
    n = np.asarray(occ, float)
    eps = np.asarray(eps, float)
    M = phi.T @ (phi * (weights * dv)[:, None])
    dn = n[:, None] - n[None, :]
    de = eps[:, None] - eps[None, :]
    K = np.zeros_like(M)
    ok = (np.abs(de) >= deg_tol) & (np.abs(dn) >= 1e-12)
    K[ok] = dn[ok] / de[ok]
    return np.einsum('gi,ij,gj->g', phi, K * M, phi, optimize=True)


def skipped_pairs(eps, occ, deg_tol=1e-6, dn_tol=1e-6):
    """Count near-degenerate pairs with UNEQUAL occupations that chi drops —
    the response singularities. Reported per geometry (n_skipped, weight) so a
    row can be annotated; weight = sum|n_i - n_j| over the dropped pairs."""
    n, e = np.asarray(occ, float), np.asarray(eps, float)
    dn = np.abs(n[:, None] - n[None, :])
    de = np.abs(e[:, None] - e[None, :])
    bad = (de < deg_tol) & (dn > dn_tol)
    np.fill_diagonal(bad, False)
    return int(bad.sum() // 2), float(dn[bad].sum() / 2.0)


def bond_mask(coords, r_ang, cyl_rad, bohr):
    z = coords[:, 2]
    rad2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
    half = (r_ang / bohr) / 2.0
    return (np.abs(z) <= half) & (rad2 < cyl_rad ** 2)


def response_deltaI(mol, coords, weights, ao, vxc_exact, D_ks, J,
                    dv, r_ang, cyl_rad, bohr, deg_tol=1e-6):
    """DeltaI_resp for one geometry and one functional, from the saved exact
    side. `dv` is vxc_exact - vxc_functional on the grid (already built by the
    caller). `ao` is the plain GTOval (ngrid, nao). Returns a dict with the
    metric plus the two honesty diagnostics (n_skipped, deg_tol sensitivity).

    The KS system is rebuilt here from vxc_exact, so no orbitals need to be
    stored in the npz. T, V, S come from the (cheap) rebuilt mol; J is passed
    in because the caller already has the exact Coulomb matrix."""
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    S = mol.intor('int1e_ovlp')
    eps, C = ks_system(mol, coords, weights, ao, vxc_exact, T, V, J, S)
    occ = ks_occupations(C, S, D_ks)
    m = bond_mask(coords, r_ang, cyl_rad, bohr)
    wm = weights[m]

    def dibar(dt):
        drho = drho_from_dv(eps, C, occ, ao, weights, dv, deg_tol=dt)
        return float(np.sum(wm * np.abs(drho[m])) / wm.sum())

    val = dibar(deg_tol)
    # deg_tol sensitivity: the SAME test kappa was held to. If this is small,
    # the metric carries no hidden free parameter.
    lo, hi = dibar(1e-4), dibar(1e-8)
    span = (max(val, lo, hi) / max(min(val, lo, hi), 1e-300)) - 1.0
    nskip, wskip = skipped_pairs(eps, occ, deg_tol=deg_tol)
    return {'dI_resp': val, 'dI_resp_degtol_span': float(span),
            'n_skipped': nskip, 'skip_weight': wskip,
            'occ_sum': float(occ.sum())}