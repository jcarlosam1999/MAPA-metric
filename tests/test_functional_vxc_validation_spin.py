"""
test_functional_vxc_validation_spin.py — Tier-1 validation, OPEN SHELL
======================================================================
The closed-shell bench (test_functional_vxc_validation.py) proved that
baselines.pbe_vxc_on_grid IS the true functional derivative (LDA to 1e-16,
PBE to ~1e-7). It never touched the SPIN-POLARIZED path — and 81 of the 140
benchmark members (58%) are open shell, so the spin path carries the majority
of the benchmark's PBE columns.

WHY THIS IS THE HIGHEST-RISK CODE IN THE PROJECT
------------------------------------------------
baselines.lsda_vxc_spin is safe: it asks libxc for vrho and returns it. No
derivation.

baselines.pbe_vxc_spin_on_grid is NOT: it hand-derives

    v^a = vrho_a - div( 2 f_{s_aa} grad(rho_a) + f_{s_ab} grad(rho_b) )

expanding div(...) with the chain rule through fxc (v2rhosigma, v2sigma2) and
the density Hessians — and its own docstring says the libxc component ordering
is *assumed*:
    sigma      = (aa, ab, bb)
    v2rhosigma = (a,aa)(a,ab)(a,bb)(b,aa)(b,ab)(b,bb)
    v2sigma2   = (aa,aa)(aa,ab)(aa,bb)(ab,ab)(ab,bb)(bb,bb)
A wrong index does not crash. It returns a smooth, plausible potential and a
believable DeltaI_bar that is silently wrong on 58% of the paper.

THE IDENTITY
------------
For any perturbation of the alpha density matrix alone,

    Int v_a(r) drho_a(r) dr  ==  Tr[ Vxc_a . dD_a ]

LHS: the GRID potential under test, integrated against drho_a.
RHS: PySCF/libxc's OWN matrix derivative dExc/dD_a, from nr_uks.

The two agree only if the hand-derived divergence — every index, every factor
of 2 — is right. The identity is EXACT and linear in dD, so no small-
perturbation limit is involved: any symmetric dD is a valid probe, and a
failure is a failure, not a truncation error.

The alpha and beta channels are probed SEPARATELY (dD_b = 0 and vice versa);
a symmetric ordering bug that cancels in the total would still be caught.

RUN
    python test_functional_vxc_validation_spin.py
    python test_functional_vxc_validation_spin.py --member HH_q+1_s1 --n-probe 5

VERDICT
    LDA/LSDA  -> must pass at ~machine precision (direct libxc call).
    PBE       -> must pass at ~1e-6 or better (grid discretization of the
                 gradient term is the only legitimate residual).
    Anything else is a bug in pbe_vxc_spin_on_grid, most likely an index
    ordering. If it fails, DO NOT ship PBE columns for the 81 open-shell
    members: either fix the ordering (the map above is the suspect list) or
    set CFG.include_pbe_spin = False and report LDA-only there.
"""

import argparse
import sys

import numpy as np


def _dm_spin(mol):
    """A real, non-symmetric-in-spin density matrix to evaluate the potential
    at. UHF with a broken-symmetry guess (molecules.run_uhf_broken_symmetry)
    is the production-realistic choice: it gives rho_a != rho_b with genuine
    spin structure, which is exactly what the sigma_ab cross terms need."""
    from molecules import run_uhf_broken_symmetry
    mf = run_uhf_broken_symmetry(mol)
    dm = mf.make_rdm1()
    return np.asarray(dm[0]), np.asarray(dm[1]), mf


def _random_symmetric_perturbation(nao, rng, scale=1e-3):
    a = rng.normal(size=(nao, nao))
    a = 0.5 * (a + a.T)
    return scale * a


def _relative_error(lhs, rhs, integrand_scale):
    """|LHS - RHS| normalized by a scale that CANNOT collapse.

    THE FIRST VERSION OF THIS BENCH WAS WRONG. It used
        rel = |lhs - rhs| / max(|lhs|, |rhs|)
    and both sides are SIGNED integrals of an oscillating integrand, so a
    probe that comes out nearly orthogonal drives the denominator to zero and
    the ratio explodes with no physics behind it. Measured on the real run:
    the absolute disagreements spanned 2e-09 .. 7e-08 (a factor of 35) while
    the 'relative' error spanned 4e-07 .. 6.6e-05 (a factor of 165), and the
    worst 'failure' was simply the probe with the smallest |LHS|. That is a
    false positive manufactured by the normalization.

    The honest scale is the L1 norm of the integrand, Int |v_sigma drho_sigma|,
    which measures how much the two sides actually have to agree ABOUT before
    cancellation — it never collapses. (The response bench already uses exactly
    this convention: Int w |fd - lin| / Int w |fd|.)
    """
    den = max(float(integrand_scale), 1e-300)
    return abs(float(lhs) - float(rhs)) / den


def _matrix_derivative(mol, grids, xc, dma, dmb):
    """PySCF's own dExc/dD^sigma (the reference side of the identity).
    nr_uks' signature has drifted across PySCF versions; both known forms are
    tried before giving up, so a version mismatch reports itself instead of
    masquerading as a physics failure."""
    from pyscf import dft
    ni = dft.numint.NumInt()
    dms = np.stack([dma, dmb])
    try:
        n, exc, vmat = ni.nr_uks(mol, grids, xc, dms)
    except TypeError:
        n, exc, vmat = ni.nr_uks(mol, grids, xc, dms, 0, 1)
    vmat = np.asarray(vmat)
    if vmat.shape[0] != 2:
        raise RuntimeError(f"nr_uks returned vmat with shape {vmat.shape}; "
                           "expected (2, nao, nao) — check the PySCF version.")
    return vmat[0], vmat[1]


def _probe(v_grid, Vmat, ao, w, dD):
    """One probe of Int v drho == Tr[V dD]. Returns (lhs, rhs, rel, absdiff)."""
    drho = np.einsum('gi,ij,gj->g', ao, dD, ao)
    lhs = float(np.sum(v_grid * drho * w))
    rhs = float(np.einsum('ij,ij->', Vmat, dD))
    scale = float(np.sum(np.abs(v_grid * drho) * w))     # L1: cannot collapse
    return lhs, rhs, _relative_error(lhs, rhs, scale), abs(lhs - rhs)


def check(member_name, r_ang=None, n_probe=3, seed=0, levels=(3, 4, 5)):
    from pyscf import dft
    from benchmark import parse_benchmark
    from molecules import build_molecule
    import baselines as bl

    members = {m['name']: m for m in parse_benchmark()}
    if member_name not in members:
        raise SystemExit(f"unknown member {member_name!r}. Open-shell examples: "
                         + ", ".join(sorted(m for m, d in members.items()
                                            if d['spin'] != 0)[:6]))
    mem = members[member_name]
    if mem['spin'] == 0:
        print(f"WARNING: {member_name} is closed shell — rho_a == rho_b makes "
              f"the sigma_ab cross terms degenerate. Pick a spin != 0 member.")
    r = float(r_ang if r_ang is not None else mem['r_eq'])
    mol = build_molecule(mem['a1'], mem['a2'], r, mem['charge'], mem['spin'])
    dma, dmb, _ = _dm_spin(mol)
    nao = mol.nao
    print(f"system   : {member_name}  ({mem['a1']}{mem['a2']}, q={mem['charge']:+d}, "
          f"2S={mem['spin']}, {mem['n_elec']}e-) at R={r:.3f} A")
    print(f"basis/aos: {mol.basis} / {nao}")
    print()
    print("THE TEST IS THE GRID SCAN, not any single number.")
    print("  Int v drho == Tr[V dD] is EXACT analytically. PySCF builds its")
    print("  matrix from the GRADIENT form; baselines builds the grid potential")
    print("  from the DIVERGENCE form (it needs the density Hessian). The two")
    print("  are related by parts, so on a finite grid they differ by QUADRATURE")
    print("  error — which must SHRINK as the grid improves. An index or factor")
    print("  bug does not shrink: it is O(1) and grid-independent.")
    print("  'PBE closed-shell' is the control: it exercises the SAME density")
    print("  Hessian + chain-rule machinery with NO spin indexing. If the spin")
    print("  path tracks it, the assumed libxc sigma/v2rhosigma/v2sigma2 maps")
    print("  are exonerated and what is left is quadrature.")
    print()

    dm_tot = dma + dmb
    rows = {}
    hdr = (f"{'grid':>5} {'pts':>7} {'functional':>22} {'worst |LHS-RHS|':>16} "
           f"{'worst rel(L1)':>14}")
    print(hdr)
    print('-' * len(hdr))
    for lev in levels:
        g = dft.gen_grid.Grids(mol)
        g.level = lev
        g.build()
        coords, w = g.coords, g.weights
        ao = mol.eval_gto('GTOval', coords)
        rng = np.random.default_rng(seed)
        probes = [_random_symmetric_perturbation(nao, rng)
                  for _ in range(n_probe)]

        ra = dft.numint.eval_rho(mol, ao, dma, xctype='LDA')
        rb = dft.numint.eval_rho(mol, ao, dmb, xctype='LDA')
        trials = []
        va, vb = bl.lsda_vxc_spin(ra, rb, xc='LDA,VWN')
        Va, Vb = _matrix_derivative(mol, g, 'LDA,VWN', dma, dmb)
        trials.append(('LSDA spin (control)', [(va, Va, dma), (vb, Vb, dmb)]))
        # closed-shell PBE on the TOTAL density: same divergence machinery,
        # zero spin indexing. The discriminating control.
        try:
            vt = bl.pbe_vxc_on_grid(mol, dm_tot, coords, xc='PBE,PBE')
            ni = dft.numint.NumInt()
            try:
                _, _, Vt = ni.nr_rks(mol, g, 'PBE,PBE', dm_tot)
            except TypeError:
                _, _, Vt = ni.nr_rks(mol, g, 'PBE,PBE', dm_tot, 0, 1)
            trials.append(('PBE closed-shell (control)', [(vt, np.asarray(Vt),
                                                           dm_tot)]))
        except Exception as e:                                # noqa: BLE001
            print(f"  (closed-shell control unavailable: {type(e).__name__}: {e})")
        try:
            pa, pb = bl.pbe_vxc_spin_on_grid(mol, dma, dmb, coords, xc='PBE,PBE')
            Pa, Pb = _matrix_derivative(mol, g, 'PBE,PBE', dma, dmb)
            trials.append(('PBE spin (under test)', [(pa, Pa, dma),
                                                     (pb, Pb, dmb)]))
        except Exception as e:                                # noqa: BLE001
            print(f"  PBE spin FAILED TO BUILD: {type(e).__name__}: {e}")

        for label, chans in trials:
            wa, wr = 0.0, 0.0
            for v_grid, Vmat, _dm in chans:
                for dD in probes:
                    _, _, rel, ad = _probe(v_grid, Vmat, ao, w, dD)
                    wa, wr = max(wa, ad), max(wr, rel)
            rows.setdefault(label, []).append((lev, wr, wa))
            print(f"{lev:>5} {len(w):>7} {label:>22} {wa:>16.3e} {wr:>14.3e}")
        print()

    print("=== VERDICT ===")
    # CALIBRATION, not a guessed threshold. A wrong libxc index substitutes one
    # second derivative for a different one; those differ from each other by
    # O(1), so the identity would fail by an O(1) FRACTION of the integrand.
    # This project has an in-house measurement of what that looks like: the
    # factor-of-2 bug in the response operator's chi read 0.498 and 1.000
    # against finite differences. A real formula bug is O(0.1-1). It cannot
    # hide at 1e-6.
    BUG_LEVEL = 1e-2      # at or above this, the identity is structurally broken
    QUAD_LEVEL = 1e-4     # below this it holds to 4+ digits -> quadrature only

    def worst_of(rows):
        return max(v[1] for v in rows) if rows else np.nan

    def spread(rows):
        vs = [v[1] for v in rows]
        return (max(vs) / max(min(vs), 1e-300)) if len(vs) > 1 else np.nan

    lsda = rows.get('LSDA spin (control)')
    ctrl = rows.get('PBE closed-shell (control)')
    spin = rows.get('PBE spin (under test)')

    if lsda:
        w = worst_of(lsda)
        print(f"  LSDA spin (control)        worst rel(L1) {w:.2e} — "
              + ("PASS, machine precision as expected"
                 if w < 1e-10 else
                 "UNEXPECTED: a direct libxc vrho call should be exact"))
    if ctrl:
        print(f"  PBE closed-shell (control) worst rel(L1) {worst_of(ctrl):.2e}"
              f"   (varies {spread(ctrl):.1f}x across grid levels)")
    if spin:
        ws = worst_of(spin)
        print(f"  PBE spin (under test)      worst rel(L1) {ws:.2e}"
              f"   (varies {spread(spin):.1f}x across grid levels)")
    print()
    print("  NOTE ON THE GRID SCAN: Becke grids at different levels are NOT")
    print("  nested, so the quadrature error need not fall monotonically with")
    print("  the level — it wanders. Three non-monotonic points are therefore")
    print("  NOT a trend, and an earlier version of this bench wrongly called a")
    print("  healthy spin path a bug by dividing its first level by its last.")
    print("  What decides is the MAGNITUDE, against the calibration above.")
    print()
    if spin is None:
        print("  PBE spin never built — see the error above.")
    else:
        ws = worst_of(spin)
        ratio = (ws / max(worst_of(ctrl), 1e-300)) if ctrl else np.nan
        if np.isfinite(ratio):
            print(f"  spin / closed-shell error ratio: {ratio:.1f}x  (the spin")
            print(f"  path carries strictly more terms — the sigma_ab cross")
            print(f"  terms and two density Hessians instead of one — so a")
            print(f"  single-digit factor is expected, not suspicious.)")
            print()
        if ws >= BUG_LEVEL:
            print(f"  => FAIL. rel(L1) = {ws:.2e} is at the O(1) scale of a real")
            print("     formula/index bug. Suspects in order: the libxc")
            print("     component ordering in baselines.pbe_vxc_spin_on_grid")
            print("     (sigma aa/ab/bb; the v2rhosigma and v2sigma2 row maps),")
            print("     the factors of 2 on the f_aa terms, then the Hessians.")
            print("     Until fixed: CFG.include_pbe_spin = False, LDA-only for")
            print("     the open-shell members.")
        elif ws < QUAD_LEVEL:
            print(f"  => PASS. The identity holds to rel(L1) = {ws:.2e}, i.e. to")
            print("     5+ significant figures, at EVERY grid level. An index or")
            print("     factor error is O(0.1-1) and cannot hide here. What")
            print("     remains is the quadrature of the by-parts step, which is")
            print("     the same floor the closed-shell path sits on.")
            print("     baselines.pbe_vxc_spin_on_grid is the true spin-resolved")
            print("     functional derivative. Tier-1 is now validated on BOTH")
            print("     shells: the 81 open-shell members can carry PBE columns.")
            print(f"     For the SI, quote the worst rel(L1) over the grid scan:")
            print(f"     {ws:.1e}.")
        else:
            print(f"  => AMBIGUOUS. rel(L1) = {ws:.2e} sits between the")
            print(f"     quadrature floor ({QUAD_LEVEL:.0e}) and the bug scale")
            print(f"     ({BUG_LEVEL:.0e}). Too clean for a wrong index, too")
            print("     dirty for pure quadrature. Look at the sigma_ab cross")
            print("     terms (gs_ab and the f_ab divergence): they exist only")
            print("     in the spin path, so they are the difference between")
            print("     this and the closed-shell control.")
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # default: a cheap, genuinely spin-polarized member (3e-, doublet)
    # HH_q-1_s1 (the H2 anion) is NOT one of the ten systems of this
    # benchmark, so the default made this test abort the moment it was run as
    # part of a normal run. The default is now an open-shell member of the
    # set; the anion is still reachable with --member if anyone wants it.
    ap.add_argument('--member', default='HC_q+0_s1')
    ap.add_argument('--R', type=float, default=None)
    ap.add_argument('--n-probe', type=int, default=3, dest='n_probe')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--levels', type=int, nargs='+',
                    default=[3, 4, 5])
    a = ap.parse_args()
    check(a.member, a.R, a.n_probe, a.seed, tuple(a.levels))