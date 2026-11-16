#!/usr/bin/env python3
"""
test_spin_channel_d2.py — regression test for FIX-D2.
=====================================================
D2: the open-shell metric formed its comparison potential as the unconditional
average (v_alpha + v_beta)/2. For H2+ the beta channel holds NO electrons, so
what libxc returns for it is a spin derivative evaluated at its internal
density floor, not a potential. Half of that entered the metric and produced
the DeltaI_bar = 8.132 Eh of Table S10, against 0.063 for LDA on the same
system.

The test rebuilds H2+ from scratch (one electron: the reference density is an
SCF orbital and v_xc = -v_H analytically, so no saved artifact is needed) and
asserts:

  T1  the beta channel of H2+ is empty to machine precision;
  T2  the OLD rule reproduces the published anomaly (>= 1 Eh) — i.e. the test
      is actually exercising the bug and not some other code path;
  T3  the NEW rule gives a value of the same order as LDA on the same system
      (ratio PBE/LDA below 5, against ~129 published);
  T4  the OLD value tracks the libxc density floor and the NEW one does not —
      the property that makes it an artifact rather than a physical number;
  T5  a genuinely open-shell system (both channels occupied) is UNCHANGED by
      the fix, to machine precision. This is the guard against "fixing" H2+ by
      quietly moving every other open-shell number.

Usage:  python test_spin_channel_d2.py
"""
import os
import sys

import numpy as np
from pyscf import gto, scf, dft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baselines as bl              # noqa: E402
import functional_eval as fe        # noqa: E402
import metrics as mt                # noqa: E402
import molecules as mo              # noqa: E402
from config import CFG              # noqa: E402


def system(atom, charge, spin, R):
    mol = gto.M(atom=atom, unit='Angstrom', basis=CFG.benchmark_basis,
                charge=charge, spin=spin, verbose=0)
    mf = (scf.ROHF(mol) if spin else scf.RHF(mol)).run()
    if not mf.converged:
        mf = mf.newton().run()
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dma, dmb = dm[0], dm[1]
    else:
        dma = dmb = 0.5 * dm
    g = dft.gen_grid.Grids(mol)
    g.level = CFG.grid_level
    g.build()
    coords, w = g.coords, g.weights
    ao = mol.eval_gto('GTOval', coords)
    ra = np.einsum('gi,ij,gj->g', ao, dma, ao)
    rb = np.einsum('gi,ij,gj->g', ao, dmb, ao)
    return dict(mol=mol, dma=dma, dmb=dmb, coords=coords, w=w, rho_a=ra,
                rho_b=rb, rho=ra + rb, R=R)


def dI(s, v):
    _, d, _ = mt.delta_metrics(s['vxc'], v, s['rho'], s['coords'], s['w'],
                               s['R'])
    return float(d)


def main():
    fails = []

    # ---------------- H2+ : one electron, empty beta channel --------------
    R = 0.64
    s = system(f'H 0 0 {-R/2}; H 0 0 {R/2}', charge=1, spin=1, R=R)
    s['vxc'] = -mo.hartree_potential_on_points(s['mol'], s['dma'], s['coords'])

    nb = float(np.sum(s['rho_b'] * s['w']))
    print(f"T1 N_beta(H2+) = {nb:.3e}")
    if abs(nb) > 1e-10:
        fails.append(f'T1: beta channel is not empty (N_beta={nb:.3e})')

    vals = {}
    for tag, xc in (('LDA', 'LDA,VWN'), ('PBE', 'PBE,PBE')):
        if tag == 'LDA':
            va, vb = bl.lsda_vxc_spin(s['rho_a'], s['rho_b'], xc=xc)
        else:
            va, vb = bl.pbe_vxc_spin_on_grid(s['mol'], s['dma'], s['dmb'],
                                             s['coords'], xc=xc)
        old = dI(s, 0.5 * (va + vb))
        vnew, prov = fe._spin_average_potential(va, vb, s['rho_a'], s['rho_b'],
                                                s['w'])
        new = dI(s, vnew)
        vals[tag] = (old, new, prov['spin_average'])
        print(f"   {tag}: old rule {old:9.4f} Eh   new rule {new:8.4f} Eh   "
              f"[{prov['spin_average']}]")

    print(f"T2 old PBE = {vals['PBE'][0]:.4f} Eh (published: 8.132)")
    if vals['PBE'][0] < 1.0:
        fails.append('T2: the old rule no longer reproduces the anomaly; the '
                     'test is not exercising the bug')
    if vals['PBE'][2] != 'alpha_only':
        fails.append(f"T2: expected alpha_only, got {vals['PBE'][2]}")

    ratio = vals['PBE'][1] / vals['LDA'][1]
    print(f"T3 PBE/LDA with the new rule = {ratio:.2f} "
          f"(published ratio {8.132/0.063:.0f})")
    if not 0.2 < ratio < 5.0:
        fails.append(f'T3: PBE/LDA = {ratio:.2f} is still not of order one')

    # ---- T4: does the number follow the density floor? -------------------
    ra, ga, Ha, la = bl.density_derivs(s['mol'], s['dma'], s['coords'])
    rb_, gb, Hb, lb = bl.density_derivs(s['mol'], s['dmb'], s['coords'])
    gs_aa = 2.0 * np.einsum('abg,bg->ag', Ha, ga)
    floors = {}
    for floor in (1e-10, 1e-15):
        sa = np.vstack([np.maximum(ra, floor), ga])
        sb = np.vstack([np.maximum(rb_, floor), gb])
        _, vxc, fxc, _ = dft.libxc.eval_xc('PBE,PBE', (sa, sb), spin=1,
                                           deriv=2)
        vrho, vs = np.asarray(vxc[0]), np.asarray(vxc[1])
        v2rs, v2ss = np.asarray(fxc[1]), np.asarray(fxc[2])
        g_fab = v2rs[:, 1] * ga + v2rs[:, 4] * gb + v2ss[:, 1] * gs_aa
        vb_f = vrho[:, 1] - (np.einsum('ag,ag->g', g_fab, ga) + vs[:, 1] * la)
        va_f = vrho[:, 0] - 2.0 * (
            np.einsum('ag,ag->g', v2rs[:, 0] * ga + v2ss[:, 0] * gs_aa, ga)
            + vs[:, 0] * la)
        floors[floor] = (dI(s, 0.5 * (va_f + vb_f)), dI(s, va_f))
    o10, n10 = floors[1e-10]
    o15, n15 = floors[1e-15]
    print(f"T4 floor 1e-10 -> 1e-15:  old rule {o10:.4f} -> {o15:.4f} "
          f"({abs(o15-o10)/max(o10,1e-30)*100:.0f}% change)   "
          f"new rule {n10:.4f} -> {n15:.4f} "
          f"({abs(n15-n10)/max(n10,1e-30)*100:.3f}% change)")
    if abs(o15 - o10) / max(o10, 1e-30) < 0.5:
        fails.append('T4: the old value does not track the density floor; '
                     'the artifact diagnosis would not hold')
    if abs(n15 - n10) / max(n10, 1e-30) > 1e-3:
        fails.append('T4: the new value tracks the density floor too')

    # ---- T5: a truly open-shell system must not move ---------------------
    R2 = 1.24
    t = system(f'He 0 0 {-R2/2}; He 0 0 {R2/2}', charge=1, spin=1, R=R2)
    t['vxc'] = np.zeros_like(t['rho'])       # value irrelevant: we compare
    na = float(np.sum(t['rho_a'] * t['w']))
    nb2 = float(np.sum(t['rho_b'] * t['w']))
    va, vb = bl.pbe_vxc_spin_on_grid(t['mol'], t['dma'], t['dmb'], t['coords'])
    old = dI(t, 0.5 * (va + vb))
    vnew, prov = fe._spin_average_potential(va, vb, t['rho_a'], t['rho_b'],
                                            t['w'])
    new = dI(t, vnew)
    print(f"T5 He2+ (N_alpha={na:.3f}, N_beta={nb2:.3f}) [{prov['spin_average']}]"
          f": old {old:.6f}  new {new:.6f}  diff {abs(new-old):.3e}")
    if prov['spin_average'] != 'both':
        fails.append(f"T5: He2+ classified as {prov['spin_average']}, "
                     "but both its channels are occupied")
    if abs(new - old) > 1e-12:
        fails.append(f'T5: a genuinely open-shell system moved by '
                     f'{abs(new-old):.3e}')

    print()
    if fails:
        for f in fails:
            print('FAIL ', f)
        return 1
    print('ALL PASS — empty spin channels no longer vote (D2 fixed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
