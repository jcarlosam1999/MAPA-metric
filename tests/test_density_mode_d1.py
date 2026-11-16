#!/usr/bin/env python3
"""
test_density_mode_d1.py — regression test for FIX-D1.
=====================================================
D1: `make_comparative_figure` evaluated the functional on the EXACT density
regardless of `density_mode`, while naming the column after the mode. Result:
`Vxc_LDA-VWN_exact` was bit-for-bit identical to `Vxc_LDA-VWN_scf` in all 203
published 1D profiles, so the error-cancellation signal the article's title
announces was not reproducible from the deposit.

This test builds a SELF-CONTAINED miniature run (one geometry of H2 in a small
basis, with a deliberately non-SCF "exact" density) and asserts:

  T1  _density_for_mode returns different matrices for 'exact' and 'scf';
  T2  the two written columns are NOT identical (the D1 signature);
  T3  the 'exact' column equals the functional evaluated on the exact DM
      (i.e. the fix did not silently change the mode that was already right);
  T4  when the scf density cannot be produced, NO column is written
      (a missing column is data; a mislabelled one is not).

Runs in seconds, needs no results tree and no .npz from a production run.
Usage:  python test_density_mode_d1.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from pyscf import gto, scf, dft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import functional_eval as fe   # noqa: E402

BASIS = '6-31G'
R_ANG = 0.740
NZ = 41


def _build_fixture(root):
    """One geometry of H2: an npz whose 'exact' density is a CISD density
    (genuinely not any functional's SCF density) plus the 1D profile CSV the
    writer merges into."""
    npz_dir = os.path.join(root, 'results_grid_data')
    vxc_dir = os.path.join(root, 'results_vxc_1D')
    os.makedirs(npz_dir)
    os.makedirs(vxc_dir)

    mol = gto.M(atom=f'H 0 0 {-R_ANG/2}; H 0 0 {R_ANG/2}', unit='Angstrom',
                basis=BASIS, charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    from pyscf import ci
    exact_dm = ci.CISD(mf).run().make_rdm1(ao_repr=True)

    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    coords, weights = grids.coords, grids.weights
    ao = mol.eval_gto('GTOval', coords)
    rho_exact = np.einsum('gi,ij,gj->g', ao, exact_dm, ao)
    # a stand-in "exact" v_xc: only its SHAPE matters here, never its value —
    # this test is about which density feeds the functional, not about physics.
    vxc_exact = -0.5 * np.ones_like(rho_exact)

    meta = dict(name='HH_q+0_s0', R_ang=R_ANG, a1='H', a2='H', charge=0,
                spin=0, benchmark_basis=BASIS, ncas=2, nelecas=[1, 1],
                anion_caveat=False)
    np.savez_compressed(
        os.path.join(npz_dir, f'HH_q+0_s0_R{R_ANG:.2f}.npz'),
        coords_bohr=coords, weights=weights, rho_exact=rho_exact,
        vxc_exact=vxc_exact, exact_dm=exact_dm, D_ks=exact_dm,
        meta=json.dumps(meta))

    z = np.linspace(-3.0, 3.0, NZ)
    pd.DataFrame({'Z_Bohr': z, 'Z_Angstrom': z * 0.52917721092,
                  'Vxc_Exact': np.zeros(NZ)}).to_csv(
        os.path.join(vxc_dir, f'HH_q+0_s0_Vxc_1D_R{R_ANG:.2f}.csv'),
        index=False)
    return npz_dir, vxc_dir, mol, exact_dm, z


def main():
    root = tempfile.mkdtemp(prefix='d1_test_')
    failures = []
    try:
        npz_dir, vxc_dir, mol, exact_dm, z = _build_fixture(root)
        finfo = fe.resolve_functional(('LDA-VWN', 'LDA,VWN'))
        csv = os.path.join(vxc_dir, f'HH_q+0_s0_Vxc_1D_R{R_ANG:.2f}.csv')

        # ---- T1: the two modes are two different densities ---------------
        ctx = fe._load_geometry_context(
            os.path.join(npz_dir, f'HH_q+0_s0_R{R_ANG:.2f}.npz'))
        d_ex = fe._density_for_mode(ctx, finfo, 'exact')['dm_tot']
        d_scf = fe._density_for_mode(ctx, finfo, 'scf')['dm_tot']
        gap = float(np.abs(np.asarray(d_ex) - np.asarray(d_scf)).max())
        print(f'T1 max|D_exact - D_scf| = {gap:.3e}')
        if gap < 1e-8:
            failures.append('T1: the two modes returned the same density '
                            'matrix; the fixture is degenerate')

        # ---- T2: the written columns differ ------------------------------
        for mode in ('exact', 'scf'):
            fe.make_comparative_figure('HH_q+0_s0', finfo, density_mode=mode,
                                       draw_figure=False, vxc_dir=vxc_dir,
                                       npz_dir=npz_dir,
                                       energies_dir=os.path.join(root, 'none'))
        df = pd.read_csv(csv)
        a, b = 'Vxc_LDA-VWN_exact', 'Vxc_LDA-VWN_scf'
        if a not in df or b not in df:
            failures.append(f'T2: columns not written ({list(df.columns)})')
        else:
            dmax = float(np.abs(df[a] - df[b]).max())
            print(f'T2 max|Vxc_exact - Vxc_scf| on the line = {dmax:.3e} Eh')
            if np.array_equal(df[a].to_numpy(), df[b].to_numpy()):
                failures.append('T2: the two columns are STILL identical '
                                'bit for bit — D1 is not fixed')
            elif dmax < 1e-10:
                failures.append(f'T2: columns differ by only {dmax:.2e} Eh, '
                                'below any plausible density-driven error')

            # ---- T3: the exact column is unchanged by the fix ------------
            line = np.column_stack([np.zeros(NZ), np.zeros(NZ), z])
            ref, _ = fe.functional_vxc_on_grid(finfo, mol, exact_dm, line)
            err = float(np.abs(df[a].to_numpy() - ref).max())
            print(f'T3 max|exact column - v_xc[D_exact]| = {err:.3e} Eh')
            if err > 1e-12:
                failures.append(f'T3: the exact column moved by {err:.2e} Eh; '
                                'the fix must not touch the mode that was '
                                'already correct')

        # ---- T4: no density -> no column ---------------------------------
        os.remove(os.path.join(npz_dir, f'HH_q+0_s0_R{R_ANG:.2f}.npz'))
        pd.DataFrame({'Z_Bohr': z, 'Z_Angstrom': z * 0.52917721092,
                      'Vxc_Exact': np.zeros(NZ)}).to_csv(csv, index=False)
        fe.make_comparative_figure('HH_q+0_s0', finfo, density_mode='scf',
                                   draw_figure=False, vxc_dir=vxc_dir,
                                   npz_dir=npz_dir,
                                   energies_dir=os.path.join(root, 'none'))
        cols = set(pd.read_csv(csv).columns)
        print(f'T4 columns written with no density available: '
              f'{sorted(c for c in cols if c.startswith("Vxc_LDA"))}')
        if any(c.startswith('Vxc_LDA') for c in cols):
            failures.append('T4: a column was written without a density')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        for f in failures:
            print('FAIL ', f)
        return 1
    print('ALL PASS — density_mode branches in the 1D writer (D1 fixed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
