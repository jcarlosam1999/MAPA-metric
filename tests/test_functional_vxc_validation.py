"""
test_functional_vxc_validation.py — pre-publication validation of Tier-1 v_xc
=============================================================================
The DeltaI metric compares the exact multiplicative v_xc against a functional's
v_xc ON THE GRID. For that grid potential to be correct, it must be the true
functional derivative of E_xc. The rigorous check (the protocol stated in
baselines.py) is the functional-derivative identity: for any density
perturbation dD,

        INT v_xc(r) drho(r) dr  ==  Tr[ V_xc_matrix . dD ]

where drho is the grid density change from dD and V_xc_matrix is pyscf's own
XC matrix (the reference). If the grid v_xc from functional_eval matches
pyscf's matrix elements to numerical precision for random perturbations, the
Tier-1 potential is validated and safe for publication figures.

This bench runs the check for LDA and GGA (Tier 1) on H2 at a few geometries.
Tier-2 (meta-GGA) will FAIL this test with the multiplicative-only convention
by construction (the tau part is missing) -- that failure is the quantitative
statement of why meta-GGA needs a convention, and the bench reports the size of
the discrepancy so it can be characterized honestly.

HOW TO RUN (from refactor/, in your qchem env):
    python test_functional_vxc_validation.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyscf import dft, gto                                   # noqa: E402
import molecules as mo                                       # noqa: E402
import baselines as bl                                       # noqa: E402
import functional_eval as fe                                 # noqa: E402


FUNCTIONALS = [
    ('LDA,VWN', 'LDA — Tier 1, expect PASS'),
    ('PBE,PBE', 'GGA — Tier 1, expect PASS'),
    # meta-GGA included to QUANTIFY the tau-omission discrepancy, not to pass:
    ('SCAN,SCAN', 'meta-GGA — Tier 2, expect FAIL (tau omitted, by design)'),
]
R_TEST = [0.74, 1.45, 2.21]


def _grids(mol):
    g = dft.gen_grid.Grids(mol)
    g.level = 4
    g.build()
    return g


def grid_vxc_dot_drho(mol, xc, dm, dD, coords, w):
    """LHS of the identity: INT v_xc(r) drho(r) dr, with v_xc from
    functional_eval's grid potential and drho from the perturbation dD."""
    finfo = fe.resolve_functional(xc)
    v_grid, prov = fe.functional_vxc_on_grid(finfo, mol, dm, coords)
    # drho(r) from dD on the grid
    ao = mol.eval_gto('GTOval_sph', coords)
    drho = np.einsum('gi,ij,gj->g', ao, dD, ao)
    return float(np.sum(v_grid * drho * w)), prov


def matrix_vxc_dot_dD(mol, xc, dm, dD):
    """RHS of the identity: Tr[V_xc_matrix . dD] using pyscf's matrix."""
    g = _grids(mol)
    ni = dft.numint.NumInt()
    _, _, vmat = ni.nr_rks(mol, g, xc, dm)
    return float(np.einsum('ij,ij->', vmat, dD))


def run_one(xc, r_ang, seed=0):
    mol = mo.build_molecule('H', 'H', r_ang, 0, 0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.kernel()
    dm = mf.make_rdm1()

    g = _grids(mol)
    coords, w = g.coords, g.weights

    # random symmetric perturbation dD (small)
    rng = np.random.default_rng(seed)
    nao = dm.shape[0]
    A = rng.standard_normal((nao, nao))
    dD = 1e-4 * (A + A.T)

    lhs, prov = grid_vxc_dot_drho(mol, xc, dm, dD, coords, w)
    rhs = matrix_vxc_dot_dD(mol, xc, dm, dD)
    rel = abs(lhs - rhs) / (abs(rhs) + 1e-30)
    return lhs, rhs, rel, prov


def main():
    print(f"{'functional':>12} {'R':>5} {'LHS(grid)':>13} {'RHS(matrix)':>13} "
          f"{'rel.err':>10}  verdict")
    print("-" * 72)
    for xc, label in FUNCTIONALS:
        for r in R_TEST:
            try:
                lhs, rhs, rel, prov = run_one(xc, r)
            except NotImplementedError as e:
                print(f"{xc:>12} {r:5.2f}  [Tier {prov if False else '2/3'}] "
                      f"not implemented: {str(e)[:40]}")
                continue
            except Exception as e:                   # noqa: BLE001
                print(f"{xc:>12} {r:5.2f}  ERROR: {type(e).__name__}: {e}")
                continue
            ok = rel < 1e-6
            verdict = "PASS" if ok else ("FAIL (expected for meta-GGA)"
                                         if 'SCAN' in xc else "FAIL <-- CHECK")
            print(f"{xc:>12} {r:5.2f} {lhs:13.3e} {rhs:13.3e} "
                  f"{rel:10.2e}  {verdict}")
        print()
    print("Interpretation:")
    print(" - LDA & PBE relative error < 1e-6 => Tier-1 grid v_xc is the true")
    print("   functional derivative. Validated for publication.")
    print(" - SCAN fails by a finite margin = the size of the omitted tau")
    print("   contribution. That margin quantifies why meta-GGA needs a")
    print("   convention; it is NOT a bug in the Tier-1 machinery.")


if __name__ == "__main__":
    main()