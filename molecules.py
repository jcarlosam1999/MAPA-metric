"""
molecules.py — builders, SCF helpers, grids, electrostatics.
============================================================
Everything that turns a benchmark member + a distance into PySCF objects and
grid-space fields, plus the SCF machinery with the convergence insurance an
unattended scan needs.

Key facts encoded here (units are a historical failure mode of this
project — see FIX-1 in the changelog): molecule inputs are ANGSTROM,
`grids.coords` and every grid-space quantity are BOHR.
"""

import numpy as np
from pyscf import gto, scf, dft

from config import CFG, BOHR, grid_level_for


# ---------------------------------------------------------------------------
# Structure builders
# ---------------------------------------------------------------------------
def build_molecule(a1, a2, r_ang, charge, spin, basis=None):
    """Diatomic on the z axis, atoms at +-R/2 (the bond midpoint is the
    origin — the metric masks and the potential-basis ghost rely on this)."""
    return gto.M(atom=f'{a1} 0 0 {-r_ang/2}; {a2} 0 0 {r_ang/2}',
                 unit='Angstrom', basis=basis or CFG.benchmark_basis,
                 charge=charge, spin=spin,
                 symmetry=CFG.use_symmetry, verbose=0)


def build_potential_mol(a1, a2, r_ang):
    """Auxiliary molecule carrying the POTENTIAL basis: identical
    even-tempered s+p sets on both atoms (benchmark homogeneity) plus an s set on
    a bond-midpoint ghost, where the static-correlation peak lives.
    Near-linear dependencies of these diffuse sets are handled downstream
    by canonical-orthogonalization pruning (inversion.py)."""
    etb_atom = gto.expand_etbs(CFG.pot_basis_atom)
    etb_mid = gto.expand_etbs(CFG.pot_basis_mid)
    basis = {a1: etb_atom, a2: etb_atom}
    for ghost in ('ghost-H', 'X-H'):        # label fallback across versions
        try:
            bg = dict(basis)
            bg[ghost] = etb_mid
            return gto.M(
                atom=f'{a1} 0 0 {-r_ang/2}; {a2} 0 0 {r_ang/2}; '
                     f'{ghost} 0 0 0',
                unit='Angstrom', basis=bg, charge=0,
                spin=(gto.charge(a1) + gto.charge(a2)) % 2,
                symmetry=False, verbose=0)
        except Exception:
            continue
    raise RuntimeError("Could not build the potential-basis molecule.")


def r_ang_of(mol):
    z = mol.atom_coords(unit='Angstrom')[:, 2]
    return float(z.max() - z.min())


# ---------------------------------------------------------------------------
# SCF references with convergence insurance
# ---------------------------------------------------------------------------
def run_scf_reference(mol):
    """ROHF/RHF with a second-order (Newton) fallback. The CASSCF quality —
    and therefore every downstream label — depends on this reference, so a
    silently unconverged SCF is not an option."""
    mf = scf.ROHF(mol) if mol.spin != 0 else scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        mf = mf.newton().run()
    return mf


def run_uhf_broken_symmetry(mol):
    """UHF that actually breaks spin symmetry for singlets (asymmetric
    guess) + one stability pass. Plain UHF converges to the RHF solution
    for stretched singlets, silently reporting the WRONG dissociation
    curve — a real failure mode caught earlier in this project."""
    mf = scf.UHF(mol)
    dm0 = mf.get_init_guess()
    if mol.spin == 0:
        dm0 = (dm0[0] * 1.05, dm0[1] * 0.95)
    mf.kernel(dm0=dm0)
    try:
        mo_i = mf.stability()[0]
        mf = mf.run(mf.make_rdm1(mo_i, mf.mo_occ))
    except Exception:
        pass
    return mf


# ---------------------------------------------------------------------------
# Grids and geometry scans
# ---------------------------------------------------------------------------
def becke_grid(mol):
    g = dft.gen_grid.Grids(mol)
    g.level = grid_level_for(mol.nelectron)
    g.build()
    return g


def make_distance_grid(r_eq):
    """Single clean scan grid: moderate sampling of the compressed region, a
    DENSE bond-breaking zone (Re..2.5 Re, where Coulson-Fischer and the
    correlation peak live), and a tail that coarsens geometrically to
    dissociation with the gap capped at max_gap.

    Replaces the old approach of concatenating two independent linspaces
    (coarse + dense), which produced near-duplicate doublets wherever the two
    grids nearly coincided (e.g. 0.94/0.95, 1.45/1.46) and a hard density
    cliff at 2.8*Re. This builds one monotone grid with no merged sub-grids,
    no blind midpoint insertion, and no duplicates."""
    r_min = max(0.6 * r_eq, CFG.r_min_floor)
    r_break_end = min(2.5 * r_eq, r_eq + 2.0)
    r_max = r_eq + CFG.r_stretch_abs

    # Zone A: compressed r_min -> Re (moderate, uniform)
    zA = np.linspace(r_min, r_eq, CFG.n_compressed, endpoint=False)
    # Zone B: bond-breaking Re -> 2.5 Re (dense, uniform — the physics)
    zB = np.linspace(r_eq, r_break_end, CFG.n_breaking, endpoint=False)
    # Zone C: dissociation tail, gaps growing geometrically from the
    # breaking-zone spacing up to max_gap (guarantees monotone spacing and
    # never exceeds max_gap, so no insertion is ever needed).
    gap_break = (r_break_end - r_eq) / CFG.n_breaking
    zC = [r_break_end]
    g = gap_break
    while zC[-1] < r_max:
        g = min(g * CFG.tail_growth, CFG.max_gap)
        zC.append(round(zC[-1] + g, 3))

    R = np.sort(np.unique(np.round(np.concatenate([zA, zB, np.array(zC)]), 3)))
    # final safety net: collapse any residual near-duplicate below min_gap
    keep = [R[0]]
    for r in R[1:]:
        if r - keep[-1] >= CFG.min_gap:
            keep.append(r)
    return np.array(keep)


def dense_axis_line(r_ang):
    """Uniform 1D line along the bond axis (BOHR) for smooth v_xc profiles —
    possible because the basis-form potential evaluates anywhere."""
    half = (r_ang / 2 + CFG.line_pad_ang) / BOHR
    z = np.linspace(-half, half, CFG.n_line_points)
    pts = np.zeros((z.size, 3))
    pts[:, 2] = z
    return z, pts


# ---------------------------------------------------------------------------
# Electrostatics on arbitrary points
# ---------------------------------------------------------------------------
def hartree_potential_on_points(mol, dm, points_bohr, batch=256):
    """v_H(r) on arbitrary points via int1e_grids, batched to bound the
    (nbatch, nao, nao) memory. Used for the Fermi-Amaldi guide (-v_H/N,
    correct -1/r asymptote for any N), the 1-electron analytic validation
    (vxc = -v_H exactly), and the exact 2e exchange (v_x = -v_H/2)."""
    n = points_bohr.shape[0]
    v = np.empty(n)
    for i0 in range(0, n, batch):
        sl = slice(i0, min(i0 + batch, n))
        ints = mol.intor('int1e_grids', grids=points_bohr[sl])
        v[sl] = np.einsum('gij,ij->g', ints, dm)
    return v


def fermi_amaldi_on_points(mol, dm, points_bohr, v_h=None):
    """-v_H/N: exact v_xc for 1 electron, exact v_x for a 2e singlet, and the
    correct -1/r asymptote for any N."""
    if v_h is None:
        v_h = hartree_potential_on_points(mol, dm, points_bohr)
    return -v_h / max(mol.nelectron, 1)


def xc_guide_on_points(mol, dm, points_bohr, rho=None, v_h=None):
    """The Wu-Yang GUIDE potential on arbitrary points (grid or 1D line).

    v_guide = v_FA + w(rho) * (v_xc^LDA[rho] - v_FA),   w = rho^2/(rho^2+rs^2)

    The guide is the PRIOR the HBY regularizer pulls toward, so it is
    inherited wherever the density does not constrain the fit. Fermi-Amaldi is
    exact for 1e/2e-singlets but is off by ~3.8 Eh in a many-electron CORE,
    where the 1s density is frozen and cannot push back — see the measured
    table in config.Config. LDA is off by 0.45 Eh there; in the valence and
    the tail the ordering reverses (only FA has the -1/r asymptote), hence the
    density switch. See CFG.guide_lda_core / CFG.guide_rho_switch.

    Pass `rho` and `v_h` when the caller already has them on these points (the
    pipeline does, on the grid) to avoid recomputing.

    Returns (v_guide, v_fa) — v_fa is returned too because the 1D-profile CSV
    and the npz record BOTH, so the guide's effect is auditable per geometry.
    """
    if v_h is None:
        v_h = hartree_potential_on_points(mol, dm, points_bohr)
    v_fa = -v_h / max(mol.nelectron, 1)
    # GATE: FA is EXACT here — never blend (1e: full v_xc; 2e singlet: v_x).
    # This keeps every already-validated light member bit-for-bit unchanged.
    if not CFG.guide_lda_core or mol.nelectron <= 2:
        return v_fa.copy(), v_fa
    if rho is None:
        ao = mol.eval_gto('GTOval', points_bohr)
        rho = dft.numint.eval_rho(mol, ao, dm, xctype='LDA')
    rho = np.maximum(np.asarray(rho), 0.0)
    _, vxc, _, _ = dft.libxc.eval_xc('LDA,VWN', np.maximum(rho, 1e-15),
                                     deriv=1)
    v_lda = np.asarray(vxc[0])
    rs2 = CFG.guide_rho_switch ** 2
    w = rho ** 2 / (rho ** 2 + rs2)          # 0 in the tail, 1 in the core
    return v_fa + w * (v_lda - v_fa), v_fa