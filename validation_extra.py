"""
validation_extra.py -- the four controls behind SI Secs. S11.4, S11.5 and S13.
==============================================================================

`validation.py` holds the seven checks that were part of the original run.
These four came later, out of objections raised against the manuscript, and
each one exists because a specific claim could otherwise be waved away:

  `run_tier1_five`
      Scores the three remaining Tier-1 multiplicative functionals (BLYP,
      PBEsol, revPBE) on the exact side already computed, and adds the
      fixed-length integration windows to DeltaI_bond for all five. Feeds the
      five-functional table: without it, "the pattern is not an artefact of
      one functional" is an assertion.

  `run_fixed_volume_response`
      Sec. 2.1 claims a fixed-volume control, but that control had only ever
      been run on the POTENTIAL metric. It matters: with the production window
      the H2+ peak sits at R = 3.311 A and the paper reads it as "the
      self-interaction signature does not switch off on dissociation"; with a
      fixed cylinder the peak of DeltaI_bond moves to the most compressed
      geometry of the scan, which is the opposite reading. If the response
      metric did the same, the abstract would not stand. It does not, and that
      is the point of running it.

  `run_basis_convergence`
      Repeats one system in a larger ORBITAL basis, changing nothing else --
      not the potential basis, not the grid, not the reference ladder, not the
      inversion. A referee asking "is this a basis artefact?" gets a number.

  `run_n2v_cross_check`
      The same densities inverted by an independent, published code (n2v). If
      the external package is not installed the stage is skipped and says so;
      it is the one stage with a dependency outside `requirements.txt`, and a
      missing optional dependency must not fail a run.

None of these re-invert anything from scratch: they all post-process the .npz
the benchmark scan already wrote, except `run_basis_convergence`, which must
re-run the inversion by construction -- that is what it measures.

Only TIER-1 multiplicative functionals appear here. Hybrids need an OEP and
meta-GGAs need a convention for the kinetic-energy density that this work has
not validated, so B3LYP and r2SCAN are out of scope, and stay out even when
asked for.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

import artifacts as art
import functional_eval as fe
import metrics as mt
import response_metric as rm
from config import CFG, BOHR

NPZ_DIR = 'results_grid_data'

# libxc note: 'REVPBE,PBE' does not exist. The revPBE exchange is GGA_X_PBE_R.
TIER1 = [('LDA-VWN', 'LDA,VWN'), ('PBE-PBE', 'PBE,PBE'),
         ('BLYP', 'B88,LYP'), ('PBEsol', 'PBESOL,PBESOL'),
         ('revPBE', 'GGA_X_PBE_R,GGA_C_PBE')]
FIXED_HALF_LENGTHS = (2.0, 3.0, 4.0)     # bohr, cylinder half-length


# ---------------------------------------------------------------------------
# shared: the functional potential on the grid, by the production route
# ---------------------------------------------------------------------------
def _functional_potential(ctx, finfo, density_mode='exact'):
    """v_xc of `finfo` on the saved grid, exactly as production computes it.

    The open-shell branch is not optional. Using the closed-shell route on a
    spin-polarized system is wrong by 20-25%, and it inverted a conclusion of
    this project twice before the branch was made explicit everywhere it is
    needed. Returns (v_on_grid, rho_used_by_the_metric).
    """
    s = fe._density_for_mode(ctx, finfo, density_mode)
    if ctx['open_shell']:
        (va, vb), _ = fe.functional_vxc_spin_on_grid(
            finfo, ctx['mol'], s['dma'], s['dmb'],
            s['rho_a'], s['rho_b'], ctx['coords'])
        v, _ = fe._spin_average_potential(va, vb, s['rho_a'], s['rho_b'],
                                          ctx['w'])
        return np.asarray(v), s['rho_tot']
    v, _ = fe.functional_vxc_on_grid(finfo, ctx['mol'], s['dm_tot'],
                                     ctx['coords'])
    v = fe._sanitize_grid_potential(v, ctx['vxc_exact'], finfo.name, ctx['R'])
    return np.asarray(v), s['rho_tot']


def _fixed_mask(coords, half_bohr, cyl_rad):
    return ((np.abs(coords[:, 2]) <= half_bohr)
            & (coords[:, 0] ** 2 + coords[:, 1] ** 2 < cyl_rad ** 2))


def _dibar_potential(dv, w, mask):
    """Volume-weighted mean |dv - c| over the mask, c the weighted median.
    Same estimator as validation.dIbar_current; kept here so the fixed-window
    variants cannot drift from the production one by an implementation
    difference rather than by the window."""
    d, ww = dv[mask], w[mask]
    o = np.argsort(d)
    cw = np.cumsum(ww[o])
    c = d[o][np.searchsorted(cw, 0.5 * cw[-1])]
    return float(np.sum(ww * np.abs(d - c)) / ww.sum())


def _inv_status(npz_path):
    """The confidence gate's verdict for this geometry, from the .row.json the
    pipeline writes beside the .npz.

    Every peak the paper quotes is a maximum over the ADMITTED stratum, and a
    table that silently maximises over the rejected geometries too is not
    slightly wrong, it is a different quantity: HF has 3 admitted geometries
    of 21, and its peak moves from 0.086 to 0.489 Eh if the gate is ignored.
    Carrying the verdict into the artifact is what lets the check apply the
    gate at all -- without this column it filtered nothing and said nothing.
    """
    import json
    rj = npz_path[:-4] + '.row.json' if npz_path.endswith('.npz') else None
    if rj and os.path.isfile(rj):
        try:
            return str(json.load(open(rj)).get('inv_status', ''))
        except Exception:                                     # noqa: BLE001
            pass
    return ''


def _npz_paths(only=None):
    pat = f"{only}_R*.npz" if only else "*_R*.npz"
    return sorted(glob.glob(os.path.join(NPZ_DIR, pat)))


# ---------------------------------------------------------------------------
# [1] five Tier-1 functionals + fixed-volume windows on the potential metric
# ---------------------------------------------------------------------------
def run_tier1_five(log=print):
    paths = _npz_paths()
    if not paths:
        raise SystemExit(f"no npz in {NPZ_DIR}: run the benchmark scan first")
    cyl = getattr(CFG, 'cyl_radius_bohr', 2.0)
    log(f"[tier1_five] {len(paths)} geometries x {len(TIER1)} functionals "
        f"(exact mode)")
    rows = []
    for p in paths:
        name = os.path.basename(p).rsplit('_R', 1)[0]
        try:
            ctx = fe._load_geometry_context(p)
        except Exception as e:                                # noqa: BLE001
            log(f"  {os.path.basename(p)}: unreadable ({type(e).__name__})")
            continue
        coords, w, R = ctx['coords'], ctx['w'], ctx['R']
        vxc = np.asarray(ctx['vxc_exact'])
        prod = rm.bond_mask(coords, R, cyl, BOHR)
        row = dict(system=name, R_Angstrom=R,
                   open_shell=bool(ctx['open_shell']),
                   inv_status=_inv_status(p))
        for label, xc in TIER1:
            finfo = fe.resolve_functional(xc)
            v, rho_m = _functional_potential(ctx, finfo, 'exact')
            dv = vxc - v
            _, dI, _ = mt.delta_metrics(vxc, v, rho_m, coords, w, R)
            row[f'DeltaI_bar_{label}_exact_Eh'] = float(dI)
            for L in FIXED_HALF_LENGTHS:
                row[f'DeltaI_bar_fixedL{int(L)}_{label}_exact_Eh'] = \
                    _dibar_potential(dv, w, _fixed_mask(coords, L, cyl))
        rows.append(row)
    art.save_rows('tier1_five', rows)
    log(f"[tier1_five] {len(rows)} rows -> {art.path('tier1_five')}")
    return rows


# ---------------------------------------------------------------------------
# [2] fixed-volume control on the RESPONSE metric
# ---------------------------------------------------------------------------
def run_fixed_volume_response(systems=None, functional='PBE,PBE', log=print):
    """Recompute DeltaI_resp with cylinders of FIXED half-length as well as
    with the production window (|z| <= R/2).

    Only the mask changes. `response_metric.bond_mask` is patched and every
    other input -- chi, deg_tol, the code path -- is left alone, so a
    difference in the result can only come from the window.
    """
    # Table S11.4 reports TWO systems: H2+ (self-interaction) and H2 (static
    # correlation). Running one of them left half the table with no data and
    # the check reporting "no data" for rows the stage was simply never asked
    # to produce.
    systems = systems or getattr(CFG, 'fixed_volume_systems',
                                 ('HH_q+1_s1', 'HH_q+0_s0'))
    if isinstance(systems, str):
        systems = (systems,)
    paths = [q for s in systems for q in _npz_paths(s)]
    if not paths:
        raise SystemExit(f"no npz for {systems} in {NPZ_DIR}")
    cyl = getattr(CFG, 'cyl_radius_bohr', 2.0)
    finfo = fe.resolve_functional(functional)
    log(f"[fixed_volume] {', '.join(systems)}: {len(paths)} geometries, "
        f"windows production + {FIXED_HALF_LENGTHS} bohr")

    original = rm.bond_mask
    rows = []
    try:
        for p in paths:
            system = os.path.basename(p).rsplit('_R', 1)[0]
            ctx = fe._load_geometry_context(p)
            coords, w, R = ctx['coords'], ctx['w'], ctx['R']
            v, _ = _functional_potential(ctx, finfo, 'exact')
            dv = np.asarray(ctx['vxc_exact']) - v
            ao = ctx['mol'].eval_gto('GTOval_sph', coords)
            import inversion as iv
            J = iv.make_J(ctx['mol'], np.asarray(ctx['exact_dm']))
            D_ks = np.asarray(ctx['D_ks'] if ctx.get('D_ks') is not None
                              else ctx['exact_dm'])
            row = dict(system=system, R_Angstrom=R, functional=functional)

            rm.bond_mask = original
            r0 = rm.response_deltaI(ctx['mol'], coords, w, ao,
                                    ctx['vxc_exact'], D_ks, J, dv, R, cyl,
                                    BOHR)
            row['DeltaI_resp_production'] = float(r0['dI_resp'])
            for L in FIXED_HALF_LENGTHS:
                rm.bond_mask = (lambda Lb: (lambda c, r_ang, cr, bohr:
                                            _fixed_mask(c, Lb, cr)))(L)
                rL = rm.response_deltaI(ctx['mol'], coords, w, ao,
                                        ctx['vxc_exact'], D_ks, J, dv, R, cyl,
                                        BOHR)
                row[f'DeltaI_resp_fixedL{int(L)}'] = float(rL['dI_resp'])
            rows.append(row)
    finally:
        rm.bond_mask = original          # restore, whatever happened

    art.save_rows('fixed_volume_response', rows)
    for s in systems:
        mine = [r for r in rows if r['system'] == s]
        if not mine:
            continue
        for col in ['DeltaI_resp_production'] + \
                   [f'DeltaI_resp_fixedL{int(L)}' for L in FIXED_HALF_LENGTHS]:
            best = max(mine, key=lambda r: r[col])
            log(f"  {s}: peak of {col:<26} at R = "
                f"{best['R_Angstrom']:.3f} A")
    return rows


# ---------------------------------------------------------------------------
# [3] orbital-basis dependence
# ---------------------------------------------------------------------------
def run_basis_convergence(system=None, basis=None, log=print):
    """Repeat one system in a larger orbital basis and compare both metrics.

    This is the one control that has to re-invert: the whole question is what
    the inversion does when the orbital basis changes. Everything else --
    potential basis, grid, reference ladder, inversion engine, kappa protocol
    -- is the production setting, which is what makes the comparison readable.

    Runs in a subdirectory so the production deposit is never touched.
    """
    import shutil
    import subprocess
    import sys as _sys

    system = system or getattr(CFG, 'basis_convergence_system', 'HH_q+0_s0')
    basis = basis or getattr(CFG, 'basis_convergence_basis', 'aug-cc-pVQZ')
    workdir = os.path.abspath(f'results_basis_{basis}')
    ref_paths = _npz_paths(system)
    if not ref_paths:
        raise SystemExit(f"no npz for {system}: run the benchmark scan first")

    # The comparison is only meaningful geometry by geometry, so the larger
    # basis is run on exactly the R values production already has.
    Rs = []
    for p in ref_paths:
        d = np.load(p, allow_pickle=True)
        Rs.append(round(float(json.loads(str(d['meta']))['R_ang']), 3))
    log(f"[basis] {system} in {basis} at {len(Rs)} geometries -> {workdir}")

    os.makedirs(workdir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    code = (
        "import os, sys, numpy as np\n"
        f"sys.path.insert(0, {here!r})\n"
        f"os.chdir({workdir!r})\n"
        "from config import CFG, make_dirs\n"
        f"CFG.benchmark_basis = {basis!r}\n"
        "CFG.resume = True\n"
        "CFG.run_functional_postpass = False\n"
        "CFG.run_validations = False\n"
        "make_dirs()\n"
        "import molecules as mo\n"
        f"mo.make_distance_grid = lambda _r: np.array({Rs!r})\n"
        "from pipeline import run_pipeline\n"
        f"run_pipeline(only=[{system!r}])\n")
    r = subprocess.run([_sys.executable, '-c', code], text=True)
    if r.returncode != 0:
        raise SystemExit(f"the {basis} run failed (exit {r.returncode})")

    cyl = getattr(CFG, 'cyl_radius_bohr', 2.0)
    finfo = fe.resolve_functional('PBE,PBE')
    rows = []
    for R in Rs:
        pa = [p for p in ref_paths if f"_R{R:.2f}" in p]
        pb = sorted(glob.glob(os.path.join(workdir, NPZ_DIR,
                                           f"{system}_R{R:.2f}.npz")))
        if not pa or not pb:
            continue
        vals = {}
        for label, path in (('production', pa[0]), (basis, pb[0])):
            ctx = fe._load_geometry_context(path)
            coords, w = ctx['coords'], ctx['w']
            v, rho_m = _functional_potential(ctx, finfo, 'exact')
            _, dI, _ = mt.delta_metrics(ctx['vxc_exact'], v, rho_m, coords, w,
                                        ctx['R'])
            ao = ctx['mol'].eval_gto('GTOval_sph', coords)
            import inversion as iv
            J = iv.make_J(ctx['mol'], np.asarray(ctx['exact_dm']))
            D_ks = np.asarray(ctx['D_ks'] if ctx.get('D_ks') is not None
                              else ctx['exact_dm'])
            res = rm.response_deltaI(ctx['mol'], coords, w, ao,
                                     ctx['vxc_exact'], D_ks, J,
                                     np.asarray(ctx['vxc_exact']) - v,
                                     ctx['R'], cyl, BOHR)
            vals[label] = (float(dI), float(res['dI_resp']))
        if len(vals) == 2:
            (b0, r0), (b1, r1) = vals['production'], vals[basis]
            rows.append(dict(system=system, R_Angstrom=R,
                             basis_reference=CFG.benchmark_basis,
                             basis_test=basis,
                             dIbar_reference=b0, dIbar_test=b1,
                             dIbar_change_pct=100 * abs(b1 - b0) / max(b0, 1e-30),
                             resp_reference=r0, resp_test=r1,
                             resp_change_pct=100 * abs(r1 - r0) / max(r0, 1e-30)))
    art.save_rows('basis_convergence', rows)
    if rows:
        mb = float(np.median([r['dIbar_change_pct'] for r in rows]))
        mr = float(np.median([r['resp_change_pct'] for r in rows]))
        log(f"[basis] median change: potential {mb:.1f}%, response {mr:.1f}% "
            f"(factor {mb / max(mr, 1e-30):.0f})")
    return rows


# ---------------------------------------------------------------------------
# [4] independent inversion code
# ---------------------------------------------------------------------------
def run_n2v_cross_check(system=None, log=print):
    """Invert the same reference densities with n2v and compare.

    n2v is not in requirements.txt on purpose: it is a cross-check, not a
    dependency, and a run must not fail because an optional package is
    absent. If it cannot be imported the stage says so and returns nothing,
    which leaves no artifact and so leaves the stage pending rather than
    recording a result that was never obtained.
    """
    ok, why = _n2v_available()
    if not ok:
        log(f"[n2v] {why}")
        log(f"[n2v] stage skipped -- it reports itself as unchecked rather "
            f"than as passed.")
        return None

    system = system or getattr(CFG, 'n2v_system', 'HH_q+0_s0')
    paths = _npz_paths(system)
    if not paths:
        raise SystemExit(f"no npz for {system} in {NPZ_DIR}")
    log(f"[n2v] {system}: {len(paths)} geometries against the production "
        f"inversion")
    rows, failed = [], 0
    for p in paths:
        ctx = fe._load_geometry_context(p)
        row = dict(system=system, R_Angstrom=ctx['R'])
        try:
            row.update(_n2v_one(ctx))
        except Exception as e:                                # noqa: BLE001
            row['note'] = f"{type(e).__name__}: {e}"[:120]
            failed += 1
        rows.append(row)
    if failed == len(rows):
        # Every geometry raised the same way: that is one broken call, not a
        # cross-check with a negative result. Writing it as an artifact would
        # let `have()` count the stage as done and let the SI cite a table of
        # tracebacks. It happened on 2026-08-13, when `pip install n2v`
        # installed an image-denoising package of the same name.
        log(f"[n2v] all {failed} geometries raised: "
            f"{rows[0].get('note', '?')}")
        log(f"[n2v] no artifact written -- this is an adapter failure, not a "
            f"result.")
        return None
    if failed:
        log(f"[n2v] {failed} of {len(rows)} geometries raised; the rest are "
            f"in the artifact")
    art.save_rows('n2v_cross_check', rows)
    return rows


# The name `n2v` on PyPI is Noise2Void, a convolutional image denoiser. The
# inversion code this stage cross-checks against lives on GitHub only. They
# collide: `import n2v` succeeds against the wrong package, and the failure
# surfaces one call later as a TypeError about an unexpected keyword. Saying
# `pip install n2v` in a README is therefore an instruction that produces a
# wall of tracebacks; this checks which package answered.
N2V_SOURCE = 'pip install git+https://github.com/wasserman-group/n2v.git'


def _n2v_available():
    try:
        import n2v
    except Exception as e:                                    # noqa: BLE001
        return False, (f"not installed ({type(e).__name__}). Install the "
                       f"inversion code with: {N2V_SOURCE}")
    if not hasattr(n2v, 'Inverter'):
        return False, (f"the installed package named 'n2v' has no Inverter "
                       f"({getattr(n2v, '__file__', '?')}); PyPI's 'n2v' is "
                       f"Noise2Void, an image denoiser. Install the inversion "
                       f"code with: {N2V_SOURCE}")
    return True, ''


def _n2v_one(ctx):
    """One geometry through n2v's WuYang, on OUR reference density.

    Feeding n2v the same density is the whole point: any difference is the
    inverter, not the input. n2v's own regularization is left at its default,
    because the comparison being made is between published codes as they ship.

    The API, since it took three attempts to get right: n2v's Inverter takes
    only an engine name; the system, the target density, the orbital guess and
    the orbital energies are set as attributes afterwards. There is a
    `from_wfn` shortcut but it reads Psi4 wavefunction objects and cannot be
    used from PySCF.

    Two conventions have to be matched by hand, and getting either wrong gives
    a plausible-looking potential that is wrong by a smooth function:

      * n2v's guide with `fermi_amaldi` is (1 - 1/N)*J, i.e. Hartree AND the
        Fermi-Amaldi xc guide together, so the exchange-correlation part alone
        is that minus J: -J/N. Ours is -v_H/N to begin with.
      * the optimised correction lives in the auxiliary (`pbs`) basis, which
        with `pbs='same'` is the orbital basis. It is evaluated on OUR grid
        here rather than on n2v's own, so the two potentials are compared
        point by point on the same quadrature.
    """
    import n2v
    from pyscf import scf, dft
    import molecules as mo
    mol = ctx['mol']
    dm = np.asarray(ctx['exact_dm'])
    closed = not ctx.get('open_shell', False)

    # Orbital guess for the optimiser. n2v needs occupied coefficients and
    # energies; a correlated density has no orbitals of its own, so a KS
    # calculation supplies the starting point. It is a guess, not the target:
    # the target is `Dt`, which is our reference density.
    mf = scf.RKS(mol) if closed else scf.UKS(mol)
    mf.xc = 'pbe'
    mf.kernel()
    na, nb = mol.nelec
    if closed:
        ca = cb = mf.mo_coeff[:, :na]
        ea = eb = mf.mo_energy
        da = db = dm / 2.0
    else:
        ca, cb = mf.mo_coeff[0][:, :na], mf.mo_coeff[1][:, :nb]
        ea, eb = mf.mo_energy[0], mf.mo_energy[1]
        dma, dmb = ctx.get('dma'), ctx.get('dmb')
        da = np.asarray(dma) if dma is not None else dm / 2.0
        db = np.asarray(dmb) if dmb is not None else dm / 2.0

    ine = n2v.Inverter(engine='pyscf')
    ine.set_system(mol, mol.basis, ref=1 if closed else 2, pbs='same')
    ine.Dt = [da, db]
    ine.ct = [ca, cb]
    ine.et = [ea, eb]
    ine.invert('WuYang', guide_components='fermi_amaldi')

    pbs = getattr(ine.eng, 'pbs', None) or mol
    phi = dft.numint.eval_ao(pbs, ctx['coords'])
    v_pbs = np.asarray(ine.v_pbs).ravel()
    corr = phi.dot(v_pbs[:ine.npbs])
    v_h = mo.hartree_potential_on_points(mol, dm, ctx['coords'])
    v = -v_h / mol.nelectron + corr

    ours = np.asarray(ctx['vxc_exact'])
    w = ctx['w']
    shift = float(np.sum(w * (ours - v)) / np.sum(w))
    d = np.abs(ours - v - shift)
    # n2v's own density-reproduction error, on our grid and by our definition,
    # so the two inverters are judged by the same criterion the gate uses.
    rho_n2v = _density_on_grid(mol, ine, ctx, closed)
    derr = (float(np.sum(w * np.abs(rho_n2v - np.asarray(ctx['rho_exact']))))
            if rho_n2v is not None else float('nan'))
    return dict(max_abs_diff_Eh=float(d.max()),
                mean_abs_diff_Eh=float(np.sum(w * d) / np.sum(w)),
                gauge_shift_Eh=shift,
                n2v_density_error_e=derr,
                n2v_npbs=int(ine.npbs),
                n2v_version=str(getattr(n2v, '__version__', 'unknown')))


def _density_on_grid(mol, ine, ctx, closed):
    """The density n2v's inverted Kohn-Sham system actually reproduces.

    Reported because it is the criterion this work admits geometries by: a
    potential that does not reproduce the reference density is not a
    disagreement between inverters, it is one inverter failing, and the two
    must not be summarised by the same number.
    """
    from pyscf import dft
    try:
        ao = dft.numint.eval_ao(mol, ctx['coords'])
        da = np.asarray(ine.Da)
        db = np.asarray(ine.Db) if not closed else da
        rho = (np.einsum('pi,ij,pj->p', ao, da, ao)
               + np.einsum('pi,ij,pj->p', ao, db, ao))
        return rho
    except Exception:                                         # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# [5] the pathology contrast at a MATCHED reduced coordinate
# ---------------------------------------------------------------------------
#
# Sec. 4.5 and the caption of Fig. 3 contrast a self-interaction signature that
# "does not switch off as the fragments separate" against static-correlation
# signatures that "decay with the closing bond". Until 2026-08-14 that contrast
# was quoted as a max/dissociation ratio, and it does not survive: on the
# response metric it gives 2.5x for H2+ against 2.3x for H2, a factor of 1.09,
# which is no discrimination at all.
#
# The statistic was the problem, not the metric. Two defects:
#
#   * the maxima are not homologous points -- H2+ peaks at 2.2 R_e and H2 at
#     0.68 R_e, so the ratio compares the compressed side of one curve against
#     the dissociated side of the other;
#   * "dissociation" is the last point of each scan, and the scans do not reach
#     equally far: H2 runs out to 7.1 R_e, He2+ only to 2.05 R_e. The ratio
#     rewards whichever system was scanned further.
#
# Both go away at a matched reduced coordinate. This stage writes, for every
# system whose admitted stratum spans R_e to 1.5 R_e, the value of each metric
# at 1.5 R_e divided by its value at R_e -- linear interpolation on the
# admitted points, no spline, no extrapolation. R_e is the benchmark table's
# own r_eq, so the coordinate is not a new free parameter.
#
# A ratio above 1 means the error is still growing where the bond is already
# half again its equilibrium length.
REDUCED_X = 1.5              # in units of R_e
# Two guards, both of which changed the published table when they were added
# on 2026-08-14, and both of which exist because the first version of this
# stage violated a rule the paper states elsewhere about its own data.
#
#   REDUCED_MAX_GAP  The value at R_e and at x*R_e is obtained by interpolating
#       between admitted geometries. If the nearest admitted point on either
#       side is far away, that "interpolation" is an extrapolation across a
#       hole in the stratum wearing a different name. Li2 is the case: R_e =
#       2.67 A is REJECTED by the gate, and the admitted points either side sit
#       0.16 and 0.15 R_e away, so its normalization was being read off a gap.
#       A system with a hole wider than this at either evaluation point is
#       declared not to span the interval rather than being interpolated
#       through.
#   the degeneracy cutoff  Sec. S4.1 states that no quantity in this work is
#       taken from a geometry where the degeneracy cutoff leaves the metric
#       indeterminate. The first version of this stage took N2's normalization
#       from R = 1.100 A, which is exactly such a geometry (the cutoff moves
#       the value by 106 times the value itself) and is one of the two points
#       Fig. 3 draws faded. LiH's R = 1.800 A is a third. They are dropped
#       here, per that rule, and the count of dropped points is written to the
#       artifact so the exclusion is visible rather than silent.
REDUCED_MAX_GAP = 0.10       # in units of R_e, either side of each node
REDUCED_INDETERMINATE = 1.0  # degtol span as a fraction of the value
_RC_METRICS = ('DeltaI_resp_PBE-PBE_exact', 'DeltaI_resp_LDA-VWN_exact',
               'DeltaI_resp_PBE-PBE_scf', 'DeltaI_resp_LDA-VWN_scf',
               'DeltaI_bar_PBE-PBE_exact_Eh', 'DeltaI_bar_LDA-VWN_exact_Eh')


_RC_EPS = 1e-9               # R/R_e is a quotient of rounded decimals


def _rc_nodes_ok(xx, x, gap):
    """Is `x` bracketed by points close enough on both sides to interpolate?

    The epsilon is not cosmetic. R and R_e are both printed decimals, so a
    geometry placed exactly at 1.5 R_e can come out as 1.4999999999999998 and
    be read as unbracketed. That happened to LiH, whose R = 2.400 A IS the
    1.5 R_e node.
    """
    lo = xx[xx <= x + _RC_EPS]
    hi = xx[xx >= x - _RC_EPS]
    if not len(lo) or not len(hi):
        return False
    return bool(x - lo.max() <= gap + _RC_EPS
                and hi.min() - x <= gap + _RC_EPS)


def run_reduced_coordinate(x=REDUCED_X, log=print):
    """Both metrics at x*R_e, normalized to their value at R_e.

    Reads the deposited dissociation curves only; nothing is re-inverted and
    nothing is re-scored, so this stage costs seconds and can always re-run.
    """
    import pandas as pd
    from benchmark import parse_benchmark

    rows = []
    for m in parse_benchmark():
        name, r_eq = m['name'], float(m['r_eq'])
        p = f"results_energies/{name}_Dissociation_Curve.csv"
        if not os.path.exists(p):
            log(f"  {name}: no curve on disk, skipped")
            continue
        d = pd.read_csv(p)
        if 'inv_status' not in d.columns:
            log(f"  {name}: curve has no confidence column, skipped")
            continue
        d = d[d['inv_status'].isin(('OK', 'MARGINAL'))].sort_values('R_Angstrom')
        row = dict(system=name, r_eq_A=r_eq, n_admitted=int(len(d)))
        if len(d):
            row['reach_Re'] = float(d['R_Angstrom'].max() / r_eq)
            row['span_ok'] = bool(d['R_Angstrom'].min() <= r_eq
                                  and d['R_Angstrom'].max() >= x * r_eq)
        else:
            row['reach_Re'] = float('nan')
            row['span_ok'] = False
        # geometries the degeneracy cutoff leaves indeterminate, over every
        # response column that carries a span diagnostic
        indet_idx = d.index[[False] * len(d)]
        for col in _RC_METRICS:
            span = f'{col}_degtol_span'
            if span in d.columns and col in d.columns:
                bad = d.index[(d[span].abs()
                               >= REDUCED_INDETERMINATE * d[col].abs())
                              .fillna(False)]
                indet_idx = indet_idx.union(bad)

        n_indet, usable, why = 0, True, ''
        for col in _RC_METRICS:
            key, raw = f'ratio_{col}', f'ratio_all_points_{col}'
            det = f'ratio_no_indeterminate_{col}'
            row[key] = row[raw] = row[det] = float('nan')
            if not row['span_ok'] or col not in d.columns:
                continue
            e0 = d.dropna(subset=[col])
            # the ratio taken over EVERY admitted point, guards off. It is
            # deposited next to the guarded one because for the two systems
            # the guards exclude, the two versions bracket the answer and both
            # lie on the same side of unity: the direction is unambiguous even
            # where the number is not quotable, and that is what Sec. S11.6
            # says about them.
            x0 = e0['R_Angstrom'].to_numpy(float) / r_eq
            v0 = e0[col].to_numpy(float)
            if len(e0) >= 3 and x0[0] <= 1.0 + _RC_EPS and x0[-1] >= x - _RC_EPS:
                b0 = float(np.interp(1.0, x0, v0))
                if b0:
                    row[raw] = float(np.interp(x, x0, v0) / b0)
            # The indeterminate set is a property of the GEOMETRY, not of the
            # column: Sec. S4.1 says no quantity is taken from one of those
            # geometries, so the same rows are dropped from every metric.
            # Applying the mask only where a `_degtol_span` sibling exists
            # would leave the potential-metric columns normalized on exactly
            # the point the guard is there to remove -- which is what happened
            # to N2 before 2026-08-14: its ratio_DeltaI_bar_* were computed
            # from R = 1.100 A in a row whose guard_reason says that geometry
            # is unusable.
            e = e0.loc[e0.index.difference(indet_idx)]
            n_indet = max(n_indet, int(len(e0)) - int(len(e)))
            xx = e['R_Angstrom'].to_numpy(float) / r_eq
            vv = e[col].to_numpy(float)
            # the same ratio with the indeterminate geometries dropped but the
            # gap guard ignored. Together with `ratio_all_points_*` it brackets
            # what the number could be for a system the guards reject, which is
            # how Sec. S11.6 reports Li2 and N2: a direction, not a value.
            if (len(e) >= 3 and xx[0] <= 1.0 + _RC_EPS
                    and xx[-1] >= x - _RC_EPS):
                bd = float(np.interp(1.0, xx, vv))
                if bd:
                    row[det] = float(np.interp(x, xx, vv) / bd)
            if len(e) < 3:
                usable, why = False, why or 'fewer than three usable points'
                continue
            if not _rc_nodes_ok(xx, 1.0, REDUCED_MAX_GAP):
                usable, why = False, why or 'no usable point within %.2f R_e of R_e' % REDUCED_MAX_GAP
                continue
            if not _rc_nodes_ok(xx, x, REDUCED_MAX_GAP):
                usable, why = False, why or 'no usable point within %.2f R_e of %.1f R_e' % (REDUCED_MAX_GAP, x)
                continue
            base = float(np.interp(1.0, xx, vv))
            row[key] = float(np.interp(x, xx, vv) / base) if base else float('nan')
        row['n_indeterminate_dropped'] = n_indet
        row['span_ok'] = bool(row['span_ok'] and usable)
        row['guard_reason'] = '' if row['span_ok'] else (
            why or ('no admitted geometry' if not row['n_admitted']
                    else 'admitted stratum does not reach %.1f R_e' % x))
        rows.append(row)
    art.save_rows('reduced_coordinate', rows)
    n = sum(1 for r in rows if r['span_ok'])
    log(f"[reduced_coordinate] {n} of {len(rows)} systems span R_e to "
        f"{x}*R_e -> {art.path('reduced_coordinate')}")
    return rows
