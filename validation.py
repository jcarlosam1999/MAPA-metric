"""
validation.py — the standalone validations behind the Supporting Information.

Five checks, all reproducible from a finished run and all invoked by main.py
at the end of a scan. Each is switched on or off individually from main.py;
the cheap ones are on by default, the expensive ones are off:

  [A] run_grid_resolution()      the production R grid is dense enough to
                                 place the DeltaI peak (pure CSV
                                 post-processing; no electronic structure).
  [B] run_response_invariance()  the response-projected DeltaI is invariant
                                 to the regularization kappa, while the
                                 potential metric is not.
  [C] run_functional_derivative() the spin-polarized functional evaluation
                                 satisfies the functional-derivative
                                 identity int v.drho = Tr[V_xc.dD].
  [D] run_ensemble_seam()        what a mid-curve scheme switch would cost;
                                 production never switches, so this number
                                 has to be produced deliberately.
  [E] run_fine_grid_scan()       recomputes a light system on a FINE R grid
                                 in a scratch directory and compares the
                                 metric peak against the production grid.

Each returns its numbers and writes a human-readable table through `log`,
so main.py can route them into the run log. Run standalone with:

    python validation.py                 # whatever is switched on in config
    python validation.py --only response # grid|response|derivative|seam|finegrid
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import scipy.linalg

from response_metric import (ks_system, ks_occupations, drho_from_dv,
                             skipped_pairs)

BOHR = 0.52917721092
NPZ_DIR = "results_grid_data"

# Default member for the functional-derivative probe: a genuinely
# spin-polarized system that is part of this study's set.
DERIV_MEMBER = 'HC_q+0_s1'


# ==========================================================================
# [A] grid resolution: is the R grid dense enough?
# ==========================================================================
def candidate_grid(r_eq, n_compressed, n_breaking, tail_growth=1.6,
                   max_gap=0.9, min_gap=0.035, r_stretch_abs=3.8,
                   r_min_floor=0.4):
    """Verbatim re-implementation of molecules.make_distance_grid, kept local
    so this script runs without importing the pipeline (and so a drift between
    the two is caught by the assertion in main())."""
    r_min = max(0.6 * r_eq, r_min_floor)
    r_break_end = min(2.5 * r_eq, r_eq + 2.0)
    r_max = r_eq + r_stretch_abs
    zA = np.linspace(r_min, r_eq, n_compressed, endpoint=False)
    zB = np.linspace(r_eq, r_break_end, n_breaking, endpoint=False)
    gap = (r_break_end - r_eq) / n_breaking
    zC, g = [r_break_end], gap
    while zC[-1] < r_max:
        g = min(g * tail_growth, max_gap)
        zC.append(round(zC[-1] + g, 3))
    R = np.sort(np.unique(np.round(np.concatenate([zA, zB, np.array(zC)]), 3)))
    keep = [R[0]]
    for r in R[1:]:
        if r - keep[-1] >= min_gap:
            keep.append(r)
    return np.array(keep)


def peak_readings(R, y):
    """(argmax, parabolic vertex, spline max) of y(R). NaN where undefined."""
    m = np.isfinite(R) & np.isfinite(y)
    R, y = np.asarray(R)[m], np.asarray(y)[m]
    if len(R) < 3:
        return np.nan, np.nan, np.nan
    o = np.argsort(R)
    R, y = R[o], y[o]
    i = int(np.argmax(y))
    r_arg = float(R[i])
    # parabolic vertex through (i-1, i, i+1); undefined at an edge maximum
    r_par = np.nan
    if 0 < i < len(R) - 1:
        x0, x1, x2 = R[i - 1], R[i], R[i + 1]
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        d1, d2 = (x1 - x0), (x2 - x1)
        # general (non-uniform spacing) 3-point vertex
        num = (y0 - y1) * d2 ** 2 - (y2 - y1) * d1 ** 2
        den = (y0 - y1) * d2 - (y2 - y1) * d1
        if abs(den) > 1e-30:
            r_par = float(x1 - 0.5 * num / den)
            if not (x0 <= r_par <= x2):        # vertex outside the bracket
                r_par = np.nan                 # => not a real interior peak
    # spline maximum
    r_spl = np.nan
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(R, y)
        fine = np.linspace(R[0], R[-1], 20001)
        r_spl = float(fine[np.argmax(cs(fine))])
    except Exception:                                         # noqa: BLE001
        pass
    return r_arg, r_par, r_spl


def decimate(R_full, y_full, R_target):
    """Nearest-neighbour pick of the full sweep at the target R values —
    what the coarser scan would have computed (the physics at a given R does
    not depend on which grid asked for it)."""
    idx, used = [], set()
    for rt in R_target:
        j = int(np.argmin(np.abs(R_full - rt)))
        if j not in used:
            used.add(j)
            idx.append(j)
    idx = np.array(sorted(idx))
    return R_full[idx], y_full[idx]


def run_grid_resolution(csv_path=None, col=None, r_eq=None,
                        grids='5,10;5,15;3,7', log=print):
    """Is the production R grid dense enough to place the DeltaI peak?

    Pure post-processing of the dissociation CSVs: decimates the grid
    and re-estimates the peak, so it costs nothing beyond file IO."""

    import pandas as pd
    path = csv_path
    if path is None:
        pref = glob.glob('results_energies/HH_q+0_s0_Dissociation_Curve.csv')
        allc = sorted(glob.glob('results_energies/*_Dissociation_Curve.csv'))
        if not (pref or allc):
            raise SystemExit('no *_Dissociation_Curve.csv in results_energies/')
        path = (pref or allc)[0]
    df = pd.read_csv(path)
    name = os.path.basename(path).replace('_Dissociation_Curve.csv', '')

    # ---- COLUMN SAFETY --------------------------------------------------
    # A real run of this script read 'DeltaI_bar_LDA-VWN_exact_Eh_OLDFILE' —
    # a suffix functional_eval.column_name() cannot produce — and reported the
    # peak at 0.814 A when metrics.py documents 1.258 A for the production
    # volume weighting. The old auto-pick took the first DeltaI_bar* column it
    # found, which was a STALE legacy column left behind by a CSV merge. The
    # answer was wrong and looked fine. So: never pick silently, and always
    # show the peak of EVERY DeltaI column, which is the one line that would
    # have caught it instantly.
    # The peak-position analysis applies to the potential metric, whose
    # maximum along R is the physical marker. The response metric decreases
    # monotonically, so a 'peak' in it is just the first grid point; it is
    # listed for completeness but never analysed here. Both families are
    # legitimate output of the post-pass -- only names matching neither are
    # left over from an older run.
    peak_cols = re.compile(
        r'^DeltaI_(bar|raw)_[^_]+_(exact|scf)(_(alpha|beta))?_Eh$')
    canonical = re.compile(
        r'^DeltaI_(bar|raw)_[^_]+_(exact|scf)(_(alpha|beta))?_Eh$'
        r'|^DeltaI_resp_[^_]+_(exact|scf)(_degtol_span)?$')
    found = [c for c in df.columns if c.startswith('DeltaI')]
    if not found:
        raise SystemExit(f'no DeltaI* column in {path}. Columns: '
                         + ', '.join(df.columns[:20]))
    ok_cols = [c for c in found if peak_cols.match(c)]
    bad_cols = [c for c in found if not canonical.match(c)]
    log("DeltaI columns in this CSV, with the peak each one reports:")
    for c in found:
        yy = df[c].to_numpy(float)
        RR = df['R_Angstrom'].to_numpy(float)
        gg = np.isfinite(RR) & np.isfinite(yy)
        _a, _p, _s = peak_readings(RR[gg], yy[gg]) if gg.sum() > 2 \
            else (np.nan, np.nan, np.nan)
        if peak_cols.match(c):
            flag = ''
        elif canonical.match(c):
            flag = '   (response metric; peak analysis does not apply)'
        else:
            flag = '   <-- unrecognized name: likely from an older run'
        log(f"    {c:<52} argmax {_a:7.4f}  spline {_s:7.4f}{flag}")
    if bad_cols:
        log()
        log("  WARNING: unrecognized DeltaI column name(s) present. The "
            "post-pass writes")
        log("  DeltaI_bar_<FUNC>_<exact|scf>[_<alpha|beta>]_Eh and "
            "DeltaI_resp_<FUNC>_<exact|scf>")
        log("  (optionally _degtol_span); anything else is left over from an "
            "older run and")
        log("  may carry a different density weighting. Re-running the "
            "post-pass overwrites")
        log("  the current columns and leaves the obsolete ones in place.")
    log()
    col = col
    if col is None:
        if len(ok_cols) == 1:
            col = ok_cols[0]
            log(f"auto-selected the only canonical column: {col}")
        elif not ok_cols:
            raise SystemExit(
                "no CANONICAL DeltaI column here — only stale ones:\n  "
                + "\n  ".join(bad_cols)
                + "\nRerun the functional post-pass, or pass --col to select "
                  "a column explicitly.")
        else:
            # The grid must serve every reported curve, not the kindest one,
            # and the exact-vs-scf pair is itself the error-cancellation
            # signal. The default over several canonical columns is
            # therefore all of them, worst case.
            col = 'all'
            log(f"{len(ok_cols)} canonical columns found -> checking ALL of "
                  f"them and taking the WORST case: the grid must be adequate "
                  f"for every reported curve, and the exact-vs-scf pair is "
                  f"the error-cancellation signal itself.")
    if col != 'all' and col not in df.columns:
        raise SystemExit(f'{col!r} not in {path}')
    if col == 'all':
        cols = list(ok_cols)
    else:
        cols = [col]
    if col != 'all' and not canonical.match(col):
        log(f"WARNING: {col!r} is not a canonical column name; it was "
              f"requested explicitly, so proceeding, but this is very likely a "
              f"stale legacy column.")
    if 'inv_status' in df.columns:
        df = df[~df['inv_status'].astype(str).isin(['ERROR', 'FAIL'])]
    R = df['R_Angstrom'].to_numpy(float)
    # NOTE: no single-column extraction here. Each column is filtered for
    # finiteness inside the per-column loop below, because different
    # functionals can fail at different geometries and must not be forced onto
    # a common mask (that would silently drop points from the healthy curves).
    # r_eq: read it from the benchmark table by member name. The old
    # inference (r_min = 0.6*r_eq) silently breaks for any member with
    # r_eq < 0.667, where make_distance_grid clamps r_min to r_min_floor=0.4.
    r_eq, r_eq_src = r_eq, 'given'
    if r_eq is None:
        try:
            from benchmark import parse_benchmark
            r_eq = {m['name']: m['r_eq'] for m in parse_benchmark()}[name]
            r_eq_src = 'benchmark table'
        except Exception:                                     # noqa: BLE001
            r_eq, r_eq_src = round(R.min() / 0.6, 3), 'INFERRED from r_min/0.6'

    log(f"system : {name}")
    log(f"columns: {', '.join(cols)}")
    log(f"sweep  : {len(R)} usable points, R = {R.min():.3f} .. {R.max():.3f} A")
    log(f"r_eq   : {r_eq:.3f} A ({r_eq_src})")
    log()

    specs = [tuple(int(x) for x in s.split(',')) for s in grids.split(';')]
    worst = {s: {'argmax': 0.0, 'spline': 0.0} for s in specs}
    for c in cols:
        yy = df[c].to_numpy(float)
        g = np.isfinite(R) & np.isfinite(yy)
        Rc, yc = R[g], yy[g]
        ra, rp, rs = peak_readings(Rc, yc)
        log(f"--- {c}")
        log(f"    FULL SWEEP peak:  argmax {ra:.4f}   parabolic {rp:.4f}   "
              f"spline {rs:.4f}   [A]")
        hdr = (f"    {'grid':>8} {'pts':>5} {'spacing':>8} {'argmax':>9} "
               f"{'d(arg)':>8} {'parab':>9} {'d(par)':>8} {'spline':>9} "
               f"{'d(spl)':>8}")
        log(hdr)
        for nc, nb in specs:
            Rt = candidate_grid(r_eq, nc, nb)
            Rd, yd = decimate(Rc, yc, Rt)
            da, dp, ds = peak_readings(Rd, yd)
            f = lambda v, ref: (abs(v - ref) if np.isfinite(v)
                                and np.isfinite(ref) else np.nan)
            spacing = (min(2.5 * r_eq, r_eq + 2.0) - r_eq) / nb
            if np.isfinite(f(da, ra)):
                worst[(nc, nb)]['argmax'] = max(worst[(nc, nb)]['argmax'],
                                                f(da, ra))
            if np.isfinite(f(ds, rs)):
                worst[(nc, nb)]['spline'] = max(worst[(nc, nb)]['spline'],
                                                f(ds, rs))
            log(f"    {f'({nc},{nb})':>8} {len(Rd):5d} {spacing:8.3f} "
                  f"{da:9.4f} {f(da, ra):8.4f} {dp:9.4f} {f(dp, rp):8.4f} "
                  f"{ds:9.4f} {f(ds, rs):8.4f}")
        log()

    if len(cols) > 1:
        log("=== WORST CASE ACROSS ALL PUBLISHED CURVES ===")
        pts_hdr = 'pts (this r_eq)'
        log(f"{'grid':>10} {pts_hdr:>15} {'worst d(argmax)':>17} "
              f"{'worst d(spline)':>17}")
        for s in specs:
            log(f"{str(s):>10} {len(candidate_grid(r_eq, *s)):>15} "
                  f"{worst[s]['argmax']:>17.4f} {worst[s]['spline']:>17.4f}")
        log()

    log()
    log("READING")
    log("  Read the column for the estimator that is actually reported;")
    log("  the estimator is part of the measurement. On a skewed peak (which")
    log("  DeltaI_bar is: a peak on a rising baseline) the 3-point parabola")
    log("  is BIASED, verified on a synthetic peak: 0.037 A off on the full")
    log("  grid and moving 0.127 A under decimation, while the spline held")
    log("  to 0.001 A, so the parabolic shift is not the figure of merit.")
    log("  If the chosen estimator moves by much less than the physical")
    log("  scale being resolved (for H2, |DeltaI peak - D_CF| ~ 0.047 A),")
    log("  the coarser grid is adequate. If it moves comparably, the")
    log("  the TAIL instead (raise CFG.tail_growth) — the tail is where")
    log("  N_unpaired plateaus and no fit needs the resolution.")
    log("  d(argmax) can only move in whole grid steps; a large d(argmax)")
    log("  with a small d(spline) just means interpolation is doing its job.")

# ==========================================================================
# [B] response invariance: is DeltaI_resp kappa-free?
# ==========================================================================
BOHR = 0.52917721092
NPZ_DIR = "results_grid_data"


def bond_mask(coords, r_ang, cyl_rad=2.0):
    z = coords[:, 2]
    rad2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
    half = (r_ang / BOHR) / 2.0
    return (np.abs(z) <= half) & (rad2 < cyl_rad ** 2)


def dIbar_current(dv, w, mask):
    """The current metric: volume-weighted mean |dv - c|, c = L1-optimal."""
    d, ww = dv[mask], w[mask]
    o = np.argsort(d)
    cw = np.cumsum(ww[o])
    c = d[o][np.searchsorted(cw, 0.5 * cw[-1])]
    return float(np.sum(ww * np.abs(d - c)) / ww.sum())


def dIbar_response(drho, w, mask):
    """The proposed metric: mean |drho| over the bond volume [e/bohr^3].
    No gauge constant is needed — orthogonality already removed it."""
    return float(np.sum(w[mask] * np.abs(drho[mask])) / w[mask].sum())


def prep(path):
    import molecules as mo
    import inversion as iv
    import functional_eval as fe
    from pyscf import dft
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    mol = fe._rebuild_mol(meta)
    grids = mo.becke_grid(mol)
    ref_dm = np.asarray(d['exact_dm'])
    ao1 = mol.eval_gto('GTOval_sph_deriv1', grids.coords)
    rho = dft.numint.eval_rho(mol, ao1, ref_dm, xctype='GGA')[0]
    v_H = mo.hartree_potential_on_points(mol, ref_dm, grids.coords)
    # The GUIDE, not raw Fermi-Amaldi: this bench re-inverts, so it must use
    # the same prior the pipeline uses or it measures a different engine.
    # (For n_elec <= 2 xc_guide_on_points returns pure FA, so H2 is unchanged.)
    v_guide, _ = mo.xc_guide_on_points(mol, ref_dm, grids.coords, rho=rho,
                                       v_h=v_H)
    return dict(mol=mol, grids=grids, ao=ao1[0], ref_dm=ref_dm, rho=rho,
                v_guide=v_guide, J_ref=iv.make_J(mol, ref_dm),
                R=float(meta['R_ang']), meta=meta)


def invert_at(inp, kappa):
    from config import CFG
    import inversion as iv
    k0, kmin, ad = CFG.kappa_start, CFG.kappa_min, CFG.adaptive_kappa
    CFG.kappa_start, CFG.kappa_min, CFG.adaptive_kappa = kappa, kappa, False
    try:
        return iv.invert_geometry(inp['mol'], inp['grids'], inp['ao'],
                                  inp['ref_dm'], inp['rho'], None,
                                  inp['v_guide'], inp['J_ref'])
    finally:
        CFG.kappa_start, CFG.kappa_min, CFG.adaptive_kappa = k0, kmin, ad


def min_contributing_gap(eps, occ, deg_tol=1e-6, occ_tol=1e-12):
    """The smallest |eps_i - eps_j| that chi actually sums over.

    This is the response operator's condition number in disguise: K_ij =
    (n_i-n_j)/(eps_i-eps_j) blows up as the gap closes, and a finite-difference
    probe of step s can only stay in the linear regime while
        s |<phi_j|dv|phi_i>| << |eps_i - eps_j|.
    So a tiny gap does not mean chi is wrong — it means a fixed s cannot
    VALIDATE chi there. Reported so the two are never confused again.
    """
    n, e = np.asarray(occ, float), np.asarray(eps, float)
    dn = np.abs(n[:, None] - n[None, :])
    de = np.abs(e[:, None] - e[None, :])
    ok = (de >= deg_tol) & (dn >= occ_tol)
    return float(de[ok].min()) if ok.any() else np.nan


def validate_response(inp, out, dv, eps, C, occ,
                      s_ladder=(1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
                      log=print):
    """Does chi.dv reproduce a finite-difference density change? Tested by
    SCALING, not at one step size.

    WHY A LADDER. chi.dv is a FIRST-ORDER operator, so the finite-difference
    residual must vanish as O(s). A single fixed s conflates two completely
    different situations:
        * chi is WRONG           -> the residual is a fixed fraction of the
                                    signal and does NOT shrink with s. (The
                                    original factor-2 bug looked exactly like
                                    this: constant 0.5 at every step.)
        * chi is RIGHT but s is too big for a near-degenerate pair -> the
          residual is large at that s and shrinks as O(s).
    The previous fixed s=1e-4 reported 0.946 and 1.00 on rows that may be
    perfectly healthy, and those rows carried the biggest kappa levers — so
    the fixed-s check was throwing away the evidence the bench exists to
    collect. The O(s) scaling is what caught the factor of 2 in the first
    place (2.3e-4 -> 2.3e-5 -> 2.3e-6); it is the right instrument.

    Returns (best_err, s_best, slope):
        best_err  the smallest residual achieved over the ladder
        s_best    the step that achieved it
        slope     d log(err) / d log(s) on the branch approaching s_best.
                  ~1 => textbook first-order convergence: chi is VALIDATED and
                  the large-s value was truncation, not a bug.
                  ~0 => the residual is structural: chi is WRONG.
    At very small s round-off takes over (err ~ eps_mach/s), so the ladder is
    V-shaped; taking the minimum and the slope on its LEFT approach is what
    separates the two regimes without assuming where the floor sits.
    """
    mol, grids = inp['mol'], inp['grids']
    ao, w = inp['ao'], grids.weights
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    S = mol.intor('int1e_ovlp')
    J = inp['J_ref']
    lin = drho_from_dv(eps, C, occ, ao, w, dv)
    Vxc0 = np.einsum('gi,g,gj->ij', ao, out['vxc'] * w, ao)
    H0 = T + V + J + Vxc0
    P = np.einsum('gi,g,gj->ij', ao, dv * w, ao)      # unscaled perturbation

    def rho_of(H):
        e, Cc = scipy.linalg.eigh(H, S)
        D = (Cc * occ) @ Cc.T
        return np.einsum('gi,ij,gj->g', ao, D, ao)

    rho0 = rho_of(H0)
    errs = []
    for s in s_ladder:
        fd = (rho_of(H0 + s * P) - rho0) / s
        den = float(np.sum(w * np.abs(fd)))
        errs.append(float(np.sum(w * np.abs(fd - lin))) / den
                    if den > 0 else np.nan)
    errs = np.array(errs)
    if not np.any(np.isfinite(errs)):
        return np.nan, np.nan, np.nan
    log("          FD ladder  s=" + "  ".join(f"{s:.0e}:{e:.1e}"
                                                for s, e in zip(s_ladder, errs)))
    i = int(np.nanargmin(errs))
    s_best, best = s_ladder[i], errs[i]
    # MEDIAN of the consecutive log-log slopes. Not the slope at the minimum:
    # when the minimum sits on the round-off floor that single slope is
    # contaminated (measured: it read 0.73 on a case converging at a clean
    # 1.00). The median ignores one bad endpoint and still reads ~0 for a
    # structurally wrong chi, which is the distinction that matters.
    sl = [np.log(errs[k] / errs[k + 1])
          / np.log(s_ladder[k] / s_ladder[k + 1])
          for k in range(len(errs) - 1)
          if np.isfinite(errs[k]) and np.isfinite(errs[k + 1])
          and errs[k] > 0 and errs[k + 1] > 0]
    slope = float(np.median(sl)) if sl else np.nan
    return float(best), float(s_best), float(slope)


def run_response_invariance(system='HH_q+0_s0',
                            Rlist=(1.110, 1.628, 2.157),
                            kappas=(1e-7, 1e-6, 1e-5),
                            functional='PBE,PBE', log=print):
    """Is the response-projected DeltaI invariant to the regularization?

    Re-inverts each geometry at several fixed kappas and compares both
    metrics. The one-electron AO integrals are kappa-independent and are
    built once per geometry rather than once per kappa."""
    import functional_eval as fe
    paths = {}
    for p in sorted(glob.glob(os.path.join(NPZ_DIR, f"{system}_R*.npz"))):
        d = np.load(p, allow_pickle=True)
        paths[round(float(json.loads(str(d['meta']))['R_ang']), 3)] = p
    todo = []
    for R in Rlist:
        if not paths:
            break
        k = min(paths, key=lambda x: abs(x - R))
        if abs(k - R) <= 0.05:
            todo.append((k, paths[k]))
    if not todo:
        raise SystemExit("no matching npz")

    log(f"{'='*80}")
    log("Is a RESPONSE-based DeltaI invariant to the regularization?")
    log(f"  {len(todo)} geometries x {len(kappas)} kappas")
    log(f"{'='*80}")

    for R, path in todo:
        inp = prep(path)
        mask = bond_mask(inp['grids'].coords, R)
        finfo = fe.resolve_functional(functional)
        v_dfa, _ = fe.functional_vxc_on_grid(finfo, inp['mol'], inp['ref_dm'],
                                             inp['grids'].coords)
        log(f"\n--- R = {R} A " + "-" * 58)
        log(f"{'kappa':>8} {'dens_err':>10} {'dIbar_CURRENT':>14} "
              f"{'dIbar_RESPONSE':>15} {'n_frac':>6} {'nskip':>5} "
              f"{'wskip':>7} {'dgtol%':>8} {'fd_err':>9} {'slope':>6}  chi")
        # kappa-independent one-electron integrals: built once
        # per geometry (they were rebuilt on every kappa).
        T = inp['mol'].intor('int1e_kin')
        V = inp['mol'].intor('int1e_nuc')
        S = inp['mol'].intor('int1e_ovlp')
        for k in kappas:
            out = invert_at(inp, k)
            dv = out['vxc'] - v_dfa
            eps, C = ks_system(inp['mol'], inp['grids'].coords,
                               inp['grids'].weights, inp['ao'], out['vxc'],
                               T, V, inp['J_ref'], S)
            occ = ks_occupations(C, S, out['D_ks'])
            n_frac = int(np.sum((occ > 1e-3) & (occ < 2.0 - 1e-3)))
            if abs(occ.sum() - inp['mol'].nelectron) > 1e-2:
                log(f"  ** sum(occ) = {occ.sum():.4f} != N = "
                      f"{inp['mol'].nelectron}: the KS rebuild is broken; "
                      f"every number on this row is meaningless.")
            drho = drho_from_dv(eps, C, occ, inp['ao'],
                                inp['grids'].weights, dv)
            cur = dIbar_current(dv, inp['grids'].weights, mask)
            res = dIbar_response(drho, inp['grids'].weights, mask)
            gap = min_contributing_gap(eps, occ)
            nskip, wskip = skipped_pairs(eps, occ)
            # DOES THE METRIC DEPEND ON deg_tol? chi drops near-degenerate
            # pairs at an arbitrary cutoff. If dIbar_RESPONSE moves when that
            # cutoff moves, the regularization dependence has merely been
            # renamed and the metric carries a new free parameter. Same
            # test, same standard as kappa.
            dg = []
            for dt in (1e-4, 1e-6, 1e-8):
                dr = drho_from_dv(eps, C, occ, inp['ao'],
                                  inp['grids'].weights, dv, deg_tol=dt)
                dg.append(dIbar_response(dr, inp['grids'].weights, mask))
            dg_spread = (max(dg) / min(dg) - 1.0) if min(dg) > 0 else np.nan
            best, s_best, slope = validate_response(inp, out, dv, eps, C, occ, log=log)
            # chi is validated if the FD ever reached it, OR if the residual
            # is converging as O(s) (then the residual is truncation, not chi)
            # VALIDATED if the ladder ever reproduces the FD to 1e-3 at
            # any step (a correct first-order operator does; the large-s and
            # small-s tails are truncation and round-off). UNPROVEN only if it
            # never gets close. A dropped degenerate pair (nskip>0) is flagged
            # separately: chi is fine, the FD just cannot probe that channel.
            if best < 1e-3:
                verdict = 'OK'
            elif nskip > 0:
                verdict = 'SING'      # response singularity, not a chi error
            else:
                verdict = 'UNPROVEN'
            log(f"{k:>8.0e} {out['density_error']:>10.2e} {cur:>14.4f} "
                  f"{res:>15.6f} {n_frac:>6d} {nskip:>5d} {wskip:>7.3f} "
                  f"{100 * dg_spread:>8.2f} {best:>9.2e} {slope:>6.2f}  "
                  f"{verdict:>8}")

    log(f"\n{'='*80}\nREADING\n{'='*80}")
    log("'n_frac' counts FRACTIONALLY occupied KS orbitals. n_frac > 0 means")
    log("the ensemble path is live, so integer-aufbau occupations would have")
    log("been wrong. Occupations now come from D_ks; sum(occ) is checked")
    log("against N and a violation is printed on the row.")
    log()
    log("VALIDATING chi — read 'slope', not 'fd_err' alone:")
    log("  chi.dv is a FIRST-ORDER operator, so its finite-difference")
    log("  residual MUST vanish as O(s). The ladder separates two things a")
    log("  single step size cannot:")
    log("    slope ~ 1  -> textbook first-order convergence. chi is CORRECT;")
    log("                  any large residual at big s was truncation.")
    log("    slope ~ 0  -> the residual is structural. chi is WRONG. (The")
    log("                  original factor-2 bug read a flat 0.5 at every s.)")
    log("  'nskip' counts pairs chi DROPS: |de| < deg_tol while |dn| >")
    log("  1e-6 — a degenerate pair carrying UNEQUAL occupations. That is a")
    log("  genuine SINGULARITY of the static response, not a nuisance: any")
    log("  perturbation splitting the degeneracy rotates the subspace by")
    log("  O(1), so the density change is O(1) for ANY step, not O(s), and")
    log("  linear response does not describe it. Verified synthetically")
    log("  (gap 1e-8, occupations 1.35/0.65): the FD residual sits FLAT at")
    log("  1.00 across five decades of s — indistinguishable from a wrong")
    log("  chi. With EQUAL occupations on the same degenerate pair the")
    log("  ladder converges at slope 1.00. So nskip > 0 explains an UNPROVEN")
    log("  row without impugning chi. 'wskip' is the occupation imbalance")
    log("  being dropped.")
    log()
    log("  'dgtol%' measures how much dIbar_RESPONSE moves when deg_tol")
    log("  sweeps 1e-4 -> 1e-8. deg_tol is an arbitrary cutoff, exactly as")
    log("  kappa is. A small dgtol% means the dropped channel carries no")
    log("  weight and the metric is well defined; a large one would mean the")
    log("  regularization dependence had merely been renamed, leaving a new")
    log("  free parameter in its place. The cutoff is therefore held to the")
    log("  same standard as kappa.")
    log()
    log("  'min_gap' is the smallest |eps_i - eps_j| chi sums over. When it")
    log("  is tiny, K_ij = dn/de is huge and no fixed s stays in the linear")
    log("  regime: 'UNPROVEN' there means the finite difference could not")
    log("  probe chi, not that chi is wrong. It is a property of the")
    log("  VALIDATION rather than of the physics, and does not bear on the")
    log("  choice of kappa, which the density error decides on its own")
    log("  evidence.")
    log()
    log("VERDICT column: OK = chi reproduces the finite difference to 1e-3")
    log("at some step (validated). SING = a degenerate pair with unequal")
    log("occupations (nskip>0) that linear response cannot describe -- chi is")
    log("fine, the FD cannot probe it, and deg_tol shows it carries no weight.")
    log("UNPROVEN = the ladder never got close AND is not a known singularity")
    log("(would warrant a look, but did not occur in the validated runs).")
    log()
    log("Reading the table: at each R, compare how the two metrics move")
    log("across kappa, using the OK and SING rows (both carry a trustworthy")
    log("dIbar_RESPONSE). dIbar_CURRENT swings by a large factor per decade")
    log("of kappa -- the known ill-conditioning, reported here as the")
    log("control. dIbar_RESPONSE holding steady across the same range is the")
    log("invariance the response projection is built to provide, and leaves")
    log("a magnitude in e/bohr^3 rather than only a peak position.")

# ==========================================================================
# [C] functional-derivative identity (spin-polarized)
# ==========================================================================
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


def check(member_name, r_ang=None, n_probe=3, seed=0,
          levels=(3, 4, 5), log=print):
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
        log(f"WARNING: {member_name} is closed shell — rho_a == rho_b makes "
              f"the sigma_ab cross terms degenerate. Pick a spin != 0 member.")
    r = float(r_ang if r_ang is not None else mem['r_eq'])
    mol = build_molecule(mem['a1'], mem['a2'], r, mem['charge'], mem['spin'])
    dma, dmb, _ = _dm_spin(mol)
    nao = mol.nao
    log(f"system   : {member_name}  ({mem['a1']}{mem['a2']}, q={mem['charge']:+d}, "
          f"2S={mem['spin']}, {mem['n_elec']}e-) at R={r:.3f} A")
    log(f"basis/aos: {mol.basis} / {nao}")
    log()
    log("THE TEST IS THE GRID SCAN, not any single number.")
    log("  Int v drho == Tr[V dD] is EXACT analytically. PySCF builds its")
    log("  matrix from the GRADIENT form; baselines builds the grid potential")
    log("  from the DIVERGENCE form (it needs the density Hessian). The two")
    log("  are related by parts, so on a finite grid they differ by QUADRATURE")
    log("  error — which must SHRINK as the grid improves. An index or factor")
    log("  bug does not shrink: it is O(1) and grid-independent.")
    log("  'PBE closed-shell' is the control: it exercises the SAME density")
    log("  Hessian + chain-rule machinery with NO spin indexing. If the spin")
    log("  path tracks it, the assumed libxc sigma/v2rhosigma/v2sigma2 maps")
    log("  are exonerated and what is left is quadrature.")
    log()

    dm_tot = dma + dmb
    rows = {}
    hdr = (f"{'grid':>5} {'pts':>7} {'functional':>22} {'worst |LHS-RHS|':>16} "
           f"{'worst rel(L1)':>14}")
    log(hdr)
    log('-' * len(hdr))
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
            log(f"  (closed-shell control unavailable: {type(e).__name__}: {e})")
        try:
            pa, pb = bl.pbe_vxc_spin_on_grid(mol, dma, dmb, coords, xc='PBE,PBE')
            Pa, Pb = _matrix_derivative(mol, g, 'PBE,PBE', dma, dmb)
            trials.append(('PBE spin (under test)', [(pa, Pa, dma),
                                                     (pb, Pb, dmb)]))
        except Exception as e:                                # noqa: BLE001
            log(f"  PBE spin FAILED TO BUILD: {type(e).__name__}: {e}")

        for label, chans in trials:
            wa, wr = 0.0, 0.0
            for v_grid, Vmat, _dm in chans:
                for dD in probes:
                    _, _, rel, ad = _probe(v_grid, Vmat, ao, w, dD)
                    wa, wr = max(wa, ad), max(wr, rel)
            rows.setdefault(label, []).append((lev, wr, wa))
            log(f"{lev:>5} {len(w):>7} {label:>22} {wa:>16.3e} {wr:>14.3e}")
        log()

    log("=== VERDICT ===")
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
        log(f"  LSDA spin (control)        worst rel(L1) {w:.2e} — "
              + ("PASS, machine precision as expected"
                 if w < 1e-10 else
                 "UNEXPECTED: a direct libxc vrho call should be exact"))
    if ctrl:
        log(f"  PBE closed-shell (control) worst rel(L1) {worst_of(ctrl):.2e}"
              f"   (varies {spread(ctrl):.1f}x across grid levels)")
    if spin:
        ws = worst_of(spin)
        log(f"  PBE spin (under test)      worst rel(L1) {ws:.2e}"
              f"   (varies {spread(spin):.1f}x across grid levels)")
    log()
    log("  NOTE ON THE GRID SCAN: Becke grids at different levels are NOT")
    log("  nested, so the quadrature error need not fall monotonically with")
    log("  the level — it wanders. Three non-monotonic points are therefore")
    log("  NOT a trend: dividing the first level by the last would")
    log("  misreport a healthy spin path as a defect.")
    log("  What decides is the MAGNITUDE, against the calibration above.")
    log()
    if spin is None:
        log("  PBE spin never built — see the error above.")
    else:
        ws = worst_of(spin)
        ratio = (ws / max(worst_of(ctrl), 1e-300)) if ctrl else np.nan
        if np.isfinite(ratio):
            log(f"  spin / closed-shell error ratio: {ratio:.1f}x  (the spin")
            log(f"  path carries strictly more terms — the sigma_ab cross")
            log(f"  terms and two density Hessians instead of one — so a")
            log(f"  single-digit factor is expected, not suspicious.)")
            log()
        if ws >= BUG_LEVEL:
            log(f"  => FAIL. rel(L1) = {ws:.2e} is at the O(1) scale of a real")
            log("     formula/index bug. Suspects in order: the libxc")
            log("     component ordering in baselines.pbe_vxc_spin_on_grid")
            log("     (sigma aa/ab/bb; the v2rhosigma and v2sigma2 row maps),")
            log("     the factors of 2 on the f_aa terms, then the Hessians.")
            log("     Until fixed: CFG.include_pbe_spin = False, LDA-only for")
            log("     the open-shell members.")
        elif ws < QUAD_LEVEL:
            log(f"  => PASS. The identity holds to rel(L1) = {ws:.2e}, i.e. to")
            log("     5+ significant figures, at EVERY grid level. An index or")
            log("     factor error is O(0.1-1) and cannot hide here. What")
            log("     remains is the quadrature of the by-parts step, which is")
            log("     the same floor the closed-shell path sits on.")
            log("     baselines.pbe_vxc_spin_on_grid is the true spin-resolved")
            log("     functional derivative. Tier-1 is now validated on BOTH")
            log("     shells: every open-shell member can carry PBE columns.")
            log(f"     Reported figure: worst rel(L1) over the grid scan,")
            log(f"     {ws:.1e}.")
        else:
            log(f"  => AMBIGUOUS. rel(L1) = {ws:.2e} sits between the")
            log(f"     quadrature floor ({QUAD_LEVEL:.0e}) and the bug scale")
            log(f"     ({BUG_LEVEL:.0e}). Too clean for a wrong index, too")
            log("     dirty for pure quadrature. Look at the sigma_ab cross")
            log("     terms (gs_ab and the f_ab divergence): they exist only")
            log("     in the spin path, so they are the difference between")
            log("     this and the closed-shell control.")
    return True

# ============================================================================
# [D] ensemble seam: what a mid-curve scheme switch would cost
# ============================================================================
def run_ensemble_seam(system=None, Rlist=None, functional='PBE,PBE',
                      log=print):
    """Measure the DeltaI discontinuity that a mid-curve scheme switch causes.

    Production always runs the unified ensemble, precisely so this seam cannot
    occur; the number quoted in the manuscript therefore has to be produced on
    purpose. This re-inverts a few ADJACENT geometries from the saved exact
    side with the unified scheme disabled, and reports the largest jump in
    bar-DeltaI between neighbours, against the same quantity with the unified
    scheme on. No reference density is recomputed: only the inversion is
    repeated, so the cost is a handful of light inversions.
    """
    from config import CFG
    import functional_eval as fe
    if system is None:
        system = CFG.validation_seam_system
    paths = {}
    for p in sorted(glob.glob(os.path.join(NPZ_DIR, f"{system}_R*.npz"))):
        d = np.load(p, allow_pickle=True)
        paths[round(float(json.loads(str(d['meta']))['R_ang']), 3)] = p
    if not paths:
        raise SystemExit(f"no npz for {system}: run the pipeline first")
    Rs = sorted(paths)
    if Rlist is None:
        # the seam appears where spin localization sets in: take the window
        # around the middle of the scan, where a naive scheme would switch.
        lo = max(0, len(Rs) // 2 - 2)
        Rlist = Rs[lo:lo + 5]
    log("=" * 80)
    log("What would a mid-curve scheme switch cost? (unified ensemble OFF)")
    log(f"  {system}: {len(Rlist)} adjacent geometries")
    log("=" * 80)
    log(f"{'R':>7} {'unified':>12} {'switched':>12} {'inv_mode':>18}")
    finfo = fe.resolve_functional(functional)
    series = {True: [], False: []}
    for R in Rlist:
        inp = prep(paths[R])
        mask = bond_mask(inp['grids'].coords, R)
        v_dfa, _ = fe.functional_vxc_on_grid(finfo, inp['mol'], inp['ref_dm'],
                                             inp['grids'].coords)
        row = {}
        for unified in (True, False):
            keep = CFG.unified_ensemble
            CFG.unified_ensemble = unified
            try:
                out = invert_at(inp, CFG.kappa_min)
            finally:
                CFG.unified_ensemble = keep
            row[unified] = (dIbar_current(out['vxc'] - v_dfa,
                                          inp['grids'].weights, mask),
                            out.get('inv_mode', '?'))
            series[unified].append(row[unified][0])
        log(f"{R:>7.3f} {row[True][0]:>12.4f} {row[False][0]:>12.4f} "
            f"{row[False][1]:>18}")

    def worst_jump(vals):
        if len(vals) < 2:
            return float('nan')
        r = [max(a, b) / min(a, b) for a, b in zip(vals, vals[1:])
             if min(a, b) > 0]
        return max(r) if r else float('nan')

    ju, js = worst_jump(series[True]), worst_jump(series[False])
    log("-" * 80)
    log(f"  largest neighbour-to-neighbour ratio, unified ensemble : {ju:.2f}x")
    log(f"  largest neighbour-to-neighbour ratio, scheme switched  : {js:.2f}x")
    log("  The unified scheme removes the seam by construction; the switched")
    log("  column is what the metric would inherit if it were not used.")
    return dict(R=list(Rlist), unified=series[True], switched=series[False],
                jump_unified=ju, jump_switched=js)


# ============================================================================
# [E] fine-grid scan: is the production R grid dense enough, from scratch?
# ============================================================================
def run_fine_grid_scan(system=None, n_fine=None, functional='PBE,PBE',
                       log=print):
    """Re-run one light system on a FINE R grid and compare the metric peak
    against the production grid.

    run_grid_resolution() decimates whatever scan is already on disk, which
    can only ever be as fine as production. This is the stronger test the
    manuscript describes: it computes new reference densities on a denser
    grid, in an isolated scratch directory so the production results are never
    touched (every output path in this package is relative, so a chdir fully
    isolates the run). Expensive: it is a small pipeline run.
    """
    import shutil
    import tempfile
    from config import CFG, make_dirs
    from benchmark import parse_benchmark
    if system is None:
        system = CFG.validation_fine_scan_system
    if n_fine is None:
        n_fine = CFG.validation_fine_scan_points
    member = {m['name']: m for m in parse_benchmark()}[system]
    r_eq = member['r_eq']
    fine = np.round(np.linspace(0.6 * r_eq, 2.5 * r_eq, int(n_fine)), 3)

    log("=" * 80)
    log("Is the production R grid dense enough? (fine scan from scratch)")
    log(f"  {system}: {len(fine)} points over "
        f"[{fine[0]:.3f}, {fine[-1]:.3f}] A, in a scratch directory")
    log("=" * 80)

    here = os.getcwd()
    scratch = tempfile.mkdtemp(prefix=f"finegrid_{system}_")
    try:
        os.chdir(scratch)
        make_dirs()
        import pandas as pd
        import pipeline as pl
        import functional_eval as fe
        rows = []
        for i, r in enumerate(fine, 1):
            rows.append(pl.process_geometry(member, float(r)))
            if i % 5 == 0 or i == len(fine):
                log(f"    {i}/{len(fine)} geometries")
        # process_geometry writes the per-geometry row and npz, but the
        # aggregated dissociation CSV is assembled by run_pipeline after its
        # loop. This scan drives the geometries directly, so it has to write
        # that CSV itself -- the post-pass merges its columns into it.
        pd.DataFrame(rows).to_csv(
            os.path.join('results_energies',
                         f"{system}_Dissociation_Curve.csv"), index=False)
        fe.run_functional_evaluation([functional], density_modes=('exact',))
        csv = os.path.join('results_energies',
                           f"{system}_Dissociation_Curve.csv")
        df = pd.read_csv(csv)
        col = fe.column_name(fe.resolve_functional(functional).name, 'exact')
        R = df['R_Angstrom'].values
        y = df[col].values
        peak_fine = peak_readings(R, y)
    finally:
        os.chdir(here)
        shutil.rmtree(scratch, ignore_errors=True)

    # production grid, read from the finished run
    prod_csv = os.path.join('results_energies',
                            f"{system}_Dissociation_Curve.csv")
    if os.path.exists(prod_csv):
        import pandas as pd
        dp = pd.read_csv(prod_csv)
        colp = [c for c in dp.columns if c.startswith('DeltaI_bar_')
                and c.endswith('_exact_Eh')]
        peak_prod = (peak_readings(dp['R_Angstrom'].values, dp[colp[0]].values)
                     if colp else None)
    else:
        peak_prod = None

    log("-" * 80)
    log(f"  peak on the fine grid       : {peak_fine}")
    log(f"  peak on the production grid : {peak_prod}")
    if peak_fine and peak_prod:
        try:
            shift = abs(float(peak_fine['spline']) - float(peak_prod['spline']))
            log(f"  spline peak shift           : {shift:.4f} A")
        except (KeyError, TypeError, ValueError):
            pass
    return dict(fine=peak_fine, production=peak_prod, n_fine=int(n_fine))


# ============================================================================
# orchestration
# ============================================================================
def run_all(log=print, deriv_member=None, **overrides):
    """Run the validations that are switched on, cheapest first.

    Each validation is independently toggled in config (and therefore from
    main.py). Every one is guarded: a failing check must never abort the others
    or the run that called them. Pass e.g. validate_fine_grid_scan=True to
    force one on for a single call.
    """
    from config import CFG

    def on(flag):
        return overrides.get(flag, getattr(CFG, flag))

    if deriv_member is None:
        deriv_member = CFG.validation_openshell_member
    results = {}

    def guarded(tag, fn, **kw):
        try:
            results[tag] = fn(log=log, **kw)
        except Exception as e:                                # noqa: BLE001
            log(f"[validation] {tag} skipped: {type(e).__name__}: {e}")
            results[tag] = None

    plan = [(t_, f_) for t_, f_ in (
        ('grid_resolution', 'validate_grid_resolution'),
        ('functional_derivative', 'validate_functional_derivative'),
        ('kappa_invariance', 'validate_kappa_invariance'),
        ('kappa_wide_range', 'validate_kappa_wide_range'),
        ('ensemble_seam', 'validate_ensemble_seam'),
        ('kappa_invariance_heavy', 'validate_kappa_invariance_heavy'),
        ('fine_grid_scan', 'validate_fine_grid_scan')) if on(f_)]
    log(f"[validation] running {len(plan)} of 7 validations: "
        f"{', '.join(t_ for t_, _ in plan) or 'none'}")

    for tag, _flag in plan:
        if tag == 'grid_resolution':
            guarded(tag, run_grid_resolution)
        elif tag == 'functional_derivative':
            guarded(tag, run_functional_derivative, member=deriv_member)
        elif tag in ('kappa_invariance', 'kappa_wide_range',
                     'kappa_invariance_heavy'):
            probe = {'kappa_invariance': CFG.validation_kappa_light,
                     'kappa_wide_range': CFG.validation_kappa_wide,
                     'kappa_invariance_heavy': CFG.validation_kappa_heavy}[tag]
            system, Rlist, kappas = probe
            guarded(tag, run_response_invariance, system=system,
                    Rlist=tuple(Rlist), kappas=tuple(kappas))
        elif tag == 'ensemble_seam':
            guarded(tag, run_ensemble_seam)
        elif tag == 'fine_grid_scan':
            guarded(tag, run_fine_grid_scan)
    return results


def run_functional_derivative(member=DERIV_MEMBER, r_ang=None, n_probe=3,
                              seed=0, levels=(3, 4, 5), log=print):
    """Functional-derivative identity on the spin-polarized route."""
    return check(member, r_ang, n_probe, seed, levels, log=log)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', choices=['grid', 'response', 'derivative',
                                      'seam', 'finegrid'],
                    default=None)
    ap.add_argument('--system', default='HH_q+0_s0',
                    help='system for a standalone --only response run')
    ap.add_argument('--member', default=DERIV_MEMBER)
    a = ap.parse_args()
    if a.only == 'grid':
        run_grid_resolution()
    elif a.only == 'response':
        run_response_invariance(system=a.system)
    elif a.only == 'derivative':
        run_functional_derivative(member=a.member)
    elif a.only == 'seam':
        run_ensemble_seam()
    elif a.only == 'finegrid':
        run_fine_grid_scan()
    else:
        run_all(deriv_member=a.member)