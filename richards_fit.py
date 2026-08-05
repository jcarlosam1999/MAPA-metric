"""
richards_fit.py — Richards (generalized-logistic) fit of the dissociation curve
==============================================================================
Fits each system's Head-Gordon unpaired-electron index N_unpaired(R) with

        N_unpaired(R) = N / (1 + exp(-k (R - x0)))^v

and reports the four parameters with their physical meaning:

    N   dissociation limit of N_unpaired — an INTEGER (a count of electrons)
    x0  critical distance (bond -> radicals; the Coulson-Fischer scale)
    k   bond stiffness (rate of orbital-overlap loss)
    v   asymmetry (electronic relaxation as the bond stretches)

WHY N IS AN INTEGER, FOUND IN TWO STAGES:
N counts the unpaired electrons of the dissociated fragments, so it is a whole
number (H2 -> 2, N2 -> 6). Fixing it is right; hardcoding it to 2 is not — that
is the H2 value, and N2's sweep plateaus near 6, so N=2 forces a curve that
saturates at 2 onto data reaching 5.963 (max residual ~4 e-, meaningless
parameters). But a free N is not physical either: the sweep only APPROACHES the
limit (H2 stops at 1.99923, N2 at 5.963), so a free fit returns 5.918 for N2.
Hence two stages:
    stage 1  N free   -> the DATA says which integer   (N2: 5.918 -> 6)
    stage 2  N pinned -> physically exact k, x0, v
Stage 2 is not cosmetic: pinning N2 to the true N=6 moves k 4.955 -> 4.532,
x0 1.804 -> 1.733, v 1.457 -> 1.865. Anchoring to the correct asymptote is what
makes the shape parameters mean anything.

N_integer_deviation (|free - integer|) is reported as a QUALITY CHECK: small
(H2 0.0003, N2 0.082) means the sweep resolves the limit; large means it stops
short of the plateau or the curve is not a clean sigmoid. Use N_fixed to
override when the integer is known independently.

CAVEAT on N/2: it is the number of electron PAIRS breaking homolytically, which
equals the bond order for homonuclear diatomics (H2: 1, N2: 3) but NOT in
general — CO has bond order 3 yet dissociates to C(3P)+O(3P) with 4 unpaired
electrons. N counts the fragments' unpaired electrons, not the bond order.

USE
  * automatic: pipeline calls fit_and_save() at the end of each system's sweep
  * standalone (re-runnable on existing curves, no recompute):
        python richards_fit.py                  # fit every curve CSV found
        python richards_fit.py --name NN_q+0_s0 # one system
Writes one JSON per system (shard-safe: no shared file, no races) plus an
aggregate table when run standalone.
"""

import glob
import json
import os

import numpy as np

CURVE_DIR = "results_energies"
OUT_DIR = "results_analysis"
MIN_POINTS = 6            # 4 free parameters — need a few more than that
MIN_SPAN = 0.5            # electrons; below this there is no sigmoid to fit


def richards(x, N0, Nf, k, x0, v):
    """Floored generalized logistic:

        N_unpaired(R) = N0 + (Nf - N0) / (1 + exp(-k (R - x0)))^v

    N0 is the COMPRESSED-geometry asymptote (unpaired electrons already present
    at short R; 0 for a closed-shell singlet, 1 for H2+, 2 for a triplet like
    B2) and Nf the DISSOCIATED asymptote (unpaired electrons of the separated
    fragments). When Nf == N0 there is no transition and the curve is constant
    at N0 (e.g. one-electron H2+ stays at 1 for all R). The old 4-parameter
    form is the N0=0 special case. Guarded exponent so the optimizer can
    explore without overflowing."""
    z = np.clip(-k * (np.asarray(x, float) - x0), -500.0, 500.0)
    return N0 + (Nf - N0) / (1.0 + np.exp(z)) ** v


def fit_richards(R, n_unpaired, N_fixed=None):
    """Fit N_unpaired(R) with BOTH asymptotes constrained to PHYSICAL INTEGERS.

    The curve runs from a compressed-geometry floor N0 (unpaired electrons
    already present at short R) to a dissociated ceiling Nf (unpaired electrons
    of the separated fragments). Both are electron counts, so both are whole
    numbers, and both are found in the same two-stage way the single asymptote
    was: stage 1 fits them free to let the DATA say which integers; stage 2
    pins both to those integers and refits the shape (k, x0, v), which is what
    anchors the shape parameters to the correct asymptotes.

    Special case Nf == N0 (no change in unpaired count, e.g. one-electron H2+):
    there is no sigmoid to fit — the curve is constant at N0. The shape
    parameters are undefined, so they are returned as neutral markers (1.0)
    with constant_marker=True, and N_unpaired is reported as the constant N0.

    N_fixed overrides the integer search. Pass an int for the old behaviour
    (that int is Nf, N0=0), or a (N0, Nf) tuple to fix both.
    """
    R = np.asarray(R, float)
    y = np.asarray(n_unpaired, float)
    m = np.isfinite(R) & np.isfinite(y)
    R, y = R[m], y[m]
    if len(R) < MIN_POINTS:
        return {'ok': False, 'reason': f'only {len(R)} valid points '
                                       f'(need >= {MIN_POINTS})'}
    order = np.argsort(R)
    R, y = R[order], y[order]

    # normalize N_fixed to (N0_fixed, Nf_fixed) or None
    N0_fixed = Nf_fixed = None
    if N_fixed is not None:
        if isinstance(N_fixed, (tuple, list)):
            N0_fixed, Nf_fixed = int(N_fixed[0]), int(N_fixed[1])
        else:
            N0_fixed, Nf_fixed = 0, int(N_fixed)   # back-compat: int == Nf

    from scipy.optimize import curve_fit

    # observed floor/ceiling from the curve endpoints (robust to noise: use the
    # compressed and dissociated extremes, not global min/max, so a non-
    # monotonic bump does not set the asymptote)
    y_lo = float(np.median(y[:max(1, len(y) // 6)]))     # compressed end
    y_hi = float(np.median(y[-max(1, len(y) // 6):]))    # dissociated end
    span = abs(y_hi - y_lo)

    # ---- constant case: no transition (Nf == N0) ------------------------
    # decided either by a user-fixed equal pair, or by the data spanning less
    # than one unpaired electron between its floor and ceiling.
    constant = ((N0_fixed is not None and N0_fixed == Nf_fixed)
                or (N_fixed is None and span < MIN_SPAN))
    if constant:
        N0c = (N0_fixed if N0_fixed is not None
               else int(round(float(np.median(y)))))
        return {
            'ok': True, 'constant_marker': True,
            'N0': int(N0c), 'Nf': int(N0c), 'N': int(N0c),
            'N_free_fit': float(np.median(y)),
            'N0_free_fit': float(y_lo), 'Nf_free_fit': float(y_hi),
            'N_integer_deviation': float(abs(np.median(y) - N0c)),
            'N_confidence': 'constant (no transition: unpaired count fixed)',
            'k': 1.0, 'x0': 1.0, 'v': 1.0,           # neutral markers
            'k_err': np.nan, 'x0_err': np.nan, 'v_err': np.nan,
            'n_broken_pairs': 0.0,
            'rms_resid_e': float(np.sqrt(np.mean((y - np.median(y)) ** 2))),
            'max_resid_e': float(np.max(np.abs(y - np.median(y)))),
            'r2': np.nan, 'n_points': int(len(R)),
            'R_min': float(R.min()), 'R_max': float(R.max()),
            'N_unpaired_max_observed': float(y.max()),
            'plateau_reached': True,
            'equation': f"N_unpaired(R) = {int(N0c)} (constant; no transition)",
        }

    x0_0 = float(np.interp(0.5 * (y_lo + y_hi), y, R))

    # ---- stage 1: N0 and Nf free — ask the data which integers ----------
    N0_free = Nf_free = np.nan
    dev0 = devf = np.nan
    if N_fixed is None:
        try:
            pf, _ = curve_fit(
                richards, R, y,
                p0=[y_lo, y_hi, 3.0, x0_0, 1.0],
                bounds=([-1.0, max(y_hi - 1.0, 0.0), 1e-3,
                         R.min() - 2.0, 1e-3],
                        [max(y_lo + 1.0, 1.0), 4.0 * max(y_hi, 1.0), 100.0,
                         R.max() + 2.0, 50.0]),
                maxfev=50000)
            N0_free, Nf_free = float(pf[0]), float(pf[1])
            seed = [float(pf[2]), float(pf[3]), float(pf[4])]
        except Exception:                                    # noqa: BLE001
            N0_free, Nf_free = y_lo, y_hi
            seed = [3.0, x0_0, 1.0]
        N0_int, Nf_int = int(round(N0_free)), int(round(Nf_free))
        dev0, devf = abs(N0_free - N0_int), abs(Nf_free - Nf_int)
    else:
        N0_int, Nf_int, seed = N0_fixed, Nf_fixed, [3.0, x0_0, 1.0]

    if Nf_int <= N0_int:
        # data did not resolve a rising transition; treat as constant at N0
        return fit_richards(R, y, N_fixed=(N0_int, N0_int))

    # ---- stage 2: N0 and Nf pinned to integers -> exact shape -----------
    def r_fixed(xx, k, x0, v):
        return richards(xx, float(N0_int), float(Nf_int), k, x0, v)

    try:
        popt, cov = curve_fit(
            r_fixed, R, y, p0=seed,
            bounds=([1e-3, R.min() - 2.0, 1e-3],
                    [100.0, R.max() + 2.0, 50.0]), maxfev=50000)
    except Exception as e:                                   # noqa: BLE001
        return {'ok': False, 'reason': f'constrained fit failed: {e}'}

    perr = (np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov))
            else np.full(3, np.nan))
    pred = r_fixed(R, *popt)
    resid = pred - y
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    k, x0, v = (float(p) for p in popt)

    dev = devf if np.isfinite(devf) else np.nan
    if N_fixed is not None:
        confidence = 'user-specified'
    elif not np.isfinite(dev):
        confidence = 'unknown'
    elif max(dev0, devf) < 0.15:
        confidence = 'high'
    elif max(dev0, devf) < 0.35:
        confidence = 'ambiguous — check the sweep reaches dissociation'
    else:
        confidence = 'LOW — free asymptote far from an integer; sweep likely ' \
                     'stops short of the plateau, or not a clean sigmoid'

    return {
        'ok': True, 'constant_marker': False,
        'N0': int(N0_int),                     # compressed-geometry floor
        'Nf': int(Nf_int),                     # dissociated ceiling
        'N': int(Nf_int),                      # back-compat alias (== Nf)
        'N_free_fit': float(Nf_free),
        'N0_free_fit': float(N0_free), 'Nf_free_fit': float(Nf_free),
        'N_integer_deviation': float(dev),
        'N0_integer_deviation': float(dev0),
        'Nf_integer_deviation': float(devf),
        'N_confidence': confidence,
        'k': k, 'x0': x0, 'v': v,
        'k_err': float(perr[0]), 'x0_err': float(perr[1]),
        'v_err': float(perr[2]),
        'n_broken_pairs': (Nf_int - N0_int) / 2.0,
        'rms_resid_e': rms,
        'max_resid_e': float(np.max(np.abs(resid))),
        'r2': float(r2),
        'n_points': int(len(R)),
        'R_min': float(R.min()), 'R_max': float(R.max()),
        'N_unpaired_max_observed': float(y.max()),
        'plateau_reached': bool(y.max() >= 0.95 * Nf_int),
        'equation': (f"N_unpaired(R) = {N0_int} + {Nf_int - N0_int} / "
                     f"(1 + exp(-{k:.4f}*(R - {x0:.4f})))^{v:.4f}"),
    }


def fit_from_curve_csv(path, N_fixed=None):
    """Fit the Richards curve from a *_Dissociation_Curve.csv, using only
    geometries whose inversion produced a usable N_unpaired."""
    import pandas as pd
    df = pd.read_csv(path)
    if 'N_unpaired' not in df.columns or 'R_Angstrom' not in df.columns:
        return {'ok': False, 'reason': 'curve CSV lacks N_unpaired / R'}
    if 'inv_status' in df.columns:                # drop hard errors/failures
        df = df[~df['inv_status'].astype(str).isin(['ERROR', 'FAIL'])]
    return fit_richards(df['R_Angstrom'].values, df['N_unpaired'].values,
                        N_fixed=N_fixed)


def fit_and_save(name, rows=None, curve_csv=None, N_fixed=None):
    """Fit one system and write results_analysis/<name>_richards.json.
    Called by the pipeline at the end of a sweep (pass `rows`), or standalone
    from an existing curve CSV. Never raises: a failed fit is reported, not
    propagated into an expensive run."""
    try:
        if rows is not None:
            R = [r.get('R_Angstrom', np.nan) for r in rows]
            nu = [r.get('N_unpaired', np.nan) for r in rows]
            res = fit_richards(R, nu, N_fixed=N_fixed)
        else:
            res = fit_from_curve_csv(
                curve_csv or f"{CURVE_DIR}/{name}_Dissociation_Curve.csv",
                N_fixed=N_fixed)
    except Exception as e:                                   # noqa: BLE001
        res = {'ok': False, 'reason': f'{type(e).__name__}: {e}'}
    res['system'] = name
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{name}_richards.json", 'w') as f:
        json.dump(res, f, indent=2)
    return res


def summary_line(res):
    """One-line human summary for the run log."""
    if not res.get('ok'):
        return f"  [richards] no fit — {res.get('reason')}"
    if res.get('constant_marker'):
        return (f"  [richards] N_unpaired = {res['N0']} constant "
                f"(no transition; {res['n_points']} pts) — one-electron or "
                f"fixed-multiplicity system")
    flag = '' if res['plateau_reached'] else '  [no plateau reached]'
    conf = res.get('N_confidence', '')
    warn = f"  [N {conf}]" if conf.startswith(('LOW', 'ambiguous')) else ''
    return (f"  [richards] N0={res['N0']}->Nf={res['Nf']} e- "
            f"(free {res['N0_free_fit']:.2f}->{res['Nf_free_fit']:.2f}), "
            f"x0={res['x0']:.3f} A, k={res['k']:.3f}, v={res['v']:.3f}, "
            f"RMS={res['rms_resid_e']:.4f} e-{flag}{warn}")


def report(res):
    """Multi-line physical report: equation, every parameter with its meaning.
    Returned as a list of lines so the caller can print and log identically."""
    if not res.get('ok'):
        return [f"  [richards] no fit — {res.get('reason')}"]
    if res.get('constant_marker'):
        return ["  [richards] unpaired-electron (Head-Gordon) curve  "
                f"({res['n_points']} pts, R={res['R_min']:.2f}-"
                f"{res['R_max']:.2f} A)",
                f"     {res['equation']}",
                f"     N_unpaired = {res['N0']} constant: the unpaired-electron "
                "count does not change along R (no static-correlation "
                "transition to fit).",
                "     This is the expected form for a one-electron system "
                "(H2+) or a fixed-multiplicity open shell: the shape "
                "parameters k, x0, v are undefined and reported as neutral 1.0.",
                f"     max deviation from constant: {res['max_resid_e']:.4f} e-"]
    L = ["  [richards] unpaired-electron (Head-Gordon) curve  "
         f"({res['n_points']} pts, R={res['R_min']:.2f}-{res['R_max']:.2f} A)",
         f"     {res['equation']}",
         f"     N0  = {res['N0']} electrons"
         "         compressed-geometry floor (unpaired electrons already "
         "present at short R)",
         f"     Nf  = {res['Nf']} electrons"
         "         dissociation limit (an integer: the unpaired electrons of "
         "the separated fragments)",
         f"     x0  = {res['x0']:.4f} +/- {res['x0_err']:.4f} A"
         "     critical distance: bond -> radicals (the Coulson-Fischer scale)",
         f"     k   = {res['k']:.4f} 1/A"
         "             bond stiffness: rate of orbital-overlap loss",
         f"     v   = {res['v']:.4f}"
         "                asymmetry: electronic relaxation as the bond stretches",
         f"     (Nf-N0)/2 = {res['n_broken_pairs']:.1f}"
         "          electron PAIRS breaking homolytically "
         "(= bond order for homonuclear diatomics only)",
         f"     fit: RMS={res['rms_resid_e']:.4f} e-, R^2={res['r2']:.6f}",
         f"     asymptotes from free fit: N0 {res['N0_free_fit']:.3f}, "
         f"Nf {res['Nf_free_fit']:.3f} — confidence: {res['N_confidence']}"]
    if not res['plateau_reached']:
        L.append("     ** the sweep does NOT reach the plateau: N is an "
                 "extrapolation and may be the wrong integer.")
    return L


def aggregate():
    """Collect every per-system JSON into one table (run after the sweep)."""
    import pandas as pd
    recs = []
    for p in sorted(glob.glob(f"{OUT_DIR}/*_richards.json")):
        with open(p) as f:
            recs.append(json.load(f))
    if not recs:
        print(f"no *_richards.json in {OUT_DIR}/")
        return None
    df = pd.DataFrame(recs)
    out = f"{OUT_DIR}/richards_fits.csv"
    df.to_csv(out, index=False)
    ok = df[df['ok']] if 'ok' in df else df
    print(f"aggregated {len(df)} systems ({len(ok)} fitted) -> {out}")
    if len(ok):
        cols = [c for c in ['system', 'N', 'n_broken_pairs', 'x0', 'k', 'v',
                            'rms_resid_e', 'N_integer_deviation',
                            'plateau_reached'] if c in ok]
        print(ok[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default=None,
                    help='fit one system (default: every curve CSV found)')
    ap.add_argument('--N-fixed', type=int, default=None, dest='n_fixed',
                    help='force the integer dissociation limit (e.g. 6 for N2)')
    args = ap.parse_args()
    if args.name:
        res = fit_and_save(args.name, N_fixed=args.n_fixed)
        print(f"{args.name}:")
        print(summary_line(res))
        if res.get('ok'):
            print(f"  {res['equation']}")
    else:
        paths = sorted(glob.glob(f"{CURVE_DIR}/*_Dissociation_Curve.csv"))
        if not paths:
            raise SystemExit(f"no *_Dissociation_Curve.csv in {CURVE_DIR}/")
        print(f"fitting {len(paths)} systems...\n")
        for p in paths:
            nm = os.path.basename(p).replace('_Dissociation_Curve.csv', '')
            res = fit_and_save(nm, curve_csv=p, N_fixed=args.n_fixed)
            print(f"{nm}:")
            print(summary_line(res))
        print()
        aggregate()