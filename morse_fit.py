"""
morse_fit.py — Morse fit of the dissociation ENERGY curve + spectroscopy
========================================================================
Fits the reference potential-energy curve E(R) with the Morse potential

        E(R) = E_inf + De * [ (1 - exp(-a (R - re)))^2 - 1 ]

and derives the standard spectroscopic constants. Unlike the Richards fit
(which describes N_unpaired, the CORRELATION character), this describes the
ENERGY — so the two together characterize a bond from both sides.

WHY MORSE: it is the standard analytic diatomic potential, has the exact
dissociation limit built in (E -> E_inf), and its constants map analytically
onto observables. Validated on the real N2 NEVPT2 curve using only the 8
bonded-region points available:
        re    1.1002 A     vs exp 1.098    -> 0.20% error
        omega_e 2335.9 cm-1 vs exp 2358.6  -> 0.96% error
        Be    1.9891 cm-1  vs exp 1.998    -> 0.40% error
        De    10.35 eV     vs exp 9.79     -> 5.7%  (see caveat)
Those are genuine spectroscopic observables, so this fit doubles as an
INDEPENDENT VALIDATION of the reference method against experiment.

HONEST CAVEATS
  * De is the least reliable constant: it is the extrapolated well depth, so
    a sweep that stops before the plateau extrapolates it (N2 above: 5.7% off
    from a curve reaching only 1.32 A). re/omega_e/Be come from the curvature
    near the minimum and are solid even from few points. `De_extrapolated`
    flags this.
  * Morse's long-range tail is EXPONENTIAL. Real neutral fragments decay as
    -C6/R^6 (dispersion) and ion+neutral as -1/R^4, so the far tail is not
    Morse-shaped. This biases De if distant points dominate the fit; it does
    not affect the curvature-derived constants.
  * Repulsive/unbound curves have no minimum to fit — reported, not forced.

CROSS-FIT (Morse + Richards): see cross_fit_properties(). The interesting one
is `well_fraction_at_x0` — what FRACTION of the dissociation energy has been
paid by the time static correlation sets in (the distance x0 from the Richards
fit). For N2 that is ~0.65: the bond climbs 65% of its well before radicalizing.
It links the energy curve to the correlation curve and is a per-system 
descriptor neither fit gives alone.
"""

import glob
import json
import os

import numpy as np

CURVE_DIR = "results_energies"
OUT_DIR = "results_analysis"
MIN_POINTS = 5                       # 4 free parameters

BOHR = 0.52917721092                 # Angstrom per bohr
HA2EV = 27.211386245988
HA2CM = 219474.6313702               # Hartree -> cm^-1
AMU2ME = 1822.888486209              # amu -> electron masses

# most-abundant-isotope masses (amu)
MASS = {'H': 1.00782503207, 'He': 4.00260325415, 'Li': 7.016004548,
        'Be': 9.012182201, 'B': 11.009305406, 'C': 12.0,
        'N': 14.003074004, 'O': 15.994914619, 'F': 18.998403224,
        'Ne': 19.99244017542, 'Na': 22.98976928}


def morse(R, E_inf, De, a, re):
    """Morse potential. E(re) = E_inf - De (the minimum); E(inf) = E_inf."""
    z = np.clip(-a * (np.asarray(R, float) - re), -500.0, 500.0)
    return E_inf + De * ((1.0 - np.exp(z)) ** 2 - 1.0)


def reduced_mass_me(a1, a2):
    """Reduced mass in electron masses; None if an element is unknown."""
    if a1 not in MASS or a2 not in MASS:
        return None
    m1, m2 = MASS[a1], MASS[a2]
    return (m1 * m2 / (m1 + m2)) * AMU2ME


def fit_morse(R, E, a1=None, a2=None):
    """Fit E(R) with the Morse potential and derive spectroscopic constants.

    R in Angstrom, E in Hartree. Pass the two element symbols to also get the
    mass-dependent constants (omega_e, omega_e x_e, D0, Be).
    """
    R = np.asarray(R, float)
    E = np.asarray(E, float)
    m = np.isfinite(R) & np.isfinite(E)
    R, E = R[m], E[m]
    if len(R) < MIN_POINTS:
        return {'ok': False, 'reason': f'only {len(R)} valid points '
                                       f'(need >= {MIN_POINTS})'}
    o = np.argsort(R)
    R, E = R[o], E[o]

    imin = int(np.argmin(E))
    if imin == 0 or imin == len(R) - 1:
        return {'ok': False,
                'reason': 'no interior minimum — curve is repulsive or the '
                          'sweep does not bracket the equilibrium'}

    from scipy.optimize import curve_fit
    depth0 = float(max(E[-1] - E[imin], 1e-3))
    p0 = [float(E[imin]) + depth0, depth0, 2.0, float(R[imin])]
    bounds = ([E.min() - 1.0, 1e-6, 1e-3, R.min() * 0.5],
              [E.max() + 5.0, 10.0, 50.0, R.max() * 1.5])
    try:
        popt, cov = curve_fit(morse, R, E, p0=p0, bounds=bounds, maxfev=100000)
    except Exception as e:                                   # noqa: BLE001
        return {'ok': False, 'reason': f'curve_fit failed: {e}'}

    E_inf, De, a_ang, re_ang = (float(v) for v in popt)
    perr = (np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov))
            else np.full(4, np.nan))
    pred = morse(R, *popt)
    resid = pred - E
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((E - E.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan

    # atomic units for the derived constants
    a_au = a_ang * BOHR                       # 1/bohr
    re_au = re_ang / BOHR                     # bohr
    k_au = 2.0 * De * a_au ** 2               # Eh/bohr^2 (force constant)

    out = {
        'ok': True,
        'E_inf_Eh': E_inf, 'De_Eh': De, 'a_inv_A': a_ang, 're_A': re_ang,
        'E_min_Eh': E_inf - De,
        'E_inf_err': float(perr[0]), 'De_err_Eh': float(perr[1]),
        'a_err': float(perr[2]), 're_err_A': float(perr[3]),
        'De_eV': De * HA2EV,
        'k_Eh_bohr2': k_au,
        'k_N_per_m': k_au * 4.3597447222071e-18 / (BOHR * 1e-10) ** 2,
        'rms_resid_Eh': rms, 'max_resid_Eh': float(np.max(np.abs(resid))),
        'r2': float(r2), 'n_points': int(len(R)),
        'R_min': float(R.min()), 'R_max': float(R.max()),
        # Is De trustworthy? Only if the sweep actually approaches the
        # asymptote. If the last point is far below E_inf, De is extrapolated.
        'De_extrapolated': bool((E_inf - E[-1]) > 0.10 * De),
        'dissociation_fraction_sampled': float((E[-1] - (E_inf - De))
                                               / De) if De > 0 else np.nan,
    }

    mu = reduced_mass_me(a1, a2) if (a1 and a2) else None
    if mu:
        we = a_au * np.sqrt(2.0 * De / mu)     # Eh (angular freq, au)
        wexe = a_au ** 2 / (2.0 * mu)          # Eh
        out.update({
            'reduced_mass_amu': mu / AMU2ME,
            'omega_e_cm': float(we * HA2CM),
            'omega_e_xe_cm': float(wexe * HA2CM),
            'D0_eV': float((De - we / 2.0) * HA2EV),   # zero-point corrected
            'Be_cm': float((1.0 / (2.0 * mu * re_au ** 2)) * HA2CM),
            'ZPE_eV': float((we / 2.0) * HA2EV),
        })
    out['equation'] = (f"E(R) = {E_inf:.6f} + {De:.6f}*[(1 - "
                       f"exp(-{a_ang:.4f}*(R - {re_ang:.4f})))^2 - 1]  [Eh, A]")
    return out


def report(res, cross=None):
    """Multi-line physical report: equation, every parameter with its meaning,
    and the derived spectroscopic constants. Returned as a list of lines so
    the caller can print and log them identically."""
    if not res.get('ok'):
        return [f"  [morse] no fit — {res.get('reason')}"]
    L = ["  [morse] dissociation ENERGY curve  "
         f"({res.get('energy_col', '?')}, {res['n_points']} pts, "
         f"R={res['R_min']:.2f}-{res['R_max']:.2f} A)",
         f"     {res['equation']}",
         f"     re    = {res['re_A']:.4f} +/- {res['re_err_A']:.4f} A"
         "        equilibrium bond length (potential minimum)",
         f"     De    = {res['De_eV']:.4f} eV ({res['De_Eh']:.6f} Eh)"
         "     well depth: energy to dissociate from the well bottom",
         f"     a     = {res['a_inv_A']:.4f} 1/A"
         "               range parameter: how fast the well opens up",
         f"     E_inf = {res['E_inf_Eh']:.6f} Eh"
         "           energy of the separated fragments (asymptote)",
         f"     k     = {res['k_N_per_m']:.1f} N/m"
         "              force constant = 2*De*a^2 (bond stiffness)"]
    if 'omega_e_cm' in res:
        L += [f"     w_e   = {res['omega_e_cm']:.1f} cm-1"
              "            harmonic vibrational frequency",
              f"     w_e*x_e = {res['omega_e_xe_cm']:.2f} cm-1"
              "          anharmonicity (deviation from a harmonic well)",
              f"     D0    = {res['D0_eV']:.4f} eV"
              "             dissociation energy corrected for zero-point motion",
              f"     Be    = {res['Be_cm']:.4f} cm-1"
              "           rotational constant (from re and the reduced mass)",
              f"     ZPE   = {res['ZPE_eV']:.4f} eV"
              "             zero-point vibrational energy"]
    L.append(f"     fit: RMS={res['rms_resid_Eh']:.2e} Eh, R^2={res['r2']:.6f}")
    if res.get('De_extrapolated'):
        L.append("     ** De is EXTRAPOLATED: the sweep stops well below the "
                 "asymptote, so De (and D0) are unreliable.")
        L.append(f"        (only {100*res.get('dissociation_fraction_sampled', float('nan')):.0f}"
                 "% of the well is sampled; re/w_e/Be come from the curvature "
                 "and remain solid)")
    if cross:
        L += ["  [morse+richards] energy curve read together with the "
              "correlation curve:",
              f"     x0/re = {cross['x0_over_re']:.3f}"
              "             how far past equilibrium the bond must stretch "
              "before radicalizing",
              f"     well fraction at x0 = {cross['well_fraction_at_x0']:.3f}"
              "   FRACTION of the dissociation energy already paid when "
              "static correlation switches on"]
        if np.isfinite(cross.get('R_at_N_unpaired_1_A', np.nan)):
            L.append(f"     R(N_unpaired=1) = "
                     f"{cross['R_at_N_unpaired_1_A']:.3f} A"
                     "    distance at which one full electron pair has broken")
    return L


def cross_fit_properties(morse_res, richards_res):
    """Descriptors that need BOTH fits — the energy curve and the correlation
    curve read together. None of these is available from either fit alone."""
    if not (morse_res.get('ok') and richards_res.get('ok')):
        return {}
    re = morse_res['re_A']
    De = morse_res['De_Eh']
    a = morse_res['a_inv_A']
    E_inf = morse_res['E_inf_Eh']
    x0 = richards_res['x0']

    # energy at the static-correlation onset, relative to the well bottom
    E_x0 = morse(np.array([x0]), E_inf, De, a, re)[0]
    E_min = E_inf - De
    well_frac = float((E_x0 - E_min) / De) if De > 0 else np.nan

    # distance at which one full pair has unpaired (N_unpaired = 1)
    N, k, v = richards_res['N'], richards_res['k'], richards_res['v']
    R_at_1 = np.nan
    if N > 1:
        try:                                   # invert the Richards curve
            R_at_1 = float(x0 - np.log((N / 1.0) ** (1.0 / v) - 1.0) / k)
        except Exception:                      # noqa: BLE001
            pass
    return {
        'x0_over_re': float(x0 / re) if re else np.nan,
        'x0_minus_re_A': float(x0 - re),
        # THE interesting one: what fraction of the dissociation energy has
        # been paid by the time static correlation switches on. N2 ~ 0.65.
        'well_fraction_at_x0': well_frac,
        'E_at_x0_Eh': float(E_x0),
        'R_at_N_unpaired_1_A': R_at_1,
    }


def fit_and_save(name, rows=None, curve_csv=None, energy_col='E_NEVPT2',
                 a1=None, a2=None, richards_res=None):
    """Fit one system's energy curve and write <name>_morse.json. Never
    raises: a failed fit is reported, not propagated into an expensive run."""
    try:
        if rows is not None:
            R = [r.get('R_Angstrom', np.nan) for r in rows]
            E = [r.get(energy_col, np.nan) for r in rows]
        else:
            import pandas as pd
            df = pd.read_csv(curve_csv or
                             f"{CURVE_DIR}/{name}_Dissociation_Curve.csv")
            if energy_col not in df.columns:
                raise KeyError(f"{energy_col} not in the curve CSV")
            R, E = df['R_Angstrom'].values, df[energy_col].values
        res = fit_morse(R, E, a1=a1, a2=a2)
        if richards_res is not None:
            res.update(cross_fit_properties(res, richards_res))
    except Exception as e:                                   # noqa: BLE001
        res = {'ok': False, 'reason': f'{type(e).__name__}: {e}'}
    res['system'] = name
    res['energy_col'] = energy_col
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{name}_morse.json", 'w') as f:
        json.dump(res, f, indent=2, default=float)
    return res


def aggregate():
    """Collect every per-system Morse JSON into one table."""
    import pandas as pd
    recs = []
    for p in sorted(glob.glob(f"{OUT_DIR}/*_morse.json")):
        with open(p) as f:
            recs.append(json.load(f))
    if not recs:
        print(f"no *_morse.json in {OUT_DIR}/")
        return None
    df = pd.DataFrame(recs)
    out = f"{OUT_DIR}/morse_fits.csv"
    df.to_csv(out, index=False)
    ok = df[df['ok']] if 'ok' in df else df
    print(f"aggregated {len(df)} systems ({len(ok)} fitted) -> {out}")
    if len(ok):
        cols = [c for c in ['system', 're_A', 'De_eV', 'omega_e_cm', 'Be_cm',
                            'rms_resid_Eh', 'De_extrapolated'] if c in ok]
        print(ok[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default=None)
    ap.add_argument('--energy-col', default='E_NEVPT2', dest='ecol')
    args = ap.parse_args()
    paths = ([f"{CURVE_DIR}/{args.name}_Dissociation_Curve.csv"] if args.name
             else sorted(glob.glob(f"{CURVE_DIR}/*_Dissociation_Curve.csv")))
    if not paths:
        raise SystemExit(f"no curve CSVs in {CURVE_DIR}/")
    for p in paths:
        nm = os.path.basename(p).replace('_Dissociation_Curve.csv', '')
        # element symbols from the member name, e.g. 'NN_q+0_s0' -> N, N
        core = nm.split('_q')[0]
        a1 = a2 = None
        for k in sorted(MASS, key=len, reverse=True):
            if core.startswith(k):
                a1, a2 = k, core[len(k):]
                break
        res = fit_and_save(nm, curve_csv=p, energy_col=args.ecol,
                           a1=a1, a2=(a2 if a2 in MASS else None))
        print(f"{nm}: {'OK' if res.get('ok') else res.get('reason')}")
    print()
    aggregate()