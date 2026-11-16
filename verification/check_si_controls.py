"""
check_si_controls.py -- the three controls the manuscript leans on.
===================================================================

Each of these exists because a specific sentence would otherwise be an
assertion. They are checked against the artifacts their stages write, so what
is verified is the run's own output rather than a number remembered from a
previous one.

  fixed volume    Sec. 2.1 claims a fixed-volume control. Under a fixed
                  cylinder the POTENTIAL metric's peak moves to the most
                  compressed geometry of the scan -- the opposite of the
                  reading the paper gives it -- while the response metric's
                  peak stays inside a narrow band. That contrast is what makes
                  the control worth reporting.
  basis           Change only the orbital basis. The potential metric moves by
                  tens of percent, the response metric by a couple. A referee
                  asking "basis artefact?" gets a number.
  n2v             An independent published inverter on the same densities.
                  Optional dependency: absent means unchecked, not passed.
  reduced coord   Sec. S11.6. The pathology contrast, evaluated at 1.5 R_e
                  instead of between each curve's own maximum and its own last
                  geometry. The old statistic gave 2.5x against 2.3x and
                  discriminated nothing; what is checked here is the claim
                  that replaced it, not a list of six digits.
  kappa hierarchy Sec. 2.6 used to say the projection annihilates the kappa
                  dependence identically. The middle level of this hierarchy
                  -- the density-reproduction error -- is the measurement that
                  refutes that, so the check asserts it is NOT flat. If a
                  rerun ever made it flat, the retraction would be the thing
                  that is wrong.
"""

from __future__ import annotations

import numpy as np

from . import paper_values as P
from ._common import ADMITTED, CURVES_DIR, LABEL, Report

import artifacts as art


def run(verbose=True):
    rep = Report('SI controls')

    # ---- fixed-volume control on the response metric ----------------------
    if not art.have('fixed_volume_response'):
        rep.fail('fixed_volume_response', 'artifact missing')
    else:
        d = art.load('fixed_volume_response')
        for system, col, R_si, ratio_si in P.FIXED_VOLUME_TABLE:
            g = d[d['system'] == system].sort_values('R_Angstrom')
            if g.empty or col not in g.columns:
                rep.fail(f'S11.4 {system} {col}', 'no data')
                continue
            y = g[col].to_numpy(float)
            R = g['R_Angstrom'].to_numpy(float)
            i = int(np.nanargmax(y))
            rep.check(f'S11.4 {system} {col:<24} R of the max', R_si,
                      float(R[i]), 5e-3, 'A')
            rep.check(f'S11.4 {system} {col:<24} max/dissociation', ratio_si,
                      float(y[i] / y[-1]), 0.05)
        # the qualitative reading the SI states in words
        g = d[d['system'] == P.FIXED_VOLUME_PEAK_SYSTEM]
        lo, hi = P.FIXED_VOLUME_RESP_PEAK_RANGE_A
        peaks = [float(g['R_Angstrom'].iloc[int(np.nanargmax(
                 g[c].to_numpy(float)))])
                 for c in g.columns if c.startswith('DeltaI_resp_')]
        rep._add(rep.OK if all(lo <= r <= hi for r in peaks) else rep.DIFF,
                 'response peak stays in the band, every window',
                 f'{lo}-{hi} A', ', '.join(f'{v:.3f}' for v in peaks), 'A')

    # ---- orbital-basis dependence ----------------------------------------
    if not art.have('basis_convergence'):
        rep.fail('basis_convergence', 'artifact missing')
    else:
        d = art.load('basis_convergence').sort_values('R_Angstrom')
        for R_si, db_si, dr_si in P.BASIS_TABLE:
            g = d[np.abs(d['R_Angstrom'] - R_si) < 5e-3]
            if g.empty:
                rep.fail(f'S11.5 R={R_si}', 'no data')
                continue
            rep.check(f'S11.5 R={R_si:<6} potential metric change', db_si,
                      float(g['dIbar_change_pct'].iloc[0]),
                      max(0.05 * db_si, 0.5), '%')
            rep.check(f'S11.5 R={R_si:<6} response metric change', dr_si,
                      float(g['resp_change_pct'].iloc[0]),
                      max(0.05 * dr_si, 0.1), '%')
        # Medians over the geometries the SI table lists, not over every row
        # the stage happened to produce: the run also covers the dissociated
        # tail, where both metrics are small and the ratio is not the quantity
        # the paper reports. Taking the median over a different set than the
        # one printed gives a different number and no error -- silent
        # disagreement between a check and the thing it checks.
        tabulated = d[d['R_Angstrom'].isin([r for r, _, _ in P.BASIS_TABLE])]
        mb = float(np.median(tabulated['dIbar_change_pct']))
        mr = float(np.median(tabulated['resp_change_pct']))
        rep.check('S11.5 ratio between the two medians',
                  P.BASIS_CHANGE_FACTOR, mb / max(mr, 1e-30),
                  0.1 * P.BASIS_CHANGE_FACTOR)
        # The declared limitation, checked rather than trusted: the position
        # of the potential metric's maximum -- the one property the paper
        # claims for it -- moves with the orbital basis.
        a_, b_ = P.BASIS_PEAK_SHIFT_A
        rep.check('S11.5 peak of DeltaI_bond, reference basis (argmax)', a_,
                  float(d.loc[d['dIbar_reference'].idxmax(), 'R_Angstrom']),
                  5e-3, 'A')
        rep.check('S11.5 peak of DeltaI_bond, larger basis (argmax)', b_,
                  float(d.loc[d['dIbar_test'].idxmax(), 'R_Angstrom']),
                  5e-3, 'A')
        _check_basis_whole_curve(rep, d)

    # ---- functional-derivative identity (Sec. S3) --------------------------
    # The routine used to return a bare True and write nothing, so the two
    # numbers Sec. S3 prints had no deposited source. Compared per route: the
    # LSDA control must be at machine precision, because its potential is
    # analytic and any error there is the machinery rather than the
    # derivative, and the two PBE routes must stay far below the scale at
    # which a wrong second derivative would show.
    if not art.have('functional_derivative'):
        rep.fail('functional_derivative', 'artifact missing')
    else:
        d = art.load('functional_derivative')
        for route, (printed, factor) in P.DERIV_ROUTES.items():
            sub = d[d['route'] == route] if 'route' in d.columns else d.iloc[0:0]
            if sub.empty:
                rep.fail(f'S3 {route}', 'no rows for this route')
                continue
            worst = float(sub['rel_L1'].max())
            label = f'S3 {route}, worst rel(L1)'
            # An order-of-magnitude claim, checked as one: the printed value
            # is a worst case over three grid levels and moves with the
            # quadrature, so pinning its digits would pin noise.
            if printed / factor <= worst <= printed * factor:
                rep.matches(label, f'{printed:.1e}', f'{worst:.2e}')
            else:
                rep.differs(label, f'{printed:.1e}', f'{worst:.2e}')
            if worst >= P.DERIV_BUG_LEVEL:
                rep.differs(f'S3 {route}, below the bug scale',
                            f'< {P.DERIV_BUG_LEVEL:.0e}', f'{worst:.2e}')

    # ---- grid resolution (Sec. S1) ----------------------------------------
    if not art.have('grid_resolution'):
        rep.fail('grid_resolution', 'artifact missing')
    else:
        d = art.load('grid_resolution')
        worst = float(d['shift_spline_A'].max())
        # Sec. S1 claims the spline peak is stable under decimation to
        # ~1e-3 A. That is the claim; check it, not the table.
        if worst <= 5e-3:
            rep.matches('S1 spline peak shift under decimation, worst',
                        '<= 0.005 A', f'{worst:.4f} A')
        else:
            rep.differs('S1 spline peak shift under decimation, worst',
                        '<= 0.005 A', f'{worst:.4f} A')

    # ---- independent inverter --------------------------------------------
    if not art.have('n2v_cross_check'):
        rep.note('n2v cross-check not run (optional dependency); this is '
                 'unchecked, not passed')
    else:
        d = art.load('n2v_cross_check')
        col = 'max_abs_diff_Eh'
        ok = int(d[col].notna().sum()) if col in d.columns else 0
        rep.note(f'n2v cross-check present: {ok} of {len(d)} geometries '
                 f'actually compared')
        if len(d) and not ok:
            rep.fail('n2v_cross_check', 'the stage ran but compared nothing; '
                     'see the note column of the artifact')

    _check_reduced_coordinate(rep)
    _check_kappa_hierarchy(rep)
    _check_gate_admits_every_kappa(rep)
    _check_neighbour_step(rep)
    return rep


# ---------------------------------------------------------------------------
# Sec. 4.1: the gate cannot choose between the regularizations
# ---------------------------------------------------------------------------
def _check_gate_admits_every_kappa(rep):
    """Every kappa in the swept window yields an inversion the gate admits.

    This is the load-bearing sentence of the answer to "the protocol never
    visits that window". If it ever stops being true the answer is wrong, so
    it is a check and not a remark.
    """
    tags = [t for t in ('kappa_invariance', 'kappa_invariance_heavy',
                        'kappa_invariance_extra_HLi_q+0_s0',
                        'kappa_invariance_extra_HF_q+0_s0') if art.have(t)]
    if not tags:
        rep.fail('4.1 gate vs kappa', 'no kappa sweep on disk')
        return
    uncond_inv = uncond_geom = cond_inv = 0
    lo, hi = float('inf'), 0.0
    for t in tags:
        d = art.load(t)
        if 'system' not in d.columns or 'density_error' not in d.columns:
            continue
        for (sysname, _R), g in d.groupby(['system', 'R_Angstrom']):
            thr = P.KAPPA_GATE_THRESHOLD.get(sysname)
            if thr is None:
                continue
            inside = bool((g['density_error'] <= thr).all())
            if sysname in P.KAPPA_UNCONDITIONAL_SYSTEMS:
                if inside:
                    uncond_inv += len(g)
                    uncond_geom += 1
                    v = g['dIbar_current'].to_numpy(float)
                    if len(v) and v.min() > 0:
                        lo = min(lo, float(v.max() / v.min()))
                        hi = max(hi, float(v.max() / v.min()))
            elif inside:
                cond_inv += len(g)
    rep.check('4.1 unconditional inversions inside the gate',
              P.KAPPA_UNCONDITIONAL_INVERSIONS, uncond_inv, 0)
    rep.check('4.1 unconditional geometries admitting every kappa',
              P.KAPPA_UNCONDITIONAL_GEOMETRIES, uncond_geom, 0)
    rep.check('4.1 conditional inversions (diffuse threshold)',
              P.KAPPA_CONDITIONAL_INVERSIONS, cond_inv, 0)
    a_, b_ = P.KAPPA_UNCONDITIONAL_SPREAD
    rep.check('4.1 spread among them, smallest', a_, lo, 0.05, 'x')
    rep.check('4.1 spread among them, largest', b_, hi, 0.15, 'x')

    # the production census the same paragraph quotes
    curves = _admitted_curves()
    if curves is None:
        rep.fail('4.1 kappa_final census', 'no curves on disk')
        return
    got = curves['kappa_final'].round(10).value_counts().to_dict()
    for k, n in P.KAPPA_FINAL_CENSUS.items():
        rep.check(f'4.1 admitted geometries at kappa = {k:.0e}', n,
                  int(got.get(round(k, 10), 0)), 0)


# ---------------------------------------------------------------------------
# Sec. 4.2: the step between neighbouring geometries
# ---------------------------------------------------------------------------
def _admitted_curves():
    import glob
    import os
    import pandas as pd
    frames = []
    for p in sorted(glob.glob(os.path.join(CURVES_DIR,
                                           '*_Dissociation_Curve.csv'))):
        d = pd.read_csv(p)
        if 'inv_status' not in d.columns:
            continue
        d['system'] = os.path.basename(p).replace('_Dissociation_Curve.csv', '')
        frames.append(d[d['inv_status'].isin(ADMITTED)])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _check_neighbour_step(rep):
    """How far apart neighbouring geometries are, against the kappa band.

    Sec. 4.2 says no feature read off one or two adjacent points is resolved,
    and sizes the secondary maximum accordingly. Both halves are checked: the
    step itself, and the rise that IS claimed.
    """
    curves = _admitted_curves()
    if curves is None:
        rep.fail('4.2 neighbour step', 'no curves on disk')
        return
    col = 'DeltaI_resp_PBE-PBE_exact'
    for sysname, want in P.NEIGHBOUR_STEP_PCT.items():
        g = curves[curves['system'] == sysname].sort_values('R_Angstrom')
        v = g[col].dropna().to_numpy(float)
        if len(v) < 4:
            rep.fail(f'4.2 {sysname} step', 'fewer than four admitted points')
            continue
        step = 100.0 * np.abs(np.diff(v)) / ((v[:-1] + v[1:]) / 2.0)
        rep.check(f'4.2 {LABEL[sysname]:<5} median step between neighbours',
                  want, float(np.median(step)), P.NEIGHBOUR_STEP_TOL_PCT, '%')

    g = curves[curves['system'] == 'HH_q+0_s0'].sort_values('R_Angstrom')
    v = g[col].dropna().to_numpy(float)
    R = g.dropna(subset=[col])['R_Angstrom'].to_numpy(float)
    if len(v) > 4:
        rep.check('4.2 H2    total range of the response curve',
                  P.H2_TOTAL_RANGE_PCT, 100.0 * (v.max() / v.min() - 1.0),
                  3.0, '%')
        try:
            a = float(v[np.argmin(np.abs(R - P.H2_SECONDARY_FROM_A))])
            b = float(v[np.argmin(np.abs(R - P.H2_SECONDARY_TO_A))])
            rep.check('4.2 H2    secondary maximum, rise over four points',
                      P.H2_SECONDARY_RISE_PCT, 100.0 * (b / a - 1.0), 0.3, '%')
            # and that the two geometries quoted as nearest really are the
            # nearest swept ones, which is where the previous wording went
            # wrong: it named 1.073 A, the farthest of the three.
            if art.have('kappa_invariance'):
                k = art.load('kappa_invariance')
                Rk = np.array(sorted(k['R_Angstrom'].unique()), float)
                mid = 0.5 * (P.H2_SECONDARY_FROM_A + P.H2_SECONDARY_TO_A)
                near = tuple(sorted(Rk[np.argsort(np.abs(Rk - mid))[:2]]))
                if near == P.H2_BAND_NEAREST_A:
                    rep.matches('4.2 the two nearest swept geometries',
                                str(P.H2_BAND_NEAREST_A), str(near), 'A')
                else:
                    rep.differs('4.2 the two nearest swept geometries',
                                str(P.H2_BAND_NEAREST_A), str(near), 'A')
        except (ValueError, IndexError):
            rep.fail('4.2 H2 secondary maximum', 'geometries not on disk')


# ---------------------------------------------------------------------------
# Sec. S11.5, whole-curve statistics
# ---------------------------------------------------------------------------
def _spline_peak(R, y):
    """The estimator that produces the peak position quoted in the text.

    Sec. S11.5 prints two displacements of the same maximum -- one from a plain
    argmax over the sampled geometries, one from this spline -- because they
    differ by more than a rounding and a reader who recomputes either should
    not read the other as a discrepancy. Both are checked.
    """
    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError:
        return None
    R = np.asarray(R, float)
    sp = UnivariateSpline(R, np.asarray(y, float), s=0, k=3)
    xx = np.linspace(R[0], R[-1], 20001)
    return float(xx[int(np.argmax(sp(xx)))])


def _spline_feature(R, y, lo, hi, want='min'):
    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError:
        return None
    R = np.asarray(R, float)
    sp = UnivariateSpline(R, np.asarray(y, float), s=0, k=3)
    xx = np.linspace(lo, hi, 6001)
    yy = sp(xx)
    if want == 'min':
        return float(xx[int(np.argmin(yy))])
    return float(xx[int(np.argmax(yy))])


def _check_basis_whole_curve(rep, d):
    R = d['R_Angstrom'].to_numpy(float)
    for name, ref, test in (('potential', 'dIbar_reference', 'dIbar_test'),
                            ('response', 'resp_reference', 'resp_test')):
        med, mx, corr = P.BASIS_WHOLE_CURVE[name]
        col = 'dIbar_change_pct' if name == 'potential' else 'resp_change_pct'
        ch = d[col].abs()
        rep.check(f'S11.5 {name:<9} median change over all {len(d)} geometries',
                  med, float(np.median(ch)), max(0.05 * med, 0.2), '%')
        rep.check(f'S11.5 {name:<9} worst change over all {len(d)} geometries',
                  mx, float(ch.max()), max(0.05 * mx, 0.2), '%')
        rep.check(f'S11.5 {name:<9} TZ-QZ curve correlation', corr,
                  float(np.corrcoef(d[ref], d[test])[0, 1]), 0.02)

    # The sentence that replaced "the position remains robust": every
    # basis-induced change in the response metric is inside the kappa bound
    # the paper already prints. This is a claim, so it is checked as one.
    worst = float(d['resp_change_pct'].abs().max())
    bound = P.BASIS_RESP_INSIDE_KAPPA_BOUND_PCT
    if worst <= bound:
        rep.matches('S11.5 every response change inside the kappa bound',
                    f'all <= {bound}%', f'worst {worst:.2f}%')
    else:
        rep.differs('S11.5 every response change inside the kappa bound',
                    f'all <= {bound}%', f'worst {worst:.2f}%')

    tol = P.BASIS_PEAK_SHIFT_TOL_A
    pk = {'potential': (P.BASIS_PEAK_SHIFT_SPLINE_A, 'dIbar_reference',
                        'dIbar_test'),
          'response': (P.BASIS_RESP_PEAK_A, 'resp_reference', 'resp_test')}
    shifts = {}
    for name, ((a_, b_), ref, test) in pk.items():
        got_ = {}
        for label, want, col in ((f'{name} peak, reference basis (spline)',
                                  a_, ref),
                                 (f'{name} peak, larger basis (spline)',
                                  b_, test)):
            got = _spline_peak(R, d[col].to_numpy(float))
            got_[col] = got
            if got is None:
                rep.fail(f'S11.5 {label}', 'scipy unavailable')
                continue
            rep.check(f'S11.5 {label}', want, got, tol, 'A')
        # The displacement is reported to three decimals in Sec. S11.5, so it
        # is measured rather than differenced from the two rounded positions:
        # 1.148 - 0.838 gives 0.310 and the text prints 0.309.
        shifts[name] = (abs(got_[ref] - got_[test])
                        if None not in got_.values() else abs(a_ - b_))

    # The retraction itself, as an inequality: the potential metric's maximum
    # moves by MORE than the physical scale Sec. S1 says the scan resolves,
    # and that is why the abstract no longer calls the position robust.
    scale = P.PEAK_TO_CF_SCALE_A
    if shifts['potential'] > scale:
        rep.matches('S11.5 potential peak shift exceeds the S1 scale',
                    f'> {scale} A', f'{shifts["potential"]:.3f} A')
    else:
        rep.differs('S11.5 potential peak shift exceeds the S1 scale',
                    f'> {scale} A', f'{shifts["potential"]:.3f} A')

    # The interior features of the response curve, which move MORE than its
    # global maximum does. Sec. S11.5 reports them for exactly that reason, so
    # a check that only looked at the global peak would be checking the
    # flattering half of the paragraph.
    for label, want, lo, hi, kind in (
            ('response interior minimum', P.BASIS_RESP_INTERIOR_MIN_A,
             0.9, 3.0, 'min'),
            ('response secondary maximum', P.BASIS_RESP_SECONDARY_MAX_A,
             1.9, 3.0, 'max')):
        for basis, col, w in (('reference basis', 'resp_reference', want[0]),
                              ('larger basis', 'resp_test', want[1])):
            got = _spline_feature(R, d[col].to_numpy(float), lo, hi, kind)
            if got is None:
                rep.fail(f'S11.5 {label}, {basis}', 'scipy unavailable')
                continue
            rep.check(f'S11.5 {label}, {basis}', w, got, 5e-3, 'A')


# ---------------------------------------------------------------------------
# Sec. S11.6, the matched reduced coordinate
# ---------------------------------------------------------------------------
def _check_reduced_coordinate(rep):
    if not art.have('reduced_coordinate'):
        rep.fail('reduced_coordinate', 'artifact missing')
        return
    d = art.load('reduced_coordinate').set_index('system')
    routes = [c for c in d.columns if c.startswith('ratio_DeltaI_resp_')]

    spanning = sorted(d.index[d['span_ok'].astype(bool)])
    rep.check('S11.6 systems spanning R_e to 1.5 R_e', P.REDUCED_SPANNING,
              len(spanning), 0)
    missing = [s for s in P.REDUCED_NOT_SPANNING if s in spanning]
    if missing:
        rep.differs('S11.6 the four declared systems stay out of the table',
                    'none of them spans', ', '.join(missing))
    else:
        rep.matches('S11.6 the four declared systems stay out of the table',
                    'none of them spans', 'none does')

    g = P.REDUCED_GROWING_SYSTEM
    if g not in d.index:
        rep.fail(f'S11.6 {g}', 'not in the artifact')
        return
    grow = d.loc[g, routes].astype(float)
    lo, hi = P.REDUCED_GROWING_RANGE
    rep.check(f'S11.6 {LABEL[g]:<5} lowest of the four routes', lo,
              float(grow.min()), P.REDUCED_TOL)
    rep.check(f'S11.6 {LABEL[g]:<5} highest of the four routes', hi,
              float(grow.max()), P.REDUCED_TOL)

    neu = d.loc[[s for s in P.REDUCED_NEUTRALS if s in d.index], routes]
    neu = neu.astype(float)
    nlo, nhi = P.REDUCED_NEUTRAL_RANGE
    rep.check('S11.6 neutrals, lowest of all routes', nlo,
              float(neu.to_numpy().min()), P.REDUCED_TOL)
    rep.check('S11.6 neutrals, highest of all routes', nhi,
              float(neu.to_numpy().max()), P.REDUCED_TOL)

    # The claim itself: no overlap, in the direction the paper states it.
    if float(grow.min()) > float(neu.to_numpy().max()):
        rep.matches('S11.6 H2+ above every neutral, every route',
                    'no overlap',
                    f'{grow.min():.3f} > {neu.to_numpy().max():.3f}')
    else:
        rep.differs('S11.6 H2+ above every neutral, every route',
                    'no overlap',
                    f'{grow.min():.3f} vs {neu.to_numpy().max():.3f}')
    if float(grow.min()) > 1.0 and float(neu.to_numpy().max()) < 1.0:
        rep.matches('S11.6 H2+ growing and every neutral decaying',
                    'H2+ > 1 > neutrals', 'as stated')
    else:
        rep.differs('S11.6 H2+ growing and every neutral decaying',
                    'H2+ > 1 > neutrals',
                    f'{grow.min():.3f} / {neu.to_numpy().max():.3f}')

    # He2+, the case Sec. S11.6 explicitly does NOT claim.
    sp = P.REDUCED_SPLIT_SYSTEM
    if sp in d.index:
        for col, want in P.REDUCED_SPLIT.items():
            rep.check(f'S11.6 {LABEL[sp]:<5} {col.split("resp_")[-1]:<14}',
                      want, float(d.loc[sp, col]), P.REDUCED_TOL)
        vals = d.loc[sp, list(P.REDUCED_SPLIT)].astype(float)
        if (vals > 1).any() and (vals < 1).any():
            rep.matches('S11.6 He2+ disagrees between functionals',
                        'one above 1, one below', 'as stated')
        else:
            rep.differs('S11.6 He2+ disagrees between functionals',
                        'one above 1, one below',
                        ', '.join(f'{v:.2f}' for v in vals))

    # The two systems Sec. S11.6 declares instead of quoting: no digit, but a
    # direction, and the direction is claimed over BOTH treatments of the
    # compromised node, so both are checked.
    for sysname in P.REDUCED_DECLARED:
        if sysname not in d.index:
            rep.fail(f'S11.6 {sysname}', 'not in the artifact')
            continue
        alt = [c for c in d.columns
               if (c.startswith('ratio_all_points_DeltaI_resp')
                   or c.startswith('ratio_no_indeterminate_DeltaI_resp'))]
        vals = d.loc[sysname, alt].astype(float).dropna()
        if not len(vals):
            rep.fail(f'S11.6 {LABEL[sysname]} direction', 'no values')
            continue
        if float(vals.max()) < P.REDUCED_DECLARED_BELOW:
            rep.matches(f'S11.6 {LABEL[sysname]:<5} decays under every treatment',
                        '< 1 throughout', f'worst {vals.max():.3f}')
        else:
            rep.differs(f'S11.6 {LABEL[sysname]:<5} decays under every treatment',
                        '< 1 throughout', f'worst {vals.max():.3f}')
        if not bool(d.loc[sysname, 'span_ok']):
            rep.matches(f'S11.6 {LABEL[sysname]:<5} declared, not quoted',
                        'span_ok False', str(d.loc[sysname, 'guard_reason']))
        else:
            rep.differs(f'S11.6 {LABEL[sysname]:<5} declared, not quoted',
                        'span_ok False', 'the guards now admit it')

    # And the negative result: the potential metric does not separate them.
    pot = 'ratio_DeltaI_bar_LDA-VWN_exact_Eh'
    if pot in d.columns and 'HH_q+0_s0' in d.index and g in d.index:
        h2, h2p = float(d.loc['HH_q+0_s0', pot]), float(d.loc[g, pot])
        above = h2 > h2p
        if above == P.REDUCED_POT_H2_ABOVE_H2PLUS_LDA:
            rep.matches('S11.6 potential metric puts H2 above H2+ under LDA',
                        'H2 > H2+', f'{h2:.2f} > {h2p:.2f}')
        else:
            rep.differs('S11.6 potential metric puts H2 above H2+ under LDA',
                        'H2 > H2+', f'{h2:.2f} vs {h2p:.2f}')


# ---------------------------------------------------------------------------
# Secs. 2.6, 4.1 and S14.2: the suppression hierarchy
# ---------------------------------------------------------------------------
def _check_kappa_hierarchy(rep):
    if not art.have('kappa_invariance'):
        rep.fail('kappa hierarchy', 'kappa_invariance artifact missing')
        return
    d = art.load('kappa_invariance')
    cols = {'potential': 'dIbar_current', 'density_error': 'density_error',
            'response': 'dIbar_response'}
    spread = {k: [] for k in cols}
    for _, g in d.groupby('R_Angstrom'):
        for k, c in cols.items():
            v = g[c].to_numpy(float)
            if len(v) < 2 or not np.isfinite(v).all() or v.min() <= 0:
                continue
            spread[k].append(100.0 * (v.max() / v.min() - 1.0))
    for k, (lo, hi) in P.KAPPA_HIERARCHY_PCT.items():
        if not spread[k]:
            rep.fail(f'kappa hierarchy, {k}', 'no usable rows')
            continue
        tol = P.KAPPA_HIERARCHY_TOL_PCT
        rep.check(f'2.6/4.1 kappa spread, {k:<14} smallest', lo,
                  float(min(spread[k])), max(tol * lo, 0.2), '%')
        rep.check(f'2.6/4.1 kappa spread, {k:<14} largest', hi,
                  float(max(spread[k])), max(tol * hi, 0.2), '%')

    # The retraction, stated as a check that can fail: the density error is
    # NOT kappa independent. If it ever came out flat, Sec. 2.6's claim that
    # the projection only damps would be the thing to revisit.
    if spread['density_error']:
        worst = 1.0 + max(spread['density_error']) / 100.0
        floor = P.KAPPA_DENSITY_ERROR_MIN_SPREAD
        if worst >= floor:
            rep.matches('2.6 density error is not kappa independent',
                        f'>= {floor:.2f}x', f'{worst:.2f}x')
        else:
            rep.differs('2.6 density error is not kappa independent',
                        f'>= {floor:.2f}x', f'{worst:.2f}x')
