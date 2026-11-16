"""
check_curves.py -- the manuscript's numbers against the dissociation curves.
============================================================================

The bulk of the paper's figures: the confidence gate, Table 2 of the main
text, Tables S4, S7 and S10, and the per-system statements of Sec. 4.5. All of
them are derived from `results_energies/*_Dissociation_Curve.csv`, which the
benchmark scan and the functional post-pass write.

Aggregations over ALL geometries are reported separately from aggregations
over the admitted stratum, because in several systems a single rejected
geometry contaminates the former. The paper quotes the admitted stratum; the
contaminated numbers are printed as a note so the difference is visible rather
than assumed.
"""

from __future__ import annotations

import numpy as np

from . import paper_values as P
from ._common import (LABEL, Report, admitted, column, integral, load_curves,
                      peak)


def run(verbose=True):
    rep = Report('paper numbers vs the dissociation curves')
    D = load_curves()
    if not D:
        rep.fail('curves', 'no *_Dissociation_Curve.csv: run the scan first')
        return rep

    # A system the run has not produced yet is a MISSING CHECK, not a crash.
    # `D['NN_q+0_s0']` used to raise straight out of the runner and take the
    # other hundred comparisons with it, so a partial run reported one failure
    # where it should have reported one per absent system and passed the rest.
    missing = [k for k in P.STRATIFICATION if k not in D]
    for k in missing:
        rep.fail(f'{LABEL.get(k, k)} curve', 'not produced by this run')
    if missing:
        rep.note(f'{len(missing)} system(s) absent; their checks are missing, '
                 f'not passed')

    def sub(key):
        """A curve, or None if this run has not got it."""
        return D.get(key)

    # ---- confidence gate --------------------------------------------------
    rep.check('geometries, total', P.GEOMETRIES_TOTAL,
              sum(len(d) for d in D.values()), 0)
    rep.check('geometries, admitted', P.GEOMETRIES_ADMITTED,
              sum(len(admitted(d)) for d in D.values()), 0)

    # ---- Table 2: where error cancellation lives (H2) ---------------------
    h2 = sub('HH_q+0_s0')
    if h2 is None:
        return rep
    for func, (pe, ps, ie, is_) in P.H2_MODES.items():
        rep.check(f'{func} peak, exact mode', pe, peak(h2, func, 'exact')[0],
                  5e-4, 'Eh')
        rep.check(f'{func} peak, scf mode', ps, peak(h2, func, 'scf')[0],
                  5e-4, 'Eh')
        rep.check(f'{func} integral, exact mode', ie,
                  integral(h2, func, 'exact'), 5e-4, 'Eh*A')
        rep.check(f'{func} integral, scf mode', is_,
                  integral(h2, func, 'scf'), 5e-4, 'Eh*A')

    me = peak(h2, 'LDA-VWN', 'exact')[0]
    ms = peak(h2, 'LDA-VWN', 'scf')[0]
    ie, is_ = integral(h2, 'LDA-VWN', 'exact'), integral(h2, 'LDA-VWN', 'scf')
    rep.check('mode gap at the peak, LDA', P.H2_MODE_GAP_PEAK_LDA_PCT,
              100 * abs(me - ms) / me, 0.01, '%')
    rep.check('mode gap in the integral, LDA',
              P.H2_MODE_GAP_INTEGRAL_LDA_PCT, 100 * abs(ie - is_) / ie,
              0.05, '%')
    iep, isp = integral(h2, 'PBE-PBE', 'exact'), integral(h2, 'PBE-PBE', 'scf')
    rep.check('mode gap in the integral, PBE',
              P.H2_MODE_GAP_INTEGRAL_PBE_PCT, 100 * abs(iep - isp) / iep,
              0.05, '%')

    # ---- Table S10 --------------------------------------------------------
    for k, (le, ls, pe, ps) in P.S10_PEAKS.items():
        if k not in D:
            continue
        for val, func, mode in ((le, 'LDA-VWN', 'exact'),
                                (ls, 'LDA-VWN', 'scf'),
                                (pe, 'PBE-PBE', 'exact'),
                                (ps, 'PBE-PBE', 'scf')):
            rep.check(f'S10 {LABEL[k]:<5} {func:<8} {mode:<5}', val,
                      peak(D[k], func, mode)[0], 5e-4, 'Eh')

    # ---- Table S7 ---------------------------------------------------------
    for k, (v, r) in P.S7_PEAK_AND_R.items():
        if k not in D:
            continue
        m, rmax = peak(D[k], 'PBE-PBE', 'exact')
        rep.check(f'S7 {LABEL[k]:<5} peak', v, m, 5e-3, 'Eh')
        rep.check(f'S7 {LABEL[k]:<5} R of the peak', r, rmax, 5e-3, 'A')

    # ---- Table S4 ---------------------------------------------------------
    for k, (n_ok, n_bad) in P.STRATIFICATION.items():
        if k not in D:
            continue
        d = D[k]
        rep.check(f'S4 {LABEL[k]:<5} admitted', n_ok, len(admitted(d)), 0)
        rep.check(f'S4 {LABEL[k]:<5} rejected', n_bad,
                  len(d) - len(admitted(d)), 0)

    # ---- Sec. 4.5, system by system ---------------------------------------
    y = column(admitted(h2), 'PBE-PBE', 'exact')
    R = admitted(h2)['R_Angstrom'].to_numpy(float)
    at_diss = float(y[np.argmin(abs(R - P.H2_DISSOCIATION_R))])
    rep.check('H2 peak', P.H2_PEAK_EH, float(np.nanmax(y)), 5e-3, 'Eh')
    rep.check('H2 at dissociation', P.H2_AT_DISSOCIATION_EH, at_diss, 5e-3,
              'Eh')
    rep.check('H2 decay factor', P.H2_DECAY_FACTOR,
              float(np.nanmax(y)) / at_diss, 0.15)

    hp = sub('HH_q+1_s1')
    if hp is not None:
        rp = column(admitted(hp), 'PBE-PBE', 'exact', kind='resp')
        Rp = admitted(hp)['R_Angstrom'].to_numpy(float)
        band = (Rp > 2.0) & (Rp < 3.5)
        rep.check('H2+ DeltaI_resp plateau', P.H2PLUS_PLATEAU_RESP,
                  float(np.median(rp[band])), 1e-4, 'e-/bohr^3')
        rep.check('H2+ one-electron identity vxc = -v_H',
                  P.H2PLUS_ONE_ELECTRON_IDENTITY_EH,
                  float(np.nanmax(hp['sie_validation_1e'].to_numpy(float))),
                  5e-6, 'Eh')

    if 'HeHe_q+1_s1' in D:
        rep.check('He2+ DeltaI_resp peak', P.HE2PLUS_PEAK_RESP,
                  float(np.nanmax(column(admitted(D['HeHe_q+1_s1']), 'PBE-PBE',
                                         'exact', kind='resp'))), 3e-4,
                  'e-/bohr^3')

    li = sub('HLi_q+0_s0')
    if li is not None:
        m, rmax = peak(li, 'PBE-PBE', 'exact')
        rep.check('LiH DeltaI_bond peak', P.LIH_PEAK_EH, m, 5e-3, 'Eh')
        rep.check('LiH R of the peak', P.LIH_PEAK_R, rmax, 0.05, 'A')
        rep.check('LiH typical DeltaI_resp', P.LIH_TYPICAL_RESP,
                  float(np.median(column(admitted(li), 'PBE-PBE', 'exact',
                                         kind='resp'))), 4e-4, 'e-/bohr^3')
    if 'HB_q+0_s0' in D:
        rep.check('BH typical DeltaI_bond', P.BH_TYPICAL_EH,
                  float(np.median(column(admitted(D['HB_q+0_s0']), 'PBE-PBE',
                                         'exact'))), 0.02, 'Eh')

    n2 = sub('NN_q+0_s0')
    if n2 is not None:
        rep.check('N2 N_unpaired at dissociation', P.N2_NUNPAIRED_DISSOCIATED,
                  float(n2['N_unpaired'].iloc[-1]), 5e-3)
        idx_eq = int(np.argmin(np.abs(n2['R_Angstrom'].to_numpy(float) - 1.100)))
        rep.check('N2 N_unpaired at equilibrium', P.N2_NUNPAIRED_EQUILIBRIUM,
                  float(n2['N_unpaired'].iloc[idx_eq]), 5e-3)
        rep.check('N2 virial gap', P.N2_VIRIAL_GAP_EH,
                  float(n2['virial_gap_Eh'].min()), 0.05, 'Eh')
        rep.check('N2 virial drift between neighbours', P.N2_VIRIAL_DRIFT_EH,
                  float(np.abs(np.diff(
                      n2['virial_gap_Eh'].to_numpy(float))).max()), 0.05, 'Eh')

    b2 = sub('BB_q+0_s2')
    collapsed = (b2[np.abs(b2['N_unpaired'] - 2.1) < 0.15]
                 if b2 is not None else [])
    if len(collapsed):
        lo, hi = P.B2_COLLAPSE_DENSITY_ERROR
        rep.check('B2 density error where the multiplet collapses, min', lo,
                  float(collapsed['inv_density_error_e'].min()), 0.15, 'e-')
        rep.check('B2 density error where the multiplet collapses, max', hi,
                  float(collapsed['inv_density_error_e'].max()), 0.15, 'e-')

    # ---- Sec. S4.1: the degeneracy-cutoff audit ---------------------------
    # Counted, not listed. Which geometries are affected moves between runs
    # because the cutoff sits on an unbounded kernel; the claim Sec. S4.1
    # makes is about how many, and about their being independent of the
    # confidence gate.
    hit, hit_admitted = [], 0
    for k, d in D.items():
        c = 'DeltaI_resp_PBE-PBE_exact'
        if c not in d.columns or c + '_degtol_span' not in d.columns:
            continue
        v = d[c].to_numpy(float)
        sp = d[c + '_degtol_span'].to_numpy(float)
        bad = np.isfinite(v) & np.isfinite(sp) & (sp >= np.abs(v))
        hit.extend([(k, float(r)) for r in d['R_Angstrom'].to_numpy()[bad]])
        if 'inv_status' in d.columns:
            hit_admitted += int(np.sum(
                bad & d['inv_status'].isin(('OK', 'MARGINAL')).to_numpy()))
    if len(hit) > P.DEGTOL_AFFECTED_MAX:
        rep.differs('S4.1 geometries with span >= value',
                    f'<= {P.DEGTOL_AFFECTED_MAX}', len(hit))
    else:
        rep.matches('S4.1 geometries with span >= value',
                    f'<= {P.DEGTOL_AFFECTED_MAX}', len(hit))
    # If none were admitted the audit would merely restate the confidence
    # gate, which is the one thing Sec. S4.1 says it does not do.
    if 0 < hit_admitted <= P.DEGTOL_AFFECTED_ADMITTED_MAX:
        rep.matches('S4.1 affected geometries inside the admitted stratum',
                    f'1..{P.DEGTOL_AFFECTED_ADMITTED_MAX}', hit_admitted)
    else:
        rep.differs('S4.1 affected geometries inside the admitted stratum',
                    f'1..{P.DEGTOL_AFFECTED_ADMITTED_MAX}', hit_admitted)

    affected = 0
    for d in D.values():
        for c in [c for c in d.columns if c.endswith('_degtol_span')]:
            base = c[:-len('_degtol_span')]
            if base in d.columns:
                v = d[base].to_numpy(float)
                sp = d[c].to_numpy(float)
                affected += int(np.sum(np.isfinite(v) & np.isfinite(sp)
                                       & (sp >= np.abs(v))))
    rep.note(f'(geometry x functional x mode) pairs whose degeneracy-cutoff '
             f'span exceeds the value itself: {affected}')

    for k in ('HC_q+0_s1', 'HeHe_q+1_s1'):
        if k not in D:
            continue
        m_all, r_all = peak(D[k], 'PBE-PBE', 'scf', only_admitted=False)
        m_adm, _ = peak(D[k], 'PBE-PBE', 'scf', only_admitted=True)
        rep.note(f'{LABEL[k]} PBE scf: all geometries {m_all:.4f} at '
                 f'R={r_all:.2f} A; admitted stratum {m_adm:.4f} -- the gap '
                 f'is why the paper quotes the stratum')
    return rep
