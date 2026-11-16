"""
check_table1.py -- Table 1, cell by cell, from the kappa-sweep artifacts.
=========================================================================

Table 1 is the central proof of the paper and, for a long time, was the only
table no script checked. It is rebuilt here from the two kappa sweeps
(`kappa_invariance` for H2/LiH/HF, `kappa_invariance_heavy` for N2) with the
three estimators the caption declares:

    column 4  max/min    = max(dIbar_current) / min(dIbar_current)
    column 5  mean       = mean of dIbar_response over the three kappas
    column 6  dispersion = max/min of dIbar_response, minus one

and column 3, N_unpaired, from the per-geometry .row.json the pipeline writes
next to each .npz -- the sweep does not print it, and without this the column
went unchecked.

Both OK and SING rows count. SING marks a degenerate pair with unequal
occupations that linear response cannot describe: chi is fine, the finite
difference simply cannot probe that channel, and the degeneracy-cutoff scan
shows it carries no weight. Dropping those rows would silently change what the
table means.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from . import paper_values as P
from ._common import GRID_DIR, Report, matches_printed, within_relative

import artifacts as art

def _sweep_tags():
    """Every kappa-sweep artifact except the wide-range one, which probes
    kappa OUTSIDE the usable window and is not what Table 1 reports. Derived
    from the registry rather than listed here, so adding a member to
    CFG.validation_kappa_extra does not silently leave its rows unchecked."""
    return [t for t in art.STAGES
            if t.startswith('kappa_invariance') and 'wide' not in t]


def _sweeps():
    """Every kappa-sweep row available, from whichever sweeps have run."""
    import pandas as pd
    frames = [art.load(t) for t in _sweep_tags() if art.have(t)]
    return pd.concat(frames, ignore_index=True) if frames else None


def run(verbose=True):
    rep = Report('Table 1')
    d = _sweeps()
    if d is None:
        rep.fail('kappa sweeps', 'no sweep artifact yet; Table 1 unchecked')
        return rep

    d['label'] = d['system'].map(P.MEMBER_TO_LABEL).fillna(d['system'])
    unknown = sorted(set(d.loc[d['label'] == d['system'], 'system']))
    if unknown:
        # A member with no label silently matches no row of the table and is
        # reported as "missing", which reads as "the sweep never ran" when in
        # fact it did. That happened once with N2; it fails loudly now.
        rep.fail('member labels', f'no label for {unknown}: the sweep ran but '
                 f'its rows cannot be matched to the table')
        return rep

    for (label, R), (ratio, mean, disp) in P.TABLE1.items():
        sub = d[(d['label'] == label) & (np.abs(d['R_Angstrom'] - R) < 5e-3)]
        if sub.empty:
            rep.fail(f'{label} {R:.3f}', 'no rows in any sweep artifact')
            continue
        cur = sub['dIbar_current'].to_numpy(float)
        res = sub['dIbar_response'].to_numpy(float)
        # A row where the degeneracy cutoff is active is not a measurement of
        # regularization sensitivity: Sec. 2.5's cutoff moves the response
        # value there by more than the regularization does, and the manuscript
        # marks those rows as indeterminate rather than quoting them. Checking
        # them against a printed digit compares two different quantities and
        # reports the difference as a failure of reproduction. It happened at
        # HF R = 0.552 A, where one run saw no cutoff activity and the next
        # saw 173 percent of the value.
        span = (sub['degtol_span_pct'].to_numpy(float)
                if 'degtol_span_pct' in sub.columns else np.zeros(len(sub)))
        if np.nanmax(span) >= P.DEGTOL_INDETERMINATE_PCT:
            rep.declared(f'{label} {R:.3f}',
                         f'degeneracy cutoff active (span '
                         f'{np.nanmax(span):.3g} % of the value); Table 1 '
                         f'marks this row and Sec. 2.5 says why')
            continue
        r_ = cur.max() / cur.min()
        tol = (P.TABLE1_TOL_RATIO_REL_HARD if label in P.TABLE1_HARD_MEMBERS
               else P.TABLE1_TOL_RATIO_REL)
        (rep.matches if within_relative(ratio, r_, tol)
         else rep.differs)(f'{label} {R:.3f}  max/min', ratio, r_)
        rep.check_printed(f'{label} {R:.3f}  mean DeltaI_resp', mean,
                          res.mean() * 1e3, '1e-3 e-/bohr^3')
        d_ = 100 * (res.max() / res.min() - 1)
        # Absolute, in percentage points: the dispersion is a ratio of nearly
        # equal numbers minus one, and two runs of the LiH/HF block showed it
        # inheriting the per-point spread additively rather than
        # proportionally. A relative tolerance would be strict where the
        # dispersion is small and slack where it is large, which is backwards.
        (rep.matches if abs(d_ - float(disp)) <= P.TABLE1_TOL_DISPERSION_ABS_PCT
         else rep.differs)(f'{label} {R:.3f}  dispersion', disp, d_, '%')

    _check_n_unpaired(rep)
    return rep


def _check_n_unpaired(rep):
    """Column 3, from the .row.json beside each .npz."""
    inverse = {v: k for k, v in P.MEMBER_TO_LABEL.items()}
    for (label, R), printed in P.TABLE1_N_UNPAIRED.items():
        member = inverse.get(label)
        hits = sorted(glob.glob(os.path.join(
            GRID_DIR, f'{member}_R{R:.2f}.row.json')))
        if not hits:
            rep.fail(f'{label} {R:.3f}  N_unpaired', 'no .row.json')
            continue
        val = json.loads(open(hits[0]).read())['N_unpaired']
        rep.check_printed(f'{label} {R:.3f}  N_unpaired', printed, val)
