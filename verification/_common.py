"""
_common.py -- the shared vocabulary of the checks.
==================================================

One Report object collects every comparison so the run ends with a single
count instead of nine scattered ones, and one loader answers "where does the
data live" so a check can never read a directory the producer does not write.

`matches_printed` is the estimator that matters. Most of these comparisons are
against a number as it appears in the manuscript, and the number of decimals
printed IS the tolerance: "4.0" and "6.42" sit in the same column of Table 1
and must not be checked to the same absolute slack. Rounding the computed
value to the printed precision and comparing strings encodes that exactly,
with nothing to choose.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

CURVES_DIR = 'results_energies'
PROFILES_DIR = 'results_vxc_1D'
GRID_DIR = 'results_grid_data'
ADMITTED = ('OK', 'MARGINAL')
TRAPZ = getattr(np, 'trapezoid', None) or np.trapz

LABEL = {'HH_q+1_s1': 'H2+', 'HeHe_q+1_s1': 'He2+', 'HH_q+0_s0': 'H2',
         'LiLi_q+0_s0': 'Li2', 'NN_q+0_s0': 'N2', 'HLi_q+0_s0': 'LiH',
         'HB_q+0_s0': 'BH', 'HF_q+0_s0': 'HF', 'HC_q+0_s1': 'CH',
         'BB_q+0_s2': 'B2'}


class Report:
    """Every comparison a check makes, and how it came out.

    Four outcomes, not two. Calling everything that is not a pass a FAIL puts
    "this number disagrees with the manuscript" and "this comparison could not
    be made because the run has not produced its input yet" in the same
    column, and both then read as "the calculation failed" -- which is a third
    thing again, and usually not what happened. The words are:

        MATCHES   the run reproduces what the paper prints
        DIFFERS   the run produced a number, and it is not that one
        no data   nothing was compared; this is unchecked, not passed
        declared  the manuscript says this quantity is not determined here,
                  and marks it as such; there is nothing to reproduce

    Only DIFFERS is a finding. `no data` is work still to do. `declared` is
    neither: a quantity the paper itself reports as indeterminate -- the rows
    Table 1 marks with a dagger, where the degeneracy cutoff moves the value
    by more than the regularization does -- cannot be compared against a
    printed digit, and counting it as outstanding work would make the suite
    fail forever at exactly the places the manuscript is most careful about. A
    check that can never go green is a check that stops being read.
    """

    OK, DIFF, ABSENT, DECLARED = 'ok', 'differs', 'absent', 'declared'

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, str, object, object, str]] = []
        self.notes: list[str] = []

    def _add(self, state, label, paper, data, unit):
        self.rows.append((state, label, paper, data, unit))
        return state == self.OK

    def check(self, label, paper, data, tol, unit=''):
        if data is None or not np.isfinite(data):
            return self._add(self.ABSENT, label, paper, data, unit)
        state = (self.OK if abs(float(data) - float(paper)) <= tol
                 else self.DIFF)
        return self._add(state, label, paper, data, unit)

    def check_printed(self, label, printed, data, unit=''):
        if data is None or not np.isfinite(data):
            return self._add(self.ABSENT, label, printed, data, unit)
        state = self.OK if matches_printed(printed, data) else self.DIFF
        return self._add(state, label, printed, data, unit)

    def fail(self, label, why):
        """An input that is not there. Not a disagreement."""
        return self._add(self.ABSENT, label, '-', why, '')

    def matches(self, label, paper, data, unit=''):
        """A comparison the caller evaluated itself, and it reproduced."""
        return self._add(self.OK, label, paper, data, unit)

    def differs(self, label, paper, data, unit=''):
        """A comparison the caller evaluated itself, and it did not."""
        return self._add(self.DIFF, label, paper, data, unit)

    def declared(self, label, why):
        """Not determined here, and the manuscript says so. Not a failure and
        not outstanding work: a reported property of the measurement."""
        return self._add(self.DECLARED, label, 'indeterminate', why, '')

    def note(self, text):
        self.notes.append(text)

    @property
    def failures(self):
        """Anything that is neither a match nor declared indeterminate --
        what the exit code is built from."""
        return [r for r in self.rows
                if r[0] not in (self.OK, self.DECLARED)]

    @property
    def disagreements(self):
        return [r for r in self.rows if r[0] == self.DIFF]

    @property
    def unchecked(self):
        return [r for r in self.rows if r[0] == self.ABSENT]

    @property
    def declared_rows(self):
        return [r for r in self.rows if r[0] == self.DECLARED]

    def render(self, verbose=True):
        word = {self.OK: 'MATCHES', self.DIFF: 'DIFFERS',
                self.ABSENT: 'no data', self.DECLARED: 'declared'}
        if verbose:
            for state, label, paper, data, unit in self.rows:
                if isinstance(data, (int, float, np.floating)):
                    shown = ('-' if data is None or not np.isfinite(data)
                             else f'{float(data):.6g}')
                else:
                    shown = str(data)
                pv = paper if isinstance(paper, str) else f'{paper:.6g}'
                print(f'  {word[state]:<8} {label:<52} paper {pv:>10} '
                      f'{unit:<11} run {shown}')
            for n in self.notes:
                print(f'  note     {n}')
        return len(self.rows) - len(self.failures), len(self.failures)


def matches_printed(printed: str, value) -> bool:
    """Does `value`, rounded to the precision it is PRINTED with, give exactly
    the printed string? No tolerance to choose and none to get wrong."""
    if value is None or not np.isfinite(value):
        return False
    dec = len(printed.split('.')[1]) if '.' in printed else 0
    return f'{float(value):.{dec}f}' == printed


def within_relative(printed: str, value, rel: float) -> bool:
    """Passes if it rounds to the printed value -- the normal criterion -- OR
    if it sits within a MEASURED relative tolerance of it. The second branch
    exists only for the columns that were shown not to be reproducible to
    their last printed digit between runs of the same code; the tolerance is
    the observed spread, and it is stated where it is defined."""
    if matches_printed(printed, value):
        return True
    if value is None or not np.isfinite(value):
        return False
    p = float(printed)
    return abs(float(value) - p) <= rel * abs(p)


def load_curves(directory=CURVES_DIR):
    """Every *_Dissociation_Curve.csv, keyed by member name."""
    out = {}
    for p in sorted(glob.glob(os.path.join(directory,
                                           '*_Dissociation_Curve.csv'))):
        key = re.sub(r'_Dissociation_Curve\.csv$', '', os.path.basename(p))
        key = key.replace('_q0_', '_q+0_').replace('_q1_', '_q+1_')
        out[key] = (pd.read_csv(p).sort_values('R_Angstrom')
                    .reset_index(drop=True))
    return out


def admitted(df):
    return df[df['inv_status'].isin(ADMITTED)]


def column(df, func, mode, kind='bar'):
    c = (f'DeltaI_bar_{func}_{mode}_Eh' if kind == 'bar'
         else f'DeltaI_resp_{func}_{mode}')
    return df[c].to_numpy(float) if c in df.columns else None


def peak(df, func, mode, kind='bar', only_admitted=True):
    d = admitted(df) if only_admitted else df
    y = column(d, func, mode, kind)
    if y is None:
        return np.nan, np.nan
    R = d['R_Angstrom'].to_numpy(float)
    g = np.isfinite(y)
    if not g.any():
        return np.nan, np.nan
    i = int(np.nanargmax(y[g]))
    return float(y[g][i]), float(R[g][i])


def integral(df, func, mode, only_admitted=True):
    d = admitted(df) if only_admitted else df
    y = column(d, func, mode)
    if y is None:
        return np.nan
    R = d['R_Angstrom'].to_numpy(float)
    g = np.isfinite(y)
    return float(TRAPZ(y[g], R[g])) if g.sum() > 1 else np.nan
