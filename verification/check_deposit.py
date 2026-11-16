"""
check_deposit.py -- hygiene of the 1D profiles before they are published.
=========================================================================

Delegates the per-file rules to `exporters.audit_line_columns`, which also
runs inside `save_line_csv`, i.e. at the moment of writing. One
implementation: if this check and the exporter could disagree, a profile could
leave one gate clean and the other dirty, which is the class of failure being
closed. What stays here is only what needs to see the whole set at once --
column schemas across files, point counts -- plus the explicit two-density-mode
test, because the functional columns are added later by the post-pass and this
is the only place the two modes are visible side by side.

It also refuses to pass while any `PENDING_*` file sits in the run directory.
A deposit can have flawless profiles and still mix conventions, because some
columns live in the dissociation curves rather than in these files. A note in
a README does not stop that: nobody runs a README. Declaring an open task
creates a file, and this check fails until the file is deliberately removed.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from ._common import PROFILES_DIR, Report

FUNCTIONALS = ('LDA-VWN', 'PBE-PBE', 'BLYP', 'PBEsol', 'revPBE')


def run(verbose=True):
    import exporters as ex

    rep = Report('deposit hygiene')
    files = sorted(glob.glob(os.path.join(PROFILES_DIR, '*_Vxc_1D_R*.csv')))
    if not files:
        rep.fail('profiles', f'no profiles in {PROFILES_DIR}')
        return rep

    schemas, sizes = {}, set()
    for path in files:
        f = os.path.basename(path)
        member = re.sub(r'_Vxc_1D_R.*', '', f)
        R = float(re.search(r'_R([\d.]+)\.csv', f).group(1))
        df = pd.read_csv(path)
        cols = {c: df[c].to_numpy() for c in df.columns}
        sizes.add(len(df))
        schemas.setdefault(member, set()).add(tuple(df.columns))

        for problem in ex.audit_line_columns(member, R, df, strict=False,
                                             warn=False):
            rep.fail(f, problem)

        if 'Vxc_Alpha' in cols and 'Vxc_Beta' in cols:
            e = np.abs(cols['Vxc_Exact']
                       - 0.5 * (cols['Vxc_Alpha'] + cols['Vxc_Beta'])).max()
            if e > 1e-10:
                rep.fail(f, f'Vxc_Exact != (alpha+beta)/2, max {e:.2e}')

        # The two density modes MUST differ. When they did not, the labels
        # branched and the physics did not.
        for fn in FUNCTIONALS:
            a, b = f'Vxc_{fn}_exact', f'Vxc_{fn}_scf'
            if a in cols and b in cols:
                dmax = np.abs(cols[a] - cols[b]).max()
                if dmax < 1e-12:
                    rep.fail(f, f'{fn}: exact and scf identical '
                                f'(max {dmax:.1e})')

    for member, s in sorted(schemas.items()):
        if len(s) > 1:
            rep.fail(member, f'{len(s)} different column schemas inside one '
                             f'system')
    if len({next(iter(s)) for s in schemas.values()}) > 1:
        rep.note('column schemas differ between systems; document them in the '
                 'deposit README')
    if len(sizes) > 1:
        rep.note(f'profiles do not all have the same number of points: '
                 f'{sorted(sizes)}')

    pending = sorted(glob.glob('PENDING_*'))
    for p in pending:
        rep.fail(p, 'declared work that has not been done; see the file')

    # A check that reports nothing reads the same whether it passed or never
    # ran. It says how many files it looked at.
    rep.check(f'{len(files)} profiles audited across {len(schemas)} systems',
              0, len(rep.failures), 0, 'problems')
    return rep
