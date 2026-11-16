"""
check_tier1_five.py -- the five-functional table of the SI.
===========================================================

Peak of DeltaI_bond over the admitted stratum for the five Tier-1
multiplicative functionals, plus the spread across them.

This table went unchecked for a long time and it showed. When the six most
stretched N2 geometries were recomputed under a single prior, LDA and PBE were
refreshed from the dissociation curves while BLYP, PBEsol and revPBE would
have kept the old convention -- two conventions inside one row of one table,
which is the same defect the recomputation existed to remove, one level down.
A number no script checks is a number that can age badly without anyone
noticing; this one already did.
"""

from __future__ import annotations

from . import paper_values as P
from ._common import LABEL, Report

import artifacts as art


def run(verbose=True):
    rep = Report('five Tier-1 functionals')
    if not art.have('tier1_five'):
        rep.fail('tier1_five', 'artifact missing; run the stage first')
        return rep
    d = art.load('tier1_five')
    # No gate, no check. Every peak here is a maximum over the ADMITTED
    # stratum; taking it over every geometry is a different quantity, not a
    # looser one -- HF alone moves from 0.086 to 0.489 Eh. An earlier version
    # filtered `if 'inv_status' in d.columns`, the column was not being
    # written, and the check quietly compared the wrong maxima against the
    # right numbers. A filter that can silently not apply is worse than no
    # filter: it reports confidently about something else.
    if 'inv_status' not in d.columns:
        rep.fail('tier1_five', 'the artifact carries no inv_status column, so '
                 'the confidence gate cannot be applied; regenerate it')
        return rep
    d = d[d['inv_status'].isin(('OK', 'MARGINAL'))]
    if d.empty:
        rep.fail('tier1_five', 'no admitted geometry in the artifact')
        return rep

    for member, printed in P.TIER1_FIVE.items():
        sub = d[d['system'] == member]
        if sub.empty:
            rep.fail(f'{LABEL[member]}', 'no rows for this system')
            continue
        # Coverage, not merely presence. A maximum over part of the admitted
        # stratum is not a rougher version of the right answer; it is a
        # different quantity wearing its name. On 2026-08-13 this artifact was
        # missing the five most stretched N2 geometries, one of which
        # (R = 4.112 A) IS the maximum, and every N2 cell would have been
        # compared against a peak taken over the other fifteen.
        want = P.STRATIFICATION.get(member, (None, None))[0]
        if want is not None and len(sub) != want:
            rep.fail(f'{LABEL[member]}',
                     f'{len(sub)} admitted rows present, {want} expected; the '
                     f'peak would be a maximum over a subset')
            continue
        peaks = []
        for func, value in zip(P.TIER1_ORDER, printed[:5]):
            col = f'DeltaI_bar_{func}_exact_Eh'
            if col not in sub.columns:
                rep.fail(f'{LABEL[member]:<5} {func}', f'no column {col}')
                continue
            m = float(sub[col].max())
            peaks.append(m)
            rep.check_printed(f'{LABEL[member]:<5} {func:<8}', value, m, 'Eh')
        if len(peaks) == len(P.TIER1_ORDER):
            rep.check_printed(f'{LABEL[member]:<5} spread', printed[5],
                              100 * (max(peaks) / min(peaks) - 1), '%')
    return rep
