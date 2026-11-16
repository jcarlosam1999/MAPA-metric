"""
runner.py -- run every check and answer one question.
=====================================================

Does this run reproduce every number the manuscript prints?

The checks are independent, so one raising is not allowed to hide the others:
each is guarded, and a check that crashes counts as failed rather than as
absent. A missing input is likewise a failure of that check, not a silent
skip -- with one stated exception, the optional n2v cross-check, which reports
itself as unchecked.

Returns the number of failures, which `main.py` uses as its exit code.
"""

from __future__ import annotations

from . import (check_curves, check_density_comparison, check_deposit,
               check_packaging, check_si_controls, check_table1,
               check_tier1_five)

CHECKS = [
    ('paper numbers vs curves', check_curves.run),
    ('Table 1', check_table1.run),
    ('five Tier-1 functionals', check_tier1_five.run),
    ('SI controls', check_si_controls.run),
    ('MAPA vs SCF density (S16)', check_density_comparison.run),
    ('deposit hygiene', check_deposit.run),
    ('packaging metadata', check_packaging.run),
]


def run_all(verbose=True):
    total_ok = total_bad = 0
    summary = []
    for title, fn in CHECKS:
        print(f'\n--- {title} ' + '-' * max(0, 60 - len(title)))
        try:
            rep = fn(verbose=verbose)
        except Exception as e:                                # noqa: BLE001
            print(f'  FAIL  the check itself raised: {type(e).__name__}: {e}')
            summary.append((title, 0, 1, 0, 0))
            total_bad += 1
            continue
        ok, bad = rep.render(verbose=verbose)
        summary.append((title, ok, len(rep.disagreements),
                        len(rep.unchecked), len(rep.declared_rows)))
        total_ok += ok
        total_bad += bad

    print('\n' + '=' * 72)
    print('VERIFICATION SUMMARY')
    print('=' * 72)
    n_diff = sum(d for _, _, d, _, _ in summary)
    n_absent = sum(a for _, _, _, a, _ in summary)
    n_decl = sum(x for _, _, _, _, x in summary)
    for title, ok, diff, absent, decl in summary:
        mark = ('ok      ' if diff == 0 and absent == 0
                else 'DIFFERS ' if diff else 'INCOMPLETE')
        extra = f'  {decl:>3} declared' if decl else ''
        print(f'  {mark:<11}{title:<40} {ok:>4} match  {diff:>3} differ  '
              f'{absent:>3} not checked{extra}')
    print(f'\n  {total_ok} match, {n_diff} differ, {n_absent} not checked'
          + (f', {n_decl} declared indeterminate' if n_decl else ''))
    if n_diff:
        print('\n  DIFFER means the run produced a number and it is not the '
              'one the\n  manuscript prints. A table value that does not '
              'reproduce from the\n  deposit is the one class of error that '
              'is not fixed in a response\n  to referees. Fix it here.')
    if n_absent:
        print('\n  NOT CHECKED means nothing was compared -- a stage has not '
              'run yet,\n  or a system is not on disk. That is work '
              'outstanding, not a result.')
    if n_decl:
        print('\n  DECLARED INDETERMINATE means the manuscript reports that '
              'quantity as\n  not determined at that point and marks it -- '
              'the dagger rows of Table 1,\n  where the degeneracy cutoff '
              'moves the value by more than the\n  regularization does. '
              'There is nothing to reproduce, so it is neither a\n  failure '
              'nor outstanding work, and it does not enter the exit code.')
    return total_bad
