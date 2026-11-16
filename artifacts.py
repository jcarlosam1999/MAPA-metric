"""
artifacts.py -- one place that knows what every stage leaves on disk.
=====================================================================

`python main.py` is meant to produce the whole paper from an empty directory,
and to be safe to run again: a second run must not redo what the first one
already finished. That property needs a single answer to one question -- *what
file proves this stage is done?* -- and it needs it in ONE place. When each
stage answers it for itself, the answers drift, and a stage that reports
"skipping, already done" while looking at the wrong file is worse than no skip
logic at all: it turns a missing result into a silent one.

So every stage registers here, and only here:

    STAGES[tag] = (relative path of its artifact, one-line description)

`have(tag)` decides whether to skip. `save_rows(tag, rows)` writes the artifact
as CSV. `load(tag)` reads it back. The verification layer reads the same table,
so a check can never look somewhere the producer does not write.

The artifacts are also what makes the results checkable by someone who did not
run the code: they are small, plain CSV, and every number the paper prints can
be traced to one of them.
"""

from __future__ import annotations

import os

DIR = 'results_validation'

# tag -> (path relative to the run directory, what it is)
#
# Anything a stage produces that a later stage or a check reads goes here. A
# stage whose only output is prose in the run log cannot be skipped safely and
# cannot be checked, so it does not belong in this table -- give it a CSV
# first.
STAGES: dict[str, tuple[str, str]] = {
    # ---- benchmark scan (per system; handled by CFG.resume, not by tag) ----
    'functional_postpass': ('results_energies/Functional_Comparison.csv',
                            'DeltaI columns merged into every curve'),

    # ---- Supporting Information validations --------------------------------
    'grid_resolution':      (f'{DIR}/grid_resolution.csv',
                             'metric vs Becke grid level'),
    'functional_derivative': (f'{DIR}/functional_derivative.csv',
                              'functional-derivative identity, spin route'),
    'kappa_invariance':     (f'{DIR}/kappa_invariance_HH_q+0_s0.csv',
                             'Table 1, light probe (H2)'),
    'kappa_invariance_heavy': (f'{DIR}/kappa_invariance_NN_q+0_s0.csv',
                               'Table 1, heavy probe (N2)'),
    'kappa_wide_range':     (f'{DIR}/kappa_wide_HH_q+0_s0.csv',
                             'kappa outside the usable window'),
    'kappa_invariance_extra_HLi_q+0_s0':
        (f'{DIR}/kappa_invariance_HLi_q+0_s0.csv', 'Table 1, LiH rows'),
    'kappa_invariance_extra_HF_q+0_s0':
        (f'{DIR}/kappa_invariance_HF_q+0_s0.csv', 'Table 1, HF rows'),
    'ensemble_seam':        (f'{DIR}/ensemble_seam.csv',
                             'unified vs switched ensemble'),
    'fine_grid_scan':       (f'{DIR}/fine_grid_scan.csv',
                             'peak position on a refined R scan'),

    # ---- post-pass extras behind SI sections --------------------------------
    'tier1_five':           (f'{DIR}/tier1_five_functionals.csv',
                             'five Tier-1 functionals + fixed-volume windows'),
    'fixed_volume_response': (f'{DIR}/fixed_volume_response.csv',
                              'fixed-volume control on the response metric'),
    'basis_convergence':    (f'{DIR}/basis_convergence.csv',
                             'orbital-basis dependence of both metrics'),
    'n2v_cross_check':      (f'{DIR}/n2v_cross_check.csv',
                             'independent inversion code, same densities'),
    'reduced_coordinate':   (f'{DIR}/reduced_coordinate.csv',
                             'both metrics at 1.5 R_e, normalized to R_e'),
    # SI S16: the response metric against the self-consistent density
    # difference. One stage writes all three; `density_comparison` is written
    # last, so it is the one whose presence means the stage is complete.
    'density_comparison_exact': (f'{DIR}/mapa_vs_density_exact_H2plus.csv',
                                 'H2+ with v_xc = -v_H, no inversion'),
    'density_comparison_model': (f'{DIR}/mapa_vs_density_model.csv',
                                 'LDA density as reference, exact v known'),
    'density_comparison':   (f'{DIR}/mapa_vs_density_production.csv',
                             'production metric vs SCF density difference'),
}


# Columns without which an artifact is not the artifact this code writes.
#
# `have()` used to ask only "is the file there, and non-empty?". That is not
# enough once the producer changes: a CSV written by an older version sits on
# disk, counts as done, gets skipped, and the check downstream then fails --
# or worse, passes -- on stale data. It happened on 2026-08-10 with
# `tier1_five`, whose confidence-gate column had just been added: the run
# skipped the stage, the check refused the file, and the only way out was to
# know to delete it by hand. The schema is part of "done".
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    'tier1_five': ('inv_status', 'DeltaI_bar_PBE-PBE_exact_Eh'),
    'fixed_volume_response': ('DeltaI_resp_production',),
    'n2v_cross_check': ('max_abs_diff_Eh', 'n2v_density_error_e'),
    'basis_convergence': ('dIbar_change_pct', 'resp_change_pct'),
    'grid_resolution': ('shift_spline_A',),
    'functional_derivative': ('rel_L1',),
    'ensemble_seam': ('switched_inv_mode',),
    'reduced_coordinate': ('span_ok', 'ratio_DeltaI_resp_PBE-PBE_exact'),
    'density_comparison': ('status', 'indeterminate_PBE-PBE', 'ratio_PBE-PBE'),
    'density_comparison_model': ('status', 'model_valid', 'ratio_model_scf',
                                 'ratio_inv_model'),
    'density_comparison_exact': ('status', 'mapa_an_PBE-PBE', 'scf_PBE-PBE'),
}


def path(tag: str) -> str:
    """Where `tag` writes. Unknown tags are a programming error, not a warning:
    a typo here would silently disable a stage's skip logic."""
    if tag not in STAGES:
        raise KeyError(f"unregistered stage {tag!r}; add it to STAGES so its "
                       f"artifact has exactly one definition")
    return STAGES[tag][0]


def describe(tag: str) -> str:
    return STAGES[tag][1]


def have(tag: str) -> bool:
    """Is this stage already done?

    Three ways it is not, all of which used to read as done:
      * the file is absent;
      * it is empty -- a crash between `open` and the first write leaves a
        zero-byte file, and treating that as done turns a missing result into
        a silent one;
      * it is there but was written by an older version of the producer and
        lacks a column this one writes. Skipping on a stale schema is the
        worst of the three, because the file looks like evidence.
    """
    p = path(tag)
    if not (os.path.isfile(p) and os.path.getsize(p) > 0):
        return False
    need = REQUIRED_COLUMNS.get(tag)
    if not need:
        return True
    try:
        import pandas as pd
        cols = set(pd.read_csv(p, nrows=0).columns)
    except Exception:                                         # noqa: BLE001
        return False                  # unreadable: redo rather than guess
    missing = [c for c in need if c not in cols]
    if missing:
        print(f"  [artifact] {p} is from an older version (missing "
              f"{', '.join(missing)}); it will be regenerated")
        return False
    return True


def save_rows(tag: str, rows) -> str:
    """Write a list of dicts as the stage's artifact. Written to a temporary
    name and renamed, so an interrupted write never leaves a half table that
    `have()` would accept."""
    import pandas as pd
    p = path(tag)
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    tmp = p + '.part'
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    os.replace(tmp, p)
    return p


def load(tag: str):
    import pandas as pd
    return pd.read_csv(path(tag))


def status() -> list[tuple[str, bool, str]]:
    """(tag, done, description) for every registered stage, for the run header
    and for the end-of-run summary."""
    return [(t, have(t), STAGES[t][1]) for t in STAGES]


def report(tag: str, read: bool, log=print) -> None:
    """Print a stage's artifact, the same way whether it was just computed or
    replayed from disk.

    A skipped stage used to leave one line -- "already on disk, skipping" --
    and a run that skipped everything read as if nothing had been checked. The
    scan solved this for geometries long ago: `pipeline` replays a saved row
    through the same reporter that a fresh one goes through, "so the log
    always reads like a full run". Stages had never been given the same
    treatment, and they are where most of the numbers are.

    So both paths call this. The rows are the rows either way; the only
    difference in the output is the word in the header, which says where they
    came from. A reader should be able to tell what a stage produced without
    having to know whether this particular launch recomputed it, and should
    never have to open the CSV to find out what was skipped.
    """
    src = 'read' if read else 'computed'
    p = path(tag)
    try:
        import pandas as pd
        d = pd.read_csv(p)
    except Exception as e:                                    # noqa: BLE001
        log(f"  [{src}] {tag}: {p} unreadable ({type(e).__name__})")
        return
    log(f"  [{src}] {tag} -- {describe(tag)}")
    log(f"           {p}, {len(d)} row(s), {len(d.columns)} column(s)")
    for line in _tabulate(d):
        log("           " + line)


def _tabulate(d, max_rows: int = 40, max_width: int = 108) -> list[str]:
    """A fixed-width rendering of a small table, wide columns dropped first.

    Truncation is stated, never silent: a table that quietly shows its first
    forty rows is a table that lies about its size.
    """
    import pandas as pd
    cols, out = list(d.columns), []
    shown = d.head(max_rows)

    def fmt(v):
        if isinstance(v, float):
            if v != v:
                return '-'
            a = abs(v)
            if a and (a < 1e-3 or a >= 1e5):
                return f"{v:.3e}"
            return f"{v:.6g}"
        return str(v)

    keep = []
    width = 0
    for c in cols:
        w = max(len(str(c)), *(len(fmt(v)) for v in shown[c])) if len(shown) \
            else len(str(c))
        if width + w + 2 > max_width and keep:
            break
        keep.append((c, w))
        width += w + 2
    out.append("  ".join(f"{c:>{w}}" for c, w in keep))
    for _, r in shown.iterrows():
        out.append("  ".join(f"{fmt(r[c]):>{w}}" for c, w in keep))
    notes = []
    if len(d) > len(shown):
        notes.append(f"{len(d) - len(shown)} more row(s) in the file")
    if len(keep) < len(cols):
        notes.append(f"{len(cols) - len(keep)} column(s) not shown: "
                     + ", ".join(cols[len(keep):]))
    for n in notes:
        out.append(f"({n})")
    return out


def save_any(tag: str, obj) -> str | None:
    """Serialize whatever a stage returned.

    The validations were written at different times and return different
    shapes -- a list of row dicts, a dict of scalars, a dict of equal-length
    lists. Rather than rewrite seven working physics routines to agree on a
    container, the shape is normalized here, once. A stage that returns
    nothing serializable gets no artifact and therefore no skip, which is the
    safe direction: it will simply run again.
    """
    import pandas as pd
    if obj is None:
        return None
    if isinstance(obj, list):
        if not obj or not isinstance(obj[0], dict):
            return None
        return save_rows(tag, obj)
    if isinstance(obj, dict):
        listish = {k: v for k, v in obj.items()
                   if isinstance(v, (list, tuple))}
        if listish and len({len(v) for v in listish.values()}) == 1:
            n = len(next(iter(listish.values())))
            flat = {k: v for k, v in obj.items()
                    if not isinstance(v, (list, tuple, dict))}
            frame = {**{k: [v] * n for k, v in flat.items()}, **listish}
            return save_rows(tag, pd.DataFrame(frame).to_dict('records'))
        scalars = {k: v for k, v in obj.items()
                   if not isinstance(v, (list, tuple, dict))}
        return save_rows(tag, [scalars]) if scalars else None
    return None
