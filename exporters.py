"""
exporters.py — everything the pipeline writes to disk.
======================================================
Three artifact families per (member, geometry):
  * the grid npz (grid fields + algebraic tensors + JSON meta) — the
    per-geometry real-space record the response metric is built from;
  * the dense 1D line CSV (human-inspectable profiles);
  * the row .json + warm-start .json that make runs FAULT-TOLERANT and
    RESUMABLE bit-for-bit (a resumed scan reloads finished rows AND the
    warm-start chain, so it is equivalent to an uninterrupted run).
"""

import os
import json
import numpy as np
import pandas as pd

from config import CFG, BOHR, run_metadata


# ---------------------------------------------------------------------------
# resume machinery
# ---------------------------------------------------------------------------
def row_path(name, r):
    return f"results_grid_data/{name}_R{r:.2f}.row.json"


def npz_path(name, r):
    """Path of the reusable exact-side npz. Kept here so the resume check and
    the save use the same convention."""
    return f"results_grid_data/{name}_R{r:.2f}.npz"


def save_row(name, r, row):
    with open(row_path(name, r), "w") as f:
        json.dump(row, f, indent=1, default=float)


def load_row(name, r):
    with open(row_path(name, r)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1D line profiles
# ---------------------------------------------------------------------------
def audit_line_columns(name, r, df, strict=True, warn=True):
    """Deposit hygiene, enforced AT THE WRITE, not by a script run afterwards.

    Task 5.3. Bug D1 shipped 203 files in which Vxc_<f>_exact and
    Vxc_<f>_scf were the same array bit for bit, and it survived every review
    because nothing ever compared two columns of the same file. One
    `np.array_equal` would have caught it on day one, which is why that check
    now lives here, in the only function that writes these files, rather than
    in a separate auditor somebody has to remember to run.

    Checks, in order of how badly each one has burned this project:
      1. no two columns identical bit for bit (D1);
      2. no non-finite values anywhere;
      3. Rho_Exact >= 0;
      4. Z_Bohr strictly increasing (a merge on a scrambled axis is silent);
      5. the analytic identities that hold for this system:
           V_FA  == -V_Hartree / N with N a positive integer,
           Vc_Exact == Vxc_Exact + V_Hartree/2  (2e closed shell; a
             DEFINITION, checked so a redefinition cannot pass unnoticed),
           Vxc_Exact == -V_Hartree              (one-electron systems).

    `strict` raises; strict=False downgrades to printed warnings, for
    re-analysing an old tree that is known to be broken.
    """
    problems = []
    cols = {c: df[c].to_numpy() for c in df.columns}

    # ---- electron count, needed before the duplicate test ---------------
    n_elec = None
    if 'V_Hartree' in cols and 'V_FA' in cols:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = -cols['V_Hartree'] / np.where(cols['V_FA'] == 0, np.nan,
                                                  cols['V_FA'])
        n = float(np.nanmedian(ratio))
        if not np.isfinite(n) or abs(n - round(n)) > 1e-6 or round(n) < 1:
            problems.append(f"V_FA is not -V_Hartree/N for integer N "
                            f"(N = {n:.6f})")
        else:
            n_elec = int(round(n))

    # ---- no two columns identical bit for bit (D1) ----------------------
    # DECLARED EXEMPTIONS ONLY. A duplicate pair is a bug unless an identity
    # forces it, and every such identity is listed here with its reason, so
    # that adding a new exemption is a visible edit rather than a silent
    # loosening. For ONE electron the Fermi-Amaldi guide is -v_H/1 = -v_H and
    # the exact v_xc is also -v_H, so those two columns coincide analytically;
    # in real data they differ by the inversion residual (~1e-5 Eh), and the
    # dedicated one-electron check below is what actually polices them.
    # ---- columns that are provably equal, declared as GROUPS ---------------
    # The duplicate check below is the D1 net: no two columns of a deposited
    # profile may be identical bit for bit. Some pairs, however, are equal by
    # an identity this very function also CHECKS, and a rule that demands an
    # identity while another forbids its consequence is a contradiction, not a
    # test.
    #
    # This bit an H2+ run on 2026-08-09, at exactly the two geometries where
    # the inversion reproduced vxc = -v_H to machine zero: the audit reported
    # `V_Guide == Vxc_Exact` and refused to write the profile. The physics was
    # not merely fine, it was as good as it can get; the two neighbouring
    # geometries passed only because their residual was 1e-8 instead of 0.
    #
    # The earlier version listed exempt PAIRS, and listing pairs is what made
    # it wrong: with one electron and the guide off, V_FA, V_Guide and
    # Vxc_Exact are all the same array, which is three pairs, and only one was
    # written down. Identities are declared as groups and every pair inside a
    # group is exempted, so an equality can no longer be half-declared.
    identities = []
    if n_elec == 1:
        # v_xc = -v_H exactly (checked below), and V_FA = -v_H/N with N = 1.
        identities.append({'V_FA', 'Vxc_Exact'})
    # V_Guide == V_FA is not a duplicate: it is the guide BEING Fermi-Amaldi.
    # With CFG.guide_lda_core = False -- the configuration this benchmark was
    # run under, and the one main.py ships -- the guide is -v_H/N everywhere by
    # definition, so the two columns coincide at every geometry of every
    # system. Without this exemption the audit fires on EVERY geometry and a
    # fresh `python main.py` writes no npz at all: found by running the
    # pipeline from scratch, which is the only way it could have been found.
    #
    # V_Guide is still written on every geometry ON PURPOSE (finding D4: the
    # N2 scan used to change column schema half way along its dissociation
    # coordinate). Dropping it when it happens to equal V_FA would bring D4
    # back, so the column stays and the identity is declared here instead.
    #
    # The exemption is CONDITIONAL: with the interpolated LDA-core guide on,
    # the two columns must differ, and the check below enforces exactly that,
    # so turning the guide on cannot silently keep the exemption alive.
    if 'V_Guide' in cols and 'V_FA' in cols:
        if not getattr(CFG, 'guide_lda_core', False):
            identities.append({'V_FA', 'V_Guide'})
        elif np.array_equal(cols['V_Guide'], cols['V_FA']) and n_elec not in (1, 2):
            # guide_lda_core is ON but the guide did not move: for N > 2 the
            # interpolated guide always shifts the core, so identical columns
            # mean the flag did not reach the inversion (this is exactly the
            # shape of bug D4).
            problems.append("guide_lda_core is on but V_Guide == V_FA bit for "
                            "bit: the LDA-core guide did not reach the "
                            "inversion")
    # Merge groups that share a column: with one electron and the guide off,
    # {V_FA, Vxc_Exact} and {V_FA, V_Guide} are one group of three, and the
    # pair {V_Guide, Vxc_Exact} follows from the other two. Deriving it beats
    # remembering it.
    merged = []
    for g in identities:
        hit = [m for m in merged if m & g]
        for m in hit:
            merged.remove(m)
            g = g | m
        merged.append(g)
    exempt = {frozenset({a, b}) for g in merged for a in g for b in g if a != b}

    names = list(cols)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset({a, b}) in exempt:
                continue
            if cols[a].shape == cols[b].shape and np.array_equal(cols[a],
                                                                 cols[b]):
                problems.append(f"columns identical bit for bit: {a} == {b}")

    for c, v in cols.items():
        if np.issubdtype(v.dtype, np.number) and not np.all(np.isfinite(v)):
            problems.append(f"{c}: {int((~np.isfinite(v)).sum())} non-finite "
                            f"values")
    if 'Rho_Exact' in cols and np.nanmin(cols['Rho_Exact']) < 0:
        problems.append(f"Rho_Exact negative (min "
                        f"{np.nanmin(cols['Rho_Exact']):.3e})")
    z = cols.get('Z_Bohr')
    if z is not None and not np.all(np.diff(z) > 0):
        problems.append("Z_Bohr is not strictly increasing")

    if n_elec == 1 and 'Vxc_Exact' in cols:
        e = float(np.abs(cols['Vxc_Exact'] + cols['V_Hartree']).max())
        if e > 3e-5:
            problems.append(f"one-electron identity vxc = -v_H violated, "
                            f"max {e:.3e} Eh")
    if {'Vc_Exact', 'Vxc_Exact', 'V_Hartree'} <= set(cols):
        e = float(np.abs((cols['Vxc_Exact'] - cols['Vc_Exact'])
                         + 0.5 * cols['V_Hartree']).max())
        if e > 1e-10:
            problems.append(f"Vc_Exact != Vxc_Exact + v_H/2 (its definition), "
                            f"max {e:.3e}")

    if problems:
        msg = (f"deposit audit failed for {name} R={r:.2f}:\n  "
               + "\n  ".join(problems))
        if strict:
            raise AssertionError(msg)
        if warn:
            print("  [DEPOSIT WARN] " + msg)
    return problems


def save_line_csv(name, r, z_bohr, columns, audit=True):
    """columns: dict of {column_name: array}; Z columns added here so the
    Bohr/Angstrom labeling stays correct in one place.

    The file is audited BEFORE it is written (task 5.3): a bad profile never
    reaches disk, so it can never reach Zenodo either."""
    df = pd.DataFrame({"Z_Bohr": z_bohr, "Z_Angstrom": z_bohr * BOHR,
                       **columns})
    if audit:
        audit_line_columns(name, r, df, strict=CFG.deposit_audit_strict)
    path = f"results_vxc_1D/{name}_Vxc_1D_R{r:.2f}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# grid npz
# ---------------------------------------------------------------------------
def export_grid_npz(member, r, mol, grids, fields, tensors, meta_extra):
    """fields: grid arrays; tensors: AO-basis matrices; meta_extra: labels.
    Empty arrays mark flag-gated fields (the HDF5 packer skips them)."""
    meta = {**run_metadata(), "name": member['name'], "R_ang": float(r),
            "a1": member['a1'], "a2": member['a2'],
            "charge": member['charge'], "spin": member['spin'],
            "ncas": member['ncas'], "nelecas": list(member['nelecas']),
            "anion_caveat": member['anion_caveat'], **meta_extra}
    np.savez_compressed(
        npz_path(member['name'], r),
        coords_bohr=grids.coords, weights=grids.weights,
        **fields, **tensors, meta=json.dumps(meta, default=float))
