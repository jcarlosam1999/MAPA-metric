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
def save_line_csv(name, r, z_bohr, columns):
    """columns: dict of {column_name: array}; Z columns added here so the
    Bohr/Angstrom labeling stays correct in one place."""
    df = {"Z_Bohr": z_bohr, "Z_Angstrom": z_bohr * BOHR, **columns}
    path = f"results_vxc_1D/{name}_Vxc_1D_R{r:.2f}.csv"
    pd.DataFrame(df).to_csv(path, index=False)
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
