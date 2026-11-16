"""
tools_gate_stability.py -- how firm is "115 of 203"?
====================================================

The confidence gate is the one criterion that decides which geometries the
article reports. It is applied to a single quantity, the density-reproduction
error |Drho|, against a threshold that depends on the system:

    heavy (n_elec >= CFG.heavy_nelec)          -> CFG.density_fail_heavy
    diffuse (ghost_leak/|Drho| >= 0.15)        -> CFG.density_fail_diffuse
    otherwise                                  -> CFG.density_fail

A count reported as a property of the data ("115 of 203 are admitted") is in
fact a property of the data AND of those constants. That is fine, and the SI
says so; what is not fine is to leave the reader unable to tell how much of
the count is data and how much is calibration. This tool answers it in two
ways that need no re-inversion:

  * the MARGIN of every admitted geometry -- |Drho| as a percentage of its own
    threshold. Exact: no reconstruction of the gate is involved, only the
    numbers the run already wrote. A stratum in which nothing sits near its
    threshold is firm; one in which several do is a stratum whose size will
    move on the next run, whatever the physics does.

  * the DISTRIBUTION of the leaked fraction ghost_leak/|Drho| around the 0.15
    cutoff. SI Sec. S5 states that reliable members leak 0.01-0.13 and the
    diffuse group 0.15-0.46, "a cutoff at 0.15 separates them". That was
    calibrated on 8 members (`config.py` says as much: "first estimates from
    8 members, not a final value"). Whether the full set still shows a gap
    there is a question with a yes-or-no answer, and it should be asked
    before a referee asks it.

Optionally it also compares two runs geometry by geometry -- which is how the
115-vs-118 discrepancy was diagnosed:

    python tools_gate_stability.py
    python tools_gate_stability.py --compare path/to/other/results_grid_data

Reads the per-geometry `.row.json` written by the scan. Writes
`results_validation/gate_stability.csv`. Nothing is re-inverted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from config import CFG

ADMITTED = ('OK', 'MARGINAL')


def _n_elec_table() -> dict[str, int]:
    """Electron count per system, from the benchmark table itself.

    Not parsed out of the tag: the tag carries the charge but not the atoms'
    Z, and a wrong electron count here would silently move N2 out of the
    heavy stratum and misreport every margin in this file.
    """
    import benchmark as bm
    return {m['name']: int(m['n_elec']) for m in bm.parse_benchmark()}


def load(npz_dir: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(npz_dir, '*_R*.row.json'))):
        try:
            d = json.load(open(p))
        except Exception:                                     # noqa: BLE001
            continue
        d['system'] = os.path.basename(p)[:-9].rsplit('_R', 1)[0]
        rows.append(d)
    return rows


def threshold(row: dict, n_elec: int | None) -> float:
    """The threshold this geometry was judged against.

    Reconstructed from the recorded inputs rather than re-derived, so it can
    be wrong in exactly one documented way: for open-shell ensembles the
    production gate takes `any(channel failed)` on the PER-CHANNEL errors,
    and only the total is written to the row. Those geometries are reported
    as unreconstructible instead of being silently mis-scored.
    """
    if n_elec is not None and n_elec >= CFG.heavy_nelec:
        return CFG.density_fail_heavy
    err = row.get('inv_density_error_e')
    leak = row.get('ghost_leak_e')
    if err and leak and err > 0 and leak / err >= CFG.diffuse_leak_frac:
        return CFG.density_fail_diffuse
    return CFG.density_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='results_grid_data')
    ap.add_argument('--compare', default=None,
                    help='a second results_grid_data to compare against')
    a = ap.parse_args()

    rows = load(a.npz)
    if not rows:
        print(f"no .row.json in {a.npz}")
        return 1
    nel = _n_elec_table()
    unknown = sorted({r['system'] for r in rows} - set(nel))
    if unknown:
        print(f"not in the benchmark table, so no electron count and no "
              f"threshold: {', '.join(unknown)}\n")

    out = []
    for r in rows:
        err = r.get('inv_density_error_e')
        leak = r.get('ghost_leak_e')
        thr = threshold(r, nel.get(r['system']))
        adm = r.get('inv_status') in ADMITTED
        out.append(dict(
            system=r['system'], R_Angstrom=r.get('R_Angstrom'),
            inv_status=r.get('inv_status'), inv_mode=r.get('inv_mode'),
            admitted=adm, density_error_e=err, ghost_leak_e=leak,
            leak_fraction=(leak / err if err and leak and err > 0 else None),
            threshold_e=thr,
            margin_pct=(100.0 * err / thr if err and thr else None),
            gate_reconstructed=((err is not None) and ((err <= thr) == adm))))

    n_adm = sum(o['admitted'] for o in out)
    bad = [o for o in out if not o['gate_reconstructed']]
    print(f"{len(out)} geometries · {n_adm} admitted "
          f"(inv_status in {ADMITTED})\n")
    print(f"gate reconstructed from the recorded inputs for "
          f"{len(out) - len(bad)} of {len(out)}"
          + ("; the rest are open-shell ensembles judged per channel, whose "
             "per-channel errors the row does not carry:" if bad else "."))
    for o in bad:
        print(f"     {o['system']:<12} R = {o['R_Angstrom']:>6.3f}  "
              f"{o['inv_status']:<9} err {o['density_error_e']:.3e} vs "
              f"threshold {o['threshold_e']:.0e}")

    # ---- how close to the edge is the admitted stratum? --------------------
    edge = sorted((o for o in out if o['admitted'] and o['margin_pct']),
                  key=lambda o: -o['margin_pct'])
    near = [o for o in edge if o['margin_pct'] > 90]
    print(f"\nAdmitted geometries within 10 % of their own threshold: "
          f"{len(near)} of {n_adm}")
    for o in near:
        print(f"     {o['system']:<12} R = {o['R_Angstrom']:>6.3f}  "
              f"{o['density_error_e']:.3e} e-  =  "
              f"{o['margin_pct']:6.1f} % of {o['threshold_e']:.0e}"
              f"   [{o['inv_status']}]")

    # ---- is there a gap at the diffuse cutoff? -----------------------------
    cut = CFG.diffuse_leak_frac
    fr = sorted(o['leak_fraction'] for o in out
                if o['leak_fraction'] is not None
                and (nel.get(o['system']) or 0) < CFG.heavy_nelec)
    band = [f for f in fr if abs(f - cut) <= 0.02]
    print(f"\nLeaked fraction ghost_leak/|Drho| across the light systems: "
          f"{len(fr)} values, {len(band)} of them within +-0.02 of the "
          f"{cut} cutoff")
    if band:
        print("     " + "  ".join(f"{f:.4f}" for f in band))
        print("     -> the cutoff falls inside a continuum, not in a gap. "
              "SI Sec. S5 should say so.")
    else:
        print("     -> the cutoff falls in a gap, as SI Sec. S5 states.")

    # ---- optional: two runs, geometry by geometry --------------------------
    if a.compare:
        other = {(r['system'], round(r.get('R_Angstrom', 0), 3)): r
                 for r in load(a.compare)}
        flips = []
        for r in rows:
            k = (r['system'], round(r.get('R_Angstrom', 0), 3))
            o = other.get(k)
            if o is None:
                continue
            if (r.get('inv_status') in ADMITTED) != (
                    o.get('inv_status') in ADMITTED):
                flips.append((r, o))
        n_o = sum(v.get('inv_status') in ADMITTED for v in other.values())
        print(f"\nAgainst {a.compare}: {n_adm} admitted here, {n_o} there; "
              f"{len(flips)} geometries change side")
        for r, o in flips:
            e1, e2 = r.get('inv_density_error_e'), o.get('inv_density_error_e')
            l1 = (r.get('ghost_leak_e') or 0) / (e1 or 1)
            l2 = (o.get('ghost_leak_e') or 0) / (e2 or 1)
            print(f"     {r['system']:<12} R = {r['R_Angstrom']:>6.3f}   "
                  f"{o.get('inv_status'):<9} -> {r.get('inv_status'):<9}"
                  f"   |Drho| {e2:.3e} -> {e1:.3e}"
                  f"   leak fraction {l2:.3f} -> {l1:.3f}")

    p = os.path.join('results_validation', 'gate_stability.csv')
    try:
        import pandas as pd
        os.makedirs('results_validation', exist_ok=True)
        pd.DataFrame(out).sort_values('margin_pct', ascending=False).to_csv(
            p, index=False)
        print(f"\n{len(out)} rows written to {p}")
    except Exception as e:                                    # noqa: BLE001
        print(f"\n(could not write {p}: {type(e).__name__}: {e})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
