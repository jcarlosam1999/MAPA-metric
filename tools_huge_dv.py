"""
tools_huge_dv.py -- what do the [HUGE-DV] points actually cost the metric?
==========================================================================

The post-pass prints, at some geometries:

    [HUGE-DV] PBE,PBE R=4.112 A: 100 grid points with finite |Dv| > 10.0 Eh
              (max 1.016e+01). NOT modified -- diagnose before trusting this
              geometry's DeltaI.

That warning is deliberate and it does not touch the data (decision D2): a
finite but physically impossible potential has to surface loudly and be
diagnosed, not clipped, because a magnitude threshold inside the metric would
be exactly the hidden free parameter the abstract says it does not have. So
the warning counts points and leaves the diagnosis to whoever reads it. This
is the diagnosis.

How the cost is measured, without reimplementing the metric
-----------------------------------------------------------
DeltaI_bond is computed TWICE with the production function
(`metrics.delta_metrics`, the same one the post-pass calls):

  1. as is;
  2. with those points neutralized -- v_func set to v_exact there, which is
     exactly what `_sanitize_grid_potential` does with non-finite points, so
     Dv = 0 and they contribute nothing to the integral.

The difference between the two IS their contribution. No mask and no gauge are
reimplemented here, so this tool cannot disagree with the metric through an
implementation difference -- which is how reassuring diagnostics get written.

It also reports where they fall: distance to the nearest nucleus, and whether
they lie inside Omega. The reasonable suspicion is that they are the innermost
radial shell of the Becke grid around a nucleus (level 3 has ~110 angular
points per shell, and the warning says 100), where the volume element is
almost zero and a volume-weighted metric barely sees them. That is a
suspicion; this prints the number.

Does the answer reach a published number?
-----------------------------------------
A contribution only matters if the geometry it belongs to survives the
confidence gate. Every row therefore carries `inv_status`, read from the same
`.row.json` the metric layer reads, and the closing summary separates admitted
geometries (`OK`/`MARGINAL` -- the stratum every table is built on) from
rejected ones. A 3 % contribution at a geometry that no table contains is a
property of the diagnostic, not of the paper, and reporting the two together
under one maximum would misstate both.

Post-processing only: nothing is re-inverted. One file is written,
`results_validation/huge_dv_diagnostic.csv`, because a console buffer is not a
record -- the run that first produced this output scrolled it away before it
could be read.

    python tools_huge_dv.py                       # every system on disk
    python tools_huge_dv.py --only NN_q+0_s0
    python tools_huge_dv.py --threshold 5         # to exercise it on H2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

import functional_eval as fe
import metrics as mt
import validation_extra as vx
from config import CFG, BOHR

FUNCTIONALS = vx.TIER1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=vx.NPZ_DIR)
    ap.add_argument('--only', default=None)
    ap.add_argument('--mode', default='exact', choices=('exact', 'scf'))
    ap.add_argument('--threshold', type=float, default=None,
                    help='default: production CFG.dv_report_Eh')
    a = ap.parse_args()

    thr = a.threshold if a.threshold is not None else getattr(
        CFG, 'dv_report_Eh', 10.0)
    radius = getattr(CFG, 'cyl_radius_bohr', 2.0)
    pat = f"{a.only}_R*.npz" if a.only else "*_R*.npz"
    paths = sorted(glob.glob(os.path.join(a.npz, pat)))
    if not paths:
        print(f"no npz matching {pat} in {a.npz}")
        return 1
    print(f"|Dv| > {thr} Eh · {a.mode} mode · {len(paths)} geometries · "
          f"Omega = |z| <= R/2, r_perp < {radius} bohr\n")

    rows: list[dict] = []
    worst, any_hit = 0.0, False
    for p in paths:
        base = os.path.basename(p)
        try:
            ctx = fe._load_geometry_context(p)
        except Exception as e:                                # noqa: BLE001
            print(f"  {base}: unreadable ({type(e).__name__})")
            continue
        status = vx._inv_status(p)          # does the confidence gate admit it?
        admitted = status in ('OK', 'MARGINAL')
        coords, w, R = ctx['coords'], ctx['w'], ctx['R']
        vxc = np.asarray(ctx['vxc_exact'])
        half = (R / BOHR) / 2.0
        omega = ((np.abs(coords[:, 2]) <= half)
                 & (coords[:, 0] ** 2 + coords[:, 1] ** 2 < radius ** 2))
        d_nuc = np.min(np.linalg.norm(coords[:, None, :]
                                      - ctx['mol'].atom_coords()[None], axis=2),
                       axis=1)

        for label, xc in FUNCTIONALS:
            finfo = fe.resolve_functional(xc)
            try:
                v, rho_m = vx._functional_potential(ctx, finfo, a.mode)
            except Exception as e:                            # noqa: BLE001
                print(f"  {base} {label}: {type(e).__name__}: {e}")
                continue
            dv = vxc - v
            big = np.isfinite(dv) & (np.abs(dv) > thr)
            if not big.any():
                continue
            any_hit = True

            _, dI, _ = mt.delta_metrics(vxc, v, rho_m, coords, w, R)
            v2 = v.copy()
            v2[big] = vxc[big]                 # Dv = 0, as for non-finite
            _, dI2, _ = mt.delta_metrics(vxc, v2, rho_m, coords, w, R)
            pct = 100.0 * abs(float(dI2) - float(dI)) / max(abs(float(dI)),
                                                            1e-30)
            if admitted:                       # only the published stratum
                worst = max(worst, pct)        # can move a published number
            inside = big & omega
            wfrac = (100.0 * float(np.sum(w[inside])) / float(np.sum(w[omega]))
                     if inside.any() else 0.0)
            print(f"  {base}  {label}  R = {R:.3f} A   "
                  f"inv_status {status or '?'}"
                  f"{'' if admitted else '  (gate-rejected)'}")
            print(f"     points over threshold {int(big.sum()):>5}   "
                  f"max |Dv| {np.abs(dv[big]).max():.3f} Eh   "
                  f"inside Omega {int(inside.sum())}")
            print(f"     distance to the nearest nucleus "
                  f"{d_nuc[big].min():.4f}-{d_nuc[big].max():.4f} bohr"
                  + (f"   volume weight {wfrac:.2e} % of Omega"
                     if inside.any() else ""))
            if wfrac < 0:
                print("     (negative share: these points carry negative "
                      "quadrature weight -- Becke partition, not an error)")
            print(f"     DeltaI_bond with them {float(dI):.6f} · without "
                  f"{float(dI2):.6f}  ->  **{pct:.4f} %**\n")

            rows.append(dict(
                system=base.split('_R')[0], R_Angstrom=float(R),
                functional=label, inv_status=status, admitted=admitted,
                n_points=int(big.sum()), n_inside_omega=int(inside.sum()),
                max_abs_dv_Eh=float(np.abs(dv[big]).max()),
                d_nucleus_min_bohr=float(d_nuc[big].min()),
                d_nucleus_max_bohr=float(d_nuc[big].max()),
                volume_weight_pct=float(wfrac),
                dIbar=float(dI), dIbar_without=float(dI2),
                contribution_pct=float(pct)))

    if not any_hit:
        print("No point exceeds the threshold anywhere.")
        print("(Lower --threshold if you meant to exercise the diagnostic.)")
        return 0

    out = os.path.join('results_validation', 'huge_dv_diagnostic.csv')
    try:
        import pandas as pd
        os.makedirs('results_validation', exist_ok=True)
        (pd.DataFrame(rows)
           .sort_values('contribution_pct', ascending=False)
           .to_csv(out, index=False))
        print(f"{len(rows)} warnings written to {out} "
              f"(sorted by contribution).\n")
    except Exception as e:                                    # noqa: BLE001
        print(f"(could not write {out}: {type(e).__name__}: {e})\n")

    adm = [r for r in rows if r['admitted']]
    rej = len(rows) - len(adm)
    if not adm:
        print("Not one of these warnings falls on an admitted geometry: all "
              f"{rej} sit at inv_status outside (OK, MARGINAL), so none of "
              "them enters a published number.")
    else:
        print(f"Largest contribution on an ADMITTED geometry: {worst:.4f} % "
              f"of DeltaI_bond   ({len(adm)} warnings admitted"
              + (f", {rej} on gate-rejected geometries and therefore in no "
                 f"table)." if rej else ")."))
        top = sorted(adm, key=lambda r: -r['contribution_pct'])[:5]
        for r in top:
            print(f"     {r['system']:<12} R = {r['R_Angstrom']:>6.3f} A  "
                  f"{r['functional']:<8} {r['contribution_pct']:8.4f} %  "
                  f"[{r['inv_status']}]")
    print()
    print("That percentage is what decides. A loud warning whose contribution")
    print("is negligible is cosmetic, and can be said to be so WITH the")
    print("number. One that is not negligible is not published without an")
    print("explanation.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
