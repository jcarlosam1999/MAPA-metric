"""
check_density_comparison.py -- Sec. 2.3 and SI Sec. S16.
========================================================

The response metric against the self-consistent density difference. Every
range the text prints is recomputed from the three tables the
`density_comparison` stage writes and compared at its printed precision.
"""

from __future__ import annotations

from . import paper_values as P
from ._common import ADMITTED, Report, matches_printed

import artifacts as art


def _rng(rep, label, printed, values):
    lo, hi = float(values.min()), float(values.max())
    ok = matches_printed(printed[0], lo) and matches_printed(printed[1], hi)
    (rep.matches if ok else rep.differs)(
        label, f'{printed[0]}--{printed[1]}', f'{lo:.4f}--{hi:.4f}')


def run(verbose=True):
    import pandas as pd
    rep = Report('MAPA vs SCF density (S16)')
    tags = ('density_comparison_exact', 'density_comparison_model',
            'density_comparison')
    missing = [t for t in tags if not art.have(t)]
    if missing:
        for t in missing:
            rep.fail(t, 'artifact missing; run the stage first')
        return rep
    ex, mo, pr = (pd.read_csv(art.path(t)) for t in tags)
    ex = ex[ex['status'].isin(ADMITTED)]
    # model rows whose own KS system does not reproduce rho_LDA are not a
    # test of anything (density_comparison.MODEL_REPRO_TOL) and are excluded
    mo = mo[mo['status'].isin(ADMITTED) & mo['model_valid'].astype(bool)]
    pr = pr[pr['status'].isin(ADMITTED)]

    r_l = ex['mapa_an_LDA-VWN'] / ex['scf_LDA-VWN']
    r_p = ex['mapa_an_PBE-PBE'] / ex['scf_PBE-PBE']
    _rng(rep, 'H2+ exact potential, ratio, LDA and PBE', P.MVD_H2PLUS_RATIO,
         pd.concat([r_l, r_p]))
    _rng(rep, 'H2+ exact potential, ratio, LDA', P.MVD_H2PLUS_RATIO_LDA, r_l)
    cmin = float(min(ex['cos_an_scf_LDA-VWN'].min(), ex['cos_an_scf_PBE-PBE'].min()))
    (rep.matches if cmin >= P.MVD_H2PLUS_COS_MIN else rep.differs)(
        'H2+ cosine, minimum', f'>= {P.MVD_H2PLUS_COS_MIN}', f'{cmin:.4f}')

    small = mo[mo['system'] != 'HLi_q+0_s0']['ratio_model_scf']
    _rng(rep, 'model H2/Li2/N2, excess in %', P.MVD_MODEL_SMALL_PCT,
         (small - 1) * 100)
    lih = mo[mo['system'] == 'HLi_q+0_s0']['ratio_model_scf']
    _rng(rep, 'model LiH, ratio', P.MVD_MODEL_LIH, lih)
    _rng(rep, 'model LiH, ratio (main text)', ('1.4', '3.5'), lih)
    dev = float((mo['ratio_inv_model'] - 1).abs().max()) * 100
    rep.check_printed('model, inverted vs exact, max change %',
                      P.MVD_MODEL_INV_PCT_MAX, dev)
    _rng(rep, 'model, inversion error / Dv in Omega', P.MVD_MODEL_INV_OVER_DV,
         mo['inv_vxc_err_omega_Eh'] / mo['dv_model_omega_Eh'])

    n2 = pr[(pr['system'] == 'NN_q+0_s0') & ~pr['indeterminate_PBE-PBE'].astype(bool)]
    _rng(rep, 'production N2, PBE ratio', P.MVD_PROD_N2_PBE, n2['ratio_PBE-PBE'])
    for (lo, hi), printed in P.MVD_PROD_N2_PBE_BANDS.items():
        band = n2[(n2['R'] >= lo) & (n2['R'] <= hi)]['ratio_PBE-PBE']
        _rng(rep, f'production N2, PBE ratio, {lo}-{hi} A', printed, band)
    return rep
