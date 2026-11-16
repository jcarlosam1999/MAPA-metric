"""
density_comparison.py -- the response metric against the self-consistent
density difference (SI Sec. S16; main text Sec. 2.3).
=========================================================================

Sec. 2.3 says that Eq. 3 (first-order displaced density on the exact KS
system) and the self-consistent difference rho_DFA^SCF - rho_ref coincide only
in the simplest case. Three comparisons stand behind that sentence, all with
the same reduction (mean |field| over Omega, e-/bohr^3) and the weighted cosine
of the two fields inside Omega:

  exact_H2plus  H2+, one electron: v_xc^exact = -v_H exactly, no inversion.
  model         rho_ref = rho_LDA, whose exact KS potential is v_LDA[rho_LDA]
                (no inversion); PBE measured both ways. The same density is
                then inverted with the production engine and the metric
                recomputed, which measures what the inversion adds when the
                answer is known.
  production    the deposited response metric (exact density mode) against
                the self-consistent difference, real references.

Every response metric recomputed from a production record must reproduce the
deposited column before it is used (relative 1e-5); otherwise the stage stops.
Rows carry the production confidence status and, for `production`, the
degeneracy-cutoff flag of Sec. 2.5 (span >= value, the test of
verification/check_curves.py); quoted ranges use admitted, determinate rows.

Needs the per-geometry records results_grid_data/<system>_R*.npz.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import artifacts as art
import functional_eval as fe
import inversion as iv
import metrics as mt
import response_metric as rm

FUNCS = (('LDA,VWN', 'LDA-VWN'), ('PBE,PBE', 'PBE-PBE'))
PRODUCTION_SYSTEMS = ('HH_q+1_s1', 'HH_q+0_s0', 'HLi_q+0_s0', 'LiLi_q+0_s0',
                      'NN_q+0_s0')
# geometry indices (into each production scan) for the model reference; the
# inversion makes this the slow part, so it samples the scan rather than
# covering it
MODEL_GEOMETRIES = {'HH_q+0_s0': (0, 5, 10, 15, 21),
                    'HLi_q+0_s0': (0, 3, 7, 11, 15, 19),
                    'LiLi_q+0_s0': (0, 3, 7, 11, 15, 19),
                    'NN_q+0_s0': (0, 3, 7, 11, 15, 19)}
MATCH_TOL = 1e-5
MODEL_REPRO_TOL = 5e-3        # e-, the compact-system confidence threshold


# ---- shared pieces ---------------------------------------------------------
def _paths(system):
    return sorted(glob.glob(f'results_grid_data/{system}_R*.npz'),
                  key=lambda p: float(p.split('_R')[-1][:-4]))


def _curve(system):
    import pandas as pd
    return pd.read_csv(f'results_energies/{system}_Dissociation_Curve.csv')


def _lookup(df, R, col):
    i = (df['R_Angstrom'] - R).abs().idxmin()
    if abs(df.loc[i, 'R_Angstrom'] - R) > 1e-4:
        raise SystemExit(f'R={R} not in the dissociation curve')
    return df.loc[i, col]


def _load(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))

    def opt(key):
        if key in d.files:
            a = np.asarray(d[key])
            return a if a.size else None
        return None

    ctx = dict(meta=meta, R=float(meta['R_ang']),
               coords=d['coords_bohr'], w=d['weights'],
               vxc_exact=d['vxc_exact'], rho_exact=d['rho_exact'],
               exact_dm=d['exact_dm'], spin=int(meta.get('spin', 0)),
               rho_a=opt('rho_alpha'), rho_b=opt('rho_beta'),
               vxc_a=opt('vxc_alpha'), vxc_b=opt('vxc_beta'),
               dma=opt('exact_dm_alpha'), dmb=opt('exact_dm_beta'),
               D_ks=opt('D_ks'), scf_cache={}, v_hartree=opt('v_hartree'))
    ctx['mol'] = fe._rebuild_mol(meta)
    ctx['open_shell'] = (ctx['spin'] != 0 and ctx['rho_a'] is not None
                         and ctx['rho_b'] is not None)
    return ctx


def _exact_density(ctx):
    return dict(dm_tot=ctx['exact_dm'], rho_tot=ctx['rho_exact'],
                dma=ctx['dma'], dmb=ctx['dmb'],
                rho_a=ctx['rho_a'], rho_b=ctx['rho_b'])


def _v_func(ctx, finfo, s):
    """The functional's potential reduced to one channel as production does."""
    mol, coords, w = ctx['mol'], ctx['coords'], ctx['w']
    if not ctx['open_shell']:
        v, _ = fe.functional_vxc_on_grid(finfo, mol, s['dm_tot'], coords)
        return fe._sanitize_grid_potential(v, ctx['vxc_exact'], finfo.name,
                                           ctx['R'], report_huge=None)
    (va, vb), _ = fe.functional_vxc_spin_on_grid(finfo, mol, s['dma'], s['dmb'],
                                                 s['rho_a'], s['rho_b'], coords)
    v, _ = fe._spin_average_potential(va, vb, s['rho_a'], s['rho_b'], w)
    return v


def _ks(mol, coords, w, ao, vxc, J, D):
    T, V, S = (mol.intor(k) for k in ('int1e_kin', 'int1e_nuc', 'int1e_ovlp'))
    eps, C = rm.ks_system(mol, coords, w, ao, vxc, T, V, J, S)
    return eps, C, rm.ks_occupations(C, S, np.asarray(D))


def _ks_ctx(ctx, vxc, ao):
    D = ctx['D_ks'] if ctx.get('D_ks') is not None else ctx['exact_dm']
    return _ks(ctx['mol'], ctx['coords'], ctx['w'], ao, vxc,
               iv.make_J(ctx['mol'], np.asarray(ctx['exact_dm'])), D)


def _reduce(f, w, m):
    return float(np.sum(w[m] * np.abs(f[m])) / w[m].sum())


def _cos(a, b, w, m):
    wm = w[m]
    return float(np.sum(wm * a[m] * b[m]) /
                 np.sqrt(np.sum(wm * a[m] ** 2) * np.sum(wm * b[m] ** 2)))


def _guard(val, dep, what):
    if np.isfinite(dep) and dep > 0 and abs(val / dep - 1) > MATCH_TOL:
        raise SystemExit(f'{what}: recomputed {val:.6e} vs deposited {dep:.6e}')


# ---- the three comparisons -------------------------------------------------
def exact_h2plus(log=print):
    system, df, rows = 'HH_q+1_s1', _curve('HH_q+1_s1'), []
    for p in _paths(system):
        ctx = _load(p)
        R, w = ctx['R'], ctx['w']
        m = mt.bond_region_mask(ctx['coords'], R)
        ao = ctx['mol'].eval_gto('GTOval', ctx['coords'])
        v_an = -np.asarray(ctx['v_hartree'])
        ks_i, ks_a = _ks_ctx(ctx, ctx['vxc_exact'], ao), _ks_ctx(ctx, v_an, ao)
        diff = ctx['vxc_exact'] - v_an
        rec = dict(system=system, status=_lookup(df, R, 'inv_status'), R=R,
                   inv_err_omega_Eh=float(np.sum(w[m] * np.abs(diff[m] - np.median(diff[m])))
                                          / w[m].sum()))
        for fn, tag in FUNCS:
            finfo = fe.resolve_functional(fn)
            v = _v_func(ctx, finfo, _exact_density(ctx))
            f_inv = -rm.drho_from_dv(*ks_i, ao, w, ctx['vxc_exact'] - v)
            f_an = -rm.drho_from_dv(*ks_a, ao, w, v_an - v)
            inv = _reduce(f_inv, w, m)
            _guard(inv, float(_lookup(df, R, f'DeltaI_resp_{tag}_exact')),
                   f'H2+ R={R} {tag}')
            dm, _, nsol = _scf_lowest(ctx['mol'], finfo.xc_string, _guesses_from(ctx))
            scf = np.einsum('gi,ij,gj->g', ao, dm, ao) - ctx['rho_exact']
            rec[f'scf_solutions_{tag}'] = nsol
            rec.update({f'mapa_an_{tag}': _reduce(f_an, w, m),
                        f'mapa_inv_{tag}': inv, f'scf_{tag}': _reduce(scf, w, m),
                        f'cos_an_scf_{tag}': _cos(f_an, scf, w, m),
                        f'cos_an_inv_{tag}': _cos(f_an, f_inv, w, m)})
        rows.append(rec)
        log(f'  H2+ R={R:.3f} exact/scf PBE '
            f"{rec['mapa_an_PBE-PBE'] / rec['scf_PBE-PBE']:.3f}")
    return rows


def production(log=print):
    rows = []
    for system in PRODUCTION_SYSTEMS:
        df = _curve(system)
        for p in _paths(system):
            ctx = _load(p)
            R, w = ctx['R'], ctx['w']
            m = mt.bond_region_mask(ctx['coords'], R)
            ao = ctx['mol'].eval_gto('GTOval', ctx['coords'])
            ks = _ks_ctx(ctx, ctx['vxc_exact'], ao)
            rec = dict(system=system, R=R, status=_lookup(df, R, 'inv_status'))
            for fn, tag in FUNCS:
                finfo = fe.resolve_functional(fn)
                col = f'DeltaI_resp_{tag}_exact'
                v = _v_func(ctx, finfo, _exact_density(ctx))
                f = -rm.drho_from_dv(*ks, ao, w, ctx['vxc_exact'] - v)
                val, dep = _reduce(f, w, m), float(_lookup(df, R, col))
                _guard(val, dep, f'{system} R={R} {tag}')
                try:
                    dm, _, nsol = _scf_lowest(ctx['mol'], finfo.xc_string,
                                              _guesses_from(ctx))
                    scf = np.einsum('gi,ij,gj->g', ao, dm, ao) - ctx['rho_exact']
                    s_val, cos = _reduce(scf, w, m), _cos(f, scf, w, m)
                except RuntimeError:
                    s_val = cos = nsol = np.nan
                rec[f'scf_solutions_{tag}'] = nsol
                span = float(_lookup(df, R, col + '_degtol_span'))
                rec.update({f'mapa_{tag}': val, f'dep_{tag}': dep,
                            f'scf_{tag}': s_val,
                            f'ratio_{tag}': val / s_val if s_val else np.nan,
                            f'cos_{tag}': cos,
                            f'indeterminate_{tag}': bool(abs(span) >= abs(dep))})
            rows.append(rec)
            log(f"  {system} R={R:.3f} {rec['status']}: MAPA/scf PBE "
                f"{rec['ratio_PBE-PBE']:.3f}")
    return rows


def _scf_lowest(mol, xc, guesses):
    """Self-consistent density of `xc`, made deterministic.

    At stretched geometries a semi-local functional has more than one SCF
    solution (H2+ localizes or not; restricted N2 and H2 can land in either
    of two states), and which one a single kernel() call reaches depends on
    round-off. Run from every guess in `guesses` (None = PySCF default),
    keep the converged solution of LOWEST energy, and report how many
    distinct solutions were found. Returns (dm_total, energy, n_distinct)."""
    from pyscf import dft
    sols = []
    for g in guesses:
        ks = (dft.RKS if mol.spin == 0 else dft.UKS)(mol)
        ks.xc, ks.verbose = xc, 0
        if isinstance(g, str):                 # a named PySCF initial guess
            ks.init_guess, g = g, None
        ks.kernel(dm0=g)
        if not ks.converged:
            ks.diis_space, ks.level_shift, ks.max_cycle = 12, 0.2, 200
            ks.kernel(dm0=ks.make_rdm1())
        if not ks.converged:
            ks = ks.newton()
            ks.kernel(dm0=ks.make_rdm1())
        if ks.converged:
            dm = ks.make_rdm1()
            sols.append((float(ks.e_tot), dm if mol.spin == 0 else dm[0] + dm[1]))
    if not sols:
        raise RuntimeError(f'{xc}: no SCF solution converged from any guess')
    sols.sort(key=lambda t: t[0])
    distinct = 1 + sum(abs(e - sols[0][0]) > 1e-6 for e, _ in sols[1:])
    return sols[0][1], sols[0][0], distinct


def _guesses_from(ctx):
    """Default guess, and the reference density itself."""
    if ctx['spin'] == 0:
        return (None, np.asarray(ctx['exact_dm']))
    if ctx['dma'] is not None and ctx['dmb'] is not None:
        return (None, np.array([ctx['dma'], ctx['dmb']]))
    return (None,)


def model(log=print):
    from pyscf import dft
    import molecules as mo
    fL, fP = fe.resolve_functional('LDA,VWN'), fe.resolve_functional('PBE,PBE')
    rows = []
    for system, idx in MODEL_GEOMETRIES.items():
        df, paths = _curve(system), _paths(system)
        for i in idx:
            meta = json.loads(str(np.load(paths[i], allow_pickle=False)['meta']))
            mol, R = fe._rebuild_mol(meta), float(meta['R_ang'])
            grids = mo.becke_grid(mol)
            coords, w = grids.coords, grids.weights
            ao1 = mol.eval_gto('GTOval_sph_deriv1', coords)
            ao = ao1[0]
            m = mt.bond_region_mask(coords, R)
            dmL, _, nL = _scf_lowest(mol, 'LDA,VWN', (None, 'atom'))
            rhoL = dft.numint.eval_rho(mol, ao1, dmL, xctype='GGA')[0]
            vL, _ = fe.functional_vxc_on_grid(fL, mol, dmL, coords)
            vP, _ = fe.functional_vxc_on_grid(fP, mol, dmL, coords)
            JL = iv.make_J(mol, dmL)
            eps, C, occ = _ks(mol, coords, w, ao, vL, JL, dmL)
            rho_check = np.einsum('gi,i,gi->g', ao @ C, occ, ao @ C)
            f_model = -rm.drho_from_dv(eps, C, occ, ao, w, vL - vP)
            dmP, _, nP = _scf_lowest(mol, 'PBE,PBE', (None, 'atom', dmL))
            dscf = dft.numint.eval_rho(mol, ao1, dmP, xctype='GGA')[0] - rhoL
            vH = mo.hartree_potential_on_points(mol, dmL, coords)
            vguide, _ = mo.xc_guide_on_points(mol, dmL, coords, rho=rhoL, v_h=vH)
            out = iv.invert_geometry(mol, grids, ao, dmL, rhoL, None, vguide, JL)
            vinv = out['vxc']
            ks_i = _ks(mol, coords, w, ao, vinv, JL, out.get('D_ks', dmL))
            f_inv = -rm.drho_from_dv(*ks_i, ao, w, vinv - vP)
            dvi, dvm = vinv - vL, vL - vP
            rec = dict(system=system, R=R, status=_lookup(df, R, 'inv_status'),
                       ks_reproduces_rhoL_e=float(np.sum(w * np.abs(rho_check - rhoL))),
                       mapa_model=_reduce(f_model, w, m), scf=_reduce(dscf, w, m),
                       mapa_inv=_reduce(f_inv, w, m),
                       cos_model_scf=_cos(f_model, dscf, w, m),
                       cos_model_inv=_cos(f_model, f_inv, w, m),
                       inv_vxc_err_omega_Eh=float(np.sum(w[m] * np.abs(dvi[m] - np.median(dvi[m])))
                                                  / w[m].sum()),
                       dv_model_omega_Eh=float(np.sum(w[m] * np.abs(dvm[m] - np.median(dvm[m])))
                                               / w[m].sum()),
                       inv_density_error_e=float(out.get('density_error', np.nan)),
                       inv_kappa=out.get('kappa_final', np.nan),
                       inv_mode=str(out.get('inv_mode', '')),
                       scf_solutions_LDA=nL, scf_solutions_PBE=nP)
            # The model rests on one premise: the KS system of v_LDA[rho_LDA]
            # reproduces rho_LDA. Where it does not (a near-degenerate
            # stretched bond whose restricted LDA solution is not the aufbau
            # state of its own potential), the "known exact potential" is not
            # exact and the row is flagged rather than used. Threshold: the
            # strictest density gate of the benchmark (compact systems).
            rec['model_valid'] = bool(rec['ks_reproduces_rhoL_e'] <= MODEL_REPRO_TOL)
            rec['ratio_model_scf'] = rec['mapa_model'] / rec['scf']
            rec['ratio_inv_model'] = rec['mapa_inv'] / rec['mapa_model']
            rows.append(rec)
            log(f"  model {system} R={R:.3f}: MAPA/scf {rec['ratio_model_scf']:.3f}, "
                f"inverted/exact {rec['ratio_inv_model']:.4f}")
    return rows


def run_density_comparison(log=print):
    import pandas as pd
    for system in set(PRODUCTION_SYSTEMS) | set(MODEL_GEOMETRIES):
        if not _paths(system):
            raise SystemExit(f'no per-geometry records for {system} in '
                             f'results_grid_data/')
    os.makedirs(art.DIR, exist_ok=True)
    log('  -- H2+, exact potential')
    pd.DataFrame(exact_h2plus(log)).to_csv(art.path('density_comparison_exact'),
                                           index=False)
    log('  -- model reference, known exact potential')
    pd.DataFrame(model(log)).to_csv(art.path('density_comparison_model'),
                                    index=False)
    log('  -- production references')
    pd.DataFrame(production(log)).to_csv(art.path('density_comparison'),
                                         index=False)
