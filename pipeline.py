"""
pipeline.py — orchestration: one geometry end-to-end, then the benchmark set scan.
========================================================================
process_geometry() is the assembly line (see the diagram in main.py):
reference ladder -> fields on the grid -> Fermi-Amaldi guide -> Wu-Yang
inversion -> DFA baselines -> metrics + exact-condition diagnostics ->
1D profiles + figures -> grid npz. run_pipeline() wraps it with preflight,
fault tolerance (a failing geometry logs ERROR and the scan continues) and
bit-for-bit resume (rows + warm-start chain reloaded from disk).
"""

import os
import traceback
import numpy as np
from tqdm import tqdm
from pyscf import dft

from config import CFG
from benchmark import parse_benchmark
import molecules as mo
import reference as rf
import inversion as iv
import metrics as mt
import exporters as ex


def process_geometry(member, r):
    name, charge, spin = member['name'], member['charge'], member['spin']
    mol = mo.build_molecule(member['a1'], member['a2'], r, charge, spin)
    n_elec = mol.nelectron
    grids = mo.becke_grid(mol)
    coords, w = grids.coords, grids.weights

    # ---- 1) reference ladder: density + energies (reference.py) -----------
    ref = rf.run_reference(mol, member['ncas'], member['nelecas'],
                           fci_auto=CFG.fci_auto_density)
    mf, ref_dm = ref['mf'], ref['ref_dm']
    mf_u = mo.run_uhf_broken_symmetry(mol)      # RHF-vs-UHF gap diagnostic

    # ---- 2) reference fields on the grid ----------------------------------
    ao1 = mol.eval_gto('GTOval_sph_deriv1', coords)
    rho_gga = dft.numint.eval_rho(mol, ao1, ref_dm, xctype='GGA')
    rho_ref, grad_rho = rho_gga[0], rho_gga[1:4]
    ao_value = ao1[0]
    tau = 0.5 * sum(np.einsum('gi,ij,gj->g', ao1[d], ref_dm, ao1[d])
                    for d in (1, 2, 3))         # 1-RDM KE density (mGGA feat.)
    n_grid_e = float(np.sum(rho_ref * w))
    if abs(n_grid_e - n_elec) > 1e-4:
        tqdm.write(f"  [GRID WARN] {name} R={r:.2f}: Int rho = "
                   f"{n_grid_e:.6f} vs N = {n_elec}")

    rho_spin = None
    dma = dmb = None
    if spin != 0 and ref['dms'] is not None:
        dma, dmb = ref['dms']
        rga = dft.numint.eval_rho(mol, ao1, dma, xctype='GGA')
        rgb = dft.numint.eval_rho(mol, ao1, dmb, xctype='GGA')
        rho_spin = (rga[0], rgb[0])

    # ---- 3) Fermi-Amaldi guide + inversion (inversion.py) -----------------
    v_H = mo.hartree_potential_on_points(mol, ref_dm, coords)
    v_fa = -v_H / max(n_elec, 1)
    J_ref = iv.make_J(mol, ref_dm)
    inv = iv.invert_geometry(mol, grids, ao_value, ref_dm, rho_ref,
                             rho_spin, v_fa, J_ref, dm_alpha=dma, dm_beta=dmb)
    mode = inv.get('inv_mode', 'UNKNOWN')
    status = ('OK' if inv['good']
              else ('FAIL' if inv['failed'] else 'MARGINAL'))
    # EVERY geometry reports, OK included. Rendered by the same _status_line
    # used by the resume path, so a fresh line and a reloaded one are
    # byte-identical to the line the fresh path printed.
    _log(_status_line(
        {'R_Angstrom': r, 'inv_status': status, 'inv_mode': mode,
         'inv_density_error_e': inv['density_error'],
         'KS_homo_lumo_gap': inv['gap'],
         'N_unpaired': inv.get('n_unpaired', float('nan')),
         'kappa_final': inv['kappa_final'],
         'ghost_leak_e': inv['ghost_leak'],
         'n_support': inv['n_support'], 'n_grid': inv['n_grid']}, name))
    np.save(f"results_convergence/{name}_R{r:.2f}_trace.npy", inv['trace'])

    sie_validation = np.nan          # 1e-: analytic ground truth vxc = -v_H
    if n_elec == 1:
        diff = inv['vxc'] + v_H
        sie_validation = float(np.max(np.abs(diff - np.median(diff))))
        tqdm.write(f"  [VALIDATE 1e-] {name}: max|vxc+v_H-c| = "
                   f"{sie_validation:.3e} Eh")

    # ---- 4) EXACT-SIDE energy decomposition (functional-agnostic) ---------
    # NOTE: functional (LDA/PBE/...) v_xc and DeltaI are NO LONGER computed
    # here. The heavy pipeline produces only the functional-AGNOSTIC exact
    # side; any functional is scored afterward by functional_eval.py over the
    # saved npz. This is the decoupling that makes the exact side reusable.
    Exc_cons, Ts, T_ref = mt.ref_exc(mol, ref['e_consistent'], inv['D_ks'],
                                     ref_dm, J_ref)
    Exc_best, _, _ = mt.ref_exc(mol, ref['e_ref'], inv['D_ks'],
                                ref_dm, J_ref)
    T_c = T_ref - Ts
    E_c_dyn = (ref['e_nevpt2'] - ref['e_cas']
               if np.isfinite(ref['e_nevpt2']) else np.nan)
    virial_gap, zero_force = mt.exact_condition_diagnostics(
        inv['vxc'], rho_ref, grad_rho, coords, w, Exc_cons, T_c)

    is_2e_singlet = (n_elec == 2 and spin == 0)
    ip_ref = (rf.compute_ip(member, r, ref['e_ref'])
              if name in CFG.ip_validation_members else np.nan)

    row = {"R_Angstrom": r, "E_SCF": ref['e_scf'], "E_UHF": mf_u.e_tot,
           "E_CASSCF": ref['e_cas'], "E_NEVPT2": ref['e_nevpt2'],
           "ref_density_level": ref['ref_level'],
           "ref_space_orbitals": ref['ref_space'],
           "ladder_gain_mEh": 1000.0 * (ref['e_cas'] - ref['e_consistent']),
           "E_FCI": ref['e_fci'],
           "Exc_consistent": Exc_cons, "Exc_ref": Exc_best,
           "E_c_dyn": E_c_dyn,
           "Ts_KS": Ts, "T_ref": T_ref, "T_c": T_c,
           "virial_gap_Eh": virial_gap, "zero_force_z": zero_force[2],
           "inv_density_error_e": inv['density_error'],
           "inv_status": status, "inv_mode": mode,
           "N_unpaired": inv.get('n_unpaired', np.nan),
           "ks_spin_square": inv.get('ks_spin_square', 0.0),
           "bs_spin_square_raw": inv.get('bs_spin_square_raw', np.nan),
           "bs_symmetrized": inv.get('symmetrized', False),
           "bs_density_error_e": inv.get('bs_density_error', np.nan),
           "n_natorb_frac": (int(np.sum((inv['natocc'] > 0.02)
                                        & (inv['natocc'] < 1.98)))
                             if inv.get('natocc') is not None else 0),
           "vxc_ens_bs_maxdiff": (float(np.max(np.abs(inv['vxc_diff'])))
                                  if inv.get('vxc_diff') is not None
                                  else np.nan),
           "KS_homo_lumo_gap": inv['gap'],
           "kappa_final": inv['kappa_final'],
           "pot_dim_kept": inv['npot_kept'],
           "ghost_leak_e": inv['ghost_leak'],
           # kept in the row so a RESUMED point can reprint the identical
           # status line (these were previously only in the inverter output)
           "n_support": inv['n_support'], "n_grid": inv['n_grid'],
           "eps_HOMO_KS": inv['eps_homo'], "IP_ref": ip_ref,
           "grid_electrons": n_grid_e, "scf_converged": ref['scf_conv'],
           "uhf_converged": bool(mf_u.converged),
           "cas_converged": ref['cas_conv'],
           "nevpt2_ok": bool(np.isfinite(ref['e_nevpt2'])),
           "anion_caveat": member['anion_caveat'],
           "sie_validation_1e": sie_validation}

    # ---- 6) dense 1D profiles + figures ------------------------------------
    z_b, line_pts = mo.dense_axis_line(r)
    v_H_line = mo.hartree_potential_on_points(mol, ref_dm, line_pts)
    v_fa_line = -v_H_line / max(n_elec, 1)
    cols = {"V_Hartree": v_H_line, "V_FA": v_fa_line}
    if len(inv['b_list']) == 2:
        va_l = v_fa_line + inv['inverters'][0].v_on_points(
            inv['b_list'][0], line_pts)
        vb_l = v_fa_line + inv['inverters'][1].v_on_points(
            inv['b_list'][1], line_pts)
        vxc_line = 0.5 * (va_l + vb_l)
        cols.update(Vxc_Alpha=va_l, Vxc_Beta=vb_l)
    else:
        vxc_line = v_fa_line + inv['inverters'][0].v_on_points(
            inv['b_list'][0], line_pts)
    ao_line = mol.eval_gto('GTOval', line_pts)
    rho_line = dft.numint.eval_rho(mol, ao_line, ref_dm, xctype='LDA')
    cols.update(Vxc_Exact=vxc_line, Rho_Exact=rho_line)
    # ensemble channel: export the rigorous ensemble v_xc, the BS diagnostic
    # v_xc, and their difference (framework-ambiguity metric) along the axis.
    if inv.get('vxc_ensemble') is not None:
        vxc_ens_line = v_fa_line + inv['inverters'][0].v_on_points(
            inv['b_list'][0], line_pts)
        cols["Vxc_Ensemble"] = vxc_ens_line
        if inv.get('vxc_bs') is not None and len(inv.get('bs_b_list', [])) == 2:
            vba = v_fa_line + inv['bs_inverters'][0].v_on_points(
                inv['bs_b_list'][0], line_pts)
            vbb = v_fa_line + inv['bs_inverters'][1].v_on_points(
                inv['bs_b_list'][1], line_pts)
            vxc_bs_line = 0.5 * (vba + vbb)
            cols["Vxc_BS"] = vxc_bs_line
            cols["Vxc_Ens_minus_BS"] = vxc_ens_line - vxc_bs_line
    if is_2e_singlet:
        # exact correlation potential (v_x = -v_H/2 analytically for 2e)
        cols["Vc_Exact"] = vxc_line + 0.5 * v_H_line
    ex.save_line_csv(name, r, z_b, cols)
    # NOTE: comparative figures (exact vs functional) are generated by the
    # functional_eval post-pass, which recomputes each functional's v_xc on
    # this saved line. The heavy pipeline exports only the exact side.

    # ---- 7) grid export -----------------------------------------------------
    empty = np.array([])
    fields = dict(
        rho_exact=rho_ref, grad_rho=grad_rho, tau=tau,
        rho_alpha=(rho_spin[0] if rho_spin else empty),
        rho_beta=(rho_spin[1] if rho_spin else empty),
        vxc_exact=inv['vxc'],
        vxc_alpha=(inv['vxc_alpha'] if inv['vxc_alpha'] is not None
                   else empty),
        vxc_beta=(inv['vxc_beta'] if inv['vxc_beta'] is not None
                  else empty),
        vc_exact=(inv['vxc'] + 0.5 * v_H if is_2e_singlet else empty),
        v_hartree=v_H, v_ref_FA=v_fa,
        pot_basis_coeffs=np.concatenate(inv['b_list']),
        D_ks=inv['D_ks'], mo_energy_ks=inv['mo_energy'])
    tensors = dict(
        S=mol.intor('int1e_ovlp'), T_kin=mol.intor('int1e_kin'),
        V_nuc=mol.intor('int1e_nuc'), J_exact_mat=J_ref,
        mo_coeff_ref=mf.mo_coeff, exact_dm=ref_dm,
        # spin-resolved exact DMs (open shell only): the functional post-pass
        # needs them to evaluate GGA spin potentials on the exact density
        exact_dm_alpha=(ref['dms'][0] if ref['dms'] is not None else empty),
        exact_dm_beta=(ref['dms'][1] if ref['dms'] is not None else empty),
        ao_value=(ao_value.astype(np.float32) if CFG.save_ao_grid
                  else empty),
        eri_ao=(mol.intor('int2e') if CFG.save_eri else empty))
    meta_extra = dict(
        E_NEVPT2=(float(ref['e_nevpt2'])
                  if np.isfinite(ref['e_nevpt2']) else None),
        ref_density_level=ref['ref_level'],
        ref_space_orbitals=int(ref['ref_space']),
        Exc_consistent=float(Exc_cons), Exc_ref=float(Exc_best),
        E_c_dyn=(float(E_c_dyn) if np.isfinite(E_c_dyn) else None),
        virial_gap_Eh=float(virial_gap),
        density_error=float(inv['density_error']),
        kappa_final=float(inv['kappa_final']), inv_status=status,
        inv_mode=mode, ks_spin_square=float(inv.get('ks_spin_square', 0.0)),
        # regime conditional + virial ingredient, mirrored from row.json
        # into the npz so the grid record is self-describing
        N_unpaired=float(inv.get('n_unpaired', np.nan)),
        T_c=float(T_c))
    ex.export_grid_npz(member, r, mol, grids, fields, tensors, meta_extra)
    return row


def preflight(members):
    """Verify every member BUILDS (basis availability, spin parity) before
    committing to a multi-day scan; unbuildable members are skipped loudly."""
    ok = []
    for m in members:
        try:
            mo.build_molecule(m['a1'], m['a2'], m['r_eq'], m['charge'],
                              m['spin'])
            ok.append(m)
        except Exception as exc:
            tqdm.write(f"[PREFLIGHT SKIP] {m['name']}: {exc}")
    tqdm.write(f"[PREFLIGHT] {len(ok)}/{len(members)} members buildable.")
    return ok


def _log(msg):
    """Print AND append to the run log, flushing immediately so a closed
    terminal or a killed process still leaves a complete record of what ran.

    When main.py has routed stdout into the log (CFG.log_is_teed), printing is
    already enough: appending here as well would double every line."""
    tqdm.write(msg)
    if CFG.log_is_teed:
        return
    try:
        os.makedirs(os.path.dirname(CFG.log_path) or '.', exist_ok=True)
        with open(CFG.log_path, 'a') as f:
            f.write(msg + "\n")
            f.flush()
    except Exception:                                        # noqa: BLE001
        pass                      # logging must never break a running scan


def _status_line(row, name, tag=""):
    """Render the per-geometry status line from a ROW dict. Shared by the
    fresh and the resumed paths so the two can never drift apart: a resumed
    point prints exactly what it printed when it was computed. EVERY geometry
    reports, OK included. Older rows lack n_support/n_grid; those render as
    '?' rather than crash.
    """
    status = row.get('inv_status', '?')
    mode = row.get('inv_mode', 'UNKNOWN')
    gap = row.get('KS_homo_lumo_gap', float('nan'))
    sup = row.get('n_support', '?')
    ngr = row.get('n_grid', '?')
    return (f"  [{status}/{mode}{tag}] {name} R={row['R_Angstrom']:.2f}: "
            f"|drho|={row.get('inv_density_error_e', float('nan')):.2e} e-, "
            f"gap={gap:.4f}, "
            f"N_unpaired={row.get('N_unpaired', float('nan')):.3f}, "
            f"kappa_final={row.get('kappa_final', float('nan')):.1e}, "
            f"ghost_leak={row.get('ghost_leak_e', float('nan')):.2e} e-, "
            f"support={sup}/{ngr}")


def _geometry_report(row, name, n_elec=None):
    """Every line a geometry contributes to the log, rendered from its ROW.

    A resumed point replays exactly what it printed when it was first computed
    -- the grid-electron warning, the status line and the one-electron
    analytic check -- so a log from a resumed run is indistinguishable from a
    log of a fresh one, and every number the manuscript rests on is present in
    both the log and the CSV.
    """
    lines = []
    n_grid_e = row.get('grid_electrons')
    if (n_elec is not None and n_grid_e is not None
            and abs(float(n_grid_e) - n_elec) > 1e-4):
        lines.append(f"  [GRID WARN] {name} R={row['R_Angstrom']:.2f}: "
                     f"Int rho = {float(n_grid_e):.6f} vs N = {n_elec}")
    lines.append(_status_line(row, name))
    sie = row.get('sie_validation_1e')
    try:
        sie = float(sie)
    except (TypeError, ValueError):
        sie = float('nan')
    if sie == sie:                                  # not NaN: one-electron case
        lines.append(f"  [VALIDATE 1e-] {name}: max|vxc+v_H-c| = "
                     f"{sie:.3e} Eh")
    return lines


def run_pipeline(only=None):
    """only: optional list of member names to run (shakedown); None = full
    benchmark set. Filtering happens before preflight so the log reflects the scope."""
    import pandas as pd
    members = parse_benchmark()
    if only:
        wanted = set(only)
        members = [m for m in members if m['name'] in wanted]
        missing = wanted - {m['name'] for m in members}
        if missing:
            tqdm.write(f"[WARN] unknown member names ignored: {missing}")
    members = preflight(members)
    for member in tqdm(members, desc="Benchmark", colour='green'):
        name = member['name']
        _log(f"\n---> {name}  (CAS({member['ncas']},"
             f"{member['nelecas']}), Re={member['r_eq']} A) <---")
        distances = mo.make_distance_grid(member['r_eq'])
        rows = []
        n_loaded = n_computed = 0
        # Single up-front statement of what is already on disk. Individual
        # geometries below are then reported identically whether they are
        # replayed from a saved row or computed now, so the log always reads
        # like a full run.
        import os as _os
        _cached = sum(1 for _r in distances
                      if _os.path.exists(ex.row_path(name, _r))
                      and _os.path.exists(ex.npz_path(name, _r)))
        if CFG.resume and _cached:
            _log(f"  [resume] {_cached}/{len(distances)} geometries already on "
                 f"disk; {len(distances) - _cached} to compute.")
        else:
            _log(f"  [scan] {len(distances)} geometries to compute.")

        for r in tqdm(distances, desc=f"Scan {name}", leave=False,
                      colour='cyan'):
            import os
            # Resume skips a point only if BOTH the row.json AND the npz exist.
            # The npz is the reusable exact-side artifact the functional
            # post-pass needs, so a stray row.json without its npz must NOT be
            # treated as "done" — recompute it instead of leaving a gap.
            row_done = os.path.exists(ex.row_path(name, r))
            npz_done = os.path.exists(ex.npz_path(name, r))
            if CFG.resume and row_done and npz_done:
                row = ex.load_row(name, r)
                rows.append(row)
                n_loaded += 1
                for line in _geometry_report(row, name, member['n_elec']):
                    _log(line)
                continue
            try:
                row = process_geometry(member, r)
            except Exception:
                _log(f"  [ERROR] {name} R={r:.2f} — continuing.\n"
                     + traceback.format_exc(limit=3))
                row = {"R_Angstrom": r, "inv_status": "ERROR"}
            n_computed += 1
            rows.append(row)
            ex.save_row(name, r, row)
        pd.DataFrame(rows).to_csv(
            f"results_energies/{name}_Dissociation_Curve.csv", index=False)
        # End-of-system characterization, for LOADED and COMPUTED points
        # alike (both are in `rows`): the Richards fit of the unpaired-electron
        # curve and the Morse fit of the energy curve, each reported with the
        # equation, every parameter and its physical meaning. Defensive: a fit
        # failure must never lose a sweep that already finished.
        try:
            import richards_fit as rf
            import morse_fit as mf
            r_res = rf.fit_and_save(name, rows=rows)
            for line in rf.report(r_res):
                _log(line)
            m_res = mf.fit_and_save(name, rows=rows,
                                    energy_col=CFG.morse_energy_col,
                                    a1=member['a1'], a2=member['a2'],
                                    richards_res=r_res)
            cross = mf.cross_fit_properties(m_res, r_res)
            for line in mf.report(m_res, cross=cross):
                _log(line)
        except Exception:                                    # noqa: BLE001
            _log("  [fits] end-of-system fits skipped (non-fatal)\n"
                 + traceback.format_exc(limit=2))
        _log(f"[{name}] done. {len(distances)} geometries.")