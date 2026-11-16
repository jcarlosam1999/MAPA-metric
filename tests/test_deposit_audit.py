#!/usr/bin/env python3
"""
test_deposit_audit.py — task 5.3.
=================================
`exporters.audit_line_columns` is the check that would have caught D1 on day
one. This test proves it catches D1's exact signature, plus the other failure
modes the deposit audit of 2026-08-07 turned up, and that a clean profile
passes.

  T1  a clean profile passes;
  T2  two identical columns are rejected            <- the D1 signature;
  T3  a non-finite value is rejected;
  T4  a negative density is rejected;
  T5  a non-monotone Z axis is rejected;
  T6  a broken one-electron identity (vxc = -v_H) is rejected;
  T7  a broken Vc_Exact definition is rejected;
  T8  save_line_csv refuses to WRITE a failing profile (the file must not
      exist afterwards) — the whole point of auditing at the write.

Usage:  python test_deposit_audit.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exporters as ex              # noqa: E402


def clean_profile(n=101, n_elec=2):
    z = np.linspace(-4, 4, n)
    vh = 1.0 / np.sqrt(z**2 + 0.7)
    return pd.DataFrame({
        'Z_Bohr': z, 'Z_Angstrom': z * 0.52917721092,
        'V_Hartree': vh, 'V_FA': -vh / n_elec,
        'Vxc_Exact': -0.61 * vh - 0.03,
        'Rho_Exact': np.exp(-z**2) + 1e-6,
    })


def expect(label, df, should_fail, results, n_elec=2):
    probs = ex.audit_line_columns('TEST', 1.0, df, strict=False)
    ok = bool(probs) == should_fail
    print(f"  {'ok  ' if ok else 'FAIL'} {label:46s} "
          f"{'rejected' if probs else 'accepted'}"
          + (f"  ({probs[0][:60]})" if probs else ''))
    if not ok:
        results.append(label)


def one_electron_profile(exact_identity, n=101):
    """H2+: one electron, so v_xc = -v_H and V_FA = -v_H/1 are the SAME array,
    and with the guide off V_Guide is that array too. `exact_identity` chooses
    whether the inversion reproduced the identity to machine zero (which it
    does at some geometries) or to a small residual."""
    z = np.linspace(-4, 4, n)
    vh = 1.0 / np.sqrt(z**2 + 0.7)
    vxc = -vh if exact_identity else -vh + 1e-8 * np.sin(z)
    return pd.DataFrame({
        'Z_Bohr': z, 'Z_Angstrom': z * 0.52917721092,
        'V_Hartree': vh, 'V_FA': -vh, 'V_Guide': -vh,
        'Vxc_Exact': vxc,
        'Rho_Exact': np.exp(-z**2) + 1e-6,
    })


def main():
    fails = []
    print('audit_line_columns:')

    # ---- one-electron identities are not duplicates ----------------------
    # An H2+ run stopped dead on 2026-08-09 at exactly the two geometries
    # where the inversion reproduced vxc = -v_H to MACHINE ZERO: the audit
    # reported `V_Guide == Vxc_Exact` and refused to write. The rule that
    # forbids identical columns was contradicting the rule, in the same
    # function, that requires this identity. Neighbouring geometries passed
    # only because their residual was 1e-8 rather than 0, which is the worst
    # possible failure mode: the check punished the best result.
    from config import CFG as _CFG
    _before = _CFG.guide_lda_core
    try:
        _CFG.guide_lda_core = False
        expect('N=1, identity exact to machine zero',
               one_electron_profile(True), False, fails, n_elec=1)
        expect('N=1, identity to 1e-8',
               one_electron_profile(False), False, fails, n_elec=1)
        # and the D1 net still catches a real duplicate in the same file
        d = one_electron_profile(True)
        d['Vxc_PBE-PBE_exact'] = d['Vxc_Exact'].to_numpy()
        d['Vxc_PBE-PBE_scf'] = d['Vxc_Exact'].to_numpy()
        expect('N=1, but exact and scf are the same array',
               d, True, fails, n_elec=1)
    finally:
        _CFG.guide_lda_core = _before

    df = clean_profile()
    expect('T1 clean profile', df, False, fails)

    d = clean_profile()
    d['Vxc_LDA_exact'] = d['Vxc_Exact'] * 0 + 1.5
    d['Vxc_LDA_scf'] = d['Vxc_LDA_exact']          # <- D1
    expect('T2 two identical columns (D1)', d, True, fails)

    d = clean_profile()
    d.loc[5, 'Vxc_Exact'] = np.nan
    expect('T3 non-finite value', d, True, fails)

    d = clean_profile()
    d.loc[5, 'Rho_Exact'] = -1e-9
    expect('T4 negative density', d, True, fails)

    d = clean_profile()
    d.loc[5, 'Z_Bohr'] = d.loc[90, 'Z_Bohr']
    expect('T5 non-monotone Z axis', d, True, fails)

    d = clean_profile(n_elec=1)
    d['Vxc_Exact'] = -d['V_Hartree']
    expect('T6a one-electron identity satisfied', d, False, fails)
    d = clean_profile(n_elec=1)
    d['Vxc_Exact'] = -d['V_Hartree'] + 1e-3
    expect('T6b one-electron identity broken', d, True, fails)

    d = clean_profile()
    d['Vc_Exact'] = d['Vxc_Exact'] + 0.5 * d['V_Hartree']
    expect('T7a Vc_Exact definition satisfied', d, False, fails)
    d['Vc_Exact'] = d['Vc_Exact'] + 1e-6
    expect('T7b Vc_Exact definition broken', d, True, fails)

    # ---- T9: V_Guide == V_FA is the guide BEING Fermi-Amaldi ------------
    # Regression for the bug that made a fresh `python main.py` write ZERO
    # npz: with CFG.guide_lda_core = False (the released configuration) the
    # guide IS -v_H/N, so V_Guide and V_FA coincide bit for bit at every
    # geometry and the duplicate-column check fired on all of them. The
    # exemption must be conditional on the flag, not unconditional.
    from config import CFG as _CFG
    _antes = _CFG.guide_lda_core
    try:
        d = clean_profile()
        d['V_Guide'] = d['V_FA']
        _CFG.guide_lda_core = False
        expect('T9a Fermi-Amaldi guide: V_Guide == V_FA is legal', d, False,
               fails)
        # ...and with the LDA-core guide ON the same file is a bug (D4): the
        # guide must have moved the core, so identical columns mean the flag
        # never reached the inversion.
        _CFG.guide_lda_core = True
        d = clean_profile(n_elec=14)
        d['V_Guide'] = d['V_FA']
        expect('T9b LDA-core guide on but V_Guide unmoved', d, True, fails)
        # and the honest case with the guide on: the columns differ.
        d = clean_profile(n_elec=14)
        d['V_Guide'] = d['V_FA'] - 0.3
        expect('T9c LDA-core guide on and V_Guide moved', d, False, fails)
    finally:
        _CFG.guide_lda_core = _antes

    # ---- T8: a failing profile must not reach disk ----------------------
    print('save_line_csv:')
    root = tempfile.mkdtemp(prefix='audit_test_')
    cwd = os.getcwd()
    try:
        os.chdir(root)
        os.makedirs('results_vxc_1D')
        good = clean_profile()
        cols = {c: good[c].to_numpy() for c in good.columns
                if c not in ('Z_Bohr', 'Z_Angstrom')}
        cols['Vxc_A'] = np.full(len(good), 2.0)
        cols['Vxc_B'] = np.full(len(good), 2.0)          # identical pair
        try:
            ex.save_line_csv('TEST', 1.0, good['Z_Bohr'].to_numpy(), cols)
            print('  FAIL T8 save_line_csv wrote a failing profile')
            fails.append('T8')
        except AssertionError:
            written = os.listdir('results_vxc_1D')
            if written:
                print(f'  FAIL T8 it raised but the file exists: {written}')
                fails.append('T8')
            else:
                print('  ok   T8 failing profile raised and wrote nothing')
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)

    print()
    if fails:
        print('FAILURES:', ', '.join(fails))
        return 1
    print('ALL PASS — the deposit audit runs at the write (task 5.3)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
