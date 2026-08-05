"""
atomic_ip.py — fragment ionization potentials for the Koopmans test
===================================================================
The Koopmans test of the inversion asks whether the KS HOMO eigenvalue tends to
minus the fragment ionization potential at dissociation:

        eps_HOMO_KS(R -> inf)  ->  -IP(fragment atom).

The reference IP is therefore an ATOMIC property of each dissociation fragment,
NOT a per-geometry molecular quantity. Computing it per R (via molecular dSCF)
is both expensive and wrong in the tail, where the molecular cation is not the
relevant reference. The correct, cheap reference is:

        IP(atom) = E(cation) - E(neutral)          (dSCF, one pair per element)

computed ONCE per element in the SAME basis and at a correlated level
consistent with the reference ladder, then cached. For a homonuclear A2 the
tail reference is IP(A); for a heteronuclear AB the bond dissociates to A + B
and the HOMO localizes on whichever fragment has the lower IP (the last
electron leaves the less tightly bound atom), so the reference is min(IP(A),
IP(B)). Both fragment IPs are returned so the caller can pick correctly.

This is a small, standalone calculation: a handful of atoms, each a neutral +
cation single point. Results are cached to results_analysis/atomic_ip.json so a
full sweep computes each element's IP once.

Usage (standalone, to pre-populate the cache):
    python atomic_ip.py --elements H He Li B C N O F

Or programmatically from the pipeline:
    from atomic_ip import fragment_ip
    ip_a, ip_b = fragment_ip('C', 'O', basis='aug-cc-pVTZ')
"""

import argparse
import json
import os

import numpy as np

OUT_DIR = "results_analysis"
CACHE = f"{OUT_DIR}/atomic_ip.json"
HA2EV = 27.211386245988

# ground-state spin multiplicity (2S) of neutral atoms and their cations, for
# the first three rows. The cation loses one electron from the neutral, so its
# spin differs by 1. These are the experimental ground-state terms.
NEUTRAL_SPIN = {'H': 1, 'He': 0, 'Li': 1, 'Be': 0, 'B': 1, 'C': 2,
                'N': 3, 'O': 2, 'F': 1, 'Ne': 0, 'Na': 1, 'Mg': 0,
                'Al': 1, 'Si': 2, 'P': 3, 'S': 2, 'Cl': 1, 'Ar': 0}
CATION_SPIN = {'H': 0, 'He': 1, 'Li': 0, 'Be': 1, 'B': 0, 'C': 1,
               'N': 2, 'O': 3, 'F': 2, 'Ne': 1, 'Na': 0, 'Mg': 1,
               'Al': 0, 'Si': 1, 'P': 2, 'S': 3, 'Cl': 2, 'Ar': 1}


def _load_cache():
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:                                    # noqa: BLE001
            return {}
    return {}


def _save_cache(cache):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CACHE, 'w') as f:
        json.dump(cache, f, indent=2)


def _atom_energy(element, charge, spin, basis, level):
    """Total energy of a single atom (neutral or cation).

    Levels: 'scf'; 'casci' (minimal valence CASCI -- biases IP LOW ~1-1.75 eV
    via unbalanced cation/neutral correlation); 'ccsdt' (CCSD(T), light-atom
    gold standard, ~0.1-0.2 eV); 'nevpt2' (CASSCF + strongly-contracted
    NEVPT2, the molecular-reference level -- method-consistent, may retain
    some imbalance on a minimal space; falls back to CCSD(T)). Default
    'nevpt2' for pipeline consistency; the __main__ experiment check reveals
    whether CCSD(T) should be preferred. Returns (energy_Ha, method_used).
    """
    from pyscf import gto, scf

    mol = gto.M(atom=f'{element} 0 0 0', basis=basis, charge=charge,
                spin=spin, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel()
    e_scf = float(mf.e_tot)
    if level == 'scf':
        return e_scf, 'UHF'

    if level == 'casci':
        try:
            from pyscf import mcscf
            ncas, nelecas = _valence_active_space(element, charge)
            cas = mcscf.CASCI(mf, ncas, nelecas)
            cas.kernel()
            return float(cas.e_tot), f'CASCI({nelecas},{ncas})'
        except Exception:                                    # noqa: BLE001
            return e_scf, 'UHF (CASCI unavailable)'

    if level == 'nevpt2':
        # CASSCF (relaxed orbitals) + strongly-contracted NEVPT2: the SAME
        # level as the molecular references. Minimal active space may retain
        # part of the cation/neutral imbalance; CCSD(T) is the fallback and
        # the experiment check decides which to trust.
        try:
            from pyscf import mcscf, mrpt
            ncas, nelecas = _valence_active_space(element, charge)
            cas = mcscf.CASSCF(mf, ncas, nelecas)
            cas.kernel()
            e_pt = mrpt.NEVPT(cas).kernel()
            return float(cas.e_tot + e_pt), f'NEVPT2/CASSCF({nelecas},{ncas})'
        except Exception:                                    # noqa: BLE001
            pass

    # default 'ccsdt' (and nevpt2 fallback): balanced correlation both sides
    try:
        from pyscf import cc
        ccsd = cc.CCSD(mf)
        ccsd.kernel()
        et = ccsd.ccsd_t()
        tag = 'CCSD(T)' if level != 'nevpt2' else 'CCSD(T) [NEVPT2 fell back]'
        return float(ccsd.e_tot + et), tag
    except Exception:                                        # noqa: BLE001
        try:
            from pyscf import mcscf
            ncas, nelecas = _valence_active_space(element, charge)
            cas = mcscf.CASCI(mf, ncas, nelecas)
            cas.kernel()
            return float(cas.e_tot), f'CASCI({nelecas},{ncas}) [fallback]'
        except Exception:                                        # noqa: BLE001
            return e_scf, 'UHF [correlated solvers unavailable]'


def _valence_active_space(element, charge):
    """(ncas, nelecas) for a modest valence CASCI of a first/second-row atom.
    Valence orbitals: 1 (H,He: 1s) or 4 (Li..Ne: 2s2p) or 4 (Na..Ar: 3s3p).
    nelecas = valence electron count after charge."""
    Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
         'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
         'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18}[element]
    ne = Z - charge
    if Z <= 2:
        return 1, max(ne, 0)                 # 1s only
    if Z <= 10:
        val = ne - 2                         # remove 1s core
        return 4, max(val, 0)                # 2s2p
    val = ne - 10                            # remove [Ne] core
    return 4, max(val, 0)                    # 3s3p


def atomic_ip(element, basis='aug-cc-pVTZ', level='nevpt2'):
    """IP(element) = E(cation) - E(neutral), in Ha. Cached per (element, basis,
    level). Returns a dict with the IP and the methods used on each side."""
    cache = _load_cache()
    key = f"{element}|{basis}|{level}"
    if key in cache:
        return cache[key]
    if element not in NEUTRAL_SPIN:
        res = {'ok': False, 'reason': f'no ground-state term for {element}'}
        cache[key] = res
        _save_cache(cache)
        return res
    try:
        e_neu, m_neu = _atom_energy(element, 0, NEUTRAL_SPIN[element],
                                    basis, level)
        e_cat, m_cat = _atom_energy(element, 1, CATION_SPIN[element],
                                    basis, level)
        ip_ha = e_cat - e_neu
        res = {'ok': True, 'element': element, 'basis': basis, 'level': level,
               'IP_Ha': float(ip_ha), 'IP_eV': float(ip_ha * HA2EV),
               'E_neutral_Ha': e_neu, 'E_cation_Ha': e_cat,
               'method_neutral': m_neu, 'method_cation': m_cat}
    except Exception as e:                                    # noqa: BLE001
        res = {'ok': False, 'element': element, 'reason': str(e)}
    cache[key] = res
    _save_cache(cache)
    return res


def fragment_ip(a1, a2, basis='aug-cc-pVTZ', level='nevpt2'):
    """Fragment IPs for a diatomic A-B. Returns (res_a1, res_a2). For the
    Koopmans tail reference use min over the finite IPs: the HOMO localizes on
    the less tightly bound fragment as the bond breaks."""
    return atomic_ip(a1, basis, level), atomic_ip(a2, basis, level)


def koopmans_reference(a1, a2, basis='aug-cc-pVTZ', level='nevpt2'):
    """The single -IP the KS HOMO should approach at dissociation: the SMALLER
    fragment IP (the last electron leaves the less bound atom)."""
    ra, rb = fragment_ip(a1, a2, basis, level)
    ips = [r['IP_Ha'] for r in (ra, rb) if r.get('ok')]
    if not ips:
        return {'ok': False, 'reason': 'no fragment IP computed'}
    ip = min(ips)
    return {'ok': True, 'IP_ref_Ha': ip, 'IP_ref_eV': ip * HA2EV,
            'minus_IP_ref_eV': -ip * HA2EV,
            'fragment_a': ra, 'fragment_b': rb}


# Experimental ground-state ionization potentials (eV), used to calibrate the
# reference protocol behind the Koopmans test reported in the manuscript.
EXP_IP_EV = {"H": 13.598, "He": 24.587, "Li": 5.392, "Be": 9.323, "B": 8.298,
             "C": 11.260, "N": 14.534, "O": 13.618, "F": 17.423, "Ne": 21.565,
             "Na": 5.139, "Mg": 7.646, "Al": 5.986, "Si": 8.152}

DEFAULT_ELEMENTS = ("H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                    "Na", "Mg", "Al", "Si")


def report_vs_experiment(elements=DEFAULT_ELEMENTS, basis=None, level='nevpt2',
                         log=print):
    """Print the atomic-IP calibration table that backs the Koopmans test.

    This is the table reported in the Supporting Information: the reference
    protocol (cation minus neutral, same basis) against experiment. Called by
    main.py so the numbers the manuscript quotes are in the run log, not only
    in the JSON cache.
    """
    if basis is None:
        from config import CFG
        basis = CFG.benchmark_basis
    log("")
    log(f"Atomic ionization potentials vs experiment ({basis}, {level}):")
    log(f"  {'el':>4} {'computed':>10} {'exp':>9} {'dev':>8}")
    log("  " + "-" * 34)
    devs = []
    for el in elements:
        r = atomic_ip(el, basis, level)
        exp = EXP_IP_EV.get(el)
        if not r.get('ok') or exp is None:
            log(f"  {el:>4} {'FAILED':>10} {'':>9} {'':>8}")
            continue
        dev = r['IP_eV'] - exp
        devs.append(abs(dev))
        log(f"  {el:>4} {r['IP_eV']:>10.3f} {exp:>9.3f} {dev:>+8.3f}")
    if devs:
        mad = sum(devs) / len(devs)
        worst = max(devs)
        el_worst = elements[devs.index(worst)]
        log("  " + "-" * 34)
        log(f"  mean |dev| = {mad:.3f} eV, max = {worst:.3f} eV ({el_worst})")
    return dict(mean_abs_dev=(sum(devs) / len(devs)) if devs else None,
                max_dev=max(devs) if devs else None, n=len(devs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--elements', nargs='+',
                    default=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',
                             'F', 'Ne', 'Na', 'Mg', 'Al', 'Si'])
    ap.add_argument('--basis', default='aug-cc-pVTZ')
    ap.add_argument('--level', default='nevpt2',
                    choices=['scf', 'casci', 'ccsdt', 'nevpt2'])
    a = ap.parse_args()
    hdr = "element"
    print(f"Atomic ionization potentials ({a.basis}, {a.level}):")
    print(f"{hdr:>8} {'IP (eV)':>10} {'method (neu/cat)':>32}")
    print('-' * 56)
    EXP_IP_EV = {"H": 13.60, "He": 24.59, "Li": 5.39, "Be": 9.32, "B": 8.30,
                 "C": 11.26, "N": 14.53, "O": 13.62, "F": 17.42, "Ne": 21.56,
                 "Na": 5.14, "Mg": 7.65, "Al": 5.99, "Si": 8.15}
    worst = 0.0
    for el in a.elements:
        r = atomic_ip(el, a.basis, a.level)
        if not r.get('ok'):
            print(f"{el:>8} {'FAILED':>10}  {r.get('reason', '')}")
            continue
        meth = r['method_neutral'] + ' / ' + r['method_cation']
        print(f"{el:>8} {r['IP_eV']:>10.3f} {meth:>32}")
        exp = EXP_IP_EV.get(el)
        if exp is not None:
            worst = max(worst, abs(r['IP_eV'] - exp))
    print(f"\ncached -> {CACHE}")
    print()
    print("Sanity check vs experiment (|dev| > 0.5 eV suspicious):")
    print(f"  {'el':>4} {'computed':>10} {'exp':>8} {'|dev|':>8}  flag")
    for el in a.elements:
        r = atomic_ip(el, a.basis, a.level)
        exp = EXP_IP_EV.get(el)
        if not r.get('ok') or exp is None:
            continue
        dev = abs(r['IP_eV'] - exp)
        flag = 'OK' if dev < 0.5 else 'CHECK (try --level ccsdt)'
        print(f"  {el:>4} {r['IP_eV']:>10.3f} {exp:>8.2f} {dev:>8.3f}  {flag}")
    if worst < 0.5:
        print(f"\n  worst deviation {worst:.3f} eV -- all within tolerance")
    else:
        print(f"\n  worst deviation {worst:.3f} eV -- investigate flagged "
              "atoms; CCSD(T) is the light-atom gold standard if NEVPT2 is biased")