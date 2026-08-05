"""
reference.py — the graded reference ladder (density & energy targets).
======================================================================
Design: PySCF's NEVPT2 provides no 1-RDM, so density
and energy references are split — and this module makes the split COHERENT:

  * Density target = best affordable VARIATIONAL level, chosen per member
    by a determinant-RAM budget:
        CASSCF(benchmark CAS)
          -> CASCI(m orbitals, on CASSCF-RELAXED orbitals; m = largest
             space fitting fci_budget_gb)     [genuine intermediate rung:
             plain CASCI on SCF orbitals is BELOW CASSCF and unreliable at
             stretched bonds, so it is never used]
          -> full-valence frozen-core FCI (the m = all case).
  * e_consistent = total energy of THAT SAME level: the variationally
    consistent partner of the inverted v_xc (its functional derivative).
  * e_ref (NEVPT2) = best total energy; E_c_dyn = E_NEVPT2 - E_CASSCF is a
    separate dynamic-correlation channel.

Everything is returned in one dict so pipeline/validation/anchor scripts
share identical semantics. Validation scripts call with fci_auto=False so
their CASSCF-vs-FCI comparisons remain meaningful.
"""

import math
import numpy as np
from pyscf import mcscf, mrpt

from config import CFG
from benchmark import n_core_orbitals_atom
from molecules import build_molecule, run_scf_reference


def ncore_frozen(mol):
    return int(sum(n_core_orbitals_atom(int(mol.atom_charge(i)))
                   for i in range(mol.natm)))


def _casci_mem_gb(m, na, nb):
    return (math.comb(m, na) * math.comb(m, nb)
            * 8 * CFG.fci_davidson_vectors / 1e9)


def _casci_cost_proxy(m, na, nb):
    """Per-iteration Davidson sigma-build cost proxy ~ N_strings * M^4, where
    N_strings = C(m,na) + C(m,nb). Tracks the contract_2e contraction against
    the M^4 active ERI -- the confirmed driver of the CASCI hang. Spin/electron
    aware, so it bounds high-spin (na,1) cases a bare orbital cap would miss."""
    return (math.comb(m, na) + math.comb(m, nb)) * m**4


def largest_affordable_casci_space(mol, ncas_benchmark):
    """Ladder selector. Returns (m, (na, nb), is_full_valence); m=None keeps
    the CASSCF density (same-space CASCI adds nothing; bigger-space CASCI on
    UNRELAXED SCF orbitals is not reliably better).

    Two constraints, BOTH necessary:
      * MEMORY: the CI vector (N_det floats x davidson_vectors) must fit
        fci_budget_gb. Governed by determinant count.
      * TIME: one contract_2e (sigma = H.c) contracts against the M^4 active
        ERI; per-iteration cost ~ N_strings * M^4 must stay under
        fci_cost_ceiling. This is the confirmed cause of the CASCI hang
        (py-spy: 100% CPU in direct_spin1.contract_2e, cycle 1 never finishing).
        Being spin/electron aware, it also bounds high-spin (na,1) members that
        a bare orbital-count cap let run away (e.g. LiC(4,1) -> M=69).
    """
    ncore = ncore_frozen(mol)
    norb_full = mol.nao - ncore
    nel = mol.nelectron - 2 * ncore
    na, nb = (nel + mol.spin) // 2, (nel - mol.spin) // 2
    if min(na, nb) < 0 or norb_full < max(na, nb):
        return None, (na, nb), False

    def ok(m):
        return (_casci_mem_gb(m, na, nb) <= CFG.fci_budget_gb
                and _casci_cost_proxy(m, na, nb) <= CFG.fci_cost_ceiling)

    # STEP-DOWN LADDER (data-driven): full FCI only for the smallest systems,
    # where it is both cheap and where the bench showed it actually helps the
    # inversion (Ne <= fci_full_max_ne). For everything else the extra CASCI
    # correlation did NOT move the inversion, so cap the space at a modest
    # casci_max_N -- cheaper, and it fixes the B2 timeout.
    m_ceiling = norb_full if mol.nelectron <= CFG.fci_full_max_ne \
        else min(norb_full, CFG.casci_max_N)

    # Full space (within the ceiling) if it satisfies memory and cost too.
    if m_ceiling >= norb_full and ok(norb_full):
        return norb_full, (na, nb), True

    # Ladder search: largest m up to m_ceiling satisfying all constraints.
    m = max(na, nb, ncas_benchmark + 1)
    if m > m_ceiling or not ok(m):
        return None, (na, nb), False
    while m < m_ceiling and ok(m + 1):
        m += 1
    return m, (na, nb), False


def run_reference(mol, ncas, nelecas, fci_auto=False):
    """One-stop reference: SCF -> CASSCF(benchmark CAS) -> NEVPT2, then (opt.)
    the ladder upgrade of the density/consistent-energy pair."""
    mf = run_scf_reference(mol)

    # -- static correlation, variational, orbital-optimized ----------------
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 100
    mc.run()
    e_cas, cas_conv = mc.e_tot, bool(mc.converged)

    # -- dynamic correlation on top (energy only; no RDM available) --------
    if mol.nelectron > 1:
        try:
            e_nevpt2 = mc.e_tot + mrpt.NEVPT(mc).kernel()
        except Exception:
            e_nevpt2 = np.nan
    else:
        e_nevpt2 = e_cas          # 1e-: CASSCF exact in basis, no dyn corr

    if mol.spin != 0:
        dma, dmb = mc.make_rdm1s()
        ref_dm, dms = dma + dmb, (dma, dmb)
    else:
        ref_dm, dms = mc.make_rdm1(), None

    # -- ladder upgrade of density + consistent energy ---------------------
    e_fci, ref_level, e_consistent, ref_space = np.nan, 'CASSCF', e_cas, ncas
    if fci_auto:
        m, nel_act, is_full = largest_affordable_casci_space(mol, ncas)
        if m is not None:
            try:
                fs = mcscf.CASCI(mf, m, nel_act)
                fs.fcisolver.max_memory = int(CFG.fci_budget_gb * 1000)
                fs.kernel(mc.mo_coeff)     # CASSCF-RELAXED orbitals: the key
                if mol.spin != 0:
                    dma, dmb = fs.make_rdm1s()
                    ref_dm, dms = dma + dmb, (dma, dmb)
                else:
                    ref_dm, dms = fs.make_rdm1(), None
                e_consistent, ref_space = fs.e_tot, m
                ref_level = 'FCI' if is_full else f'CASCI+{m}o'
                if is_full:
                    e_fci = fs.e_tot
            except Exception:
                pass                       # keep CASSCF density on failure

    e_ref = e_nevpt2 if np.isfinite(e_nevpt2) else e_consistent
    return dict(mf=mf, e_scf=mf.e_tot, e_cas=e_cas, e_nevpt2=e_nevpt2,
                e_fci=e_fci, e_ref=e_ref, e_consistent=e_consistent,
                ref_level=ref_level, ref_space=ref_space, ref_dm=ref_dm,
                dms=dms, cas_conv=cas_conv, scf_conv=bool(mf.converged))


def compute_ip(member, r_ang, e_neutral):
    """Ionization-theorem validation: IP = E(cation) - E(neutral) at the
    same CASSCF+NEVPT2 level (one alpha removed, same ncas). Enabled per
    member via CFG.ip_validation_members (doubles the cost of a point)."""
    try:
        na, nb = member['nelecas']
        if na > 0:
            nelecas_c, spin_c = (na - 1, nb), abs(member['spin'] - 1)
        else:
            nelecas_c, spin_c = (na, nb - 1), member['spin'] + 1
        mol_c = build_molecule(member['a1'], member['a2'], r_ang,
                               member['charge'] + 1, spin_c)
        if mol_c.nelectron == 0:
            return float(mol_c.energy_nuc() - e_neutral)
        ref_c = run_reference(mol_c, member['ncas'], nelecas_c)
        return float(ref_c['e_ref'] - e_neutral)
    except Exception:
        return np.nan
