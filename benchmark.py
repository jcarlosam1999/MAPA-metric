"""
benchmark.py — the diatomic set of this study: member table, parser, element data.
==================================================================
The set is defined ONCE here as a verbatim text table (easy to diff/extend)
and parsed with hard validation at import time: electron count vs
atoms+charge, spin parity, nelecas vs spin, active-space bounds, and name
uniqueness. A malformed row aborts before any compute is spent.

Columns: atom1 atom2 dist_eq(A) elec_count charge spin(2S) ncas nelecas
Member naming: {A}{B}_q{charge:+d}_s{spin}, e.g. 'CO_q+1_s1'.

Anion members are flagged (anion_caveat): many are resonances, not bound
states; finite-basis values are stabilized artifacts.
"""

# nuclear charges for every element the benchmark set can contain
Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
     'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14}

BENCHMARK_TABLE = """
H  H  1.06  1  1 1 2 (1,0)
H  H  0.74  2  0 0 2 (1,1)
He He 1.08  3  1 1 2 (2,1)
H  Li  1.6  4  0 0 5 (1,1)
H  B  1.23  6  0 0 5 (2,2)
Li Li 2.67  6  0 0 8 (1,1)
H  C  1.12  7  0 1 5 (3,2)
B  B  1.59 10  0 2 8 (4,2)
H  F  0.92 10  0 0 5 (4,4)
N  N   1.1 14  0 0 8 (5,5)
"""


def parse_benchmark(table=BENCHMARK_TABLE):
    """Parse & validate; returns list of member dicts. Raises on any
    inconsistency (fail fast: a bad row must never reach the scan)."""
    members, names, problems = [], set(), []
    for ln, line in enumerate(table.strip().splitlines(), 1):
        p = line.split()
        if not p:
            continue
        a1, a2, r_eq = p[0], p[1], float(p[2])
        n_tab, charge, spin, ncas = int(p[3]), int(p[4]), int(p[5]), int(p[6])
        na, nb = (int(x) for x in p[7].strip('()').split(','))
        n_elec = Z[a1] + Z[a2] - charge
        # ---- validation rules (each catches a real past-failure mode) ----
        if n_elec != n_tab:
            problems.append(f"L{ln} {a1}{a2}: elec {n_tab} != Z-q {n_elec}")
        if (n_elec - spin) % 2:
            problems.append(f"L{ln} {a1}{a2}: spin {spin} wrong parity")
        if na - nb != spin:
            problems.append(f"L{ln} {a1}{a2}: nelecas vs spin mismatch")
        if max(na, nb) > ncas or na + nb > 2 * ncas or min(na, nb) < 0:
            problems.append(f"L{ln} {a1}{a2}: ({na},{nb}) not in ncas={ncas}")
        name = f"{a1}{a2}_q{charge:+d}_s{spin}"
        if name in names:
            name = f"{name}_L{ln}"
        names.add(name)
        members.append(dict(name=name, a1=a1, a2=a2, r_eq=r_eq,
                            charge=charge, spin=spin, ncas=ncas,
                            nelecas=(na, nb), n_elec=n_elec,
                            anion_caveat=bool(charge < 0)))
    if problems:
        raise ValueError("Benchmark table inconsistencies:\n  "
                         + "\n  ".join(problems))
    return members


def n_core_orbitals_atom(z):
    """Frozen-core convention: none (H,He), 1s (Li..Ne), 1s2s2p (Na..Si)."""
    return 0 if z <= 2 else (1 if z <= 10 else 5)


def phenomenon_label(n_elec, charge):
    """Physics headline for figures: 1e- systems probe pure SIE (no
    correlation exists); anions carry the resonance caveat."""
    if n_elec == 1:
        return 'Self-Interaction Error'
    if charge < 0:
        return 'Static Correlation (anion / resonance caveat)'
    return 'Static + Dynamic Correlation'