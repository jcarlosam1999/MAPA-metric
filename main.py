"""
main.py — control panel of the Wu-Yang inversion pipeline.
=================================================================
This is the only file that needs editing for a run: every setting below
mutates the shared config.CFG object that all modules read. Launch with:
 
    python main.py
 
PROCESS DIAGRAM
---------------
  benchmark.py (member table,         main.py  <-- parameters set here
  validated at import)              |
        |                           v
        |                    config.CFG (shared by every module)
        v                           |
   parse_benchmark + preflight  <---------+
        |
        v  per member: molecules.make_distance_grid(Re)
        v  per geometry R  ..................... [pipeline.process_geometry]
   [1] reference.py : SCF(Newton fallback) -> CASSCF(benchmark CAS) -> NEVPT2
                      -> ladder: CASCI+m (CASSCF orbitals) -> FCI if it
                      fits the RAM budget  => density + consistent energy
   [2] grid fields  : rho, grad rho, tau, spin densities   (molecules.py)
   [3] inversion.py : Fermi-Amaldi guide -v_H/N -> Wu-Yang inversion
                      (pruned potential basis, LBFGS+Newton, adaptive
                      kappa continuation, cold-restart guards)
   [4] baselines.py : LDA/LSDA potentials; PBE potential (closed shell,
                      + EXPERIMENTAL spin-polarized); spin-correct Exc
   [5] metrics.py   : DeltaI_bar (total & per spin), consistent/NEVPT2
                      Exc pair, E_c_dyn, Levy-Perdew virial, zero-force
   [6] plotting.py / exporters.py : 1D profiles, figures, grid npz,
                      row.json + warm-start persistence (RESUME-able)
        |
        v
  results_energies/  results_vxc_1D/  results_grid_data/  results_convergence/
        |
        +--> aggregate figures + stats
        +--> validate_reference_fci.py / _dmrg.py   (reference tiers)
        +--> atomic_references.py   (fragment anchors, size consistency)
 
SHAKEDOWN BEFORE THE FULL SCAN (1-3 weeks on a 4-core box): set
ONLY_MEMBERS to one light and one heavy member, check that
ref_density_level reads FCI / CASCI+{m}o, that virial_gap_Eh is small,
and that no [FAIL] lines appear; then empty the list and launch.
"""
 
# ============================================================================
# THREADING — must run BEFORE numpy/pyscf import (BLAS reads these at load time)
# ============================================================================
# Some OpenBLAS/OpenMP builds deadlock non-deterministically inside PySCF's
# dense linear algebra (observed: certain members hang with the CPU idle,
# immune to Ctrl-C because the block is in native code below the Python
# interpreter, killable only with SIGKILL). Pinning the numerical libraries to
# a single thread removes the deadlock; for systems this small the dense-BLAS
# speed cost is negligible. If a large run later needs threads back, raise
# these once the deadlock source is identified.
import atexit
import datetime
import os
import re
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
 
import json
 
from config import CFG, make_dirs, run_metadata
 
# runtime thread pin too (some PySCF ops read this rather than the env var)
try:
    from pyscf import lib as _pyscf_lib
    _pyscf_lib.num_threads(1)
except Exception:                                            # noqa: BLE001
    pass
 
# ============================================================================
# PARAMETERS — edit here, nowhere else
# ============================================================================
# ---- chemistry --------------------------------------------------------------
CFG.benchmark_basis = 'aug-cc-pVTZ'     # homogeneous benchmark set basis (anions need aug-)
CFG.grid_level = 3                # PySCF Becke grid level
 
# ---- reference ladder -------------------------------------------------------
CFG.fci_auto_density = True       # best affordable variational density
CFG.fci_budget_gb = 4.0           # determinant RAM budget for the ladder
 
# ---- inversion engine -------------------------------------------------------
CFG.kappa_start = 1e-4            # HBY smoothness; continuation start
CFG.adaptive_kappa = True         # kappa/10 until density target
CFG.density_target = 5e-4         # e-: 'good inversion' criterion
CFG.unified_ensemble = True       # ensemble machinery at every closed-shell
                                  # R: no RKS->ENSEMBLE seam in DeltaI
CFG.unified_ensemble_openshell = True   # open-shell (spin!=0) also runs the
                                  # per-channel ensemble (fixed fractional
                                  # natural occupations), NOT aufbau UKS:
                                  # one homogeneous framework across the whole
                                  # benchmark. Validated on H2+ (bit-identical
                                  # to the prior path) and opens the CH gap
                                  # 0->0.042.
CFG.prune_tol = 1e-8              # potential-basis pruning (raise to 1e-7
                                  # if a heavy member reports MARGINAL)
 
# ---- scan geometry ----------------------------------------------------------
# single clean grid: dense bond-breaking zone, tapering tail (see
# molecules.make_distance_grid). ~30-34 unique R per member, no duplicates.
CFG.n_compressed, CFG.n_breaking = 5, 10
CFG.r_stretch_abs = 3.8              # Angstrom beyond Re (tail end)
 
# ---- optional physics -------------------------------------------------------
CFG.include_pbe = True            # closed-shell PBE potential [EXPERIMENTAL]
CFG.include_pbe_spin = True       # open-shell PBE potential   [EXPERIMENTAL]
CFG.bs_diagnostic = False         # False => ~2.7x faster full benchmark
CFG.ip_validation_members = []    # e.g. ['HF_q+0_s0'] for eps_HOMO vs -IP
 
# ---- exports & operations ---------------------------------------------------
CFG.save_ao_grid = True           # ao_value float32 in npz (grid featurizer)
CFG.save_eri = False              # nao^4 tensor: keep off, recompute in JAX
CFG.resume = True                 # reload finished rows + warm-start chain
 
# ---- functional evaluation (control room) -----------------------------------
# The heavy pipeline computes the functional-AGNOSTIC exact side once. After it
# finishes, evaluate DeltaI_bond for these functionals over the saved npz —
# no inversion rerun. Add any libxc functional here; Tier-1 (LDA/GGA) are
# validated, meta-GGA/hybrids need their convention validated first.
CFG.run_functional_postpass = True
CFG.functional_postpass_force = True   # True: re-score even if the
                                       # columns are already on disk

# ---- validations (Supporting Information) ------------------------------------
# Every claim in the paper comes out of this run. Each validation is separately
# switchable; the annotated cost is for a 4-core desktop. The cheap ones are on,
# the expensive ones are off — turn them on when preparing the final numbers.
CFG.run_validations = True                    # master switch
CFG.validate_grid_resolution = True           # seconds  (CSV post-processing)
CFG.validate_functional_derivative = True     # minutes  (no inversion)
CFG.validate_kappa_invariance = True          # ~30 min  (H2, 3R x 3 kappa)
CFG.validate_kappa_wide_range = True         # ~15 min  (H2, 1e-9..1e-5)
CFG.validate_ensemble_seam = True            # ~20 min  (H2, seam exposed)
CFG.validate_kappa_invariance_heavy = True   # ~4 h     (N2, 3R x 3 kappa)
CFG.validate_fine_grid_scan = True          # ~2-3 h   (fine H2 scan)
CFG.functionals_to_eval = ['LDA,VWN', 'PBE,PBE']   # e.g. add 'SCAN,SCAN' once
                                                   # its tau convention is set
CFG.functional_density_modes = ('exact', 'scf')    # both = error cancellation
 
# ---- run scope ----------------------------------------------------------------
ONLY_MEMBERS = [
    ]                 # [] runs every member of the set; a shorter
                      # list restricts the run to those members

# ---- parallel sharding (machine-usable 2-core run) ----------------------------
# Run several INDEPENDENT processes, each taking a cost-balanced slice of the
# benchmark. Heavy systems are distributed evenly across shards (snake draft on
# estimated cost ~ n_elec^1.29) so all shards finish at about the same time.
# Within a shard, systems run CHEAPEST-FIRST so light systems (and any early
# failures) surface fast. Example for 2 cores, machine stays usable:
#     Terminal A:  SHARD_ID=0  ->  nice -n 10 python main.py
#     Terminal B:  SHARD_ID=1  ->  nice -n 10 python main.py
# Each process resumes independently; if one crashes the other continues.
SHARD_ID = 0                      # which shard THIS process runs (0-based)
SHARD_COUNT = 1                   # total shards (1 = no sharding, full set)
# ============================================================================
 
 
def _cost_balanced_shard(members):
    """Assign members to shards by a snake draft on estimated cost, then return
    this process's shard ordered cheapest-first. Physics-cost model matches the
    timing bench (~n_elec^1.29). Deterministic: every process computes the same
    assignment and picks its own SHARD_ID."""
    def cost(m):
        return m['n_elec'] ** 1.29          # timing-bench model: t ~ ne^1.29
    ranked = sorted(members, key=cost, reverse=True)      # heaviest first
    shards = [[] for _ in range(SHARD_COUNT)]
    loads = [0.0] * SHARD_COUNT
    for m in ranked:                                      # greedy least-loaded
        i = min(range(SHARD_COUNT), key=lambda k: loads[k])
        shards[i].append(m)
        loads[i] += cost(m)
    mine = sorted(shards[SHARD_ID], key=cost)             # cheapest-first
    return mine, loads
 
 
_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


class _Tee:
    """Duplicate a stream into the run log.

    Everything printed anywhere in the package -- pipeline progress, warnings,
    the post-pass, the validations and any traceback -- lands in the log, so a
    single file is the complete record of what a run did. Terminal control
    sequences are stripped on the way to the file: progress-bar repaints
    (carriage returns) and the cursor-movement codes tqdm emits belong on a
    terminal, not in a record meant to be read months later.
    """

    def __init__(self, stream, handle, drop_progress=False):
        self._stream = stream
        self._handle = handle
        self._drop = drop_progress

    def write(self, s):
        n = self._stream.write(s)
        if self._drop and '\r' in s and '\n' not in s:
            return n                        # bare progress repaint: console only
        clean = _ANSI_RE.sub('', s).replace('\r', '')
        if clean:
            try:
                self._handle.write(clean)
                self._handle.flush()
            except (OSError, ValueError):
                pass              # logging must never break a running scan
        return n

    def flush(self):
        self._stream.flush()
        try:
            self._handle.flush()
        except (OSError, ValueError):
            pass

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()


def _install_log_tee(path):
    """Route stdout and stderr into `path` as well as the console."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    handle = open(path, 'a', buffering=1, encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle, drop_progress=True)
    CFG.log_is_teed = True
    atexit.register(handle.close)
    return handle


def _log_header():
    """Write the run's identity at the top of the log: when, what, and how."""
    import platform
    print("=" * 78)
    print("RUN " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 78)
    print(f"  python        : {platform.python_version()} on {platform.system()}")
    try:
        from pyscf import __version__ as _pv
        print(f"  pyscf         : {_pv}")
    except Exception:                                        # noqa: BLE001
        pass
    print(f"  basis         : {CFG.benchmark_basis}")
    print(f"  members       : {', '.join(ONLY_MEMBERS) if ONLY_MEMBERS else 'all'}")
    print(f"  functionals   : {', '.join(map(str, CFG.functionals_to_eval))}")
    print(f"  density modes : {', '.join(CFG.functional_density_modes)}")
    if SHARD_COUNT > 1:
        print(f"  shard         : {SHARD_ID + 1}/{SHARD_COUNT}")
    on = [k[len('validate_'):] for k in sorted(dir(CFG))
          if k.startswith('validate_') and getattr(CFG, k)]
    print(f"  validations   : {', '.join(on) if on else 'none'}"
          if CFG.run_validations else "  validations   : disabled")
    print("=" * 78)



def main():
    make_dirs()
    # One log per RUN, stamped with the moment the run started, so successive
    # runs never overwrite each other and a log can be matched to its results
    # by time alone. Shards additionally get their own file: two parallel
    # processes must not share one log (interleaved writes corrupt the record).
    run_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    CFG.log_path = (f"results_logs/run_{run_stamp}_shard{SHARD_ID}.log"
                    if SHARD_COUNT > 1 else f"results_logs/run_{run_stamp}.log")
    print(f"[setup] logging this run to {CFG.log_path}")
    _install_log_tee(CFG.log_path)
    _log_header()
    with open("results_grid_data/run_metadata.json", "w") as f:
        json.dump(run_metadata(), f, indent=2)
    # import AFTER CFG mutation so heavy modules see the final settings
    from pipeline import run_pipeline
    # Atomic-IP cache for the Koopmans test: build it once up front if
    # it is missing, so a fresh clone reproduces the asymptotic
    # validation without a separate manual step.
    try:
        import atomic_ip as aip
        if not os.path.exists(aip.CACHE):
            print('[setup] atomic IP cache not found; building '
                  f'{aip.CACHE} ...')
            for _el in ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',
                        'F', 'Ne', 'Na', 'Mg', 'Al', 'Si'):
                aip.atomic_ip(_el, CFG.benchmark_basis, 'nevpt2')
            print('[setup] atomic IP cache ready.')
        # The calibration table behind the Koopmans test goes into the log,
        # so every number the manuscript quotes is in the run record and not
        # only in the JSON cache. Cached entries make this nearly free.
        aip.report_vs_experiment(basis=CFG.benchmark_basis, level='nevpt2')
    except Exception as _e:                              # noqa: BLE001
        print(f'[setup] atomic IP cache unavailable ({_e}); '
              'Koopmans test will report NaN.')

    if ONLY_MEMBERS:
        run_pipeline(only=ONLY_MEMBERS)
    elif SHARD_COUNT > 1:
        from benchmark import parse_benchmark
        mine, loads = _cost_balanced_shard(parse_benchmark())
        names = [m['name'] for m in mine]
        print(f"[shard {SHARD_ID}/{SHARD_COUNT}] {len(names)} systems, "
              f"est. cost {loads[SHARD_ID]:.0f} of {sum(loads):.0f} "
              f"(balance: {min(loads):.0f}-{max(loads):.0f})")
        run_pipeline(only=names)
    else:
        run_pipeline(only=None)
 
    # control-room post-pass: score functionals against the saved exact side.
    # Idempotent AND cheap on repeat runs: functionals whose columns are already
    # present in every dissociation CSV are skipped, so re-running main.py after
    # adding one functional scores only that one.
    if CFG.run_functional_postpass:
        from functional_eval import (run_functional_evaluation,
                                     pending_functionals)
        if CFG.functional_postpass_force:
            pending, done = list(CFG.functionals_to_eval), []
        else:
            pending, done = pending_functionals(
                CFG.functionals_to_eval,
                density_modes=CFG.functional_density_modes)
        print("\n=== Functional evaluation post-pass ===")
        if CFG.functional_postpass_force:
            print("  forced re-score (functional_postpass_force = True)")
        if done:
            print(f"  already scored, skipping: {', '.join(map(str, done))}")
        if pending:
            print(f"  scoring: {', '.join(map(str, pending))}")
            run_functional_evaluation(
                pending, density_modes=CFG.functional_density_modes)
            print("  see results_energies/Functional_Comparison.csv")
        else:
            print("  nothing to do: every requested functional is on disk.")

    # Validations behind the Supporting Information. They read what this run
    # just produced, so every number in the paper comes out of `python main.py`.
    if CFG.run_validations:
        import validation
        print("\n=== Validations (Supporting Information) ===")
        # stdout is teed into the run log, so plain print is enough here.
        validation.run_all(log=print,
                           deriv_member=CFG.validation_openshell_member)
 
 
if __name__ == "__main__":
    main()