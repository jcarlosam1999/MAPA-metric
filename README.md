# A Regularization-Invariant Real-Space Metric for DFT Error — code and data

Reference implementation accompanying the manuscript. Everything reported in
the paper is produced by this package: a single entry point (`main.py`) runs
the Wu–Yang inversion of correlated reference densities for the ten diatomic
systems of the study, evaluates the functionals on the recovered exact
potential, and writes every table value as a column in the output CSVs.

## Requirements

- Python ≥ 3.10
- [PySCF](https://pyscf.org/) ≥ 2.3
- numpy, scipy, pandas, matplotlib

```
pip install pyscf numpy scipy pandas matplotlib
```

The reference calculations are the expensive part. On a 4-core desktop the
full ten-system scan takes on the order of days; a single light system
(H2+, H2) finishes in minutes. Runs are resumable: finished geometries are
reloaded from disk, so an interrupted scan continues where it stopped.

## Reproducing the study

```
python main.py
```

`main.py` is the control panel and is meant to be read before it is run. It

1. builds the atomic ionization-potential cache used by the Koopmans test
   (once, if `results_analysis/atomic_ip.json` is absent);
2. runs the heavy pipeline (`pipeline.py`): for every system and internuclear
   distance, it builds the correlated reference density, inverts it with the
   Wu–Yang engine, and writes the **functional-agnostic exact side** (exact
   v_xc on the grid, reference density, energy decomposition, N_unpaired,
   inversion diagnostics);
3. runs the functional post-pass (`functional_eval.py`), which scores each
   functional against the saved exact side **without re-running the
   inversion**, and merges the metric columns into each dissociation CSV.

4. runs the three validations of the Supporting Information (`validation.py`).

To run a subset, edit `ONLY_MEMBERS` at the top of `main.py`.

Re-running `main.py` is cheap and safe. Finished geometries are reloaded from
disk rather than recomputed; functionals whose columns are already present in
every dissociation CSV are skipped by the post-pass, so adding one functional
to `CFG.functionals_to_eval` and re-running scores only that functional. A
partially merged functional is redone rather than trusted.

Each run writes its own log, `results_logs/run_<YYYYMMDD_HHMMSS>.log`, stamped
with the moment the run started, so a log is always matched to the results it
produced. The log is a **complete** record: `main.py` routes both stdout and
stderr into it, so pipeline progress, grid warnings, the post-pass, the
validations and any traceback all appear there as well as on the console. It
opens with a header identifying the run — timestamp, Python and PySCF
versions, basis, members, functionals, density modes and which validations are
switched on. Only progress-bar repaints are kept out, to keep it readable.

Resumed geometries are replayed from their saved rows and print exactly what
they printed when first computed — the status line, the grid-electron warning
and the one-electron analytic check alike — so a log from a resumed run is
indistinguishable from a log of a full one. Each system opens with a single
statement of how many of its geometries were already on disk. The atomic-IP
calibration table behind the Koopmans test is printed too. Together with the
per-system CSVs, the log is a complete record of every number the manuscript
rests on.

## Where each result comes from

| Manuscript item | Produced by | File / column |
|---|---|---|
| Metric values, per geometry | `main.py` | `results_energies/<system>_Dissociation_Curve.csv`, columns `DeltaI_bar_<functional>_<mode>_Eh` and `DeltaI_resp_<functional>_<mode>` |
| Confidence stratum (OK/MARGINAL/FAIL) | `main.py` | same CSV, columns `inv_status`, `inv_density_error_e`, `ghost_leak_e` |
| Unpaired-electron index and its saturation | `main.py` (`richards_fit.py`) | same CSV, column `N_unpaired`; fit in `results_analysis/<system>_richards.json` |
| Spectroscopic constants | `main.py` (`morse_fit.py`) | `results_analysis/<system>_morse.json` |
| Functional comparison and error cancellation | `main.py` | `results_energies/Functional_Comparison.csv` and `..._ErrorCancellation.csv` |
| Koopmans test against experiment | `atomic_ip.py` | `results_analysis/atomic_ip.json` |
| 1D potential profiles, exact **and** per functional | `main.py` | `results_vxc_1D/<system>_Vxc_1D_R<r>.csv`, columns `Vxc_Exact` and `Vxc_<functional>_<mode>` |
| Regularization (κ) invariance of the response metric | `main.py` (`validation.py`) | run log, `results_logs/run_<timestamp>.log` |
| Grid-resolution robustness of the metric peak | `main.py` (`validation.py`) | same log |
| Functional-derivative identity, spin-polarized route | `main.py` (`validation.py`) | same log |
| Cost of a mid-curve scheme switch (the seam) | `main.py`, toggle on | same log |
| Fine-grid check of the metric peak | `main.py`, toggle on | same log |

Re-scoring is normally handled by `main.py` itself (see above). For a
targeted re-score of selected systems, without touching the inversion:

```
python rerun_postpass.py --only HH_q+0_s0 NN_q+0_s0 --modes exact scf
```

## Validations

Five checks reproduce the Supporting Information. Each is switched
independently at the top of `main.py`, annotated with its cost; the cheap ones
run by default and the expensive ones are off until you want the final
numbers:

```python
CFG.validate_grid_resolution = True           # seconds  (CSV post-processing)
CFG.validate_functional_derivative = True     # minutes  (no inversion)
CFG.validate_kappa_invariance = True          # ~30 min  (H2, 3R x 3 kappa)
CFG.validate_kappa_wide_range = False         # ~15 min  (H2, 1e-9..1e-5)
CFG.validate_ensemble_seam = False            # ~20 min  (H2, seam exposed)
CFG.validate_kappa_invariance_heavy = False   # ~4 h     (N2, 3R x 3 kappa)
CFG.validate_fine_grid_scan = False           # ~2-3 h   (fine H2 scan)
```

Two of them deserve a note. The **seam** check measures the discontinuity a
mid-curve scheme switch would introduce: production never switches, so that
number has to be produced on purpose, and the check does it by re-inverting a
few adjacent geometries with the unified scheme disabled. The **fine-grid**
check recomputes a light system on a denser R grid and compares the metric
peak with the production grid; it runs in a scratch directory, so production
results are never touched.

They can also be run on their own:

```
python validation.py --only response      # grid|response|derivative|seam|finegrid
```

## The confidence gate

The metric is defined on the density recovered by the inversion, so it is
reported only where that inversion is reliable. Every geometry carries a
status assigned **solely** from its density-reproduction error
`inv_density_error_e`: `OK`, `MARGINAL`, or `FAIL`. Thresholds are stratified
by physics rather than by electron count (`config.py`, `density_fail_for`):
a base threshold, a relaxed one for many-electron systems, and a third for
diffuse densities detected through the fraction of the error leaking to the
far field. Failed geometries are written to disk like any other — they are
not deleted — so the reliable/failed contrast reported in the manuscript can
be reproduced directly from the CSVs.

## Inversion framework

A single ensemble Kohn–Sham framework is used throughout, with fixed
fractional occupations taken from the natural occupations of the reference
density. Closed shells invert against one ensemble (`inv_mode` `ENSEMBLE` or
`ENSEMBLE_NOSPIN`); open shells apply the same machinery per spin channel
(`ENSEMBLE_OS`). No scheme switch occurs mid-curve, which is what prevents a
spurious step in the metric. The legacy single-determinant paths are retained
in the code, disabled, and labelled `*_LEGACY` so that the production route is
unambiguous in every log line.

## Module map

| Module | Role |
|---|---|
| `main.py` | control panel: configuration, system selection, run order |
| `config.py` | all tunable parameters and the confidence thresholds |
| `benchmark.py` | the ten-system table, parsed and validated at import |
| `molecules.py` | molecule builders, SCF helpers, grids, electrostatics |
| `reference.py` | the graded reference ladder (SCF → CASSCF → CASCI → FCI) |
| `inversion.py` | the Wu–Yang inversion engine |
| `metrics.py` | the potential metric and exact-physics diagnostics |
| `response_metric.py` | the regularization-invariant response metric |
| `baselines.py` | DFA potentials and energies on the reference density |
| `functional_eval.py` | functional scoring in exact and self-consistent modes |
| `exporters.py` | everything written to disk (CSV, npz, row json) |
| `pipeline.py` | orchestration of one geometry, then the whole set |
| `plotting.py` | per-geometry figures |
| `atomic_ip.py` | atomic and fragment ionization potentials (Koopmans) |
| `richards_fit.py`, `morse_fit.py` | end-of-system curve characterization |
| `rerun_postpass.py` | re-score functionals without re-inverting |
| `PHYSICS_FINDINGS.md` | numbered record of the physical results and design decisions the code rests on; source comments refer to its items by number |
| `validation.py` | the three validations behind the SI, run automatically by `main.py` (or standalone: `python validation.py --only grid\|response\|derivative`) |

## Output layout

```
results_energies/     <system>_Dissociation_Curve.csv   (all metric columns)
                      Functional_Comparison*.csv
results_grid_data/    <system>_R<r>.npz                 (exact side, reusable)
                      <system>_R<r>.row.json            (per-geometry record)
results_vxc_1D/       1D v_xc profiles and figures
results_analysis/     atomic_ip.json, richards/morse fits
results_convergence/  convergence diagnostics
results_logs/         per-run logs
```

## License and citation

Released under the MIT License (see `LICENSE`).

### What goes in the dataset record

The software record holds this repository. The dataset record holds what a
full run produces, which is far larger and is what makes the results
verifiable without repeating days of computation:

```
results_energies/     per-system dissociation CSVs (all metric columns)
                      Functional_Comparison*.csv
results_grid_data/    per-geometry .npz (exact side) and .row.json
results_vxc_1D/       1D potential profiles, exact and per functional
results_analysis/     atomic_ip.json, Richards and Morse fits
results_logs/         run logs
```

The `.npz` files dominate the size; if the deposition limit becomes a problem,
they are the part to compress or subset, since every table in the article can
be rebuilt from the CSVs alone.
