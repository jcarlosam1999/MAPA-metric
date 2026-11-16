# MAPA-metric

A real-space, regularization-invariant metric for density-functional error.
Reference implementation for *A Real-Space, Regularization-Invariant Metric for
Density-Functional Error: A Proof of Concept on Fundamental Diatomics*.

```bash
pip install -r requirements.txt
python main.py
```

One command produces the benchmark, the metrics,
the Supporting Information validations, the regression tests and the
verification, and it exits non-zero if any number the paper prints fails to
come out of the run.

## Design

**Everything comes from `main.py`.** There is no second script and no manual step
between the raw calculation and a published figure. If a number
appears in the manuscript, some stage of this run produces it and some check
compares the two.

**Nothing is computed twice.** Every stage registers its output in
`artifacts.py`, and a stage whose artifact is already on disk is skipped and
says so. A full run from empty is days of compute; a second run is minutes.
Interrupting and relaunching is the normal way to use it: the geometry scan
resumes mid-scan (`CFG.resume`), the functional post-processing scores only
what is missing, and the validations skip what they already wrote. Set
`CFG.validation_force = True` to redo them anyway.

## Configuration

The top of `main.py` is the only place where the run is configured: chemistry, inversion
engine, scan geometry, which validations run, whether the self-tests and the
verification run. Everything is on by default. To turn a stage off, edit one line
there; there are no command-line flags, so a run's configuration can be
compared with a diff.

The expensive stages: the N2 kappa sweep takes about four hours, the fine H2
scan a few hours, and the basis-convergence control re-inverts one system in a
larger basis. They are on because the paper reports them.

## Layout

```
main.py               configuration and entry point
config.py             every setting, with its default and why
artifacts.py          what each stage leaves on disk; the skip logic
molecules.py benchmark.py reference.py     systems and reference densities
inversion.py          Wu-Yang inversion with the HBY smoothness norm
metrics.py response_metric.py              the two metrics
functional_eval.py    functional potentials on the grid; the post-processing
pipeline.py exporters.py                   the scan and what it writes
validation.py         the Supporting Information validations
validation_extra.py   the four controls of SI S11.4, S11.5 and S13
density_comparison.py the response metric against the SCF density difference (SI S16)
verification/         does this run reproduce the paper?
tests/                regression tests, run as a stage
tools_import_tree.py  proves the repository contains nothing unreachable
tools_huge_dv.py      what the [HUGE-DV] warning actually costs the metric
```

`python tools_import_tree.py` walks the import graph from `main.py` and fails
if a module is neither reachable nor explicitly accounted for. The repository
is meant to hold the code the paper needs and nothing else, and this check
keeps it that way.

### The `[HUGE-DV]` warning

The post-processing reports, and never modifies, grid points where the functional
potential differs from the exact one by more than `CFG.dv_report_Eh`. The
warning is printed prominently, and `tools_huge_dv.py` measures what those
points contribute. It recomputes `DeltaI_bond` twice with the production routine,
as is and with those points neutralised; the difference between the
two is their contribution. It runs on its own, after a scan, over the `.npz`
already on disk:

```
python tools_huge_dv.py                        # every geometry
python tools_huge_dv.py --only NN_q+0_s0
python tools_huge_dv.py --threshold 5          # to exercise it on H2
```

Every row carries the geometry's `inv_status`, so a contribution at a
gate-rejected geometry is never included in a maximum reported in a table. The output is also written to
`results_validation/huge_dv_diagnostic.csv`, sorted by contribution, so that a
record remains after the console is closed.

Result for the published scan: the flagged points sit 0.00-0.03 bohr from
a nucleus, carry ~1e-7 % of Omega's volume weight, and move `DeltaI_bond` by
0.00-0.09 %, which is negligible. The only
large contribution (HF at R = 4.339 A, 3.0-3.7 % across all five
functionals, points ~4 bohr out carrying *negative* Becke weight) falls at
`inv_status = FAIL` and therefore enters no reported number.

## Verification

`verification/` compares what this run produced against what the manuscript
prints. The expected values live in exactly one file, `paper_values.py`; a
check imports them. Where the comparison is against a printed number, the
printed precision is the tolerance: the computed value is rounded to the
decimals shown and compared as a string, so no tolerance has to be chosen. Where
a measured tolerance is used instead, it is the observed spread across
independent runs and its provenance is stated next to it.

The aim is that every table value can be reproduced from the data deposit.

## Scope

Ten diatomics, 203 geometries, of which a density-reproduction criterion
evaluated independently of the metric admits 118. The metric compares two
*multiplicative* potentials, so it applies directly to local-density and GGA
functionals. Only multiplicative functionals are scored: LDA/SVWN,
PBE, BLYP, PBEsol, revPBE. Hybrids need an OEP and meta-GGAs need a convention
for the kinetic-energy density that this work has not validated; they are out
of scope.

## Data

The deposit (1D profiles, dissociation curves and the per-geometry grid data)
is archived separately on Zenodo; see `CITATION.cff`. This repository is the
code that produces it.

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

`main.py` holds the configuration and should be read before it is run. It

1. builds the atomic ionization-potential cache used by the Koopmans test
   (once, if `results_analysis/atomic_ip.json` is absent);
2. runs the heavy pipeline (`pipeline.py`): for every system and internuclear
   distance, it builds the correlated reference density, inverts it with the
   Wu–Yang engine, and writes the **functional-agnostic exact side** (exact
   v_xc on the grid, reference density, energy decomposition, N_unpaired,
   inversion diagnostics);
3. runs the functional post-processing (`functional_eval.py`), which scores
   each functional against the saved exact side **without re-running the
   inversion**, and merges the metric columns into each dissociation CSV;
4. runs the Supporting Information validations (`validation.py`), the
   additional controls (`validation_extra.py`, `density_comparison.py`), the
   regression tests and the verification.

To run a subset, edit `ONLY_MEMBERS` at the top of `main.py`.

Re-running `main.py` is cheap and safe. Finished geometries are reloaded from
disk rather than recomputed; functionals whose columns are already present in
every dissociation CSV are skipped by the post-processing, so adding one functional
to `CFG.functionals_to_eval` and re-running scores only that functional. A
partially merged functional is redone rather than trusted.

Each run writes its own log, `results_logs/run_<YYYYMMDD_HHMMSS>.log`, stamped
with the moment the run started, so a log is always matched to the results it
produced. The log is a **complete** record: `main.py` routes both stdout and
stderr into it, so pipeline progress, grid warnings, the post-processing, the
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

| Result in the article | Produced by | File / column |
|---|---|---|
| Metric values, per geometry | `main.py` | `results_energies/<system>_Dissociation_Curve.csv`, columns `DeltaI_bar_<functional>_<mode>_Eh` and `DeltaI_resp_<functional>_<mode>` |
| Confidence class (OK/MARGINAL/FAIL) | `main.py` | same CSV, columns `inv_status`, `inv_density_error_e`, `ghost_leak_e` |
| Unpaired-electron index and its saturation | `main.py` (`richards_fit.py`) | same CSV, column `N_unpaired`; fit in `results_analysis/<system>_richards.json` |
| Spectroscopic constants | `main.py` (`morse_fit.py`) | `results_analysis/<system>_morse.json` |
| Functional comparison and error cancellation | `main.py` | `results_energies/Functional_Comparison.csv` and `..._ErrorCancellation.csv` |
| Koopmans test against experiment | `atomic_ip.py` | `results_analysis/atomic_ip.json` |
| 1D potential profiles, exact **and** per functional | `main.py` | `results_vxc_1D/<system>_Vxc_1D_R<r>.csv`, columns `Vxc_Exact` and `Vxc_<functional>_<mode>` |
| Regularization (κ) invariance of the response metric | `main.py` (`validation.py`) | `results_validation/kappa_invariance_<system>.csv` |
| Grid-resolution robustness of the metric peak | `main.py` (`validation.py`) | `results_validation/grid_resolution.csv` |
| Functional-derivative identity, spin-polarized route | `main.py` (`validation.py`) | `results_validation/functional_derivative.csv` |
| Cost of a mid-curve scheme switch (the seam) | `main.py` (`validation.py`) | `results_validation/ensemble_seam.csv` |
| Fine-grid check of the metric peak | `main.py` (`validation.py`) | `results_validation/fine_grid_scan.csv` |
| Five functionals, fixed windows, orbital basis, n2v, reduced coordinate | `main.py` (`validation_extra.py`) | `results_validation/*.csv` |
| Response metric against the self-consistent density difference | `main.py` (`density_comparison.py`) | `results_validation/mapa_vs_density_*.csv` |

Re-scoring is handled automatically by the post-processing in `main.py`, which
only computes what is missing:
to re-score selected systems or functionals without touching the inversion, set
`ONLY_MEMBERS` or `CFG.functionals_to_eval` at the top of `main.py` and re-run.

## Validations

These checks reproduce the Supporting Information. Each is switched
independently at the top of `main.py`, annotated with its cost, and all are on
by default:

```python
CFG.validate_grid_resolution = True           # seconds  (CSV post-processing)
CFG.validate_functional_derivative = True     # minutes  (no inversion)
CFG.validate_kappa_invariance = True          # ~30 min  (H2, 3R x 3 kappa)
CFG.validate_kappa_wide_range = True          # ~15 min  (H2, 1e-9..1e-5)
CFG.validate_ensemble_seam = True             # ~20 min  (H2, seam exposed)
CFG.validate_kappa_invariance_heavy = True    # ~4 h     (N2, 3R x 3 kappa)
CFG.validate_kappa_invariance_extra = True    # ~1 h     (LiH and HF rows of Table 2)
CFG.validate_fine_grid_scan = True            # ~2-3 h   (fine H2 scan)
```

Two of them need a note. The **seam** check measures the discontinuity a
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
(`ENSEMBLE_OS`). No scheme switch occurs mid-curve, which prevents a
spurious step in the metric. The legacy single-determinant paths are retained
in the code, disabled, and labelled `*_LEGACY` so that the production route is
unambiguous in every log line.

## Module map

| Module | Role |
|---|---|
| `main.py` | configuration, system selection, run order |
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
| `PHYSICS_FINDINGS.md` | numbered record of the physical results and design decisions the code rests on; source comments refer to its items by number |
| `density_comparison.py` | the response metric against the self-consistent density difference (SI S16): H2+ with the exact potential, a model reference with a known potential, and the production references |
| `validation.py` | the validations behind the SI, run automatically by `main.py` (or standalone: `python validation.py --only grid\|response\|derivative`) |

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

This work is archived as three citable objects, each with its own DOI:

| Object | DOI |
|---|---|
| Software (this repository), archived at Zenodo | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21813195.svg)](https://doi.org/10.5281/zenodo.21813195) |
| Dataset (results produced by this software), archived at Zenodo | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21812943.svg)](https://doi.org/10.5281/zenodo.21812943) |
| Preprint, ChemRxiv | [![DOI](https://img.shields.io/badge/DOI-10.26434/chemrxiv.15007130/v1-blue.svg)](https://doi.org/10.26434/chemrxiv.15007130/v1) |

All three identifiers are the same in `README.md` and `CITATION.cff`, and
`verification/check_packaging.py` checks that they stay that way.
Machine-readable metadata for the software, dataset and preprint is in `CITATION.cff`; GitHub
and Zenodo both read it, and most reference managers import it directly.

If you use the code, cite the software record and the article. If you use the
numbers without re-running anything, cite the dataset record as well.

### What goes in the dataset record

The software record holds this repository. The dataset record holds what a
full run produces. It is far larger, and it lets the results be checked
without repeating days of computation:

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