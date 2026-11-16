# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the
interface. A minor release can change computed numbers; every such change is
listed under **Results affected**, with its size.

---

## [1.1.0] - 2026-11-08

The version cited by the article. `python main.py` is now the only entry point:
it runs the benchmark, the metrics, every validation and control in the
Supporting Information, the regression tests, and a verification that compares
the run with the values the article prints. Anyone holding results from 1.0.0
should regenerate them.

### Results affected

Changes large enough to alter a statement:

| Quantity | 1.0.0 | 1.1.0 | Cause |
|---|---|---|---|
| H2+ response metric, plateau | 0.11 | 1.2e-3 e-/bohr^3 | empty spin channel included in the spin average (fixed, see below) |
| H2+ potential metric, peak (PBE, exact mode) | 8.132 Eh | 0.031 Eh | same cause |
| N2 virial-gap drift between neighbouring geometries | 6.5 Eh | 4.9 Eh | six geometries had used a different prior (fixed, see below) |
| N2 density-reproduction error at R = 5.012 A | 3.6e-3 e- | 2.1e-3 e- | same cause |
| N2 response metric (PBE, self-consistent) at R = 4.112 A | 2.645e-3 | 1.512e-3 | the self-consistent calculation had converged to a different solution |
| Geometries admitted by the confidence gate | 115 of 203 | 118 of 203 | He2+ at 1.566 A, LiH at 1.800 A and HF at 0.846 A now pass; none of them is the maximum of its system |

Removing the LDA-core prior from the six most stretched N2 geometries also
lowers every potential-metric peak of N2 by 1.2-1.4 %.

Changes at the last printed digit are not listed. Repeating the whole
calculation three times, on two machines, moves the max/min ratio of the potential metric by
up to 0.5 %, its dispersion over the regularization parameter by up to 1.3 %,
and the mean of the response metric not at all.

### Added

- `artifacts.py`: one registry of what each stage writes. A stage whose output
  is already on disk is skipped, so an interrupted run can be relaunched.
  Set `CFG.validation_force = True` to redo the validations.
- `verification/`: compares the run with the values printed in the article and
  its Supporting Information. Expected values live in `paper_values.py`. A
  printed value is compared at its printed precision; where a measured
  tolerance is used, its source is recorded next to it.
- `validation_extra.py`: three further multiplicative functionals (BLYP,
  PBEsol, revPBE), the fixed-volume control on the response metric,
  orbital-basis dependence (aug-cc-pVQZ), cross-validation against the
  independent `n2v` inversion code, and both metrics at a matched reduced
  coordinate (1.5 R_e against R_e). The `n2v` check is optional and is
  reported as unchecked when the package is absent.
- `density_comparison.py`: the response metric against the self-consistent
  density difference, for H2+ with the exact potential, for a model reference
  whose exact potential is known (the LDA density), and for the production
  references. Each recomputed metric must reproduce the deposited value before
  it is used, and each self-consistent density is the lowest-energy solution
  found from several starting guesses.
- `validation.py` now runs the LiH and HF regularization sweeps and writes
  every validation to a CSV file instead of only printing it.
- `tests/`: five regression tests, run as a stage of the normal run.
- `tools_import_tree.py`: fails if a module is neither reachable from
  `main.py` nor explicitly accounted for.
- `tools_huge_dv.py` and `CFG.dv_report_Eh`: report grid points where the
  functional potential differs from the exact one by an implausible amount,
  and measure what they contribute to the potential metric (0.00-0.09 % on
  admitted geometries). The threshold only controls a warning; no reported
  value depends on it.
- `tools_gate_stability.py`: the distance of each geometry from its
  confidence threshold.
- `CFG.deposit_audit_strict`: a 1D profile that fails the deposit audit is not
  written.
- `response_metric.drho_from_dv(..., occ_occ=False)`: the textbook
  occupied-virtual response, to measure what discarding the occupied-occupied
  pairs costs. Production always uses `occ_occ=True`.
- `.gitignore` for the `results_*/` directories.

### Changed

- `CFG.guide_lda_core` defaults to `False`: a single Fermi-Amaldi prior is
  used at every geometry.
- `CFG.functional_postpass_force` defaults to `False`; it was re-scoring every
  functional on every launch.
- Every per-geometry row and 1D profile carries the same columns, including
  the guide settings where they are zero.
- The default system of the spin-channel validation is CH (`HC_q+0_s1`).
- `README.md` rewritten.

### Fixed

- **Exact and self-consistent modes wrote the same array.** The 1D profiles
  stored identical `Vxc_<f>_exact` and `Vxc_<f>_scf` columns. Profiles are now
  audited when they are written.
- **An empty spin channel was included in the spin average.** For H2+ the
  comparison potential was `(v_a + v_b)/2`, although the beta channel holds no
  electrons; libxc returns a value at its density floor there, not a
  potential. A channel without electrons is now excluded.
- **The position of the potential-metric maximum was computed in more than one
  place.** It now has one generator.
- **Six N2 geometries used a different prior.** The geometries with
  R >= 2.585 A had been inverted with an LDA-core guide and the rest with pure
  Fermi-Amaldi. All 203 geometries now use the same prior.
- The reduced-coordinate stage read LiH's 1.5 R_e node as absent because
  `2.400/1.600` evaluates to `1.4999999999999998`.
- The spin-channel validation stopped at once in a normal run, because its
  default system was not part of the benchmark.

### Removed

- `rerun_postpass.py`. The post-processing in `main.py` scores only what is
  missing, which covers what this script did.

### Notes

- The benchmark is unchanged: the same ten diatomics, the same 203 geometries,
  the same scan.
- Only multiplicative functionals are scored (LDA/SVWN, PBE, BLYP, PBEsol,
  revPBE). Meta-GGAs and hybrids are out of scope.

---

## [1.0.0] - 2026-08-05

Initial public release accompanying the preprint: Kohn-Sham inversion applied
to multi-reference diatomics, the potential and response metrics in exact and
self-consistent density modes, and the supporting validations.

[1.1.0]: https://github.com/jcarlosam1999/MAPA-metric/releases/tag/v1.1.0
[1.0.0]: https://github.com/jcarlosam1999/MAPA-metric/releases/tag/v1.0
