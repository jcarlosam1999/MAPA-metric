"""
config.py — shared configuration for the Wu-Yang benchmark pipeline.
==============================================================
Every module reads its knobs from the single `CFG` instance defined here.
`main.py` is the ONLY place where these values should be edited/overridden
(edit them there, not here), which keeps parameter provenance in one file
while avoiding circular imports (modules import config; main imports the
modules and mutates CFG before launching).

Physical constants and the run-metadata builder also live here so that every
artifact (npz meta, CSVs) records the exact configuration that produced it.
"""

import os
from dataclasses import dataclass, field, asdict

BOHR = 0.52917721092          # Angstrom per Bohr — the one unit constant.


@dataclass
class Config:
    # ---- chemistry -------------------------------------------------------
    benchmark_basis: str = 'aug-cc-pVTZ'   # homogeneous across the benchmark set (anions
                                     # need diffuse functions; homogeneity
                                     # keeps DeltaI comparable)
    grid_level: int = 3
    # heavy-system knobs exist but default to the SAME as light until a real
    # diagnosis justifies changing them. (An earlier grid_level_heavy=4 inflated
    # N2's grid to ~56k points and crashed WSL on memory WITHOUT fixing the
    # inversion's spurious zero-gap — so it was reverted. The root cause of the
    # N2 gap=0 is still under isolated-bench diagnosis; do not raise these
    # blindly.)
    heavy_nelec: int = 12            # threshold: n_elec >= this = 'heavy'
    grid_level_heavy: int = 3        # reverted to 3 (was 4, caused OOM crash)
    prune_tol_heavy: float = 1e-8    # reverted to 1e-8 (loosening didn't help)

    # ---- reference ladder (density targets) ------------------------------
    # CASSCF(benchmark CAS) -> CASCI(m, CASSCF-relaxed orbitals) -> FCI, with m =
    # largest space fitting the determinant RAM budget. NEVPT2 gives the
    # best total energy; the CONSISTENT energy pairs with the density level.
    fci_auto_density: bool = True
    fci_budget_gb: float = 4.0
    fci_davidson_vectors: int = 8
    # CEILING on the Davidson sigma-build cost per iteration, the real driver of
    # the CASCI hang (confirmed by py-spy: 100% CPU in direct_spin1.contract_2e,
    # first cycle never finishing). contract_2e contracts against the M^4 active
    # ERI, so per-iteration cost scales ~ N_strings * M^4, where
    # N_strings = C(M,na) + C(M,nb). This is spin- and electron-aware, unlike a
    # bare orbital-count cap: it correctly limits high-spin cases like (4,1) or
    # (3,1) where one spin has many strings even though min(na,nb)=1 (those broke
    # a naive M cap, e.g. LiC(4,1) reaching M=69 at ~5000x the BH cost).
    # Calibration: BH ran at CASCI+40o, (2,2) -> proxy 4.0e9. Setting the ceiling
    # just above that (5e9 ~ 1.25x BH) forces every member to <= the one cost
    # that is CONFIRMED to run. A full audit of the benchmark showed this clears
    # all 26 hanging members while leaving BH-like members essentially unchanged.
    # NOTE: this is a per-ITERATION cost. Davidson iteration COUNT can still vary
    # (denser spectra at stretched geometries), so a member may be slow without
    # hanging; raise/lower this knob if needed.
    fci_cost_ceiling: float = 5e9
    # LADDER STEP-DOWN (data-driven): the casscf-vs-ladder bench showed that for
    # BH/LiH/Li2/HF the inversion |drho| did NOT improve when the reference went
    # from CASSCF to large CASCI/FCI -- i.e. beyond a modest space the extra
    # correlation is wasted effort (and CASCI+40o cost 500-1500s, B2 even hung).
    # So: full FCI only where it is cheap AND helps (Ne <= fci_full_max_ne, which
    # covers H2+/H2/He2+/LiH), and cap the CASCI space for everything else at a
    # modest casci_max_N instead of pushing to the cost ceiling. This cuts time
    # and fixes the B2 timeout without losing inversion quality where we checked.
    fci_full_max_ne: int = 4         # electrons at/below which we keep full FCI
    casci_max_N: int = 20            # orbital cap for Ne > fci_full_max_ne


    # ---- Wu-Yang inversion engine ----------------------------------------
    kappa_start: float = 1e-4        # HBY smoothness strength (continuation
    adaptive_kappa: bool = True      # divides by 10 until density target)
    kappa_min: float = 1e-6          # validated floor on H2: kappa_final=1e-6
                                     # gave clean |drho|~1e-3 across the scan.
                                     # (1e-8 over-relaxes and was never the
                                     # validated setting.)
    # GAP-AWARE KAPPA FLOOR (experiment, opt-in). The continuation drives kappa
    # to kappa_min; for small-gap systems (LiH/Li2/BH/HF) that floor is too low,
    # the Hessian goes near-singular and the solve breaks (ENSEMBLE_FIT_FAIL). A
    # gap-aware floor RAISES the minimum regularization as the KS gap collapses,
    # trading a slightly larger (but STABLE, finite) density error for avoiding
    # the numerical breakdown. Valid to tune freely: the reported metrics are
    # kappa-independent (per preprint.tex), so a stable higher-kappa result still
    # yields the same physics. floor = clip(kappa_min * kappa_gap_ref/gap,
    # kappa_min, kappa_floor_max). OFF by default; the factorial bench turns it on.
    kappa_gap_aware: bool = False
    kappa_gap_ref: float = 0.30      # 'healthy' gap; at this gap floor == kappa_min
    kappa_floor_max: float = 1e-3    # hard cap so gap->0 doesn't over-regularize
    # SPATIAL SYMMETRY (experiment, opt-in). build_molecule uses symmetry=False;
    # open-shell systems with spatial degeneracy (CH, B2) then invert to gap=0
    # garbage (|drho| ~ whole electrons). Enforcing point-group symmetry pins the
    # correct degenerate occupation. OFF by default; the factorial bench turns on.
    use_symmetry: bool = False
    # UNRESTRICTED ENSEMBLE (experiment, opt-in). Open-shell (mol.spin!=0)
    # currently uses _two_channel_inversion: each spin channel is an aufbau
    # WuYangBasisInverter (INTEGER n_occ). This is a DIFFERENT method from the
    # closed-shell path (EnsembleInverter, FIXED FRACTIONAL natural occupations)
    # -- the exact heterogeneity flagged as a concern. It may also be the CH/B2
    # root cause: near a spatial (pi) degeneracy the reference alpha/beta natural
    # occupations are genuinely fractional, and NO integer-occupation aufbau
    # channel can reproduce a fractional-occupation target density -- exactly
    # the failure signature seen (|drho| ~ 1-2 whole electrons, gap=0).
    # This flag routes open-shell through the SAME EnsembleInverter class used
    # for closed shell, called once per spin channel with that channel's own
    # natural occupations (from the reference dm_alpha/dm_beta). Homogenizes
    # the method across the whole benchmark. OFF by default; validate on
    # H2+ (already-working control) before trusting it on CH/B2.
    unified_ensemble_openshell: bool = False
    lbfgs_maxiter: int = 800
    newton_polish_maxit: int = 15
    density_target: float = 5e-4     # e-, 'good inversion' tier
    density_fail: float = 5e-3       # e-, hard-fail tier (light systems)
    # Heavy systems (n_elec >= heavy_nelec; 14e- is the benchmark ceiling)
    # invert to a PHYSICAL vxc — smooth and left-right symmetric to ~2e-3 Eh,
    # verified on N2 — but only ~1% density error, because reproducing a 14e-
    # density with the auxiliary potential basis is intrinsically harder. The
    # H2-calibrated 5e-3 fail threshold wrongly hard-rejects these good
    # inversions. A looser heavy threshold marks them MARGINAL (usable
    # potential, imperfect density) so they enter the dataset with their
    # quality recorded (trusted flag + density_error weights), not discarded.
    # This is an honest statement that ~1% is the physical floor for the
    # heaviest benchmark members, not a workaround.
    density_fail_heavy: float = 6e-2      # e-, hard-fail tier (heavy systems)
    # DIFFUSE tier (data-driven, tunable). A light system that leaks a large
    # FRACTION of its density error to the diffuse far field inverts at the same
    # ~1% Wu-Yang floor as a heavy one. Bench separation: reliable members leak
    # 0.01-0.13 of |drho| to the tail; the diffuse FAIL group (LiH/Li2/BH) leaks
    # 0.17-0.46. A cut at 0.15 catches the diffuse group; the threshold then
    # relaxes to density_fail_diffuse. Calibrate on the full sweep -- these are
    # first estimates from 8 members, not a final value.
    diffuse_leak_frac: float = 0.15   # ghost_leak/|drho| above this => diffuse
    density_fail_diffuse: float = 2e-2  # e-, hard-fail tier (diffuse systems)
    gap_warn: float = 1e-3           # Eh, KS near-degeneracy flag

    # potential basis (rides on both atoms + bond-midpoint ghost);
    # pruned by canonical orthogonalization in the grid metric.
    pot_basis_atom: list = field(default_factory=lambda:
                                 [(0, 16, 0.15, 2.2), (1, 6, 0.25, 2.2)])
    pot_basis_mid: list = field(default_factory=lambda:
                                [(0, 10, 0.10, 2.2)])
    prune_tol: float = 1e-8
    penalty_ridge: float = 1e-6
    density_floor: float = 1e-6      # ghost-state cure: grid points with
                                     # rho_target < density_floor * max(rho)
                                     # are excluded from the potential fit
                                     # (v_xc there stays at the guide).

    # ---- the GUIDE potential (the prior the regularizer pulls toward) -----
    # v_xc = v_guide + sum_t b_t g_t, and HBY penalizes kappa*Int|grad(v_xc -
    # v_guide)|^2. So wherever the density does NOT constrain the fit, v_xc
    # relaxes to v_guide: the guide is the PRIOR, and its error is inherited
    # exactly where the fit is blind.
    #
    # Fermi-Amaldi (-v_H/N) is EXACT for 1 electron (the full v_xc) and for
    # 2-electron singlets (the exchange potential) — i.e. perfect for the
    # light half of the benchmark. It is structurally WRONG in the CORE of a
    # many-electron atom: dividing v_H by N underestimates the exchange hole
    # of a 1s electron, whose local self-interaction is that of one pair, not
    # 1/N of the total. Measured on the real N2 inversion at R=1.10 A,
    # averaged over rho*w around the N nucleus:
    #
    #     shell      rho     v_xc(inverted)   v_FA    FA err   v_LDA   LDA err
    #     0-0.05b    124        -5.35        -1.51    +3.84    -4.90    +0.45
    #     0.05-0.1b   68        -4.67        -1.43    +3.25    -3.99    +0.68
    #     0.1-0.2b    28        -4.15        -1.28    +2.87    -2.94    +1.21
    #
    # FA is off by ~3.8 Eh where LDA is off by 0.45 — and the 1s core density
    # is frozen (bound by -Z/r), so the fit cannot push back against the pull
    # toward that error. Result: the core v_xc wanders with kappa and along
    # the warm-start chain (measured: ~2.4 Eh drift across the N2 scan at
    # FROZEN core density), which destroys the Levy-Perdew virial (its
    # integrand weights v_xc by 3rho + r.grad rho, ~60-85% core for N2).
    #
    # THE CURE IS THE PRIOR, not more regularization:
    #     v_guide = v_FA + w(rho) * (v_xc^LDA[rho_ref] - v_FA)
    #     w(rho)  = rho^2 / (rho^2 + guide_rho_switch^2)
    # LDA in the core (where it is 8x better than FA), FA in the valence and
    # the tail (where FA's exact -1/r asymptote is what matters and LDA has
    # none). At guide_rho_switch=4 e/bohr^3 the measured weights are 0.98-0.999
    # in the N2 core, 0.015 at the N2 bond midpoint and 0 in the tail — so the
    # BOND REGION, where DeltaI_bar is measured, keeps the validated FA guide.
    #
    # GATE: members with n_elec <= 2 keep PURE Fermi-Amaldi, because there it
    # is exact and any blend would perturb already-validated results (H2, He,
    # HHe+, and the 1e SIE-validation path where v_xc = -v_H analytically).
    # The guide is spin-unpolarized by construction: FA has no spin, and the
    # 1s core it corrects is paired, so LSDA == LDA there to <1 mEh.
    #
    # REPORT IN THE PAPER: guide_rho_switch is a new parameter, so its
    # irrelevance must be shown, not asserted. Run the sweep at 2 and 8 and
    # report that DeltaI_bar does not move (it cannot: the fit, not the guide,
    # determines v_xc wherever the density has support).
    # PRODUCTION VALUE: False = pure Fermi-Amaldi. Set to False on 2026-08-07
    # (finding D4). It had been True while pipeline.py never called
    # xc_guide_on_points, so 197 of the 203 published geometries were run with
    # pure Fermi-Amaldi and 6 (N2, R >= 2.585 A) with the interpolated guide.
    # The flag now means what it says: whatever it is set to is what the whole
    # scan uses. Turning it back on re-inverts every system with N > 2 — 8 of
    # the 10 members, 161 of the 203 geometries — so it is a decision for a
    # fresh benchmark, not a switch to flip mid-project.
    guide_lda_core: bool = False     # True = interpolated LDA-core guide
    guide_rho_switch: float = 4.0    # e/bohr^3, FA <-> LDA crossover density
    # closed-shell representability breakdown -> broken-symmetry fallback.
    # When the restricted KS gap collapses below rks_breakdown_gap while the
    # density residual stays above target, the correlated density is not
    # closed-shell representable; retry with a broken-symmetry two-channel
    # inversion seeded by bs_seed (L/R density skew, 0..1).
    rks_breakdown_gap: float = 1e-3
    unified_ensemble: bool = True    # PRODUCTION DEFAULT: run the ensemble
                                     # machinery at EVERY closed-shell R
                                     # (validated by test_ensemble_unified_H2:
                                     # removes the RKS->ENSEMBLE DeltaI seam,
                                     # e.g. 1.66->0.24 across one step of H2).
                                     # False = legacy RKS-first path (for
                                     # comparison/debugging only).
    bs_seed: float = 0.3
    # DeltaI_bar weighting: 'volume' (corrected) or 'density' (legacy).
    # The exact vxc's mid-bond correlation barrier lives where the density is
    # LOWEST, so rho-weighting suppresses the signal the metric exists to
    # detect and lets the nuclear region dominate — it moved the DeltaI_bar
    # maximum out of the bond-breaking region and up against equilibrium.
    # 'density' is kept only to reproduce legacy numbers.
    # NO PEAK POSITION IS QUOTED HERE, ON PURPOSE (see D3). This comment used
    # to carry R=0.740 / R=1.258 while the paper carried 0.814 / 1.146: four
    # numbers, two sources, no pair agreeing — and the peak POSITION is the
    # one thing the paper claims about this metric. The peak now has exactly
    # one generator (validation.run_grid_resolution, over the production
    # *_Dissociation_Curve.csv) and exactly one written home
    # (HECHOS_Y_CIFRAS.md). Do not re-inline it anywhere.
    dIbar_weighting: str = 'volume'
    # D2 diagnostic: report — never modify — finite grid points whose
    # deviation from the exact potential exceeds this, in Eh. A PRINTED
    # WARNING THRESHOLD, not a metric parameter: no value of it changes any
    # number the paper reports. None silences it.
    dv_report_Eh: float = 10.0
    # Task 5.3: exporters.save_line_csv audits every 1D profile before writing
    # it. True = a failing profile aborts the run. Set False only to
    # re-analyse an existing tree that is already known to be broken.
    deposit_audit_strict: bool = True
    # Run log: every per-geometry status line and every end-of-system fit
    # report is appended here as well as printed, so a closed terminal or a
    # crash never loses the record of what was done. main.py gives each shard
    # its own file — two parallel processes must not share one log.
    log_path: str = "results_logs/run.log"
    # which energy column the Morse fit uses (the correlated reference)
    morse_energy_col: str = "E_NEVPT2"
    # BROKEN-SYMMETRY DIAGNOSTIC (closed-shell singlets only).
    # The BS channel runs _two_channel_inversion = TWO extra full inversions
    # (alpha + beta) and fires exactly where `localized` is true — i.e. on the
    # STRETCHED points, which are already the slowest (small KS gap => harder
    # convergence). That makes an ENSEMBLE point ~5x an ENSEMBLE_NOSPIN one:
    # ~3x from doing 3 inversions instead of 1, ~1.7x from harder convergence.
    #
    # For closed-shell singlets it is PURELY diagnostic: the primary vxc and
    # D_ks come from the ensemble, and its vxc_alpha/vxc_beta are never
    # consumed downstream (functional_eval gates the spin-resolved path on
    # spin != 0). Turning it off loses only the
    # diagnostic columns bs_spin_square_raw, bs_density_error_e and
    # vxc_ens_bs_maxdiff (the ensemble-vs-BS framework-ambiguity marker),
    # and leaves the metric of this study untouched.
    # Genuine open-shell systems (spin != 0) are unaffected: there the
    # two-channel inversion is the PRIMARY path, not this diagnostic.
    #
    # Note: the BS channel is known to be unusable for N2 anyway
    # (|vxc_ens - vxc_bs| > 1000 Eh), so on heavy systems it is paying 3x for
    # a broken marker. Consider running it on a few representative light
    # systems to characterize framework ambiguity, and off for production.
    bs_diagnostic: bool = True       # False => ~2.7x faster full benchmark
    bs_localization_min: float = 0.05    # <S^2> above this = genuine spin
                                         # localization (real representability
                                         # breakdown); at or below = the gap
                                         # collapse was numerical, not physical
    natocc_keep_tol: float = 1e-4        # ensemble channel keeps natural
                                         # orbitals with occupation above this
                                         # (core ~2, active fractional, drop ~0)
    natocc_frac_lo: float = 0.02         # occupation window that counts as
    natocc_frac_hi: float = 1.98         # genuinely fractional for the <S^2>
                                         # derivation (excludes core, empty,
                                         # and the basis-incompleteness tail)
    n_unpaired_localization_min: float = 0.05   # N_unpaired above this marks a
                                         # genuine static-correlation breakdown
                                         # (ENSEMBLE vs ENSEMBLE_NOSPIN).
                                         # Head-Gordon index: ~0.007 at
                                         # equilibrium, lifts through ~0.05 at
                                         # the Coulson-Fischer onset.

    # ---- metric & profiles -------------------------------------------------
    cyl_radius_bohr: float = 2.0     # bond-region cylinder for DeltaI
    n_line_points: int = 601
    line_pad_ang: float = 3.0

    # ---- functional evaluation (post-pass over the exact side) -------------
    # Functionals to score with DeltaI_bond after the heavy run. Each entry is
    # a libxc string or (name, string). Only Tier-1 (LDA/GGA, multiplicative)
    # are validated for publication; meta-GGA/hybrids need their convention/OEP
    # validated first (see functional_eval.py tiers).
    functionals_to_eval: list = field(default_factory=lambda:
                                      ['LDA,VWN', 'PBE,PBE'])
    functional_density_modes: tuple = ('exact', 'scf')
    run_functional_postpass: bool = True   # main.py runs the post-pass
    # Force a re-score even when the dissociation CSVs already carry the
    # columns. The skip test looks at those CSVs only, so anything else the
    # post-pass writes -- the per-functional v_xc columns of the 1D profiles,
    # the comparative figures -- is not regenerated unless this is set. It is
    # required after any change to what the post-pass produces.
    functional_postpass_force: bool = False
    # Validations behind the Supporting Information, run by main.py after the
    # post-pass so every reported number comes from one `python main.py`.
    # The two members below are the ones the validations are cheapest on; both
    # must be present in the set being run.
    # set by main.py when it routes stdout/stderr into the run log; modules
    # that also append to the log check it so lines are not written twice.
    log_is_teed: bool = False
    run_validations: bool = True     # master switch for the whole stage
    # ---- individual validations -------------------------------------------
    # Each one is independent and separately switchable from main.py. The
    # annotated cost is for a 4-core desktop; the cheap ones are on by default,
    # the expensive ones are off and must be turned on deliberately.
    #
    #   seconds   grid_resolution        pure CSV post-processing, no QC
    #   minutes   functional_derivative  one molecule, no inversion
    #   ~30 min   kappa_invariance       H2: 3 geometries x 3 kappas
    #   ~15 min   kappa_wide_range       H2: 1 geometry x 5 kappas (1e-9..1e-5)
    #   ~20 min   ensemble_seam          H2: re-inversion with the seam exposed
    #   ~4 h      kappa_invariance_heavy N2: 3 geometries x 3 kappas
    #   ~2-3 h    fine_grid_scan         a fine H2 scan in a scratch directory
    validate_grid_resolution: bool = True
    validate_functional_derivative: bool = True
    validate_kappa_invariance: bool = True
    validate_kappa_wide_range: bool = False
    validate_ensemble_seam: bool = False
    validate_kappa_invariance_heavy: bool = False
    validate_kappa_invariance_extra: bool = False
    validate_fine_grid_scan: bool = False
    # parameters of the individual validations
    # The R values below must be POINTS OF THE PRODUCTION GRID, otherwise the
    # probe silently falls back to the nearest saved geometry (or skips it).
    # These are grid points of the current scan.
    validation_kappa_light: tuple = ('HH_q+0_s0', (1.073, 1.628, 2.028),
                                     (1e-7, 1e-6, 1e-5))
    validation_kappa_heavy: tuple = ('NN_q+0_s0', (1.595, 2.090, 2.420),
                                     (1e-7, 1e-6, 1e-5))
    validation_kappa_wide: tuple = ('HH_q+0_s0', (1.073,),
                                    (1e-9, 1e-8, 1e-7, 1e-6, 1e-5))
    # Table 1 has LiH and HF rows too: a polar closed shell and the most
    # restrictive member of the set. They were produced by hand once, which
    # meant three of the twelve rows of the central table were not reachable
    # from `python main.py`. They are now a probe like the others.
    validation_kappa_extra: tuple = (
        ('HLi_q+0_s0', (1.600, 2.000, 2.400), (1e-7, 1e-6, 1e-5)),
        ('HF_q+0_s0', (0.552, 0.626, 0.699), (1e-7, 1e-6, 1e-5)),
    )
    validation_seam_system: str = 'HH_q+0_s0'
    validation_fine_scan_system: str = 'HH_q+0_s0'
    validation_fine_scan_points: int = 27
    validation_openshell_member: str = 'HC_q+0_s1'  # spin-polarized derivative

    # ---- distance scan (relative to each member's r_eq) --------------------
    # single clean grid: moderate compressed zone, dense bond-breaking zone,
    # geometrically-tapering tail. See molecules.make_distance_grid.
    # ~15 points/curve (was ~34): 2x fewer geometries at equal physics.
    # Zone sizes are NOT cut uniformly — the bond-breaking zone keeps enough
    # points to resolve Coulson-Fischer, while the tail coarsens hard because
    # N_unpaired plateaus there (asymptotically flat physics). Validated on
    # real H2 data with the Richards/Head-Gordon sigmoid: the 15-point grid
    # reproduces the 34-point fit's k and x0 (critical distance) to 0.08% and
    # the whole N_unpaired curve to 2e-4 electrons.
    n_compressed: int = 3            # points in [0.6 Re, Re) (compressed)
    n_breaking: int = 7              # points in [Re, 2.5 Re) (bond-breaking)
    tail_growth: float = 1.6         # geometric gap growth in the tail
    max_gap: float = 0.9             # Angstrom, cap on tail spacing
    min_gap: float = 0.035           # Angstrom, below this = near-duplicate
    r_stretch_abs: float = 3.8       # Angstrom beyond r_eq (tail end)
    r_min_floor: float = 0.4

    # ---- baselines & optional physics --------------------------------------
    include_pbe: bool = True         # closed-shell PBE grid potential
    include_pbe_spin: bool = True    # EXPERIMENTAL spin-polarized PBE pot.
    ip_validation_members: list = field(default_factory=list)  # names for
                                     # the eps_HOMO vs -IP check (costly)

    # ---- exports & operations ----------------------------------------------
    save_ao_grid: bool = True        # ao_value as float32 in npz
    save_eri: bool = False           # nao^4: 0.1-1 GB/geometry; recompute
                                     # or density-fit downstream instead
    resume: bool = True

    dirs: tuple = ("results_energies", "results_vxc_1D",
                   "results_vxc_1D/pdf", "results_grid_data",
                   "results_convergence", "results_analysis",
                   "results_logs")


CFG = Config()


def make_dirs():
    for d in CFG.dirs:
        os.makedirs(d, exist_ok=True)


def grid_level_for(n_elec):
    """Finer grid for heavy systems (diagnosed need: N2 and up)."""
    return CFG.grid_level_heavy if n_elec >= CFG.heavy_nelec else CFG.grid_level


def prune_tol_for(n_elec):
    """Looser prune (larger potential basis) for heavy systems."""
    return CFG.prune_tol_heavy if n_elec >= CFG.heavy_nelec else CFG.prune_tol


def density_fail_for(n_elec, ghost_leak=None, density_error=None):
    """Hard-fail density threshold. Two physical regimes invert only to ~1%
    density error, and BOTH deserve the relaxed ceiling rather than the
    H2-calibrated 5e-3:

      * HEAVY (n_elec >= heavy_nelec): reproducing a many-electron density with
        the auxiliary potential basis is intrinsically harder (verified on N2).

      * DIFFUSE (data-driven): systems whose density has a large diffuse tail
        (Li-bearing, anions, stretched) leak a significant FRACTION of the
        density error into the far field, where Wu-Yang has a ~1% floor. The
        electron count misses these (LiH has 4e- yet behaves like a heavy
        system; N2 has 14e- and inverts cleanly). We detect them by the
        leaked FRACTION ghost_leak/density_error exceeding diffuse_leak_frac,
        which the bench data separates far better than n_elec alone.

    Passing ghost_leak/density_error is optional (back-compatible): with neither,
    the classifier falls back to the electron-count rule only."""
    if n_elec >= CFG.heavy_nelec:
        return CFG.density_fail_heavy
    if (ghost_leak is not None and density_error is not None
            and density_error > 0
            and ghost_leak / density_error >= CFG.diffuse_leak_frac):
        return CFG.density_fail_diffuse
    return CFG.density_fail


def run_metadata():
    """Everything needed to reproduce a run, embedded in each artifact."""
    from pyscf import __version__ as pyscf_version
    md = {k: v for k, v in asdict(CFG).items() if k != 'dirs'}
    md.update(pyscf_version=pyscf_version,
              reference_level="density=ladder(CASSCF->CASCI+->FCI), "
                              "energy=NEVPT2; consistent pair exported",
              engine="auxiliary-basis Wu-Yang, hybrid LBFGSB+Newton, "
                     "HBY-regularized, adaptive kappa, pruned")
    return md