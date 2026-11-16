"""
paper_values.py -- every number the paper prints, in one place.
===============================================================

The project rule these checks exist to enforce: *a table value that does not
reproduce from the data deposit is the one class of error that is not fixed in
a response to referees*. Enforcing it means comparing what is printed against
what the code produces, and that comparison is only trustworthy if "what is
printed" lives in exactly one place. Spread across six checkers, it drifts:
one gets updated after a rerun and the others keep asserting the old value,
and then two scripts disagree about what the paper says.

So the manuscript's numbers are here, and nowhere else. A check imports them.

Values are stored the way they are PRINTED, as strings, wherever the printed
precision is itself the tolerance -- "4.0" and "6.42" appear in the same column
of Table 1 and must not be checked to the same absolute slack. Where the
comparison is genuinely numeric with a measured tolerance, the tolerance is
stored next to the value and its provenance is stated.
"""

# --------------------------------------------------------------------------
# scope: the confidence gate
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# release metadata. Not a result, but four defects hid here through five
# adversarial rounds because no check opened these files: a CITATION.cff that
# did not parse, a README quoting a superseded admitted count, a bibliography
# labelling the software v1.0, and an RSC template placeholder left in the
# manuscript reading as a DOI. check_packaging.py now reads them.
# --------------------------------------------------------------------------
SOFTWARE_VERSION = '1.1.0'

GEOMETRIES_TOTAL = 203
GEOMETRIES_ADMITTED = 118

# system -> (admitted, rejected).  Table S4.
STRATIFICATION = {
    'HH_q+1_s1': (20, 0), 'HH_q+0_s0': (22, 0), 'NN_q+0_s0': (20, 0),
    'HeHe_q+1_s1': (12, 8), 'HLi_q+0_s0': (16, 4), 'LiLi_q+0_s0': (13, 7),
    'HB_q+0_s0': (6, 14), 'HF_q+0_s0': (4, 17), 'BB_q+0_s2': (5, 15),
    'HC_q+0_s1': (0, 20),
}

# --------------------------------------------------------------------------
# Table 2 of the main text: where error cancellation lives (H2)
#   functional -> (peak exact, peak scf, integral exact, integral scf)
# --------------------------------------------------------------------------
H2_MODES = {
    'LDA-VWN': (0.4154, 0.4146, 0.6254, 0.6602),
    'PBE-PBE': (0.4194, 0.4194, 0.6227, 0.6483),
}
H2_MODE_GAP_PEAK_LDA_PCT = 0.17
H2_MODE_GAP_INTEGRAL_LDA_PCT = 5.6
H2_MODE_GAP_INTEGRAL_PBE_PCT = 4.1

# --------------------------------------------------------------------------
# Table S10: peak of DeltaI_bond over the admitted stratum
#   system -> (LDA exact, LDA scf, PBE exact, PBE scf)
#
# H2+ carries the D2 correction (an empty spin channel does not vote in the
# spin average). N2 carries the A1 correction: its six most stretched
# geometries were recomputed under the single pure Fermi-Amaldi convention,
# and the N2 peak sits at R = 4.112 A, which is one of them.
# --------------------------------------------------------------------------
S10_PEAKS = {
    'HH_q+1_s1': (0.026, 0.031, 0.031, 0.042),
    'HeHe_q+1_s1': (0.250, 0.248, 0.253, 0.254),
    'HH_q+0_s0': (0.415, 0.415, 0.419, 0.419),
    'LiLi_q+0_s0': (0.181, 0.182, 0.171, 0.172),
    'NN_q+0_s0': (0.169, 0.166, 0.166, 0.165),
    'HLi_q+0_s0': (0.259, 0.260, 0.261, 0.261),
    'HB_q+0_s0': (0.306, 0.307, 0.298, 0.298),
    'HF_q+0_s0': (0.085, 0.086, 0.078, 0.080),
    'BB_q+0_s2': (0.159, 0.157, 0.153, 0.152),
}

# Table S7: PBE peak (exact mode) and the R at which it sits.
S7_PEAK_AND_R = {
    'HH_q+1_s1': (0.031, 3.31), 'HeHe_q+1_s1': (0.25, 2.21),
    'HH_q+0_s0': (0.42, 1.18), 'LiLi_q+0_s0': (0.17, 2.03),
    'HLi_q+0_s0': (0.26, 2.00), 'HB_q+0_s0': (0.30, 1.03),
    'HF_q+0_s0': (0.08, 0.55), 'BB_q+0_s2': (0.15, 1.59),
}

# --------------------------------------------------------------------------
# Table 1: invariance to the regularization.
#   (system, R) -> (max/min, mean DeltaI_resp in 1e-3, dispersion %)
# printed as strings: the printed precision IS the tolerance.
# --------------------------------------------------------------------------
TABLE1 = {
    ('H2', 1.073): ('12.8', '1.00', '8.4'),
    ('H2', 1.628): ('7.4', '0.84', '1.2'),
    ('H2', 2.028): ('4.0', '0.91', '9.1'),
    ('N2', 1.595): ('1.25', '2.43', '0.8'),
    ('N2', 2.090): ('1.72', '1.44', '0.8'),
    ('N2', 2.420): ('1.48', '1.65', '1.0'),
    ('LiH', 1.600): ('6.42', '0.95', '9.4'),
    ('LiH', 2.000): ('6.29', '0.78', '6.6'),
    ('LiH', 2.400): ('3.06', '0.59', '4.4'),
    ('HF', 0.552): ('2.12', '4.36', '2.1'),
    ('HF', 0.626): ('1.76', '3.34', '0.6'),
    ('HF', 0.699): ('1.40', '3.18', '0.8'),
}
# Column 3 of Table 1: N_unpaired, which the kappa sweep does not print.
TABLE1_N_UNPAIRED = {
    ('H2', 1.073): '0.03', ('H2', 1.628): '0.30', ('H2', 2.028): '0.91',
    ('N2', 1.595): '0.81', ('N2', 2.090): '4.35', ('N2', 2.420): '5.49',
}
MEMBER_TO_LABEL = {'HH_q+0_s0': 'H2', 'NN_q+0_s0': 'N2',
                   'HLi_q+0_s0': 'LiH', 'HF_q+0_s0': 'HF'}

# Tolerances for Table 1, MEASURED across three independent runs of the same
# code (twice on one machine, once on another), not chosen by eye:
#   max/min      12.7435 / 12.7593 / 12.6960 at H2 R=1.073  -> 0.50 % spread
#   dispersion   8.3851 / 8.3851 / 8.2816                   -> 1.25 % spread
#   mean resp    identical in every printed digit
# The asymmetry is exactly what the paper claims, so the potential column is
# checked to 1 %, the dispersion to 2 %, and the response mean to the printed
# digit with no slack at all -- it is the only column that earned that.
TABLE1_TOL_RATIO_REL = 0.01
TABLE1_TOL_DISPERSION_REL = 0.02

# The three runs above are all H2, whose inversions are the best in the
# benchmark (density error ~2e-3 e-). Applying their spread to the LiH and HF
# block applies a tolerance measured on the wrong system: those inversions
# carry ~1.3e-2 e-, five times worse, and HF's admitted geometries are all
# MARGINAL and all compressed. A fourth run, on a second machine, measured
# that block directly (18 points, LiH and HF):
#
#   max/min      LiH reproduces to < 1 %; HF moves by 11 % at both determinate
#                geometries (1.76 -> 1.57, 1.40 -> 1.25)
#   dispersion   moves ADDITIVELY, by about the per-point relative change in
#                percentage points: LiH 1.600 9.42 -> 8.53 from 0.90 %,
#                HF 0.626 0.57 -> 1.53 from 0.89 %
#   mean resp    17 of 18 values reproduce to 0.9 % or better, median 0.19 %
#
# So the dispersion is checked against an ABSOLUTE tolerance in percentage
# points rather than a relative one -- that is what the measurement says it
# does -- and the ratio against the worst measured spread. Both numbers come
# from the comparison, not from what would make the table pass.
TABLE1_TOL_RATIO_REL_HARD = 0.12          # HF; LiH and H2 do far better
TABLE1_TOL_DISPERSION_ABS_PCT = 1.0       # percentage points, both blocks
TABLE1_HARD_MEMBERS = ('HF',)

# --------------------------------------------------------------------------
# Five Tier-1 functionals, peak over the admitted stratum (SI).
#   system -> (LDA, PBE, BLYP, PBEsol, revPBE, spread %)
# --------------------------------------------------------------------------
TIER1_FIVE = {
    'HH_q+1_s1': ('0.026', '0.031', '0.033', '0.030', '0.034', '32.1'),
    'HeHe_q+1_s1': ('0.250', '0.253', '0.255', '0.250', '0.256', '2.3'),
    'HH_q+0_s0': ('0.415', '0.419', '0.419', '0.417', '0.420', '1.1'),
    'LiLi_q+0_s0': ('0.181', '0.171', '0.180', '0.169', '0.163', '10.5'),
    'NN_q+0_s0': ('0.169', '0.166', '0.167', '0.169', '0.166', '1.9'),
    'HLi_q+0_s0': ('0.259', '0.261', '0.264', '0.259', '0.261', '2.2'),
    'HB_q+0_s0': ('0.306', '0.298', '0.302', '0.301', '0.295', '3.9'),
    'HF_q+0_s0': ('0.085', '0.078', '0.084', '0.079', '0.075', '13.2'),
    'BB_q+0_s2': ('0.159', '0.153', '0.155', '0.154', '0.150', '5.8'),
}
TIER1_ORDER = ('LDA-VWN', 'PBE-PBE', 'BLYP', 'PBEsol', 'revPBE')

# --------------------------------------------------------------------------
# per-system figures quoted in Sec. 4.5
# --------------------------------------------------------------------------
H2_PEAK_EH = 0.42
H2_AT_DISSOCIATION_EH = 0.05
H2_DISSOCIATION_R = 5.29
H2_DECAY_FACTOR = 8.9
H2PLUS_PLATEAU_RESP = 1.2e-3          # e-/bohr^3, median over 2.0 < R < 3.5 A
HE2PLUS_PEAK_RESP = 1.5e-3
LIH_PEAK_EH = 0.26
LIH_PEAK_R = 2.00
LIH_TYPICAL_RESP = 1.0e-3
BH_TYPICAL_EH = 0.29
N2_NUNPAIRED_DISSOCIATED = 6.00
N2_NUNPAIRED_EQUILIBRIUM = 0.07
N2_VIRIAL_GAP_EH = -11.9
# 4.9 and not 6.5: the 6.5 was measured on the MIXED curve (14 geometries with
# a pure Fermi-Amaldi prior and 6 with the LDA-core guide), and the two that
# set it were among the six. Under the single convention the drift is 4.9, and
# the guide makes it worse -- the opposite of what the draft claimed.
N2_VIRIAL_DRIFT_EH = 4.9
B2_COLLAPSE_DENSITY_ERROR = (1.2, 2.6)     # e-, min and max
H2PLUS_ONE_ELECTRON_IDENTITY_EH = 3e-5     # vxc = -v_H, max violation

# Sec. S4.1: the degeneracy-cutoff audit.
#
# This used to pin four N2 geometries to the span/value ratios they showed in
# one run (R = 0.660, 0.924, 1.012, 1.430 -> 246, 2.7, 0.5, 368). That is not
# a reproducible quantity and pinning it was a mistake: which geometry is
# affected depends on where its near-degenerate pair falls relative to the
# cutoff, and the orbital energies move between runs. In the run of
# 2026-08-13 those four are all below 1.5e-3 of the value and two different
# N2 geometries are affected instead. What IS stable, and what Sec. S4.1 now
# claims, is the shape of the phenomenon: a small number of geometries across
# the benchmark, concentrated where orbitals are degenerate or the gap
# collapses, uncorrelated with the density-error gate -- and no reported
# quantity taken from one of them.
# A Table 1 row whose degeneracy-cutoff span reaches this fraction of the
# value is indeterminate by Sec. 2.5's own criterion and is not compared
# against a printed digit; Table 1 marks those rows with a dagger.
DEGTOL_INDETERMINATE_PCT = 100.0

DEGTOL_AFFECTED_MAX = 30          # of 203, PBE exact; a count, not a list
DEGTOL_AFFECTED_ADMITTED_MAX = 6  # some must be admitted, else it just
                                  # restates the gate

# --------------------------------------------------------------------------
# Sec. 4.1: the answer to "the protocol never visits the swept window".
#
# The objection is that the dispersion is measured over kappa values the
# discrepancy principle never returns, so it describes an unreachable region.
# It does not, and this is the measurement that settles it: at every geometry
# of Table 1, ALL THREE regularizations produce an inversion the confidence
# gate admits, and among those equally admissible inversions the potential
# metric spans up to 12.7x. If a rerun ever made the gate reject one of them,
# the paragraph in Sec. 4.1 would be wrong and would have to be rewritten, so
# this is checked rather than asserted.
#
# Thresholds are the production ones, per system: light, heavy (n_elec >= 12),
# or diffuse. HF's admitted geometries carry ghost/|drho| = 0.17-0.43, well
# above the 0.15 diffuse classifier, so they are gated at the diffuse level.
# --------------------------------------------------------------------------
# Bound over the twelve geometries once the indeterminate row is removed.
# 6.6 % is the LOWER BLOCK's bound, not the whole table's; conflating them was
# an error in the abstract until 2026-08-14.
KAPPA_BOUND_ALL_TWELVE_PCT = 9.4
KAPPA_BOUND_WITHOUT_LIH1600_PCT = 9.1
KAPPA_BOUND_LOWER_BLOCK_PCT = 6.6

KAPPA_GATE_THRESHOLD = {
    'HH_q+0_s0': 5e-3, 'NN_q+0_s0': 6e-2,
    'HLi_q+0_s0': 2e-2, 'HF_q+0_s0': 2e-2,
}
# Sec. 4.1 rests this argument only on the systems whose threshold assignment
# is unconditional. H2 (strict) and N2 (heavy) are assigned by electron count
# alone, so their eighteen inversions either pass or do not with nothing left
# to assume; they all pass. The LiH and HF blocks pass the DIFFUSE threshold
# their geometries carry in production, but that assignment is made by the
# ghost-leak fraction of each individual inversion and the sweep does not
# deposit it -- so eighteen further inversions are admissible only
# conditionally, and the paragraph says so instead of counting them.
KAPPA_UNCONDITIONAL_SYSTEMS = ('HH_q+0_s0', 'NN_q+0_s0')
KAPPA_UNCONDITIONAL_INVERSIONS = 18       # all of them inside the gate
KAPPA_UNCONDITIONAL_GEOMETRIES = 6
KAPPA_CONDITIONAL_INVERSIONS = 18         # LiH and HF, diffuse threshold
KAPPA_UNCONDITIONAL_SPREAD = (1.25, 12.7)  # x, over those six geometries
KAPPA_ADMISSIBLE_WORST_SPREAD = 12.7      # x, over inversions the gate admits
# Production kappa census over the admitted stratum, Sec. 4.1.
KAPPA_FINAL_CENSUS = {1e-6: 97, 1e-4: 20, 1e-5: 1}

# --------------------------------------------------------------------------
# Sec. 4.2: signal against noise. The median difference between neighbouring
# geometries, compared with the kappa band the paper publishes for the same
# system. In H2 the step is INSIDE the band, which is why no claim is made
# from one or two adjacent points, and why the secondary maximum is quoted
# with its size and its span rather than merely pointed at.
#   system -> median step between neighbours, %
# --------------------------------------------------------------------------
NEIGHBOUR_STEP_PCT = {
    'HH_q+0_s0': 2.3, 'NN_q+0_s0': 8.5, 'HLi_q+0_s0': 9.6,
    'HH_q+1_s1': 6.2, 'LiLi_q+0_s0': 9.2, 'HeHe_q+1_s1': 8.7,
}
NEIGHBOUR_STEP_TOL_PCT = 0.3
H2_TOTAL_RANGE_PCT = 128            # gross shape, far above the band
# The secondary maximum of Fig. 2b: from the interior minimum to the secondary
# peak, over four consecutive geometries.
H2_SECONDARY_RISE_PCT = 16.0
# The kappa band at the two geometries NEAREST the feature -- 1.628 and 2.028 A.
# It used to be quoted as 8.4 and 9.1 %, but 8.4 % is 1.073 A, the FARTHEST of
# the three swept geometries.
H2_BAND_NEAREST_PCT = (1.2, 9.1)
H2_BAND_NEAREST_A = (1.628, 2.028)
H2_SECONDARY_FROM_A = 1.739
H2_SECONDARY_TO_A = 2.312

# --------------------------------------------------------------------------
# Sec. S3: the functional-derivative identity, worst relative L1 over grid
# levels 3, 4 and 5 on CH. Three routes, and the point of the first is that
# it is exact: LSDA's potential is analytic, so any error there would be the
# machinery rather than the derivative expression.
#   route -> (printed worst rel(L1), tolerance as a factor)
# --------------------------------------------------------------------------
DERIV_ROUTES = {
    'LSDA spin (control)': (2.4e-14, 10.0),
    'PBE closed-shell (control)': (1.6e-07, 3.0),
    'PBE spin (under test)': (1.1e-05, 3.0),
}
# Above this the identity is structurally broken rather than quadrature-limited
# (a wrong second derivative changes the integrand by an O(0.1-1) fraction).
DERIV_BUG_LEVEL = 1e-2

# --------------------------------------------------------------------------
# SI controls (Secs. S11.4 and S11.5)
# --------------------------------------------------------------------------
# Table S11.4: (system, window, R of the maximum, max/dissociation ratio).
# The response peak stays inside a narrow band across every window; the
# potential metric's does not. That contrast is the claim, so both the
# positions and the ratios are checked, not just the qualitative reading.
FIXED_VOLUME_TABLE = [
    ('HH_q+1_s1', 'DeltaI_resp_production', 2.332, 2.5),
    ('HH_q+1_s1', 'DeltaI_resp_fixedL2',    2.014, 11.5),
    ('HH_q+1_s1', 'DeltaI_resp_fixedL3',    2.491, 7.5),
    ('HH_q+1_s1', 'DeltaI_resp_fixedL4',    2.904, 4.7),
    ('HH_q+0_s0', 'DeltaI_resp_production', 0.503, 2.3),
    ('HH_q+0_s0', 'DeltaI_resp_fixedL2',    2.028, 18.4),
    ('HH_q+0_s0', 'DeltaI_resp_fixedL3',    2.312, 10.6),
    ('HH_q+0_s0', 'DeltaI_resp_fixedL4',    2.767, 6.6),
]
# The band quoted in the text is 2.0-2.9 A; 2.95 is that band's upper edge at
# the precision it is printed with, not slack added to make a value fit.
FIXED_VOLUME_RESP_PEAK_RANGE_A = (2.0, 2.95)
FIXED_VOLUME_PEAK_SYSTEM = 'HH_q+1_s1'

# Table S11.5: (R, change in the potential metric %, change in the response
# metric %) between aug-cc-pVTZ and aug-cc-pVQZ, orbital basis only.
BASIS_TABLE = [(0.444, 74, 2.4), (0.681, 54, 0.6), (1.073, 91, 4.8),
               (1.184, 89, 6.3), (1.628, 48, 0.9), (2.028, 70, 6.3),
               (2.767, 4.6, 1.1)]
BASIS_CHANGE_FACTOR = 29          # median potential / median response
# The declared limitation: the ONE thing the paper claims about the potential
# metric -- where its maximum sits -- is itself basis dependent. Two estimators
# of the same displacement, both printed in Sec. S11.5 and both checked, so
# that a reader who recomputes either does not read the other as a discrepancy:
#   argmax over the sampled geometries  1.184 -> 0.851 A
#   spline, the estimator behind \PeakVol   1.148 -> 0.838 A
BASIS_PEAK_SHIFT_A = (1.184, 0.851)
BASIS_PEAK_SHIFT_SPLINE_A = (1.148, 0.838)
BASIS_PEAK_SHIFT_TOL_A = 2e-3

# Sec. S11.5, the numbers that replaced the retracted claim that the peak
# position is "robust". Whole-curve statistics over all 22 geometries of the
# comparison -- not the seven-row subset BASIS_TABLE prints, which is why they
# are stored separately: a median over a different set is a different number.
#   (median |change| %, max |change| %, TZ-QZ correlation)
BASIS_WHOLE_CURVE = {
    'potential': (59.7, 90.9, 0.281),
    'response':  (2.4, 6.5, 0.983),
}
# The claim that closes the basis objection: every basis-induced change in the
# response metric falls inside the kappa-invariance bound the paper already
# prints. If a rerun ever breaks this, the sentence in Sec. S11.5 and in the
# abstract is wrong, not the tolerance.
BASIS_RESP_INSIDE_KAPPA_BOUND_PCT = 9.4
# Peak of the RESPONSE curve, which is what the abstract's stability claim now
# rests on, and its interior features, which move more and are reported as
# moving more.
BASIS_RESP_PEAK_A = (0.502, 0.525)
BASIS_RESP_INTERIOR_MIN_A = (1.700, 1.837)
BASIS_RESP_SECONDARY_MAX_A = (2.426, 2.506)
# Sec. S1's physical scale: the peak-to-Coulson-Fischer separation. The
# potential metric's shift must exceed it (that is the retraction) and the
# response curve's secondary maximum must stay below it.
PEAK_TO_CF_SCALE_A = 0.157

# --------------------------------------------------------------------------
# Sec. S11.6: the pathology contrast at a matched reduced coordinate.
#
# This replaced a max/dissociation ratio that does not discriminate (2.5x for
# H2+ against 2.3x for H2, a factor of 1.09). What is checked here is the
# CLAIM, not a list of digits: one system grows at 1.5 R_e by every route and
# no other does, and the two populations do not overlap. Pinning the six
# ratios individually would pin interpolation noise; pinning the separation
# pins the sentence the paper prints.
# --------------------------------------------------------------------------
REDUCED_X = 1.5
REDUCED_GROWING_SYSTEM = 'HH_q+1_s1'
# Only the systems the two guards leave quotable. Li2 and N2 are NOT here:
# Li2's equilibrium separation is rejected by the gate and N2's is the
# degeneracy-cutoff geometry, so neither can be normalized at R_e. They are
# checked separately, as a direction rather than a value.
REDUCED_NEUTRALS = ('HH_q+0_s0', 'HLi_q+0_s0')
REDUCED_GROWING_RANGE = (1.19, 1.39)      # over the four functional/mode routes
REDUCED_NEUTRAL_RANGE = (0.62, 0.85)
# Sec. S11.6 quotes no digit for these two but does claim a direction, over
# both treatments of the compromised node. That claim is checked.
REDUCED_DECLARED = ('LiLi_q+0_s0', 'NN_q+0_s0')
REDUCED_DECLARED_BELOW = 1.0
REDUCED_TOL = 0.02
# He2+ is the case that is NOT claimed: it grows under PBE and decays under
# LDA, and Sec. S11.6 reports that disagreement rather than the persistence.
REDUCED_SPLIT_SYSTEM = 'HeHe_q+1_s1'
REDUCED_SPLIT = {'ratio_DeltaI_resp_PBE-PBE_exact': 1.23,
                 'ratio_DeltaI_resp_LDA-VWN_exact': 0.57}
# The potential metric fails to separate: it places H2 between the two cations
# under PBE and above H2+ under LDA. Checked, because Sec. S11.6 asserts it.
REDUCED_POT_H2_ABOVE_H2PLUS_LDA = True
# Four systems do not span R_e .. 1.5 R_e and are declared rather than
# estimated; the count is checked so that a rerun which silently gains or
# loses one is caught.
REDUCED_SPANNING = 4
REDUCED_NOT_SPANNING = ('HB_q+0_s0', 'HC_q+0_s1', 'BB_q+0_s2', 'HF_q+0_s0',
                        'LiLi_q+0_s0', 'NN_q+0_s0')

# --------------------------------------------------------------------------
# Sec. 2.6, Sec. 4.1 and Sec. S14.2: the three-stage suppression hierarchy.
#
# The paper used to say the projection annihilates the kappa dependence
# identically. It does not: if it did, the density-reproduction error would be
# kappa independent, and it moves. These are the three levels as printed.
#   (min, max) percentage change across kappa in [1e-7, 1e-5], H2 block
# --------------------------------------------------------------------------
KAPPA_HIERARCHY_PCT = {
    'potential': (304.0, 1169.0),
    'density_error': (10.0, 96.0),
    'response': (1.2, 9.1),
}
KAPPA_HIERARCHY_TOL_PCT = 0.10        # relative; these are ratios of ratios
# The point of the middle row: it is NOT flat. If a rerun ever made it flat,
# the retraction in Sec. 2.6 would be wrong and would have to be undone.
KAPPA_DENSITY_ERROR_MIN_SPREAD = 1.05

# --------------------------------------------------------------------------
# Sec. 2.3 and SI Sec. S16: the response metric against the self-consistent
# density difference. Ratios are MAPA / |rho_SCF - rho_ref| reduced the same
# way; ranges over admitted (and, for production, determinate) geometries.
# --------------------------------------------------------------------------
MVD_H2PLUS_RATIO = ('0.90', '1.10')            # LDA and PBE together
MVD_H2PLUS_RATIO_LDA = ('0.95', '1.07')        # SI S16.1
MVD_H2PLUS_COS_MIN = 0.996                     # "at least 0.996"
MVD_MODEL_SMALL_PCT = ('2', '35')              # H2, Li2, N2: MAPA exceeds by
MVD_MODEL_LIH = ('1.40', '3.47')               # SI; main text prints 1.4--3.5
MVD_MODEL_INV_PCT_MAX = '0.6'                  # |inverted/exact - 1|, percent
MVD_MODEL_INV_OVER_DV = ('0.07', '1.40')       # SI S16.2
MVD_PROD_N2_PBE = ('0.33', '4.35')
MVD_PROD_N2_PBE_BANDS = {(0.0, 0.95): ('1.14', '1.29'),
                         (1.0, 1.6): ('1.48', '4.35'),
                         (1.9, 2.8): ('0.33', '0.46')}
