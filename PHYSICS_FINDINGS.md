# Physics findings

Numbered record of the physical results and design decisions that shape this
implementation. Comments in the source refer to these items by number
(`see PHYSICS_FINDINGS.md #6`), so the numbering is stable: append rather than
renumber.

Each entry states what was found, what the alternative was, and what decided
between them. Where a decision rests on a measurement, the measurement is
quoted.

---

## 1. The magnitude of a potential-domain metric is set by the regularizer

Recovering `v_xc` from a correlated density is ill-posed, and the inversion is
stabilized by a smoothness penalty of weight `kappa`. Measured on the H2
density, a relative error of 1e-3 in `rho` amplifies to roughly 53x in the
potential. Consequently a metric defined directly on the potential difference
inherits `kappa` rather than the physics: across `kappa = 1e-9 .. 1e-5` the
mid-bond barrier of H2 moves over more than an order of magnitude while the
density error stays pinned at its basis-set floor.

**Consequence.** The magnitude of the potential-domain metric is not reported
as a physical quantity. What survives is its *position*: the location of the
maximum along the dissociation coordinate is stable against `kappa`.

## 2. The response projection removes that dependence

Projecting the potential error through the static Kohn-Sham linear response
function measures the density the error would displace instead of the error
itself. Two properties follow. Analytically, a constant shift of the potential
is annihilated by orthogonality of the orbitals, so the quantity is gauge
invariant by construction, and the null-space component that regularization
introduces is filtered out. Numerically, the projected magnitude varies by a
few percent across the same `kappa` range over which the unprojected metric
varies by factors of several.

**Consequence.** Position is read from the potential metric, magnitude from the
response metric. The two are complementary and neither replaces the other.

## 3. The static-correlation marker: unpaired-electron index, not spin squared

Three candidate markers of the onset of static correlation were compared along
dissociation curves:

| candidate | behaviour |
|---|---|
| Head-Gordon `N_unpaired` from the reference natural occupations | smooth, branch-free, parameter-free, and ~0 at equilibrium |
| occupation-based `<S^2>` (square-root form) | recognizable, but amplifies the dynamic-correlation tail |
| raw broken-symmetry `<S^2>` | oscillates with the symmetry-breaking branch |

`N_unpaired` is the primary marker. It suppresses the dynamic-correlation tail
that contaminates the `<S^2>` forms, it does not depend on which
symmetry-broken solution the SCF happens to find, and it saturates at the
integer number of unpaired electrons of the separated fragments — 2 for a
broken single bond, 4 for the B2 triplet, 6 for the N2 triple bond — which
makes it verifiable against known chemistry rather than merely monotonic.

The two `<S^2>` forms are retained as secondary diagnostics and exported, but
no classification depends on them.

## 4. Two distinct objects are both called `<S^2>`

The `<S^2>` of the *reference wavefunction* and the `<S^2>` of the *inverted
ensemble Kohn-Sham system* are different quantities and must not be conflated.
For a singlet reference the former is exactly zero: the wavefunction is a spin
eigenstate. The latter grows sigmoidally along dissociation.

This is representational, not spin contamination. An auxiliary Kohn-Sham
determinant requires spin polarization in order to reproduce a multideterminant
density that no restricted determinant can generate. Reporting the second
quantity as if it were the first would misdescribe the reference.

## 5. Fractional occupations demand occupied-occupied transitions

The linear response function is summed over all ordered orbital pairs, with no
restriction of the first index to the occupied set and no compensating factor
of two. With integer occupations the occupied-occupied terms vanish
identically, because `n_i - n_j = 0` inside the occupied block, and the
familiar occupied-virtual form is recovered. That cancellation fails as soon as
the reference is multi-reference and the Kohn-Sham occupations are fractional:
occupied-occupied pairs with `n_i != n_j` then contribute, and most strongly
where static correlation is strongest.

Retaining the occupied-virtual form and doubling it double-counts those pairs.
This is a requirement for operating on stretched bonds, not a refinement.

## 6. Architecture: ensemble primary, broken symmetry diagnostic

A stretched correlated closed-shell density is not, in general, representable
by a single restricted Kohn-Sham determinant. Two routes were available.

The **ensemble** route inverts against an auxiliary system with fixed
fractional occupations taken from the reference natural occupations. It
reproduces the density exactly and yields a `v_xc` that is symmetric by
construction: there is no branch to choose.

The **broken-symmetry** route reaches a comparable density by breaking spatial
or spin symmetry, but the result depends on which branch the SCF finds, and the
potential must be symmetrized after the fact.

The ensemble channel is therefore **primary** and supplies the reported
potential. Broken symmetry is retained as a **diagnostic** only, because it
supplies the raw `<S^2>` marker (item 3). Both potentials and their difference
are exported so the choice remains auditable.

The ensemble scheme is applied at *every* geometry, not only after a gap
collapse is detected. Switching schemes mid-curve introduces a spurious step in
the metric — the largest neighbour-to-neighbour ratio of the metric across the transition rising from 1.73 with the unified scheme to 2.04 when a mid-curve switch is forced — which the
unified scheme removes by construction.

Open shells run the same machinery per spin channel, each channel carrying the
fractional natural occupations of its own reference spin-density matrix rather
than an integer aufbau filling. This gives one inversion framework across the
whole set.

## 7. The reference ladder is limited by sigma-build cost, not memory

Reference densities come from a graded ladder (SCF -> CASSCF -> CASCI -> FCI).
The binding constraint on how large a CASCI space is affordable is not the
memory of the CI vector but the cost of the Davidson sigma build, which scales
as `N_str * M^4` in the number of active orbitals `M` and dominates the run
time. Selecting the space by a memory budget admits spaces that are
intractable in time: a system with few valence electrons fits a large orbital
space in memory while being far more expensive than a system with many valence
electrons in a smaller space.

**Consequence.** The ladder is capped by a ceiling on the sigma-build cost,
with full-valence FCI reserved for the smallest systems, where it is both
affordable and where it measurably improves the inversion.

## 8. Validation of the ensemble inverter

The ensemble inverter is a small, targeted override of the validated
engine: only the density build, the occupied-eigenvalue sum in the objective,
and the occupation weights of the response Hessian change. The occupations are
physical inputs and are not optimized.

Measured performance:

| system | result |
|---|---|
| H2 | density error 4.75e-4; potential asymmetry 2.1e-5 |
| N2, CAS(6,6) | bond-region error ~1.9e-3 |

For the one-electron cation the inversion is analytic (`v_xc = -v_H`), and the
implementation reproduces it to <= 3e-5 Eh across the full scan (worst case 2.6e-5). That case
serves as a standing check on the exact side of the pipeline.

## 9. The confidence criterion is the density error, and nothing else

Every geometry is classified solely by how well the inverted density
reproduces the reference. Thresholds are stratified by physics rather than by
electron count: a base threshold for compact light systems, a relaxed one for
many-electron systems whose densities the auxiliary basis reproduces to about
1%, and a third for diffuse densities, detected by the fraction of the error
that leaks to the far field.

The metric magnitude is deliberately *not* part of the criterion. The response
metric remains finite on a failed inversion — it is a valid response of
whatever auxiliary system the inversion returned — so a plausible-looking value
can accompany a reference that has collapsed to the wrong state. Agreement with
the reference density is what certifies that the potential belongs to the right
physical state.

Failed geometries are written to disk like any other. The contrast between the
admitted and rejected strata is evidence that the criterion works, not
something to be pruned.

## 10. Spatial signatures separate the pathologies

Self-interaction error and static correlation leave opposite spatial
signatures along a dissociation curve. The self-interaction signature is
long-ranged and persistent: it does not switch off as the fragments separate,
and the response metric falls to a non-vanishing plateau. The
static-correlation signature is confined to the bond-breaking region and
vanishes at dissociation.

The one-electron limit is the sharpest case. There the exact potential is
`-v_H`, and a gradient correction has no physical counterpart to reproduce: PBE
deposits a large error there while the local functional does not, differing by
two orders of magnitude on H2+ while the two agree within a few percent
everywhere else.

## 11. Error cancellation lives in the integral, not at the peak

Comparing each functional evaluated on the exact density against the same
functional evaluated on its own self-consistent density isolates the
cancellation between a wrong potential and the density it produces. At the peak
of the dissociation curve the two evaluations agree closely for every system
studied except the one-electron cation. Integrated along the curve the
difference is several percent.

**Consequence.** The cancellation is sought in the integral of the curve, which
is where it resides, rather than at its maximum.

## 12. The virial diagnostic does not certify the bonding region

The Levy-Perdew virial relation is dominated by the core. On N2 with a pure
Fermi-Amaldi prior the virial gap reaches -10.9 Eh, concentrated within 0.2
bohr of the nuclei, and drifts by 2.4 Eh between geometries because the
saturated core density cannot counteract the regularizer's pull toward the
guide. Optimizing the prior with an LDA-core guide removes the drift without
stiffening `kappa`.

**Consequence.** The virial gap is reported as a diagnostic of the core, and is
not used to certify the interatomic region that the metric measures.
