# Physics findings

Numbered record of the physical results and design decisions that shape this
implementation. Comments in the source refer to these items by number
(`see PHYSICS_FINDINGS.md #6`), so the numbering is stable: append rather than
renumber. The figures quoted here are those of the article; where an item and
the article differ, the article is correct.

Each entry states what was found, what the alternative was, and what decided
between them. Where a decision rests on a measurement, the measurement is
quoted.

---

## 1. The magnitude of a potential-domain metric is set by the regularizer

Recovering `v_xc` from a correlated density is ill-posed, and the inversion is
stabilized by a smoothness penalty of weight `kappa`. Measured on the H2
density, a relative perturbation of 1e-3 in `rho` over the bond region
produces a relative change in the potential about 53 times larger. A metric
defined directly on the potential difference therefore depends on `kappa`
rather than on the physics: over the usable window `kappa = 1e-7 .. 1e-5` it
varies by up to 12.8x, while the density-reproduction error moves by at most a
factor of 2.1.

**Consequence.** The magnitude of the potential-domain metric is not reported
as a physical quantity. The position of its maximum along the dissociation
coordinate is stable against `kappa` and, in H2, against fixed integration
windows, but it moves by 0.31 A when the orbital basis is doubled
(aug-cc-pVTZ to aug-cc-pVQZ). It is therefore reported only as a localization
in a stated basis.

## 2. The response projection damps that dependence

Projecting the potential error through the static Kohn-Sham linear response
function measures the density the error would displace instead of the error
itself. Analytically, a constant shift of the potential gives zero by
orthogonality of the orbitals, so the quantity is gauge invariant, and the
null-space component of `chi` is removed exactly. The `kappa` dependence is
not confined to that null space; it lies mostly in ill-conditioned directions,
which the projection damps rather than removes. Numerically, over
`kappa = 1e-7 .. 1e-5` the projected metric varies by at most 9.4 %, against
up to 12.8x for the unprojected one, and by at most 6.5 % when the orbital
basis is doubled.

**Consequence.** The location of a failure is read from the potential metric,
in a stated basis; its magnitude is read from the response metric.

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
integer number of unpaired electrons of the separated fragments (2 for a
broken single bond, 4 for the B2 triplet, 6 for the N2 triple bond), so it
can be checked against known chemistry.

The two `<S^2>` forms are retained as secondary diagnostics and exported, but
no classification depends on them.

## 4. Two distinct objects are both called `<S^2>`

The `<S^2>` of the *reference wavefunction* and the `<S^2>` of the *inverted
ensemble Kohn-Sham system* are different quantities and must not be conflated.
For a singlet reference the former is exactly zero: the wavefunction is a spin
eigenstate. The latter grows sigmoidally along dissociation.

This is a feature of the representation, not spin contamination. An auxiliary
Kohn-Sham determinant requires spin polarization in order to reproduce a
multideterminant density that no restricted determinant can generate.
Reporting the second quantity as if it were the first would describe the
reference incorrectly.

## 5. Fractional occupations require occupied-occupied transitions

The linear response function is summed over all ordered orbital pairs, with no
restriction of the first index to the occupied set and no compensating factor
of two. With integer occupations the occupied-occupied terms vanish
identically, because `n_i - n_j = 0` inside the occupied block, and the
familiar occupied-virtual form is recovered. That cancellation fails as soon as
the reference is multi-reference and the Kohn-Sham occupations are fractional:
occupied-occupied pairs with `n_i != n_j` then contribute, and most strongly
where static correlation is strongest.

Retaining the occupied-virtual form and doubling it double-counts those pairs.
Restricting the sum to occupied-virtual pairs changes the response metric by a
median of 29 % over the admitted geometries and by more than 90 % in He2+, so
these pairs are required on stretched bonds.

## 6. Architecture: ensemble primary, broken symmetry diagnostic

A stretched correlated closed-shell density is not, in general, representable
by a single restricted Kohn-Sham determinant. Two routes were available.

The **ensemble** route inverts against an auxiliary system with fixed
fractional occupations taken from the reference natural occupations. It
reproduces the density without breaking symmetry and yields a `v_xc` that is
symmetric by construction: there is no branch to choose.

The **broken-symmetry** route reaches a comparable density by breaking spatial
or spin symmetry, but the result depends on which branch the SCF finds, and the
potential must be symmetrized after the fact.

The ensemble channel is therefore **primary** and supplies the reported
potential. Broken symmetry is retained as a **diagnostic** only, because it
supplies the raw `<S^2>` marker (item 3). Both potentials and their difference
are exported so the choice remains auditable.

The ensemble scheme is applied at *every* geometry, not only after a gap
collapse is detected. Switching schemes mid-curve introduces a spurious step in
the metric: the largest neighbour-to-neighbour ratio of the metric across the
transition is 1.73 with the unified scheme, and 2.04 or 6.66 when a mid-curve
switch is forced, depending on where the switch is triggered. The unified
scheme removes that step by construction.

Open shells run the same machinery per spin channel, each channel carrying the
fractional natural occupations of its own reference spin-density matrix rather
than an integer aufbau filling. This gives one inversion framework across the
whole set.

## 7. The reference ladder is limited by sigma-build cost, not memory

Reference densities come from a graded ladder (SCF -> CASSCF -> CASCI -> FCI).
The limit on how large a CASCI space is affordable is not the memory of the CI
vector but the cost of the Davidson sigma build, which scales as
`N_str * M^4` in the number of active orbitals `M` and dominates the run time.
Selecting the space by a memory budget admits spaces that are intractable in
time: a system with few valence electrons fits a large orbital space in memory
while being far more expensive than a system with many valence electrons in a
smaller space.

**Consequence.** The ladder is capped by a ceiling on the sigma-build cost,
with full-valence FCI reserved for the smallest systems (four electrons or
fewer), where it is both affordable and where it measurably improves the
inversion.

## 8. Validation of the ensemble inverter

The ensemble inverter changes only a small part of the validated engine: the
density build, the occupied-eigenvalue sum in the objective, and the
occupation weights of the response Hessian. The occupations are physical
inputs and are not optimized.

Measured performance:

| system | result |
|---|---|
| H2 | density error 4.75e-4; potential asymmetry 2.1e-5 |
| N2, CAS(6,6) | bond-region error ~1.9e-3 |

For the one-electron cation the exact potential is known analytically
(`v_xc = -v_H`), and the inversion reproduces it to <= 3e-5 Eh across the full
scan (worst case 2.6e-5). That case is a permanent check on the exact side of
the pipeline.

## 9. The confidence criterion is the density error, and nothing else

Every geometry is classified solely by how well the inverted density
reproduces the reference. Thresholds depend on the system rather than on
electron count alone: 5e-3 e- for compact light systems, 6e-2 e- for
many-electron systems whose densities the auxiliary basis reproduces to about
1 %, and 2e-2 e- for diffuse densities, detected by the fraction of the error
that leaks to the far field. 118 of the 203 geometries are admitted.

The metric magnitude is deliberately *not* part of the criterion. The response
metric remains finite on a failed inversion, because it is a valid response of
whatever auxiliary system the inversion returned, so a plausible-looking value
can accompany a reference that has collapsed to the wrong state. Agreement with
the reference density is what shows that the potential belongs to the right
physical state.

Failed geometries are written to disk like any other. The contrast between
admitted and rejected geometries shows that the criterion works, so the
rejected ones are kept.

## 10. Spatial signatures separate the pathologies

Self-interaction error and static correlation leave different spatial
signatures along a dissociation curve. The self-interaction signature of H2+
does not switch off as the fragments separate: its response metric rises to a
plateau of about 1.2e-3 e-/bohr^3 between 2.2 and 2.7 A. The
static-correlation signatures of H2, Li2, N2 and LiH are confined to the
bond-breaking region and decay as the bond opens.

The comparison is made at a matched reduced coordinate: the response metric at
1.5 R_e divided by its value at R_e. H2+ is the only system still growing by
every route (1.19-1.39 across LDA and PBE and both density modes), against
0.62-0.85 for H2 and LiH. The potential metric does not reproduce this
separation.

## 11. Error cancellation appears in the integral, not at the peak

Comparing each functional evaluated on the exact density against the same
functional evaluated on its own self-consistent density separates the
cancellation between a wrong potential and the density it produces. In H2,
across five LDA and GGA functionals, the two evaluations differ by at most
0.17 % at the peak of the dissociation curve and by 2.7-5.6 % in its integral.

**Consequence.** The cancellation is measured in the integral of the curve,
not at its maximum.

## 12. The virial diagnostic does not certify the bonding region

The Levy-Perdew virial relation is dominated by the core. On N2 with the
Fermi-Amaldi prior the virial gap reaches -11.9 Eh, concentrated within
0.2 bohr of the nuclei, and drifts by up to 4.9 Eh between neighbouring
geometries. An LDA-core guide at the six most stretched geometries does not
repair it (drift 6.5 Eh) and leaves the response metric unchanged to 0.6 %.

**Consequence.** The virial gap is reported as a diagnostic of the core, and is
not used to certify the interatomic region that the metric measures. A single
Fermi-Amaldi prior is used at every geometry.
