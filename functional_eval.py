"""
functional_eval.py — real-space DeltaI evaluation for ANY functional
=====================================================================
This module is DECOUPLED from the heavy pipeline. The pipeline computes the
functional-agnostic exact side ONCE (Wu-Yang inversion -> exact multiplicative
v_xc, reference density) and saves it per geometry as an .npz. This module
reads those .npz files, rebuilds the (cheap) molecule from the stored metadata,
and evaluates the real-space deviation metric DeltaI_bond for whatever
functional(s) the control room (main.py) requests -- without ever rerunning
the expensive inversion or CASSCF. A new functional invented years later can be
scored against the saved exact side with only this cheap post-process.

TWO DENSITY MODES (both requested; their DIFFERENCE is the error-cancellation
signal):
  * 'exact' : the functional's v_xc evaluated ON THE EXACT (reference) density.
              Isolates the pure functional-FORM error.
  * 'scf'   : the functional's v_xc on its OWN self-consistent density (a cheap
              KS-SCF with that functional). Includes the density-driven error;
              exact-vs-scf difference = error cancellation.

THREE FUNCTIONAL TIERS (the real dividing line is multiplicative vs
orbital-dependent, NOT semi-local vs hybrid):
  * TIER 1  MULTIPLICATIVE (LDA, GGA): v_xc is a true multiplicative grid
            potential; the DeltaI comparison against the exact multiplicative
            v_xc is rigorous. FULLY IMPLEMENTED & (to be) VALIDATED.
  * TIER 2  META-GGA (SCAN, r2SCAN, ...): v_xc has the multiplicative rho/sigma
            part PLUS a tau-dependent ORBITAL-DEPENDENT part (generalized KS).
            Comparing to a multiplicative exact v_xc requires a CONVENTION for
            the tau part. Provided as a clearly-flagged extension point.
            [NOT YET VALIDATED — see tau_convention below]
  * TIER 3  HYBRID (B3LYP, ...): nonlocal exact-exchange fraction; the KS
            multiplicative potential requires an OEP treatment. Extension
            point only. [NOT YET VALIDATED — needs OEP]

Validate TIER 1 before publication with test_functional_vxc_validation.py,
which checks Int v(r) drho(r) dr == Tr[V_xc_mat dDM] (the functional-derivative
identity) against pyscf's own matrix elements.
"""

import os
import glob
import json
import numpy as np
from pyscf import dft, gto, scf

from config import CFG, BOHR
import inversion as iv
import scipy.linalg
import molecules as mo
import metrics as mt
import baselines as bl


# ===========================================================================
# functional resolution: name/string/callable -> rung + evaluator
# ===========================================================================
class FunctionalInfo:
    """Describes a functional and how to evaluate its v_xc. Constructed from a
    libxc string (e.g. 'PBE,PBE', 'SCAN,SCAN'), or a user callable for a
    brand-new functional not in libxc."""

    def __init__(self, name, xc_string=None, custom_vxc=None,
                 tau_convention='none'):
        self.name = name
        self.xc_string = xc_string
        self.custom_vxc = custom_vxc          # callable(rho,grho,tau,...) -> v
        self.tau_convention = tau_convention  # for meta-GGA (see Tier 2)
        self.rung = self._detect_rung()
        self.hybrid_coeff = self._detect_hybrid()

    def _detect_rung(self):
        if self.custom_vxc is not None:
            return 'CUSTOM'
        t = dft.libxc.xc_type(self.xc_string)   # 'LDA'|'GGA'|'MGGA'|'HF'|...
        return {'LDA': 'LDA', 'GGA': 'GGA', 'MGGA': 'MGGA',
                'HF': 'HYBRID'}.get(t, t)

    def _detect_hybrid(self):
        if self.custom_vxc is not None:
            return 0.0
        try:
            return float(dft.libxc.hybrid_coeff(self.xc_string))
        except Exception:                       # noqa: BLE001
            return 0.0

    @property
    def is_multiplicative(self):
        """True only for the rigorous Tier-1 case: pure KS multiplicative
        potential. Meta-GGA and hybrids are generalized KS."""
        return self.rung in ('LDA', 'GGA') and self.hybrid_coeff == 0.0


def resolve_functional(spec):
    """Accept a functional in any of the supported forms:
      * str            -> libxc name, e.g. 'PBE,PBE' or 'SCAN,SCAN'
      * (name, str)    -> display name + libxc string
      * FunctionalInfo -> passed through
      * dict           -> {'name','xc_string'|'custom_vxc','tau_convention'}
    Returns a FunctionalInfo."""
    if isinstance(spec, FunctionalInfo):
        return spec
    if isinstance(spec, str):
        return FunctionalInfo(spec, xc_string=spec)
    if isinstance(spec, tuple) and len(spec) == 2:
        return FunctionalInfo(spec[0], xc_string=spec[1])
    if isinstance(spec, dict):
        return FunctionalInfo(spec['name'],
                              xc_string=spec.get('xc_string'),
                              custom_vxc=spec.get('custom_vxc'),
                              tau_convention=spec.get('tau_convention', 'none'))
    raise ValueError(f"Unrecognized functional spec: {spec!r}")


# ===========================================================================
# v_xc on the grid, dispatched by tier
# ===========================================================================
def functional_vxc_on_grid(finfo, mol, dm, coords):
    """Return the functional's v_xc on the grid for density matrix dm.
    TIER 1 is rigorous; TIER 2/3 raise unless a convention is set.

    NOTE (closed shell here for clarity; the pipeline's spin-polarized path
    reuses baselines.pbe_vxc_spin_on_grid for GGA — the same tier logic
    applies)."""
    rung = finfo.rung

    if rung == 'LDA':
        rho, _, _, _ = bl.density_derivs(mol, dm, coords)
        _, vxc, _, _ = dft.libxc.eval_xc(finfo.xc_string,
                                         np.maximum(rho, 1e-15), deriv=1)
        return vxc[0], {'tier': 1}

    if rung == 'GGA':
        # rigorous multiplicative GGA potential: vrho - 2 div(vsigma grad rho)
        rho, grho, hess, lap = bl.density_derivs(mol, dm, coords)
        rho_gga = np.vstack([np.maximum(rho, 1e-15), grho])
        _, vxc, fxc, _ = dft.libxc.eval_xc(finfo.xc_string, rho_gga, deriv=2)
        vrho, vsigma = vxc[0], vxc[1]
        v2rhosigma, v2sigma2 = fxc[1], fxc[2]
        grad_sigma = 2.0 * np.einsum('abg,bg->ag', hess, grho)
        grad_vsigma = v2rhosigma * grho + v2sigma2 * grad_sigma
        v = vrho - 2.0 * (np.einsum('ag,ag->g', grad_vsigma, grho)
                          + vsigma * lap)
        return v, {'tier': 1}

    if rung == 'MGGA':
        # TIER 2 EXTENSION POINT. The rho/sigma part is multiplicative; the
        # tau part is orbital-dependent (generalized KS). A convention is
        # required to project it onto a grid potential for the DeltaI
        # comparison against the exact MULTIPLICATIVE v_xc.
        if finfo.tau_convention == 'none':
            raise NotImplementedError(
                f"'{finfo.name}' is meta-GGA: its tau-dependent potential is "
                "orbital-dependent (generalized KS), so comparing it to the "
                "exact multiplicative v_xc needs a tau convention. Set "
                "tau_convention explicitly (e.g. 'multiplicative_rho_sigma' to "
                "compare only the multiplicative part, or a future OEP-style "
                "projection). NOT YET VALIDATED — do not use for publication "
                "figures until test_functional_vxc_validation.py passes for it.")
        if finfo.tau_convention == 'multiplicative_rho_sigma':
            # STUB — NOT IMPLEMENTED CORRECTLY YET. A previous draft returned
            # vrho only, silently dropping the sigma gradient term its name
            # promises (and tau). Until the full vrho - 2 div(vsigma grad rho)
            # meta-GGA evaluation is implemented AND passes
            # test_functional_vxc_validation.py, refuse rather than return a
            # wrong potential with a plausible-looking name.
            raise NotImplementedError(
                f"'{finfo.name}' tau_convention='multiplicative_rho_sigma' is "
                "a stub: the sigma-gradient term of the multiplicative part "
                "is not implemented yet (a draft returned vrho only, which is "
                "NOT rho+sigma). Implement + validate before use.")
        raise ValueError(f"Unknown tau_convention {finfo.tau_convention!r}")

    if rung == 'HYBRID' or finfo.hybrid_coeff != 0.0:
        # TIER 3 EXTENSION POINT. Nonlocal exact exchange -> the KS
        # multiplicative potential requires OEP. Not implemented.
        raise NotImplementedError(
            f"'{finfo.name}' is a hybrid (exact-exchange fraction "
            f"{finfo.hybrid_coeff:.3f}): its KS multiplicative potential needs "
            "an OEP treatment, not yet implemented. Extension point only.")

    if rung == 'CUSTOM':
        rho, grho, _, _ = bl.density_derivs(mol, dm, coords)
        tau = _tau_on_grid(mol, dm, coords)
        return finfo.custom_vxc(rho, grho, tau, coords), {'tier': 'custom'}

    raise ValueError(f"Unhandled functional rung {rung!r} for {finfo.name!r}")


def functional_vxc_spin_on_grid(finfo, mol, dma, dmb, rho_a, rho_b,
                                 coords):
    """Spin-resolved (v_alpha, v_beta) for open-shell systems, same tier
    logic as the closed-shell dispatcher. LDA needs only the spin densities;
    GGA needs the spin DMs (for the gradient terms). Tier 2/3 gate as usual."""
    rung = finfo.rung
    if rung == 'LDA':
        return bl.lsda_vxc_spin(rho_a, rho_b, xc=finfo.xc_string), {'tier': 1}
    if rung == 'GGA':
        if dma is None or dmb is None:
            raise RuntimeError(
                "spin-resolved GGA needs exact_dm_alpha/beta in the npz "
                "(saved by pipeline versions >= this one); re-run the point "
                "or use the total-density column only")
        va, vb = bl.pbe_vxc_spin_on_grid(mol, dma, dmb, coords,
                                         xc=finfo.xc_string)
        return (va, vb), {'tier': 1}
    if rung == 'MGGA':
        raise NotImplementedError(
            f"'{finfo.name}' is meta-GGA: spin-resolved comparison needs the "
            "same tau convention as the total-density path (Tier 2, not yet "
            "validated).")
    raise NotImplementedError(
        f"'{finfo.name}' (rung {rung}): spin-resolved path implemented only "
        "for Tier-1 LDA/GGA.")


def _tau_on_grid(mol, dm, coords):
    """Kinetic energy density tau(r) = 1/2 sum_i |grad psi_i|^2, built from the
    density matrix as 1/2 sum_uv D_uv grad phi_u . grad phi_v."""
    ao1 = mol.eval_gto('GTOval_sph_deriv1', coords)   # (4,N,nao)
    g = ao1[1:4]                                        # (3,N,nao)
    tau = 0.5 * np.einsum('dgi,ij,dgj->g', g, dm, g)
    return tau


# ===========================================================================
# per-geometry and per-system evaluation
# ===========================================================================
def _rkey(R):
    """Canonical R key for the curve dict and the CSV merge. MUST be identical
    on both sides. Python's round() and NumPy's round() DISAGREE on values with
    a 5 in the third decimal (banker's rounding of the float repr): e.g.
    round(1.595,2)=1.59 but np.round(1.595,2)=1.6. Storing keys with one and
    looking them up with the other silently dropped every such geometry to NaN
    (that was the H2 R=1.295 and N2 R=1.595/1.925/2.255 failures). Fixed-format
    string is deterministic and identical everywhere."""
    return f"{float(R):.2f}"


def _rebuild_mol(meta):
    """Rebuild the (cheap) molecule from the npz metadata, so functional eval
    never needs the original expensive run."""
    basis = meta.get('benchmark_basis', meta.get('basis', 'aug-cc-pVTZ'))
    return mo.build_molecule(meta['a1'], meta['a2'], float(meta['R_ang']),
                             int(meta['charge']), int(meta['spin']),
                             basis=basis)


def _load_geometry_context(npz_path):
    """Load ONE geometry's npz ONCE and package everything every functional x
    density-mode evaluation needs. np.load is lazy, so only the accessed
    arrays are read (ao_value etc. stay on disk). The context also carries a
    per-functional SCF cache so 'scf' mode runs one KS-SCF per functional per
    geometry, never more."""
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d['meta']))

    def opt(key):
        if key in d.files:
            a = np.asarray(d[key])
            return a if a.size else None
        return None

    ctx = dict(meta=meta, R=float(meta['R_ang']),
               coords=d['coords_bohr'], w=d['weights'],
               vxc_exact=d['vxc_exact'], rho_exact=d['rho_exact'],
               exact_dm=d['exact_dm'], spin=int(meta.get('spin', 0)),
               rho_a=opt('rho_alpha'), rho_b=opt('rho_beta'),
               vxc_a=opt('vxc_alpha'), vxc_b=opt('vxc_beta'),
               dma=opt('exact_dm_alpha'), dmb=opt('exact_dm_beta'),
               D_ks=opt('D_ks'), scf_cache={})
    ctx['mol'] = _rebuild_mol(meta)
    ctx['open_shell'] = (ctx['spin'] != 0 and ctx['rho_a'] is not None
                         and ctx['rho_b'] is not None)
    return ctx


def _sanitize_grid_potential(v_func, v_exact, name, R,
                             abs_floor=64, frac_cap=1e-2,
                             report_huge=CFG.dv_report_Eh):
    """Replace non-finite points in a functional's grid potential before the
    metric. A GGA potential built from libxc fxc components can produce a
    handful of NaN/inf grid points where the reference density is near-zero or
    the sigma factors blow up; a single such point makes delta_metrics return
    NaN for the ENTIRE geometry (that was 3 of 20 N2 exact-mode points). These
    are integration-measure-zero artifacts, not physics.

    An artifact is a SMALL number of ISOLATED points: at most `abs_floor`
    points (a fixed handful, grid-size independent) OR under `frac_cap` of the
    grid. Anything larger is a genuine evaluation failure and is refused so it
    surfaces as missing data rather than a silently patched wrong number.
    Neutralized points are set to v_exact, i.e. Dv=0 there, contributing
    nothing to the |Dv| integral.

    DELIBERATELY NOT A MAGNITUDE THRESHOLD (see D2). The obvious-looking
    extension — also replace points where |v_func - v_exact| exceeds some cap
    — was considered and rejected. It would have masked the H2+/PBE artifact
    instead of explaining it (the real cause was averaging in a spin channel
    with no electrons; see _spin_average_potential), and it would have put a
    tunable energy scale inside the metric, which is exactly the "hidden free
    parameter" the abstract says the metric does not have. A finite but
    physically impossible potential must surface as a LOUD diagnostic and be
    diagnosed, not be clipped. Hence `report_huge` below: it warns, it counts,
    it never edits the data.
    """
    if report_huge is not None:
        dv = np.abs(np.asarray(v_func) - np.asarray(v_exact))
        huge = int(np.sum(np.isfinite(dv) & (dv > report_huge)))
        if huge:
            print(f"  [HUGE-DV] {name} R={R:.3f} A: {huge} grid points with "
                  f"finite |Dv| > {report_huge} Eh (max "
                  f"{np.nanmax(dv[np.isfinite(dv)]):.3e}). NOT modified — "
                  f"diagnose before trusting this geometry's DeltaI.")
    bad = ~np.isfinite(v_func)
    nbad = int(bad.sum())
    if nbad == 0:
        return v_func
    frac = nbad / v_func.size
    if nbad > abs_floor and frac > frac_cap:
        raise RuntimeError(
            f"{name} grid potential has {nbad} non-finite points "
            f"({100*frac:.2f}%) at R={R:.3f} A -- too many to be point "
            f"artifacts; the functional evaluation failed here. Reported as "
            f"missing.")
    v = v_func.copy()
    v[bad] = v_exact[bad]            # Dv = 0 there: contributes nothing to |Dv|
    return v


def _spin_average_potential(va, vb, rho_a, rho_b, w):
    """The single spin-averaged potential the total-channel metric compares
    against the (spin-averaged) exact v_xc. Returns (v_tot, provenance).

    FIX-D2. The rule used to be an unconditional v_tot = (v_a + v_b)/2. That
    is right when both channels are occupied and WRONG when one of them is
    empty, because an empty channel has no potential to average in — what
    libxc returns there is the value of a derivative evaluated at the density
    FLOOR, not a physical quantity.

    Measured on H2+ (one electron, rho_beta identically zero), PBE, R=0.64 A,
    aug-cc-pVTZ, grid level 3:

        v_tot = (v_a+v_b)/2      DeltaI_bar = 8.119 Eh   <- Table S10's 8.132
        alpha channel only       DeltaI_bar = 0.0138 Eh

    and the 8.119 tracks the floor rather than any physics: pushing the
    beta-density floor from 1e-15 to 1e-10 moves it to 1.786 Eh while the
    alpha channel does not move in the fourth decimal. The mass of the
    integral also sits at rho > 1e-2 and 0.5 < r < 1.0 bohr, i.e. in the DENSE
    part of Omega, which rules out the low-density-corona reading as well.
    The beta term is an artifact of evaluating a spin derivative on a channel
    with no electrons; it is not a large-but-finite GGA value that a magnitude
    threshold should be invented to catch.

    So: a channel that holds no electrons does not vote. Every genuinely
    open-shell system (He2+, B2, CH) has both channels occupied and is
    numerically unchanged by this; H2+ is the only member of the benchmark
    with an empty channel, and it is the only row that moves. NO free
    parameter enters the metric — `occ_tol` is an integer-electron-count test
    (channels hold 0 or >= 1 electrons; nothing sits near 1e-8), not a
    tunable knob.
    """
    occ_tol = 1e-8
    na = float(np.sum(np.asarray(rho_a) * w)) if rho_a is not None else 0.0
    nb = float(np.sum(np.asarray(rho_b) * w)) if rho_b is not None else 0.0
    occ_a, occ_b = na > occ_tol, nb > occ_tol
    if occ_a and occ_b:
        return 0.5 * (va + vb), {'spin_average': 'both',
                                 'n_alpha': na, 'n_beta': nb}
    if occ_a:
        return va, {'spin_average': 'alpha_only', 'n_alpha': na,
                    'n_beta': nb}
    if occ_b:
        return vb, {'spin_average': 'beta_only', 'n_alpha': na,
                    'n_beta': nb}
    raise RuntimeError(
        "both spin channels are empty on the grid — the density fed to the "
        "metric integrates to zero; this is a broken geometry, not a result.")


def _density_for_mode(ctx, finfo, density_mode):
    """THE single place that answers 'which density does this (functional,
    mode) pair use?'. Returns a dict with the total density matrix and grid
    density, plus the spin-resolved pieces when they exist.

    FIX-D1. This used to be inlined in deltaI_for_geometry only, so the 1D
    profile writer had no way to ask the question and silently used the exact
    density for BOTH modes — the labels branched, the physics did not. One
    function, two callers, no way for them to disagree.
    """
    if density_mode == 'exact':
        return dict(dm_tot=ctx['exact_dm'], rho_tot=ctx['rho_exact'],
                    dma=ctx['dma'], dmb=ctx['dmb'],
                    rho_a=ctx['rho_a'], rho_b=ctx['rho_b'])
    if density_mode == 'scf':
        return _scf_density(ctx, finfo)
    raise ValueError(f"density_mode must be 'exact' or 'scf', "
                     f"got {density_mode!r}")


def deltaI_for_geometry(finfo, ctx, density_mode):
    """DeltaI_bond for one functional at one geometry, from a preloaded
    context. Closed shell: total-density comparison (as before). Open shell:
    spin-resolved Tier-1 evaluation -> total + alpha + beta DeltaI columns
    (exact channels vxc_alpha/beta vs the functional's v_a/v_b, each weighted
    by its own spin density; the total compares the spin-averaged potentials
    weighted by rho_tot, matching the old pipeline's semantics)."""
    mol, coords, w, R = ctx['mol'], ctx['coords'], ctx['w'], ctx['R']

    if not ctx['open_shell']:
        s = _density_for_mode(ctx, finfo, density_mode)
        dm, rho_metric = s['dm_tot'], s['rho_tot']
        vxc_func, prov = functional_vxc_on_grid(finfo, mol, dm, coords)
        vxc_func = _sanitize_grid_potential(vxc_func, ctx['vxc_exact'],
                                            finfo.name, R)
        _, dI_bar, _ = mt.delta_metrics(ctx['vxc_exact'], vxc_func,
                                        rho_metric, coords, w, R)
        out = {'R': R, 'dI_bar': float(dI_bar),
               'density_mode': density_mode, **prov}
        # The response columns are SECONDARY: a failure there (e.g. a rebuilt-
        # mol integral quirk) must not discard the primary dI_bar, which is
        # already computed and finite. Guard them separately and record the
        # reason so a NaN response column is self-explaining in the CSV.
        try:
            out.update(_response_columns(ctx, ctx['vxc_exact'] - vxc_func))
        except Exception as e:                               # noqa: BLE001
            out.update({'dI_resp': np.nan, 'dI_resp_degtol_span': np.nan,
                        'resp_n_skipped': -1,
                        'resp_note': f"{type(e).__name__}: {e}"[:120]})
        return out

    # ---- open shell: spin-resolved Tier-1 path -------------------------
    s = _density_for_mode(ctx, finfo, density_mode)
    ra, rb = s['rho_a'], s['rho_b']
    dma, dmb = s['dma'], s['dmb']
    rho_metric_tot = s['rho_tot']

    (va, vb), prov = functional_vxc_spin_on_grid(finfo, mol, dma, dmb,
                                                 ra, rb, coords)
    v_tot, spin_prov = _spin_average_potential(va, vb, ra, rb, ctx['w'])
    prov = {**prov, **spin_prov}
    _, dI_bar, _ = mt.delta_metrics(ctx['vxc_exact'], v_tot,
                                    rho_metric_tot, coords, w, R)
    out = {'R': R, 'dI_bar': float(dI_bar),
           'density_mode': density_mode, **prov}
    # response metric uses the spin-AVERAGED Dv, matching dI_bar's total
    # channel (chi here is the spin-summed KS response; a fully spin-resolved
    # chi is a possible refinement, not needed for the magnitude reported here).
    try:
        out.update(_response_columns(ctx, ctx['vxc_exact'] - v_tot))
    except Exception as e:                                   # noqa: BLE001
        out.update({'dI_resp': np.nan, 'dI_resp_degtol_span': np.nan,
                    'resp_n_skipped': -1,
                    'resp_note': f"{type(e).__name__}: {e}"[:120]})
    # per-channel DeltaI when the exact spin potentials were saved. FIX-D2:
    # an EMPTY channel gets no number. With volume weighting the integral of
    # an empty channel is perfectly well defined arithmetically and perfectly
    # meaningless physically — which is how the 8.132 got published.
    if ctx['vxc_a'] is not None and ctx['vxc_b'] is not None:
        if prov.get('spin_average') in (None, 'both', 'alpha_only'):
            _, dIa, _ = mt.delta_metrics(ctx['vxc_a'], va, ra, coords, w, R)
            out['dI_bar_alpha'] = float(dIa)
        if prov.get('spin_average') in (None, 'both', 'beta_only'):
            _, dIb, _ = mt.delta_metrics(ctx['vxc_b'], vb, rb, coords, w, R)
            out['dI_bar_beta'] = float(dIb)
    return out


def _response_columns(ctx, dv):
    """Regularization-invariant DeltaI_resp for one (functional, mode) from a
    preloaded geometry context and its Dv = vxc_exact - vxc_func on the grid.

    Reuses the SINGLE response implementation in response_metric.py (the same
    code the diagnostic bench validated), so the production column and the
    bench can never diverge. Rebuilds the KS system from the saved vxc_exact;
    needs the plain GTOval and the exact Coulomb matrix, both cheap.

    Never raises: a failure here must not sink the whole post-pass, so it
    returns NaNs and a note instead (a degenerate geometry is data, not a
    crash).
    """
    import response_metric as rm
    mol, coords, w = ctx['mol'], ctx['coords'], ctx['w']
    ao = mol.eval_gto('GTOval', coords)
    J = iv.make_J(mol, np.asarray(ctx['exact_dm']))
    D_ks = (np.asarray(ctx['D_ks']) if ctx.get('D_ks') is not None
            else np.asarray(ctx['exact_dm']))
    # NO blanket try/except here. A structural failure (missing import, wrong
    # shape, absent field) must abort the run on the first geometry, not turn
    # into 2817 silent NaNs — that exact mistake hid a NameError across every
    # response column once already. linalg failure on a genuinely singular
    # geometry is the ONLY tolerated case, and it is narrow and labelled.
    try:
        r = rm.response_deltaI(
            mol, coords, w, ao, ctx['vxc_exact'], D_ks, J, dv,
            ctx['R'], CFG.cyl_radius_bohr, BOHR)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as e:
        return {'dI_resp': np.nan, 'dI_resp_degtol_span': np.nan,
                'resp_n_skipped': -1,
                'resp_note': f"singular KS system: {e}"}
    return {'dI_resp': r['dI_resp'],
            'dI_resp_degtol_span': r['dI_resp_degtol_span'],
            'resp_n_skipped': r['n_skipped']}


def _scf_density(ctx, finfo):
    """Self-consistent density of the functional (cached per functional in
    the geometry context: exactly ONE KS-SCF per functional per geometry).
    Fails loudly on non-convergence — stretched-bond SCF often diverges and
    an unconverged density would yield silent-garbage DeltaI."""
    key = finfo.name
    if key in ctx['scf_cache']:
        return ctx['scf_cache'][key]
    if finfo.xc_string is None:
        raise ValueError("self-consistent mode needs a libxc xc_string "
                         "(custom callables: supply density externally)")
    mol, coords = ctx['mol'], ctx['coords']
    ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    ks.xc = finfo.xc_string
    # Stretched-bond KS-SCF is stiff. Give it the same convergence insurance
    # the reference SCF uses (molecules.run_scf_reference): a plain kernel()
    # diverges on N2 past ~1.5 A and returned RuntimeError -> NaN for 6 of 20
    # geometries. Escalate: default DIIS -> denser grid + level shift -> a
    # second-order (Newton) solve seeded from the best DIIS density. Only after
    # all three fail is the point genuinely non-representable and reported NaN.
    ks.kernel()
    if not ks.converged:
        ks.diis_space = 12
        ks.level_shift = 0.2
        ks.max_cycle = 200
        dm_seed = ks.make_rdm1()
        ks.kernel(dm0=dm_seed)
    if not ks.converged:
        try:
            ks = ks.newton()
            ks.kernel(dm0=dm_seed)
        except Exception:                                    # noqa: BLE001
            pass
    if not ks.converged:
        raise RuntimeError(
            f"KS-SCF ({finfo.name}) did not converge after DIIS, level-shift, "
            f"and Newton at R={ctx['R']:.3f} A. This geometry has no stable "
            f"self-consistent {finfo.name} density; reported as missing.")
    dm = ks.make_rdm1()
    ao = mol.eval_gto('GTOval_sph', coords)
    if mol.spin != 0:
        dma, dmb = dm[0], dm[1]
        ra = np.einsum('gi,ij,gj->g', ao, dma, ao)
        rb = np.einsum('gi,ij,gj->g', ao, dmb, ao)
        out = dict(dm_tot=dma + dmb, dma=dma, dmb=dmb,
                   rho_tot=ra + rb, rho_a=ra, rho_b=rb)
    else:
        rho = np.einsum('gi,ij,gj->g', ao, dm, ao)
        out = dict(dm_tot=dm, dma=None, dmb=None,
                   rho_tot=rho, rho_a=None, rho_b=None)
    ctx['scf_cache'][key] = out
    return out


def aggregate_scan(R, dI_bar, stretch_ref=1.5, r_eq=None):
    """Turn a DeltaI_bond(R) curve into comparison scalars.
      * dIbar_max     : PRIMARY reference (worst real-space error)
      * dIbar_integral: trapezoid over R (accumulated error, single-point-robust)
      * dIbar_stretch : value at stretch_ref * r_eq (standard stretched regime)
    All points are retained by the caller; these are convenience aggregates."""
    R = np.asarray(R, float)
    d = np.asarray(dI_bar, float)
    order = np.argsort(R)
    R, d = R[order], d[order]
    good = ~np.isnan(d)
    # trapezoid works across numpy versions (np.trapz removed in numpy 2.0)
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    out = {'dIbar_max': float(np.nanmax(d)) if good.any() else np.nan,
           'dIbar_integral': (float(_trapz(d[good], R[good]))
                              if good.sum() > 1 else np.nan)}
    if r_eq is not None and good.any():
        rt = stretch_ref * r_eq
        out['dIbar_stretch'] = float(np.interp(rt, R[good], d[good]))
        out['stretch_R'] = float(rt)
    return out


def evaluate_system(name, spec, density_modes=('exact', 'scf'),
                    npz_dir='results_grid_data', r_eq=None):
    """Evaluate one system's dissociation curve for ONE functional (thin
    wrapper over the shared single-load engine; run_functional_evaluation
    uses the same engine for all functionals at once)."""
    finfo = resolve_functional(spec)
    curves, dead, _ = _evaluate_system_all(name, [finfo], density_modes,
                                           npz_dir)
    result = {'system': name, 'functional': finfo.name, 'rung': finfo.rung,
              'is_multiplicative': finfo.is_multiplicative, 'modes': {}}
    for mode in density_modes:
        if (finfo.name, mode) in dead:
            result['modes'][mode] = {'error': dead[(finfo.name, mode)]}
            continue
        cur = curves.get((finfo.name, mode, None), {})
        R = sorted(cur)
        dI = [cur[r] for r in R]
        result['modes'][mode] = {'R': R, 'dI_bar': dI,
                                 **aggregate_scan(R, dI, r_eq=r_eq)}
    return result


def _evaluate_system_all(name, finfos, density_modes, npz_dir):
    """THE single-load engine: each geometry's npz is loaded ONCE and every
    functional x density-mode is evaluated inside (with the per-functional
    SCF cache in the context). Returns
      curves: {(func_name, mode, channel): {R: dI}}   channel in (None,
              'alpha', 'beta'),
      dead:   {(func_name, mode): tier-gate error}   (NotImplementedError
              applies to the whole functional, so it is raised once and the
              pair is skipped from then on),
      dm_bank: {(func_name, mode, rkey): density matrix used for the metric}
              — see FIX-D1. The 1D profile writer (make_comparative_figure)
              MUST evaluate each labelled column on the SAME density the
              metric used, and the only place that density exists is the
              per-geometry context created here. Handing back the (small,
              nao x nao) matrices is what lets the deposit and the tables be
              the same numbers; recomputing the SCF downstream would let the
              two converge to different solutions, which is the very failure
              being fixed."""
    paths = sorted(glob.glob(os.path.join(npz_dir, f"{name}_R*.npz")))
    if not paths:
        raise FileNotFoundError(f"no npz for system {name!r} in {npz_dir}")
    curves, dead, dm_bank = {}, {}, {}
    for p in paths:
        try:
            ctx = _load_geometry_context(p)
        except Exception as e:                           # noqa: BLE001
            print(f"  [npz skip] {os.path.basename(p)}: "
                  f"{type(e).__name__}: {e}")
            continue
        for finfo in finfos:
            for mode in density_modes:
                if (finfo.name, mode) in dead:
                    continue
                try:
                    dm_bank[(finfo.name, mode, _rkey(ctx['R']))] = \
                        _density_for_mode(ctx, finfo, mode)
                except Exception:                        # noqa: BLE001
                    pass    # absent key => the writer refuses to write, and
                            # says so; it never falls back to another density
        for finfo in finfos:
            for mode in density_modes:
                if (finfo.name, mode) in dead:
                    continue
                try:
                    res = deltaI_for_geometry(finfo, ctx, mode)
                except NotImplementedError as e:
                    dead[(finfo.name, mode)] = str(e)
                    continue
                except Exception as e:                   # noqa: BLE001
                    res = {'R': ctx['R'], 'dI_bar': float('nan'),
                           'point_error':
                               f"{type(e).__name__}: {e}"[:80]}
                for key, ch in (('dI_bar', None),
                                ('dI_bar_alpha', 'alpha'),
                                ('dI_bar_beta', 'beta'),
                                ('dI_resp', 'resp'),
                                ('dI_resp_degtol_span', 'resp_degtol'),
                                ('resp_n_skipped', 'resp_nskip'),
                                ('point_error', '_diag_point_error'),
                                ('resp_note', '_diag_resp_note')):
                    if key in res:
                        # keys rounded to 2 decimals: the CSV merge and the
                        # file naming both use R rounded, and raw meta R (e.g.
                        # 0.444) would silently never match the lookup
                        curves.setdefault((finfo.name, mode, ch),
                                          {})[_rkey(res['R'])] = res[key]
    return curves, dead, dm_bank


def make_comparative_figure(name, finfo, density_mode='exact',
                            draw_figure=True,
                            vxc_dir='results_vxc_1D',
                            npz_dir='results_grid_data',
                            energies_dir='results_energies',
                            dm_bank=None):
    """Regenerate the exact-vs-functional comparative v_xc figure from SAVED
    artifacts, with NO expensive rerun: the exact 1-RDM and identity come from
    the npz (geometry-level, grid-independent), the exact profile + line
    coordinates from the 1D CSV. The functional's v_xc is evaluated on that
    line using the density THAT MODE actually uses. The heavy pipeline no
    longer makes figures.

    `dm_bank`: {(func_name, mode, rkey): density dict} handed down by
    _evaluate_system_all, which already ran (and cached) the KS-SCF for the
    metric. Passing it costs nothing and guarantees the deposit column and the
    table value come from the SAME density.

    FIX-D1. The previous version evaluated the functional on `exact_dm` for
    every mode while naming the column after the mode, so
    Vxc_<f>_exact and Vxc_<f>_scf were the same array in all 203 published
    profiles. Two rules now hold, and both matter:
      1. the density comes from _density_for_mode, the same function the
         metric uses;
      2. if that density cannot be obtained (no dm_bank entry and no npz to
         rebuild one, or a non-converged SCF), the column is NOT WRITTEN and
         the reason is printed. A missing column is data; a column labelled
         'scf' holding the exact density is not.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import plotting as pl

    prof_paths = sorted(glob.glob(os.path.join(
        vxc_dir, f"{name}_Vxc_1D_R*.csv")))
    dfe_path = os.path.join(energies_dir, f"{name}_Dissociation_Curve.csv")
    dfe = pd.read_csv(dfe_path) if os.path.exists(dfe_path) else None

    for p in prof_paths:
        prof = pd.read_csv(p)
        if 'Z_Bohr' not in prof or 'Vxc_Exact' not in prof:
            continue
        R = _R_from_profile_name(p)
        npz_path = os.path.join(npz_dir, f"{name}_R{R:.2f}.npz")
        if not os.path.exists(npz_path):
            continue
        d = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        exact_dm = d['exact_dm']
        # ---- FIX-D1: the density this labelled column is entitled to -------
        key = (finfo.name, density_mode, _rkey(R))
        dens = (dm_bank or {}).get(key)
        if dens is None:
            if density_mode == 'exact':
                dens = {'dm_tot': exact_dm}
            else:
                # standalone call (no bank): rebuild the context and run the
                # SCF here. Costs one KS-SCF; refuses rather than substitutes.
                try:
                    ctx = _load_geometry_context(npz_path)
                    dens = _density_for_mode(ctx, finfo, density_mode)
                except Exception as e:                       # noqa: BLE001
                    print(f"  [1D column skipped] {name} R={R:.2f} "
                          f"{finfo.name}/{density_mode}: "
                          f"{type(e).__name__}: {e}")
                    continue
        dm_line = dens['dm_tot']
        from benchmark import Z
        member = dict(name=name, a1=meta['a1'], a2=meta['a2'],
                      charge=int(meta['charge']), spin=int(meta['spin']),
                      # TOTAL electrons (Z1+Z2-q), NOT the active count —
                      # sum(nelecas) mislabels e.g. Li2+ (5 e-, active 1) as
                      # a 1-electron SIE system in the figure title.
                      n_elec=Z[meta['a1']] + Z[meta['a2']]
                             - int(meta['charge']))
        mol = _rebuild_mol(meta)
        line_pts = np.column_stack([np.zeros(len(prof)), np.zeros(len(prof)),
                                    prof['Z_Bohr'].values])
        try:
            vxc_func_line, _ = functional_vxc_on_grid(finfo, mol, dm_line,
                                                      line_pts)
        except NotImplementedError:
            return   # Tier 2/3 without convention: no figure, already flagged
        # Persist this functional's v_xc alongside the exact one in the same 1D
        # profile, so the CSV carries every potential that enters the metric.
        # The array is already in hand for the figure, so this costs one write.
        safe = finfo.name.replace(',', '-').replace(' ', '')
        prof[f"Vxc_{safe}_{density_mode}"] = vxc_func_line
        try:
            prof.to_csv(p, index=False)
        except OSError:
            pass                      # a read-only results tree must not abort
        n_unp = _lookup(dfe, R, 'N_unpaired')
        mode = _lookup(dfe, R, 'inv_mode', as_str=True)
        dbar = _lookup(dfe, R, column_name(finfo.name, density_mode))
        if draw_figure:
            pl.plot_comparative_vxc(prof['Z_Bohr'].values,
                                    prof['Vxc_Exact'].values, vxc_func_line, R,
                                    dbar if dbar is not None else np.nan,
                                    member, n_unpaired=n_unp, inv_mode=mode,
                                    func_label=finfo.name)


def _R_from_profile_name(path):
    import re
    m = re.search(r'_R(-?\d+\.?\d*)', os.path.basename(path))
    return round(float(m.group(1)), 2) if m else np.nan


def _lookup(dfe, R, col, as_str=False):
    import pandas as pd  # noqa: F401
    if dfe is None or col not in dfe.columns:
        return None
    hit = dfe.loc[dfe['R_Angstrom'].map(_rkey) == _rkey(R), col]
    if len(hit) == 0:
        return None
    v = hit.iloc[0]
    return str(v) if as_str else float(v)


def column_name(func_name, density_mode, kind='dIbar', channel=None):
    """Explicit per-functional column name, e.g.
       DeltaI_bar_PBE-PBE_exact_Eh, DeltaI_bar_LDA-VWN_scf_alpha_Eh.
    channel: None (total) | 'alpha' | 'beta' (open-shell spin channels).
    Sanitizes the libxc comma so column names stay clean."""
    safe = func_name.replace(',', '-').replace(' ', '')
    tag = {'dIbar': 'DeltaI_bar', 'raw': 'DeltaI_raw', 'dIresp': 'DeltaI_resp'}.get(kind, kind)
    ch = f"_{channel}" if channel else ''
    unit = '_Eh' if kind == 'dIbar' else ''
    return f"{tag}_{safe}_{density_mode}{ch}{unit}"


# ===========================================================================
# ORCHESTRATION — the control-room post-pass (called from main.py)
# ===========================================================================
def _build_r_eq_map():
    """name -> equilibrium distance, from the benchmark table. Used for the
    dIbar_stretch aggregate (the dissociation CSV carries no r_eq column)."""
    try:
        from benchmark import parse_benchmark
        return {m['name']: m['r_eq'] for m in parse_benchmark()}
    except Exception:                                       # noqa: BLE001
        return {}


_R_EQ_MAP = _build_r_eq_map()


def pending_functionals(functionals, density_modes=('exact', 'scf'),
                        energies_dir='results_energies'):
    """Which functionals still need scoring, and which are already on disk.

    The post-pass is idempotent but not free: re-scoring a functional reloads
    every geometry's npz. This lets the caller skip the ones whose columns are
    already present in EVERY dissociation CSV, and run only the new ones.
    A functional counts as done only if all of its (mode) columns exist in all
    systems, so a partially-merged run is redone rather than silently trusted.

    Returns (pending, done) as lists of the original functional specs.
    """
    import glob
    import pandas as pd
    curves = sorted(glob.glob(os.path.join(energies_dir,
                                           '*_Dissociation_Curve.csv')))
    if not curves:
        return list(functionals), []      # nothing scored yet
    have = []
    for path in curves:
        try:
            have.append(set(pd.read_csv(path, nrows=0).columns))
        except Exception:                                    # noqa: BLE001
            return list(functionals), []  # unreadable: redo rather than guess
    pending, done = [], []
    for spec in functionals:
        try:
            fname = resolve_functional(spec).name
        except Exception:                                    # noqa: BLE001
            pending.append(spec)
            continue
        need = {column_name(fname, m) for m in density_modes}
        if all(need <= cols for cols in have):
            done.append(spec)
        else:
            pending.append(spec)
    return pending, done


def run_functional_evaluation(functionals, density_modes=('exact', 'scf'),
                              npz_dir='results_grid_data',
                              energies_dir='results_energies',
                              out_summary='results_energies/'
                                          'Functional_Comparison.csv',
                              force=False):
    """Post-pass over the saved exact side, restructured so each geometry's
    npz is loaded ONCE and every functional x density-mode (and, for open
    shell, every spin channel) is evaluated inside. Merges explicit
    per-functional columns into each system's *_Dissociation_Curve.csv and
    writes the benchmark-level comparison summary."""
    import pandas as pd

    npz_files = glob.glob(os.path.join(npz_dir, "*_R*.npz"))
    systems = sorted({os.path.basename(p).rsplit('_R', 1)[0]
                      for p in npz_files})
    if not systems:
        raise FileNotFoundError(f"no npz systems found in {npz_dir!r}")

    finfos = [resolve_functional(s) for s in functionals]
    summary_rows = []

    # Rows of a system that is skipped below still belong in the benchmark
    # summary: rebuilding it from only the systems recomputed this time would
    # silently shrink it.
    previous = (pd.read_csv(out_summary) if os.path.exists(out_summary)
                else pd.DataFrame())

    def _already_scored(df_):
        """Has this system got every column the requested (functional, mode)
        pairs would write?

        The skip used to live one level up, in `pending_functionals`, which
        asks whether a functional is scored in EVERY system at once. That is
        the right question for deciding which functionals to pass in, and the
        wrong one for deciding what to recompute: one system missing its
        columns made the post-pass re-score all ten, and a user who had set
        `functional_postpass_force = False` watched it do exactly what that
        flag is supposed to prevent. Reported 2026-08-09. The decision is now
        taken per system, where the cost is."""
        if df_ is None or force:
            return False
        cols = set(df_.columns)
        for fi_ in finfos:
            for m_ in density_modes:
                need = {column_name(fi_.name, m_),
                        column_name(fi_.name, m_, kind='dIresp')}
                if not need <= cols:
                    return False
                if df_[list(need)].isna().all().any():
                    return False          # column present but never filled
        # A geometry recomputed by the scan has had its metric cells blanked
        # by `pipeline._write_curve`, which is invisible above: one NaN in a
        # full column is not an empty column. Without this the system would be
        # reported as already scored and that geometry would keep its hole.
        if 'postpass_cleared' in df_.columns and df_['postpass_cleared'].any():
            return False
        return True

    for name in systems:
        csv_path = os.path.join(energies_dir, f"{name}_Dissociation_Curve.csv")
        df = (pd.read_csv(csv_path) if os.path.exists(csv_path) else None)
        r_eq = _R_EQ_MAP.get(name)

        if _already_scored(df):
            keep = (previous[previous['system'] == name]
                    if 'system' in previous.columns else pd.DataFrame())
            summary_rows.extend(keep.to_dict('records'))
            print(f"  {name}: already scored for "
                  f"{', '.join(f.name for f in finfos)}; skipping "
                  f"({len(keep)} summary rows carried over)")
            continue

        curves, dead, dm_bank = _evaluate_system_all(name, finfos,
                                                     density_modes, npz_dir)

        for finfo in finfos:
            for mode in density_modes:
                if (finfo.name, mode) in dead:
                    summary_rows.append(dict(
                        system=name, functional=finfo.name, rung=finfo.rung,
                        density_mode=mode, dIbar_max=np.nan,
                        dIbar_integral=np.nan,
                        note=dead[(finfo.name, mode)][:80]))
                    continue
                # merge every channel's column into the dissociation CSV
                for ch in (None, 'alpha', 'beta',
                           '_diag_point_error', '_diag_resp_note'):
                    cur = curves.get((finfo.name, mode, ch))
                    if cur is None or df is None:
                        continue
                    if ch in ('_diag_point_error', '_diag_resp_note'):
                        # flat diagnostic columns: name the cause per functional
                        # x mode so a NaN metric self-explains in the CSV.
                        safe = finfo.name.replace(',', '-').replace(' ', '')
                        col = f"{ch[6:]}_{safe}_{mode}"
                        df[col] = [cur.get(rr, '') for rr in
                                   df['R_Angstrom'].map(_rkey)]
                        continue
                    col = column_name(finfo.name, mode, channel=ch)
                    df[col] = [cur.get(rr, np.nan)
                               for rr in df['R_Angstrom'].map(_rkey)]
                # the response metric (kappa-invariant): one column per
                # functional x mode, plus its deg_tol-sensitivity audit column.
                resp = curves.get((finfo.name, mode, 'resp'))
                if resp is not None and df is not None:
                    rc = column_name(finfo.name, mode, kind='dIresp')
                    df[rc] = [resp.get(rr, np.nan)
                              for rr in df['R_Angstrom'].map(_rkey)]
                    span = curves.get((finfo.name, mode, 'resp_degtol'), {})
                    df[rc + '_degtol_span'] = [span.get(rr, np.nan)
                                               for rr in df['R_Angstrom'].map(_rkey)]
                # aggregates from the TOTAL channel
                cur = curves.get((finfo.name, mode, None), {})
                R = sorted(cur)
                dI = [cur[r] for r in R]
                agg = aggregate_scan(R, dI, r_eq=r_eq) if R else {}
                summary_rows.append(dict(
                    system=name, functional=finfo.name, rung=finfo.rung,
                    density_mode=mode,
                    dIbar_max=agg.get('dIbar_max', np.nan),
                    dIbar_integral=agg.get('dIbar_integral', np.nan),
                    dIbar_stretch=agg.get('dIbar_stretch', np.nan),
                    is_multiplicative=finfo.is_multiplicative, note=''))

        if df is not None:
            if 'postpass_cleared' in df.columns:
                df = df.drop(columns=['postpass_cleared'])
            df.to_csv(csv_path, index=False)

        # comparative figures (exact vs functional), one per functional
        for finfo in finfos:
            try:
                # every density mode contributes its own v_xc column to the
                # 1D profile; the illustrative figure is drawn once, for the
                # first mode, so the modes do not overwrite each other's plot.
                for _i, _dm in enumerate(density_modes):
                    make_comparative_figure(name, finfo, density_mode=_dm,
                                            draw_figure=(_i == 0),
                                            dm_bank=dm_bank)
            except Exception:                            # noqa: BLE001
                pass   # figures are illustrative; never block the data pass

    summary = pd.DataFrame(summary_rows)
    if {'exact', 'scf'} <= set(density_modes) and not summary.empty:
        piv = summary.pivot_table(index=['system', 'functional'],
                                  columns='density_mode', values='dIbar_max')
        if {'exact', 'scf'} <= set(piv.columns):
            piv['error_cancellation'] = piv['exact'] - piv['scf']
            piv.reset_index().to_csv(
                out_summary.replace('.csv', '_ErrorCancellation.csv'),
                index=False)
    summary.to_csv(out_summary, index=False)
    return summary_rows