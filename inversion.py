"""
inversion.py — the Wu-Yang engine (the numerically verified core).
==================================================================
Parametrization: v_xc(r) = v_FA(r) + sum_t b_t g_t(r) in an even-tempered
Gaussian potential basis (Wu & Yang, JCP 118, 2498 (2003)), regularized by
the Heaton-Burgess/Bulat/Yang smoothness norm on the CORRECTION only
(kappa * Int |grad(v_xc - v_FA)|^2). The Fermi-Amaldi guide -v_H/N gives
the correct -1/r asymptote and equals the exact v_x for 2e singlets / the
exact v_xc for 1e systems, so the optimizer only 'discovers' correlation.

Every design element below was forced by a verified failure mode:
  * PRUNING (canonical orthogonalization in the grid metric): diffuse etb
    sets are near-linearly dependent; without pruning the Newton system is
    numerically singular (rcond ~1e-17) and coefficients run away along
    null directions -> nonsense KS spectra.
  * HYBRID OPTIMIZER: full-step Newton from scratch jumps into HOMO-LUMO
    crossing regions (aufbau kinks of the concave sum-of-occupied-
    eigenvalues) and stalls; L-BFGS-B reaches the smooth basin in tens of
    iterations, then Newton with the ANALYTIC response Hessian
        H_st = 2 f_occ Sum_ia <i|g_s|a><a|g_t|i> / (eps_a - eps_i)
    finishes quadratically (verified: 3.9e-4 -> 7.8e-8 -> 2.3e-12).
  * ADAPTIVE KAPPA: at fixed kappa the Tikhonov bias makes the residual
    scale with the size of the required correction, which GROWS with R;
    continuation (kappa/10, warm-started) meets the density target with
    the LARGEST kappa that satisfies it (discrepancy-principle choice,
    reported as kappa_final).
  * COLD-RESTART GUARD: a FAILed warm start must never cascade.

Success is judged by the physical criterion Int|drho| dr, never by the
scipy flag. Open shell: with J fixed at the reference density, the alpha
and beta channels DECOUPLE exactly -> two independent single-channel
inversions against the FCI/CASCI/CASSCF spin densities.
"""

import numpy as np
import scipy.linalg
from scipy.optimize import minimize
from pyscf import dft, scf

from config import CFG, prune_tol_for, density_fail_for
from molecules import build_potential_mol, r_ang_of


class WuYangBasisInverter:
    """Single spin channel. occ_fac = 2 (closed shell) or 1 (spin channel)."""

    def __init__(self, mol, mol_pot, grids, ao_value, H_fixed, S,
                 rho_target, n_occ, occ_fac, v_ref_grid):
        self.S = S
        self.rho_t, self.w = rho_target, grids.weights
        self.n_occ, self.occ_fac = n_occ, occ_fac
        self.mol, self.mol_pot = mol, mol_pot
        self.ao = ao_value

        # ---- density-floor mask (ghost-state cure) ------------------------
        # The Wu-Yang objective is nearly flat where rho_target ~ 0, so a
        # too-free potential basis digs unpenalized "ghost" wells in the
        # far field (symptom: KS gap collapses to 0, |drho| blows up while
        # kappa runs to its floor). We fit the potential only where the
        # reference density is non-negligible; elsewhere v_xc is fixed at
        # the Fermi-Amaldi guide, which already has the correct -1/r tail.
        rho_ref_scale = float(np.max(np.abs(self.rho_t)))
        self.fit_mask = self.rho_t > CFG.density_floor * rho_ref_scale
        self.n_support = int(np.sum(self.fit_mask))     # diagnostic
        self.n_grid = len(self.fit_mask)
        w_fit = self.w * self.fit_mask         # zero weight off-support

        # ---- potential basis on the grid + PRUNING ------------------------
        gpot_full = mol_pot.eval_gto('GTOval', grids.coords)
        self.npot_full = gpot_full.shape[1]
        self.Mgram = gpot_full.T @ (w_fit[:, None] * gpot_full)
        ev, U = scipy.linalg.eigh(self.Mgram)
        keep = ev > prune_tol_for(self.mol.nelectron) * ev.max()
        self.X = U[:, keep] / np.sqrt(ev[keep])   # canonical orthogonalization
        self.gpot = gpot_full @ self.X            # orthonormal on the grid
        self.npot = self.gpot.shape[1]

        # ---- fixed Hamiltonian (v_ref folded in) + coupling tensors -------
        vref_w = v_ref_grid * self.w
        self.H_fixed = H_fixed + ao_value.T @ (vref_w[:, None] * ao_value)
        self.G = np.einsum('gi,gt,g,gj->tij', ao_value, self.gpot,
                           self.w, ao_value, optimize=True)

        # HBY smoothness norm (2*kin = Int grad g . grad g) + relative ridge
        T_red = self.X.T @ (2.0 * mol_pot.intor('int1e_kin')) @ self.X
        self.T_reg = T_red + (CFG.penalty_ridge * np.trace(T_red)
                              / max(self.npot, 1)) * np.eye(self.npot)
        self.kappa = CFG.kappa_start
        self.trace = []                           # accepted iterates only


    # -- KS solve at the current coefficients -------------------------------
    def _solve(self, c):
        # Single-entry exact cache: Newton polish calls _f_grad(c) and
        # _hessian(c) back-to-back with the SAME c, each otherwise repeating
        # the dominant generalized eigendecomposition. The cache returns only
        # on an EXACT match (np.array_equal) — any change in c recomputes — so
        # the result is bit-for-bit identical to the uncached path.
        cached = getattr(self, '_solve_cache', None)
        if cached is not None and np.array_equal(cached[0], c):
            return cached[1]
        F = self.H_fixed + np.einsum('t,tij->ij', c, self.G)
        eps, C = scipy.linalg.eigh(F, self.S)
        Cocc = C[:, :self.n_occ]
        D = self.occ_fac * (Cocc @ Cocc.T)
        rho = dft.numint.eval_rho(self.mol, self.ao, D, xctype='LDA')
        result = (eps, C, D, rho)
        self._solve_cache = (c.copy(), result)
        return result

    # -- Wu-Yang objective (concave W -> minimize f = -W + penalty) ---------
    def _f_grad(self, c, record=False):
        eps, C, D, rho = self._solve(c)
        drho = (rho - self.rho_t) * self.fit_mask     # on-support only
        W = (self.occ_fac * np.sum(eps[:self.n_occ])
             - np.sum((self.gpot @ c) * self.rho_t * self.w))
        f = -W + 0.5 * self.kappa * c @ self.T_reg @ c
        g = (-(self.gpot.T @ (drho * self.w))
             + self.kappa * (self.T_reg @ c))
        if record:
            self.trace.append(float(np.sum(np.abs(drho) * self.w)))
        return f, g

    # -- analytic response Hessian (verified vs finite differences) ---------
    def _hessian(self, c):
        eps, C, D, rho = self._solve(c)
        Gmo = np.einsum('pi,tpq,qj->tij', C, self.G, C, optimize=True)
        occ, vir = slice(0, self.n_occ), slice(self.n_occ, len(eps))
        denom = np.maximum(eps[vir][None, :] - eps[occ][:, None], 1e-8)
        Gia = Gmo[:, occ, vir]
        H = (2.0 * self.occ_fac
             * np.einsum('sia,tia,ia->st', Gia, Gia, 1.0 / denom))
        return H + self.kappa * self.T_reg

    # -- one (L-BFGS-B -> Newton polish) cycle at the current kappa ----------
    def _optimize_at_kappa(self, c):
        res = minimize(lambda cc: self._f_grad(cc, record=True), c,
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': CFG.lbfgs_maxiter, 'maxcor': 30,
                                'ftol': 1e-14, 'gtol': 1e-10})
        c = res.x
        for _ in range(CFG.newton_polish_maxit):
            f0, g = self._f_grad(c, record=True)
            if self.trace[-1] < 1e-8:
                break
            H = self._hessian(c)
            mu = max(1e-9 * np.trace(H) / max(self.npot, 1), 1e-12)  # rel.
            solved = False
            while mu < 1e6:
                try:
                    step = scipy.linalg.solve(H + mu * np.eye(self.npot),
                                              -g, assume_a='pos')
                    solved = True
                    break
                except scipy.linalg.LinAlgError:
                    mu *= 100
            if not solved:
                break
            slope, alpha, ok = g @ step, 1.0, False
            for _ in range(30):                          # Armijo backtracking
                f1, _ = self._f_grad(c + alpha * step)
                if f1 <= f0 + 1e-4 * alpha * slope:
                    ok = True
                    break
                alpha *= 0.5
            if ok:
                c = c + alpha * step
        return c

    def run(self):
        c = np.zeros(self.npot)              # always cold start (no warm-start)
        self.kappa = CFG.kappa_start
        while True:                                      # kappa continuation
            derr = float(np.sum(np.abs(self._solve(c)[3] - self.rho_t)
                                * self.w * self.fit_mask))
            c = self._optimize_at_kappa(c)
            eps_now, _, _, rho_now = self._solve(c)
            derr = float(np.sum(np.abs(rho_now - self.rho_t)
                                * self.w * self.fit_mask))
            # gap-aware floor: as the KS gap collapses, raise the minimum
            # regularization to keep the Hessian conditioned (opt-in).
            if CFG.kappa_gap_aware and len(eps_now) > self.n_occ:
                gap_now = abs(float(eps_now[self.n_occ] - eps_now[self.n_occ - 1]))
                floor = CFG.kappa_min * CFG.kappa_gap_ref / max(gap_now, 1e-4)
                floor = min(max(floor, CFG.kappa_min), CFG.kappa_floor_max)
            else:
                floor = CFG.kappa_min
            if (not CFG.adaptive_kappa) or derr <= CFG.density_target \
                    or self.kappa <= floor:
                break
            self.kappa = max(self.kappa / 10.0, floor)

        eps, C, D, rho = self._solve(c)
        derr = float(np.sum(np.abs(rho - self.rho_t) * self.w * self.fit_mask))
        # off-support leakage: how much density escaped to the ghost region
        ghost_leak = float(np.sum(np.abs(rho - self.rho_t) * self.w
                                  * (~self.fit_mask)))
        # RAW gap = eps[n_occ]-eps[n_occ-1]. For the ensemble this straddles a
        # tiny-occupation (~1e-4) tail orbital and is spuriously ~0, which
        # mislabels healthy inversions as breakdowns. The SIGNIFICANT gap
        # (last orbital with occ >= 0.5 vs the next) is the physical one
        # (A/B bench: stable 0.338 at equilibrium, 0.081 stretched). Report
        # both; mode arbitration uses the significant gap.
        gap = (float(eps[self.n_occ] - eps[self.n_occ - 1])
               if len(eps) > self.n_occ else np.nan)
        occ_vec = getattr(self, 'occ', None)
        if occ_vec is not None:
            n_sig = int(np.sum(np.asarray(occ_vec) >= 0.5))
        else:
            n_sig = self.n_occ            # single-determinant: all occ = occ_fac
        gap_significant = (float(eps[n_sig] - eps[n_sig - 1])
                           if 0 < n_sig < len(eps) else gap)
        return dict(b=self.X @ c,                        # full-length, warm
                    D=D, mo_energy=eps, density_error=derr, gap=gap_significant,
                    gap_raw=gap, gap_significant=gap_significant,
                    ghost_leak=ghost_leak,
                    eps_homo=float(eps[max(n_sig - 1, 0)]),
                    trace=np.array(self.trace),
                    kappa_final=float(self.kappa), npot_kept=int(self.npot),
                    n_support=self.n_support, n_grid=self.n_grid,
                    good=derr < CFG.density_target,
                    failed=derr > density_fail_for(self.mol.nelectron,
                                                   ghost_leak=ghost_leak,
                                                   density_error=derr))

    def v_on_points(self, b_full, points_bohr):
        """Basis form pays off here: the potential evaluates ANYWHERE —
        dense smooth 1D profiles without any smoothing filter."""
        return self.mol_pot.eval_gto('GTOval', points_bohr) @ b_full


class EnsembleInverter(WuYangBasisInverter):
    """Ensemble (fractional-occupation) Kohn-Sham inversion.

    For a closed-shell density past the Coulson-Fischer point, no single
    determinant reproduces it (natural occupations -> (1,1,...)). Instead of
    breaking symmetry we keep the RESTRICTED framework and fill the KS
    orbitals with the FIXED fractional natural occupations read from the
    reference 1-RDM:  rho_KS = sum_i n_i |phi_i|^2. The occupations are NOT
    optimized -- they are physical inputs -- so this is a small override on
    the validated engine: only the density build, the objective's occupied-
    eigenvalue sum, and the response-Hessian occupation weights change.

    Advantages over broken symmetry: reproduces the density exactly and the
    v_xc is symmetric BY CONSTRUCTION (no branch choice). Validated on H2
    (density error 4.75e-4, asymmetry 2.1e-5) and N2 CAS(6,6) (bond-region
    error ~1.9e-3). See PHYSICS_FINDINGS.md items 6 and 8.
    """

    def __init__(self, mol, mol_pot, grids, ao_value, H_fixed, S,
                 rho_target, occ, v_ref_grid):
        # occ = full natural-occupation vector (core ~2, active fractional,
        # rest ~0), already truncated to the kept orbitals by the caller.
        self.occ = np.asarray(occ)
        n_orb = len(self.occ)
        # reuse the base constructor; n_occ = number of natural orbitals,
        # occ_fac = 1 (occupations already carry the factor).
        super().__init__(mol, mol_pot, grids, ao_value, H_fixed, S,
                         rho_target, n_orb, 1.0, v_ref_grid)

    def _solve(self, c):
        cached = getattr(self, '_solve_cache', None)
        if cached is not None and np.array_equal(cached[0], c):
            return cached[1]
        F = self.H_fixed + np.einsum('t,tij->ij', c, self.G)
        eps, C = scipy.linalg.eigh(F, self.S)
        # fractional density: sum_i occ_i |phi_i|^2 over the natural orbitals
        D = (C[:, :self.n_occ] * self.occ) @ C[:, :self.n_occ].T
        rho = dft.numint.eval_rho(self.mol, self.ao, D, xctype='LDA')
        result = (eps, C, D, rho)
        self._solve_cache = (c.copy(), result)
        return result

    def _f_grad(self, c, record=False):
        eps, C, D, rho = self._solve(c)
        drho = (rho - self.rho_t) * self.fit_mask
        W = (np.sum(self.occ * eps[:self.n_occ])
             - np.sum((self.gpot @ c) * self.rho_t * self.w))
        f = -W + 0.5 * self.kappa * c @ self.T_reg @ c
        g = (-(self.gpot.T @ (drho * self.w))
             + self.kappa * (self.T_reg @ c))
        if record:
            self.trace.append(float(np.sum(np.abs(drho) * self.w)))
        return f, g

    def _hessian(self, c):
        eps, C, D, rho = self._solve(c)
        Gmo = np.einsum('pi,tpq,qj->tij', C, self.G, C, optimize=True)
        n = len(eps)
        occ_full = np.zeros(n)
        occ_full[:self.n_occ] = self.occ
        # occupation-weighted response over ALL orbital pairs; weight
        # (f_i - f_a)/(eps_i - eps_a). Vectorized; verified vs the explicit
        # double loop to 2.8e-14 on the N2 test.
        df = occ_full[:, None] - occ_full[None, :]
        de = eps[:, None] - eps[None, :]
        mask = (np.abs(df) > 1e-12) & (np.abs(de) > 1e-8)
        weight = np.where(mask, df / np.where(mask, de, 1.0), 0.0)
        H = np.einsum('ia,sia,tia->st', weight, Gmo, Gmo, optimize=True)
        return H + self.kappa * self.T_reg


def _natural_occupations(mol, S, ref_dm, tol=1e-4):
    """Natural occupations from the reference 1-RDM: eigenvalues of the RDM
    in an orthonormal basis. Returns the occupations above `tol`, sorted
    descending (core ~2, active fractional, virtual ~0 dropped). These are
    the fixed fractional occupations the ensemble inverter fills."""
    A = scipy.linalg.sqrtm(S).real
    occ = np.linalg.eigvalsh(A @ ref_dm @ A)[::-1]
    return occ[occ > tol]


def n_unpaired_from_occupations(occ):
    """PRIMARY static-correlation marker: the Head-Gordon nonlinear number of
    effectively-unpaired electrons,

        N_unpaired = sum_i  n_i^2 (2 - n_i)^2          [Head-Gordon et al 2003]

    over the natural occupations n_i of the reference 1-RDM. It is smooth in
    geometry (occupations come from the smooth reference RDM), parameter-free,
    and -- crucially -- scales as ~n_u^2 for a small antibonding occupation,
    so it SUPPRESSES the dynamic-correlation tail that is always present near
    equilibrium. That is why it reads ~0 for a closed-shell molecule and lifts
    sharply only at the Coulson-Fischer onset of genuine static correlation,
    unlike the sqrt-based <S^2> map which amplifies the dynamic tail (~sqrt(n_u))
    and reads spuriously large at equilibrium.

    Physical meaning: the number of effectively unpaired electrons. Closed
    shell -> 0. Fully dissociated H2 (occupations 1,1) -> 2 (one unpaired e-
    per atom). Stretched N2 (three broken pairs) -> ~6. Reported RAW (not
    normalized) so the count is interpretable across the benchmark set.

    Note: every natural orbital contributes; core (~2) and virtual (~0) give
    ~0 automatically because n^2 (2-n)^2 vanishes at the integer endpoints, so
    no fractional window is needed here (contrast the <S^2> map, which does).

    Verified limits: (2,0)->0, (1,1)->2, (1.5,0.5)->2*0.5625=1.125 per... see
    tests; closed shell exactly 0.
    """
    occ = np.asarray(occ, dtype=float)
    return float(np.sum(occ ** 2 * (2.0 - occ) ** 2))


def spin_square_from_occupations(occ):
    """SECONDARY diagnostic: <S^2> of the equivalent broken-symmetry state,
    derived from the reference natural occupations. Retained because it is a
    recognizable spin-contamination number, but it is NOT the primary marker:
    its sqrt(n_u) scaling amplifies the dynamic-correlation tail and reads
    spuriously large (~0.42) for closed-shell H2 at equilibrium. Use
    n_unpaired_from_occupations for the static-correlation marker; keep this
    only for cross-reference against broken-symmetry <S^2> values.

    For each fractionally-occupied natural-orbital PAIR (n, 2-n) the
    equivalent two-configuration singlet has orbital overlap
    S_ov = (sqrt(n) - sqrt(2-n)) / (sqrt(n) + sqrt(2-n)) and contributes
    (1 - S_ov^2). Pairs at (2,0) -> 0, pairs at (1,1) -> 1.

    Verified limits: (2,0)->0, (1,1)->1, (1.5,0.5)->0.93.
    """
    occ = np.asarray(occ, dtype=float)
    # Only GENUINELY fractional orbitals form broken pairs. Occupations near 2
    # (core/inactive), near 0 (virtual), or in the basis-incompleteness tail
    # (< frac_lo) are NOT broken pairs and must be excluded, else the tail
    # gets mis-paired with the real fractional partner and corrupts <S^2>.
    frac_lo, frac_hi = CFG.natocc_frac_lo, CFG.natocc_frac_hi
    frac = np.sort(occ[(occ > frac_lo) & (occ < frac_hi)])[::-1]  # descending
    s2 = 0.0
    i, j = 0, len(frac) - 1
    while i < j:
        # pair the most-occupied bonding orbital with the most-occupied
        # antibonding one (largest with smallest of the fractional set)
        n_b, n_a = frac[i], frac[j]
        cg, cu = np.sqrt(n_b / 2.0), np.sqrt(n_a / 2.0)
        denom = cg + cu
        if denom > 1e-12:
            s2 += 1.0 - ((cg - cu) / denom) ** 2
        i += 1
        j -= 1
    if i == j:
        # lone half-filled (~1.0) orbital: pair with its own complement (2-n)
        n = frac[i]
        cg, cu = np.sqrt(n / 2.0), np.sqrt(max(2.0 - n, 0.0) / 2.0)
        denom = cg + cu
        if denom > 1e-12:
            s2 += 1.0 - ((cg - cu) / denom) ** 2
    return float(s2)


def _ks_spin_square(mol, S, Ca, Cb, n_a, n_b):
    """<S^2> of the inverted KS single determinant with distinct alpha/beta
    spatial orbitals (broken symmetry). Exact single-determinant formula:
        <S^2> = Sz(Sz+1) + n_b - sum_ij |<alpha_i|beta_j>|^2
    For a restricted singlet the occupied alpha/beta orbitals coincide and
    the overlap sum equals n_b, giving <S^2> = 0. As the bond stretches and
    the channels localize on opposite atoms, the overlaps drop and <S^2>
    rises toward 1 (50/50 singlet-triplet) -- the static-correlation marker.
    """
    sz = 0.5 * (n_a - n_b)
    Ca_occ, Cb_occ = Ca[:, :n_a], Cb[:, :n_b]
    ovlp = Ca_occ.T @ S @ Cb_occ            # <alpha_i | beta_j>
    return float(sz * (sz + 1) + n_b - np.sum(ovlp ** 2))


def _two_channel_inversion(mol, mol_pot, grids, ao_value, H_fixed, S,
                           v_ref_grid, rho_a, rho_b, n_a, n_b):
    """Invert two spin channels independently against target spin densities
    rho_a, rho_b (summing to the reference total). Returns the assembled
    result dict plus the KS <S^2>. Used for genuine open shell AND for the
    broken-symmetry closed-shell fallback. Each channel is inverted from a
    cold start (no warm-start: it caused branch-inconsistent BS solutions)."""
    res_ch, v_ch, invs, bl, Cocc = [], [], [], [], []
    for n_occ, rho_ch in ((n_a, rho_a), (n_b, rho_b)):
        if n_occ == 0:
            v_ch.append(None)
            Cocc.append(None)
            continue
        inv = WuYangBasisInverter(mol, mol_pot, grids, ao_value, H_fixed,
                                  S, rho_ch, n_occ, 1.0, v_ref_grid)
        r = inv.run()
        res_ch.append(r)
        v_ch.append(v_ref_grid + inv.v_on_points(r['b'], grids.coords))
        invs.append(inv)
        bl.append(r['b'])
        _, C, _, _ = inv._solve(inv.X.T @ (inv.Mgram @ r['b']))
        Cocc.append(C)
    D_ks = sum(r['D'] for r in res_ch)
    va = v_ch[0]
    s2 = 0.0
    if len(res_ch) == 2 and Cocc[0] is not None and Cocc[1] is not None:
        s2 = _ks_spin_square(mol, S, Cocc[0], Cocc[1], n_a, n_b)
    return dict(vxc=(0.5 * (va + v_ch[1]) if v_ch[1] is not None else va),
                vxc_alpha=va, vxc_beta=v_ch[1], D_ks=D_ks,
                mo_energy=np.sort(np.concatenate(
                    [r['mo_energy'] for r in res_ch])),
                density_error=sum(r['density_error'] for r in res_ch),
                gap=min(r['gap'] for r in res_ch),
                kappa_final=min(r['kappa_final'] for r in res_ch),
                npot_kept=min(r['npot_kept'] for r in res_ch),
                ghost_leak=max(r['ghost_leak'] for r in res_ch),
                n_support=min(r['n_support'] for r in res_ch),
                n_grid=res_ch[0]['n_grid'],
                eps_homo=max(r['eps_homo'] for r in res_ch),
                trace=np.concatenate([r['trace'] for r in res_ch]),
                n_unpaired=float('nan'),   # open-shell path: S^2 from spin
                ks_spin_square=s2,         # channels, not occupations
                good=all(r['good'] for r in res_ch),
                failed=any(r['failed'] for r in res_ch),
                inverters=invs, b_list=bl)


def _two_channel_ensemble_inversion(mol, mol_pot, grids, ao_value, H_fixed, S,
                                    v_ref_grid, rho_a, rho_b, dm_a, dm_b):
    """Open-shell inversion using the SAME EnsembleInverter machinery as the
    closed-shell path, called once per spin channel with that channel's own
    FIXED FRACTIONAL natural occupations (from dm_a, dm_b) instead of the
    aufbau integer n_occ that _two_channel_inversion uses. Homogenizes the
    method: every geometry in the benchmark now runs one framework (fixed
    physical natural occupations), whether closed- or open-shell, spin-summed
    or per-channel. Mirrors _two_channel_inversion's assembly exactly so the
    output dict shape is identical (drop-in alternative, gated by
    CFG.unified_ensemble_openshell)."""
    res_ch, v_ch, invs, bl, Cocc = [], [], [], [], []
    for dm_ch, rho_ch in ((dm_a, rho_a), (dm_b, rho_b)):
        occ_ch = _natural_occupations(mol, S, dm_ch, tol=CFG.natocc_keep_tol)
        if len(occ_ch) == 0:
            v_ch.append(None)
            Cocc.append(None)
            continue
        inv = EnsembleInverter(mol, mol_pot, grids, ao_value, H_fixed, S,
                              rho_ch, occ_ch, v_ref_grid)
        r = inv.run()
        res_ch.append(r)
        v_ch.append(v_ref_grid + inv.v_on_points(r['b'], grids.coords))
        invs.append(inv)
        bl.append(r['b'])
        _, C, _, _ = inv._solve(inv.X.T @ (inv.Mgram @ r['b']))
        Cocc.append(C)
    D_ks = sum(r['D'] for r in res_ch)
    va = v_ch[0]
    s2 = 0.0
    if len(res_ch) == 2 and Cocc[0] is not None and Cocc[1] is not None:
        n_a_int = int(round(float(np.sum(_natural_occupations(
            mol, S, dm_a, tol=CFG.natocc_keep_tol)))))
        n_b_int = int(round(float(np.sum(_natural_occupations(
            mol, S, dm_b, tol=CFG.natocc_keep_tol)))))
        s2 = _ks_spin_square(mol, S, Cocc[0], Cocc[1], n_a_int, n_b_int)
    return dict(vxc=(0.5 * (va + v_ch[1]) if v_ch[1] is not None else va),
                vxc_alpha=va, vxc_beta=v_ch[1], D_ks=D_ks,
                mo_energy=np.sort(np.concatenate(
                    [r['mo_energy'] for r in res_ch])),
                density_error=sum(r['density_error'] for r in res_ch),
                gap=min(r['gap'] for r in res_ch),
                kappa_final=min(r['kappa_final'] for r in res_ch),
                npot_kept=min(r['npot_kept'] for r in res_ch),
                ghost_leak=max(r['ghost_leak'] for r in res_ch),
                n_support=min(r['n_support'] for r in res_ch),
                n_grid=res_ch[0]['n_grid'],
                eps_homo=max(r['eps_homo'] for r in res_ch),
                trace=np.concatenate([r['trace'] for r in res_ch]),
                n_unpaired=float('nan'),   # channel occupations already fractional
                ks_spin_square=s2,
                good=all(r['good'] for r in res_ch),
                failed=any(r['failed'] for r in res_ch),
                inverters=invs, b_list=bl)


def invert_geometry(mol, grids, ao_value, ref_dm_tot, rho_tot,
                    rho_spin, v_ref_grid, J_ref, dm_alpha=None, dm_beta=None):
    """Full inversion for one geometry. Every inversion is cold-started.

    Closed-shell singlets (PRODUCTION, CFG.unified_ensemble=True): the
    ENSEMBLE machinery runs at EVERY R, filling the fixed fractional natural
    occupations of the reference 1-RDM. Near equilibrium (occ ~ 2, eps, ...)
    it reduces smoothly to the restricted solution, so there is no
    RKS->ENSEMBLE seam in DeltaI (validated by test_ensemble_unified_H2.py).
    The BROKEN-SYMMETRY two-channel inversion runs only as a DIAGNOSTIC where
    it is meaningful (spin localization or a collapsing gap), supplying the
    raw <S^2>_BS marker and the framework-ambiguity v_xc difference. inv_mode
    is arbitrated by the smooth N_unpaired marker: ENSEMBLE (localized) /
    ENSEMBLE_NOSPIN (weakly correlated); failures map to RKS_BREAKDOWN /
    NUMERICAL_FAIL.

    Legacy path (CFG.unified_ensemble=False, comparison only): restricted
    first, ensemble+BS only after breakdown — this is the flow whose mode
    switch produced the DeltaI seam.

    No warm-start: every inversion is cold. Warm-starts caused branch-
    inconsistent broken-symmetry solutions (the localized state from one
    geometry poisoned the next), and the modest time saving did not justify
    the failure modes it introduced."""
    S = mol.intor('int1e_ovlp')
    H_fixed = mol.intor('int1e_kin') + mol.intor('int1e_nuc') + J_ref
    n_a = (mol.nelectron + mol.spin) // 2
    n_b = (mol.nelectron - mol.spin) // 2
    mol_pot = build_potential_mol(mol.atom_symbol(0), mol.atom_symbol(1),
                                  r_ang_of(mol))
    out = {}
    if mol.spin == 0:
        if CFG.unified_ensemble:
            # ---- UNIFIED ENSEMBLE (production default) --------------------
            # Validated by test_ensemble_unified_H2.py: running the ensemble
            # machinery at EVERY closed-shell R removes the RKS->ENSEMBLE
            # seam in DeltaI (a 5-7x cliff at the mode switch, e.g. H2
            # 1.66 -> 0.24 Eh between R=1.41 and 1.48 A) because near
            # equilibrium the natural occupations are ~(2, eps, ...) and the
            # ensemble solution reduces smoothly to the RKS one THROUGH THE
            # SAME MACHINERY. The dIbar_max scalar then lands on physics, not
            # on the numerical stress of the pre-breakdown restricted fit.
            occ = _natural_occupations(mol, S, ref_dm_tot,
                                       tol=CFG.natocc_keep_tol)
            ens = EnsembleInverter(mol, mol_pot, grids, ao_value, H_fixed, S,
                                   rho_tot, occ, v_ref_grid)
            re = ens.run()
            vxc_ens = v_ref_grid + ens.v_on_points(re['b'], grids.coords)
            n_unpaired = n_unpaired_from_occupations(occ)
            s2 = spin_square_from_occupations(occ)
            localized = n_unpaired > CFG.n_unpaired_localization_min

            # BS DIAGNOSTIC only where it is meaningful (genuine spin
            # localization, or a collapsing effective gap): it supplies the
            # raw <S^2>_BS marker and the framework-ambiguity v_xc difference
            # without doubling the cost of every near-equilibrium point.
            bs = None
            if CFG.bs_diagnostic and (localized
                                      or re['gap'] < CFG.rks_breakdown_gap):
                za = mol.atom_coords()[0, 2]
                bias = 1.0 / (1.0 + np.exp(-(grids.coords[:, 2] - za) * 2.0))
                half = 0.5 * rho_tot
                skew = CFG.bs_seed * (2 * bias - 1)
                rho_a, rho_b = half * (1.0 + skew), half * (1.0 - skew)
                wsum = np.sum(half * grids.weights)
                rho_a *= wsum / max(np.sum(rho_a * grids.weights), 1e-30)
                rho_b *= wsum / max(np.sum(rho_b * grids.weights), 1e-30)
                bs = _two_channel_inversion(
                    mol, mol_pot, grids, ao_value, H_fixed, S, v_ref_grid,
                    rho_a, rho_b, n_a, n_a)
                _symmetrize_bs_potential(mol, grids, bs)

            if not re['failed']:
                mode = 'ENSEMBLE' if localized else 'ENSEMBLE_NOSPIN'
            else:
                # The ensemble fit did not reach target. Name by WHAT RAN
                # (ensemble) and WHY it is hard, not by 'RKS' (never used here):
                # localized -> genuine static-correlation regime; else -> a
                # numerical shortfall with no static-correlation marker.
                mode = 'ENSEMBLE_STATIC_FAIL' if localized else 'ENSEMBLE_FIT_FAIL'

            vxc_diff = (vxc_ens - bs['vxc']
                        if bs is not None and bs.get('vxc') is not None
                        else None)
            out.update(
                vxc=vxc_ens,                    # PRIMARY = ensemble
                vxc_ensemble=vxc_ens,
                vxc_bs=(bs.get('vxc') if bs is not None else None),
                vxc_diff=vxc_diff,
                vxc_alpha=(bs.get('vxc_alpha') if bs is not None else None),
                vxc_beta=(bs.get('vxc_beta') if bs is not None else None),
                D_ks=re['D'], mo_energy=re['mo_energy'],
                density_error=re['density_error'], gap=re['gap'],
                eps_homo=re['eps_homo'], trace=re['trace'],
                kappa_final=re['kappa_final'], npot_kept=re['npot_kept'],
                ghost_leak=re['ghost_leak'],
                n_support=re['n_support'], n_grid=re['n_grid'],
                n_unpaired=n_unpaired, ks_spin_square=s2,
                bs_spin_square_raw=(bs['ks_spin_square']
                                    if bs is not None else np.nan),
                inv_mode=mode, natocc=occ,
                bs_density_error=(bs['density_error']
                                  if bs is not None else np.nan),
                bs_symmetrized=(bs.get('symmetrized', False)
                                if bs is not None else False),
                good=re['good'], failed=re['failed'],
                inverters=[ens], b_list=[re['b']],
                bs_inverters=(bs.get('inverters', [])
                              if bs is not None else []),
                bs_b_list=(bs.get('b_list', []) if bs is not None else []))
        else:
            # ---- LEGACY RKS-first path (comparison/debugging only) --------
            inv = WuYangBasisInverter(mol, mol_pot, grids, ao_value, H_fixed, S,
                                      rho_tot, n_a, 2.0, v_ref_grid)
            r = inv.run()
            broke = (r['gap'] < CFG.rks_breakdown_gap
                     and r['density_error'] > CFG.density_target)
            if not broke:
                # single-determinant representable: N_unpaired ~0 by construction,
                # but compute it honestly from the occupations so a surprising
                # value would surface rather than be hidden by a hardcoded zero.
                occ_rks = _natural_occupations(mol, S, ref_dm_tot,
                                               tol=CFG.natocc_keep_tol)
                out.update(vxc=v_ref_grid + inv.v_on_points(r['b'], grids.coords),
                           vxc_alpha=None, vxc_beta=None, D_ks=r['D'],
                           mo_energy=r['mo_energy'],
                           density_error=r['density_error'], gap=r['gap'],
                           eps_homo=r['eps_homo'], trace=r['trace'],
                           kappa_final=r['kappa_final'], npot_kept=r['npot_kept'],
                           ghost_leak=r['ghost_leak'],
                           n_support=r['n_support'], n_grid=r['n_grid'],
                           n_unpaired=n_unpaired_from_occupations(occ_rks),
                           ks_spin_square=0.0,   # RKS determinant: no spin
                           inv_mode='RESTRICTED_LEGACY',  # legacy path only
                           good=r['good'], failed=r['failed'],
                           inverters=[inv], b_list=[r['b']])
            else:
                # Representability breakdown: no single closed-shell determinant
                # reproduces the correlated density. Architecture (see
                # PHYSICS_FINDINGS.md #6): the ENSEMBLE channel is PRIMARY (it
                # supplies the rigorous, symmetric-by-construction v_xc), while
                # BROKEN SYMMETRY is retained as a DIAGNOSTIC that supplies the
                # <S^2> static-correlation marker. Both v_xc and their difference
                # are exported.

                # --- primary: ensemble (fixed fractional natural occupations) ---
                occ = _natural_occupations(mol, S, ref_dm_tot,
                                           tol=CFG.natocc_keep_tol)
                ens = EnsembleInverter(mol, mol_pot, grids, ao_value, H_fixed, S,
                                       rho_tot, occ, v_ref_grid)
                re = ens.run()
                vxc_ens = v_ref_grid + ens.v_on_points(re['b'], grids.coords)

                # --- diagnostic: broken symmetry (supplies <S^2>) ----------------
                za = mol.atom_coords()[0, 2]
                bias = 1.0 / (1.0 + np.exp(-(grids.coords[:, 2] - za) * 2.0))
                half = 0.5 * rho_tot
                skew = CFG.bs_seed * (2 * bias - 1)
                rho_a, rho_b = half * (1.0 + skew), half * (1.0 - skew)
                wsum = np.sum(half * grids.weights)
                rho_a *= wsum / max(np.sum(rho_a * grids.weights), 1e-30)
                rho_b *= wsum / max(np.sum(rho_b * grids.weights), 1e-30)
                bs = _two_channel_inversion(
                    mol, mol_pot, grids, ao_value, H_fixed, S, v_ref_grid,
                    rho_a, rho_b, n_a, n_a)
                _symmetrize_bs_potential(mol, grids, bs)
                # PRIMARY marker: Head-Gordon N_unpaired from the reference natural
                # occupations — smooth, branch-free, parameter-free, and ~0 at
                # equilibrium (suppresses the dynamic-correlation tail). See
                # PHYSICS_FINDINGS.md #3. Two secondary diagnostics are retained:
                # the sqrt-based occupation <S^2> (recognizable but amplifies the
                # dynamic tail) and the raw broken-symmetry <S^2> (oscillates).
                n_unpaired = n_unpaired_from_occupations(occ)
                s2 = spin_square_from_occupations(occ)
                s2_bs_raw = bs['ks_spin_square']
                localized = n_unpaired > CFG.n_unpaired_localization_min

                # inv_mode: arbitrated by the SMOOTH occupation-derived marker so
                # the classification does not flip between neighboring geometries.
                if not re['failed'] and localized:
                    mode = 'ENSEMBLE'          # rigorous fit + genuine breakdown
                elif not re['failed']:
                    mode = 'ENSEMBLE_NOSPIN'   # ensemble fit ok, weak correlation
                elif localized:
                    mode = 'ENSEMBLE_STATIC_FAIL'  # fit short; static-corr regime
                else:
                    mode = 'ENSEMBLE_FIT_FAIL'     # fit short; numerical

                # v_xc difference (ensemble - BS-symmetrized): a per-geometry
                # metric of the single-determinant framework ambiguity.
                vxc_diff = (vxc_ens - bs['vxc']
                            if bs.get('vxc') is not None else None)

                out.update(
                    vxc=vxc_ens,                    # PRIMARY = ensemble
                    vxc_ensemble=vxc_ens,
                    vxc_bs=bs.get('vxc'),
                    vxc_diff=vxc_diff,
                    vxc_alpha=bs.get('vxc_alpha'), vxc_beta=bs.get('vxc_beta'),
                    D_ks=re['D'], mo_energy=re['mo_energy'],
                    density_error=re['density_error'], gap=re['gap'],
                    eps_homo=re['eps_homo'], trace=re['trace'],
                    kappa_final=re['kappa_final'], npot_kept=re['npot_kept'],
                    ghost_leak=re['ghost_leak'],
                    n_support=re['n_support'], n_grid=re['n_grid'],
                    n_unpaired=n_unpaired,
                    ks_spin_square=s2, bs_spin_square_raw=s2_bs_raw,
                    inv_mode=mode,
                    natocc=occ,
                    bs_density_error=bs['density_error'],
                    bs_symmetrized=bs.get('symmetrized', False),
                    good=re['good'], failed=re['failed'],
                    inverters=[ens], b_list=[re['b']],
                    bs_inverters=bs.get('inverters', []),
                    bs_b_list=bs.get('b_list', []))
    else:
        use_ens_os = (CFG.unified_ensemble_openshell
                     and dm_alpha is not None and dm_beta is not None)
        if use_ens_os:
            out.update(_two_channel_ensemble_inversion(
                mol, mol_pot, grids, ao_value, H_fixed, S, v_ref_grid,
                rho_spin[0], rho_spin[1], dm_alpha, dm_beta))
            out['inv_mode'] = 'ENSEMBLE_OS'   # PRODUCTION: per-channel ensemble
        else:
            # Legacy aufbau two-channel path. Only reached if
            # unified_ensemble_openshell is OFF or the reference spin DMs are
            # missing -- NOT the production configuration. Labeled explicitly.
            out.update(_two_channel_inversion(
                mol, mol_pot, grids, ao_value, H_fixed, S, v_ref_grid,
                rho_spin[0], rho_spin[1], n_a, n_b))
            out['inv_mode'] = 'AUFBAU_LEGACY'
        # Homogenize the static-correlation marker across the benchmark: the
        # two-channel path leaves n_unpaired = NaN, but the Head-Gordon index
        # is well defined on the reference TOTAL 1-RDM natural occupations for
        # open shells too (a one-electron H2+ gives ~1: one unpaired electron,
        # no pair to break). ks_spin_square stays the secondary diagnostic.
        occ_os = _natural_occupations(mol, S, ref_dm_tot,
                                      tol=CFG.natocc_keep_tol)
        out['n_unpaired'] = n_unpaired_from_occupations(occ_os)
        out['natocc'] = occ_os
    return out


def _symmetrize_bs_potential(mol, grids, bs):
    """A broken-symmetry solution picks ONE of two degenerate branches
    (alpha-left/beta-right or its mirror), so its spin-averaged potential is
    artificially lopsided. For a HOMONUCLEAR diatomic the density is exactly
    mirror-symmetric (z -> -z about the bond midpoint at the origin), hence
    the physical v_xc MUST be too. We restore it by averaging the potential
    with its mirror image -- equivalently averaging vxc_alpha(z) with
    vxc_beta(-z). Heteronuclear molecules have NO such symmetry (the charge-
    transfer step is physically asymmetric), so we leave them untouched."""
    if mol.atom_symbol(0) != mol.atom_symbol(1):
        return                                   # heteronuclear: asymmetry is real
    if bs.get('vxc_alpha') is None or bs.get('vxc_beta') is None:
        return
    z = grids.coords[:, 2]
    # nearest-mirror index for each grid point (match -z). Grid is not a
    # regular lattice, so match by nearest coordinate in all 3 dims.
    mirror = np.column_stack([grids.coords[:, 0], grids.coords[:, 1], -z])
    from scipy.spatial import cKDTree
    tree = cKDTree(grids.coords)
    _, idx = tree.query(mirror, k=1)
    va, vb = bs['vxc_alpha'], bs['vxc_beta']
    # symmetric physical potential: average the branch with its mirror
    vxc_sym = 0.25 * (va + vb + va[idx] + vb[idx])
    bs['vxc'] = vxc_sym
    bs['vxc_alpha'] = 0.5 * (va + vb[idx])       # symmetrized spin channels
    bs['vxc_beta'] = 0.5 * (vb + va[idx])
    bs['symmetrized'] = True


def make_J(mol, dm):
    J, _ = scf.hf.get_jk(mol, dm)
    return J