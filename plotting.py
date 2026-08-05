"""
plotting.py — publication figures (per geometry).
=================================================
Profiles are evaluated on a DENSE UNIFORM LINE via the basis-form potential
(molecules.dense_axis_line + inverter.v_on_points), so no smoothing filter
touches the physics — a raw plot IS the topology (an early Savitzky-Golay
stage would distort the very peak the metric measures).
"""

import matplotlib
matplotlib.use('Agg')            # non-interactive backend: batch figure
                                 # generation with no GUI. Prevents the Tk
                                 # 'main thread is not in main loop' /
                                 # Tcl_AsyncDelete crash at interpreter exit.
import matplotlib.pyplot as plt

from config import BOHR
from benchmark import phenomenon_label

plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})


def plot_comparative_vxc(z_bohr, vxc_line, vlda_line, r_ang, dbar, member,
                         n_unpaired=None, inv_mode=None, func_label='LDA'):
    z = z_bohr * BOHR
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(z, vxc_line, color='black', lw=3, zorder=10,
            label=r'Reference $v_{xc}$ (Wu-Yang inversion)')
    ax.plot(z, vlda_line, color='red', lw=2.5, ls='--', zorder=9,
            label=rf'{func_label} $v_{{xc}}[\rho_{{ref}}]$')
    nuc = r_ang / 2
    bm = (z >= -nuc) & (z <= nuc)
    ax.fill_between(z[bm], vxc_line[bm], vlda_line[bm], color='purple',
                    alpha=0.2,
                    label=rf'$\bar{{\Delta}}I_{{bond}}$ = {dbar:.4f} Eh')
    ax.axvline(-nuc, color='gray', ls=':')
    ax.axvline(nuc, color='gray', ls=':')
    ax.set_xlabel(r'Internuclear Axis $Z$ ($\mathrm{\AA}$)', fontweight='bold')
    ax.set_ylabel(r'$v_{xc}$ ($E_h$)', fontweight='bold')
    ax.set_title(f"{phenomenon_label(member['n_elec'], member['charge'])}\n"
                 f"{member['a1']}{member['a2']} (q={member['charge']:+d}, "
                 f"2S={member['spin']}) at "
                 rf"R = {r_ang:.2f} $\mathrm{{\AA}}$", fontweight='bold')
    # static-correlation marker (Head-Gordon N_unpaired) + inversion mode
    if n_unpaired is not None and n_unpaired == n_unpaired:  # not NaN
        txt = rf'$N_\mathrm{{unpaired}}$ = {n_unpaired:.3f}'
        if inv_mode is not None:
            txt += f'\nmode: {inv_mode}'
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    import numpy as np
    lo = min(np.percentile(vxc_line, 1), np.percentile(vlda_line, 1), -1.0)
    ax.set_ylim(lo - 0.1, max(vxc_line.max(), vlda_line.max()) + 0.15)
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, ls=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results_vxc_1D/pdf/Comparative_Vxc_{member['name']}"
                f"_R{r_ang:.2f}.pdf", dpi=300)
    plt.close(fig)


def plot_correlation_peak(z_bohr, vc_exact, vc_vwn, r_ang, dbar_c, member):
    """2e singlets only: v_x = -v_H/2 is analytic, so the CORRELATION
    potential is isolated exactly — the sharpest version of the
    argument (the static-correlation peak vs the local VWN guess)."""
    z = z_bohr * BOHR
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(z, vc_exact, color='black', lw=3, zorder=10,
            label=r'$v_c = v_{xc} + v_H/2$ (analytic $v_x$)')
    ax.plot(z, vc_vwn, color='tab:orange', lw=2.5, ls='--', zorder=9,
            label=r'VWN $v_c[\rho_{ref}]$')
    nuc = r_ang / 2
    ax.axvline(-nuc, color='gray', ls=':')
    ax.axvline(nuc, color='gray', ls=':')
    ax.set_xlabel(r'Internuclear Axis $Z$ ($\mathrm{\AA}$)', fontweight='bold')
    ax.set_ylabel(r'$v_c$ ($E_h$)', fontweight='bold')
    ax.set_title(f"Isolated correlation potential, "
                 f"{member['a1']}{member['a2']} at "
                 rf"R = {r_ang:.2f} $\mathrm{{\AA}}$  "
                 rf"($\bar{{\Delta}}I_c$ = {dbar_c:.4f} Eh)",
                 fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, ls=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results_vxc_1D/pdf/Correlation_Vc_{member['name']}"
                f"_R{r_ang:.2f}.pdf", dpi=300)
    plt.close(fig)