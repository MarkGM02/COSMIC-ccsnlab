from sn_types import sn_subtypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

default_sigma = 265.0
default_alpha1 = 1.0
default_met_cosmic = 0.020243

LEGEND_FONT_SIZE = 24
LABEL_FONT_SIZE = 40
TICK_FONT_SIZE = 30

sn_color_dict = {
    'IIP' : ("#F58464", "#E1512A"),
    'IIL' : ("#FFB06A", "#F4904D"),
    'IIn' : ("#E0E775", "#CCC158"),
    'IIb' : ("#9E77C1", "#7440A2"),
    'Ib' :  ("#738FEC", "#3554B8"),
    'Ic' :  ("#6DD961", "#3BAA2F")
}

def add_metallicity_label(ax, z_dim, x=0.02, y=0.95):
    Z_str = r'$Z \approx$' + str(z_dim) + r'$\,Z_\odot$' if z_dim != 1.0 else r'$Z \approx Z_\odot$'
    ax.text(x, y, Z_str, transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FONT_SIZE)

def replace_closest_bin(bins, m_ccsn_min):
    bins = np.asarray(bins)
    idx = np.argmin(np.abs(bins - m_ccsn_min))
    bins_new = bins.copy()
    bins_new[idx] = m_ccsn_min
    return bins_new

def plot_single_subtype_mass_type(data, mass_type, subtype, ax, bins, max_remnant=15.0, min_zams_mass=8.0):
    # Pick the mass columns for SN1 and SN2 depending on mass_type
    if mass_type == 'zams':
        col1, col2 = 'zams_mass_1', 'zams_mass_2'
    elif mass_type == 'explosion':
        col1, col2 = 'sn_1_mass_1', 'sn_2_mass_2'
    elif mass_type == 'remnant':
        col1, col2 = 'sn_1_remnant_mass', 'sn_2_remnant_mass'
    else:
        raise ValueError(f"mass_type {mass_type!r} not recognized")
    
    #create a mask for zams mass to be above min_zams_mass
    mask_above_1 = data['zams_mass_1'] >= min_zams_mass
    mask_above_2 = data['zams_mass_2'] >= min_zams_mass

    # Filter to desired subtype and reasonable remnant-mass cap (consistent with your plots)
    sn1 = data[(data['sn_1_subtype'] == subtype) & (data['sn_1_remnant_mass'] < max_remnant)]
    sn2 = data[(data['sn_2_subtype'] == subtype) & (data['sn_2_remnant_mass'] < max_remnant)]

    above_masses = pd.concat([sn1.loc[mask_above_1, col1], sn2.loc[mask_above_2, col2]])
    below_masses = pd.concat([sn1.loc[~mask_above_1, col1], sn2.loc[~mask_above_2, col2]])

    # Color by subtype if available
    colorlo, colorhi = sn_color_dict.get(subtype)

    #stacked histogram: plot below as histtype='barstacked', with hatch circle then dashed, filled with color
    weight_above = np.ones(len(above_masses)) * (1e6 / np.max(data['sample_mass']))
    weight_below = np.ones(len(below_masses)) * (1e6 / np.max(data['sample_mass']))

    ax.hist([below_masses, above_masses], bins=bins, color=[colorlo, colorhi], weights=[weight_below, weight_above],
            histtype='barstacked', label=[r'$M_{\rm ZAMS}<M_{\rm CCSN,min}$', r'$M_{\rm ZAMS}>M_{\rm CCSN,min}$'])

def make_mass_type_plots(data, z=0.020243, Ic_ratio=0.43, bins=30, max_remnant=15.0):
    # --- classify ---
    data = sn_subtypes(data, IIP_scheme='include n', Ic_scheme='relative', Ic_ratio=Ic_ratio)
    data = data[(data.sigma == default_sigma) &
                (data.alpha1 == default_alpha1) &
                (data.met_cosmic == z)]
    
    fig, axs = plt.subplots(6, 3, figsize=(28, 36), sharex=False, sharey=False)
    axs = np.asarray(axs)

    # --- Grab min ZAMS mass for binning ---
    min_zams_mass = data[(data.is_single) &
                         (data.met_cosmic == 0.020243) &
                         (data.sn_1_type.isin(['I', 'II']))].zams_mass_1.min()

    # --- shared bins ---
    bins_Z  = np.logspace(np.log10(0.77), np.log10(150.0), bins)
    bins_Z = replace_closest_bin(bins_Z, min_zams_mass)
    bins_CC = np.logspace(np.log10(1.4),  np.log10(25.0),  bins)
    bins_R  = np.logspace(np.log10(1.0),  np.log10(15.0),  bins)

    # --- ticks per column ---
    ticks_Z  = [1, 2, 4, 8, 18, 40, 80, 150]
    ticks_CC = [1, 2, 3, 5, 7, 10, 16, 25]
    ticks_R  = [1, 2, 3, 5, 10, 15]

    # --- titles only on top row ---
    axs[0, 0].set_title('Mass at ZAMS', fontsize=36, fontweight='semibold')
    axs[0, 1].set_title('Mass at core collapse', fontsize=36, fontweight='semibold')
    axs[0, 2].set_title('Remnant mass', fontsize=36, fontweight='semibold')

    def plot_row(row_idx, subtype, df):
        for col_idx, (mass_type, bins) in enumerate(zip(
            ['zams', 'explosion', 'remnant'],
            [bins_Z, bins_CC, bins_R]
        )):
            ax = axs[row_idx, col_idx]
            plot_single_subtype_mass_type(df, mass_type, subtype, ax, bins, min_zams_mass=min_zams_mass, max_remnant=max_remnant)
            if col_idx == 1: ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')
            ax.set_xscale('log')
            ax.minorticks_off()
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

        axs[row_idx, 0].text(0.98, 0.95, f'Type {subtype}',
                             transform=axs[row_idx, 0].transAxes,
                             ha='right', va='top', fontsize=LEGEND_FONT_SIZE,
                             fontweight='semibold')
        axs[row_idx, 0].set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)

    # --- plot all rows ---
    for i, subtype in enumerate(['IIn', 'IIP', 'IIL', 'IIb']):
        plot_row(i, subtype, data)
    for i, subtype in enumerate(['Ib', 'Ic'], start=4):
        plot_row(i, subtype, data)

    for ax in axs[:, 0]:
        #plot a horizontal line at min zams mass
        ax.axvline(min_zams_mass, color='black', linestyle='--', lw=2, label = f'{min_zams_mass:.1f}' + r'$\,M_\odot$')
        #shade the left side of the plot
        ax.axvspan(0, min_zams_mass, facecolor='lightgrey', alpha=0.75, zorder=0)
        #write from binaries in the middle left
        if ax == axs[0, 0]:
            ax.text(0.03, 0.58, 'Mergers +\naccretors\nonly', transform=ax.transAxes, ha='left', va='center',
                    fontsize=LEGEND_FONT_SIZE+8, fontweight='semibold', color='crimson', zorder=5)

            

    for ax in axs[:, 1]:
        #add a legend in the upper left
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')

    # --- ticks & xlabels for every row ---
    for r in range(6):
        axs[r, 0].set_xticks(ticks_Z,  labels=[str(t) for t in ticks_Z])
        axs[r, 1].set_xticks(ticks_CC, labels=[str(t) for t in ticks_CC])
        axs[r, 2].set_xticks(ticks_R,  labels=[str(t) for t in ticks_R])

    # --- x-axis labels on bottom row only ---
    axs[5, 0].set_xlabel(r'$M_{\rm ZAMS}\, (M_\odot)$', fontsize=LABEL_FONT_SIZE)
    axs[5, 1].set_xlabel(r'$M_{\rm CC}\, (M_\odot)$', fontsize=LABEL_FONT_SIZE)
    axs[5, 2].set_xlabel(r'$M_{\rm rem}\, (M_\odot)$', fontsize=LABEL_FONT_SIZE)

    # --- metallicity label top-left of first panel ---
    add_metallicity_label(axs[0, 0], 1.0, x=0.02, y=0.95)

    #random y-ticks
    axs[2, 2].set_yticks([2, 4, 6, 8, 10])

    for r in range(6):
        axs[r, 1].set_xlim(0.7, None)

    plt.tight_layout()
    fig.savefig('final_figs/II_and_I_mass_types_combined.png', dpi=300)
    plt.show()
