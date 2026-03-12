from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from ccsnlab.sn_types import sn_subtypes
import matplotlib.gridspec as gridspec

LEGEND_FONT_SIZE = 22
LABEL_FONT_SIZE = 40
TICK_FONT_SIZE = 30
FIG_WIDTH = 30
FIG_HEIGHT = 14
DPI = 300

def filter_and_mask_sne(data,
                        binfrac='offner23',
                        kicks=(5, None),
                        alpha=1.0,
                        qcflag=5,
                        remnantflag=6,
                        fryer_limit=('sn_1_remnant_mass', 'sn_2_remnant_mass', 3),
                        met_cosmic=0.02):
    
    binfrac_mask = data.binfrac == binfrac
    kickflag_mask = data.kickflag == kicks[0]
    sigma_mask = data.sigma == kicks[1] if kicks[1] != None else pd.Series(True, index=data.index)
    alpha_mask = data.alpha == alpha
    qcflag_mask = data.qcflag == qcflag
    remnantflag_mask = data.remnantflag == remnantflag
    met_cosmic_mask = data.met_cosmic == met_cosmic
    keep_mask = binfrac_mask & kickflag_mask & sigma_mask & alpha_mask & qcflag_mask & remnantflag_mask & met_cosmic_mask
    data = data[keep_mask].copy()

    #mask out the CCSNe considered to not explode:
    if remnantflag == 6:
        #mask on maltsev_region
        explodes_mask_1 = data.sn_1_maltsev_region != 'Direct BH'
        explodes_mask_2 = data.sn_2_maltsev_region != 'Direct BH'
    elif remnantflag == 4:
        #mask on the columns provided in fryer_limit
        col_1, col_2, limit = fryer_limit
        explodes_mask_1 = data[col_1] < limit
        explodes_mask_2 = data[col_2] < limit
    else:
        raise ValueError(f"Invalid remnantflag {remnantflag} provided. Must be 4 or 6.")
    # overwrite the sn_types as None where there is no explosion
    data.loc[~explodes_mask_1, 'sn_1_type'] = None
    data.loc[~explodes_mask_2, 'sn_2_type'] = None

    return data

def plot_single_delay_histogram(data,
                                ax,
                                binfrac='offner23',
                                kicks=(5, None),
                                alpha=1.0,
                                qcflag=5,
                                remnantflag=6,
                                fryer_limit=('sn_1_remnant_mass', 'sn_2_remnant_mass', 3),
                                met_cosmic=0.02,
                                sn_types_to_include = ['I', 'II'],
                                linestyle = '-',
                                linewidth=2,
                                mask_list = None,
                                labels = None,
                                color_list = ['black'],
                                histtype='step',
                                bin_range=(1, 1000),
                                n_bins=30):

    filtered_data = filter_and_mask_sne(data, binfrac, kicks, alpha, qcflag, remnantflag, fryer_limit, met_cosmic)
    data_list = []
    weight_list = []

    #if the user desires to create a stacked histogram where the mask_list is used to create the different components, aggregate data for each mask
    if mask_list is not None:
        for mask in mask_list:
            mask_1, mask_2 = mask 
            masked_sn_1_times = filtered_data[(mask_1(filtered_data)) & (filtered_data.sn_1_type.isin(sn_types_to_include))].sn_1_time
            masked_sn_2_times = filtered_data[(mask_2(filtered_data)) & (filtered_data.sn_2_type.isin(sn_types_to_include))].sn_2_time
            masked_sn_times = pd.concat([masked_sn_1_times, masked_sn_2_times])
            weights = np.ones(len(masked_sn_times)) * 1e6 / np.max(filtered_data['sample_mass'])
            data_list.append(masked_sn_times)
            weight_list.append(weights)
    else:
        sn_1_times = filtered_data[filtered_data.sn_1_type.isin(sn_types_to_include)].sn_1_time
        sn_2_times = filtered_data[filtered_data.sn_2_type.isin(sn_types_to_include)].sn_2_time
        sn_times = pd.concat([sn_1_times, sn_2_times])
        data_list.append(sn_times)
        weights = np.ones(len(sn_times)) * 1e6 / np.max(filtered_data['sample_mass'])
        weight_list.append(weights)

    # plot the entire data_list
    bins = np.logspace(np.log10(bin_range[0]), np.log10(bin_range[1]), n_bins)
    if histtype == 'step':
        lines = ax.hist(data_list, bins=bins, color=color_list, histtype=histtype, linestyle=linestyle, weights=weight_list,
                        label=labels, linewidth=linewidth, alpha=alpha)
    elif histtype == 'barstacked':
        lines = ax.hist(data_list, bins=bins, color=color_list, histtype=histtype, label=labels, weights=weight_list, alpha=alpha)
    else:
        raise ValueError(f"Invalid histtype {histtype} provided. Must be 'step' or 'barstacked'.")
    return lines

def add_descriptor(ax, sn_types, x=0.98, y=0.96):
    if len(sn_types) == 2:
        label = 'All CCSNe'
    elif sn_types[0] == 'I':
        label = 'Type I'
    elif sn_types[0] == 'II':
        label = 'Type II'
    ax.text(x, y, label, transform=ax.transAxes,
            ha='right', va='top', fontsize=LEGEND_FONT_SIZE + 5)

def add_metallicity_label(ax, z_dim, x=0.02, y=0.95):
    Z_str = r'$Z = $' + str(z_dim) + r'$\,Z_\odot$' if z_dim != 1.0 else r'$Z = Z_\odot$'
    ax.text(x, y, Z_str, transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FONT_SIZE)

xticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

def plot_remnant_types(data,
                       bin_range = (1, 500),
                       n_bins = 30,
                       axis_range = (1, 500),
                       color_list = ["#f26359", "#f7c067"],
                       savepath = 'delay_times_remnant_types.png'):
    
    # 3x3 grid, rows of binfrac = 0.0, 0.6, offner23, and cols of All, I, II
    fig, axs = plt.subplots(3, 3, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.5), sharex=False, sharey='row')

    mask_list = [(lambda df: df.sn_1_ns, lambda df: df.sn_2_ns),
                 (lambda df: ~df.sn_1_ns, lambda df: ~df.sn_2_ns)]

    for ax, sn_types, binfrac in zip(axs.flatten(),
                                          [['I', 'II'], ['I'], ['II']] * 3,
                                          ['0.0'] * 3 + ['0.6'] * 3 + ['offner23'] * 3):

        #plot first the divided histogram of the remnant types
        lines = plot_single_delay_histogram(data,
                                            ax,
                                            binfrac=binfrac,
                                            kicks=(5, None),
                                            alpha=1.0,
                                            qcflag=5,
                                            remnantflag=6,
                                            fryer_limit=('sn_1_remnant_mass', 'sn_2_remnant_mass', 3),
                                            met_cosmic=0.02,
                                            sn_types_to_include = sn_types,
                                            linestyle = None,
                                            linewidth=None,
                                            mask_list = mask_list,
                                            labels = ['NS', 'Fallback BH'],
                                            color_list = color_list,
                                            histtype='barstacked',
                                            bin_range=bin_range,
                                            n_bins=n_bins)

        #if sn_types == ['I', 'II'] and binfrac == '0.0':
            #legend = ax.legend(handles=lines, title='Remnant Type', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(1.0, 0.88), loc='upper right')
            #ax.add_artist(legend)
        
        #plot last a black line outlining the sum of all the ccsne
        plot_single_delay_histogram(data,
                                ax,
                                binfrac=binfrac,
                                kicks=(5, None),
                                alpha=1.0,
                                qcflag=5,
                                remnantflag=6,
                                fryer_limit=('sn_1_remnant_mass', 'sn_2_remnant_mass', 3),
                                met_cosmic=0.02,
                                sn_types_to_include = ['I', 'II'],
                                linestyle = '-',
                                linewidth=2,
                                mask_list = None,
                                labels = ['Total DTD'],
                                color_list = ['black'],
                                histtype='step',
                                bin_range=bin_range,
                                n_bins=n_bins)
        
        #if sn_types == ['I', 'II'] and binfrac == '0.0':
        #    legend = ax.legend(handles=lines, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(1.0, 0.6), loc='upper right')
        
        add_descriptor(ax, sn_types, y=0.96)
        ax.set_xscale('log')
        if binfrac == 'offner23':
            ax.set_xlabel('Time after starburst (Myr)', fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])    
        ax.tick_params(axis='both', labelsize=24)
        ax.set_xlim(axis_range[0], axis_range[1])
        
        if sn_types == ['I', 'II']:
            ax.set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)
            binfrac_labels = {'0.0': '0%', '0.6': '60%', 'offner23': 'Offner+23'}
            ax.text(0.02, 0.96, r'$f_{\rm bin} = $' + binfrac_labels[binfrac], transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FONT_SIZE+10)

            if binfrac == '0.0':
                add_metallicity_label(ax, 1.0, x=0.02, y=0.85)

    fig.tight_layout()
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.show()



def plot_metallicities(data, remcap=15.0):
    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=False, sharey='row')
    
    METALLICITIES = [0.0001,
                     0.000325,
                     0.000587,
                     0.00087,
                     0.00157,
                     0.002832,
                     0.005109,
                     0.009218,
                     0.016629,
                     0.03]

    for ax, sn_types, singles_only in zip(axs.flatten(),
                                          [['All'], ['I'], ['II']] * 2,
                                          [True, True, True, False, False, False]):

        bin_range = (2, 1600)
        linestyle = '-'
        
        plot_single_delay_histogram(data, METALLICITIES, 'metallicity', sn_types,
                                    ax, linestyle=linestyle, singles_only=singles_only,
                                    sn_cmaps=supernova_colormaps, bin_range=bin_range, histtype='step', remcap=remcap)

        legend_title = r'$Z/Z_\odot$'
        if singles_only:
            ax.legend(ncols=2, title=legend_title, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(1.0, 0.9), loc='upper right')
        
        add_descriptor(ax, sn_types, y=0.96)
        
        ax.set_xscale('log')

        if not singles_only:
            ax.set_xlabel('Time after starburst (Myr)', fontsize=LABEL_FONT_SIZE)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])

        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        
        if sn_types[0] == 'All':
            ax.set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)
            if singles_only:
                ax.text(0.02, 0.96, 'Singles', transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FONT_SIZE+10, fontweight='semibold')
            else:
                ax.text(0.02, 0.96, 'Singles +\nbinaries', transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FONT_SIZE+10, fontweight='semibold')

        ax.set_xlim(None, bin_range[1] + 1000)

    fig.tight_layout()
    fig.savefig('final_figs/delay_times_metallicity.png', dpi=DPI, bbox_inches='tight')
    plt.show()

def plot_sigma_and_alpha_and_klencki(sigma_data, alpha1_data, klencki_data, remcap=15.0):
    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=False, sharey=True)
    
    SIGMA_VALUES = [50.0, 73.9, 97.8, 121.7, 145.6, 169.4, 193.3, 217.2, 241.1, 265.0]
    ALPHA1_VALUES = [0.05, 0.083, 0.139, 0.232, 0.387, 0.646, 1.077, 1.797, 2.997, 5.0]


    for ax, sn_types, varied_thing_name in zip(axs.flatten(),
                                          [['All'], ['I'], ['II']] * 2,
                                          ['sigma']*3 + ['alpha1']*3):

        bin_range = (2, 1600)

        if varied_thing_name == 'sigma':
            varied_things = SIGMA_VALUES
            data = sigma_data
            legend_title = r'$\sigma$ (km$\,$s$^{-1}$)'
        else:
            varied_things = ALPHA1_VALUES
            data = alpha1_data
            legend_title = r'$\alpha$'

        histtype = 'bar' if varied_thing_name == 'Klencki' else 'step'
        plot_single_delay_histogram(data, varied_things, varied_thing_name, sn_types,
                                    ax, linestyle='-', singles_only=False,
                                    sn_cmaps=supernova_colormaps, bin_range=bin_range, histtype=histtype, remcap=remcap)

        leg = ax.legend(ncols=2, title=legend_title, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
                        bbox_to_anchor=(1.0, 0.92), loc='upper right')
        ax.add_artist(leg)
        
        #if this is the alpha row -- plot the klencki data
        if varied_thing_name == 'alpha1':
            plot_single_delay_histogram(klencki_data, [None], 'Klencki', sn_types,
                                        ax, linestyle='-', singles_only=False,
                                        sn_cmaps=supernova_colormaps, what_is_all=['I', 'II'],
                                        bin_range=bin_range, histtype='step', alpha=1.0, linewidth=1, remcap=remcap)
            #now we are going to create a legend. First we need to create the black line object to pass in
            hist_patch = Patch(facecolor='white', edgecolor='black', lw=1)
            ax.legend(handles=[hist_patch], labels=[r'$\alpha=0.7$'], title='Klencki+2021',
                      fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(1.0, 0.42), loc='upper right')

        add_descriptor(ax, sn_types, y=0.96)

        ax.set_xscale('log')

        if varied_thing_name == 'alpha1':
            ax.set_xlabel('Time after starburst (Myr)', fontsize=LABEL_FONT_SIZE)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.set_xlim(None, bin_range[1] + 1000)
        ax.set_ylim(None, 1400)
        
        if sn_types[0] == 'All':
            ax.set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)
            if varied_thing_name == 'sigma':
                ax.text(0.02, 0.96, 'Natal kicks', transform=ax.transAxes,
                        ha='left', va='top', fontsize=LEGEND_FONT_SIZE+10, fontweight='semibold')
            else:
                ax.text(0.02, 0.96, 'Common\nenvelope', transform=ax.transAxes,
                        ha='left', va='top', fontsize=LEGEND_FONT_SIZE+10, fontweight='semibold')

    for ax in axs.flatten():
        add_metallicity_label(ax, 1.0, x=0.02, y=0.85)
        break

    fig.tight_layout()
    fig.savefig('final_figs/delay_times_sigma_alpha.png', dpi=DPI, bbox_inches='tight')
    plt.show()


sn_color_dict = {
    'IIP' : "#FF7955",
    'IIL' : "#FFB06A",
    'IIn' : "#F1DF7A",
    'IIb' : "#AA8BC5",
    'Ib' : "#7E98EB",
    'Ibn' : "#37DDCF",
    'Ic' : "#7EDE74"
}


def plot_subtypes_dtd(data, I_ax, II_ax, legend=True, legend_total=True):
    # goal -- construct all the data, then make one call to stackplot
    # We're only considering NS progenitor SNe here btw.

    I_results_dict = {}
    II_results_dict = {}
    I_weights_dict = {}
    II_weights_dict = {}

    for subtype in sn_color_dict.keys():
        # get the number of SNe of this subtype
        sn_1_times = data[(data.sn_1_subtype == subtype) & (data.sn_1_remnant_mass < 3.0)]['sn_1_time']
        sn_2_times = data[(data.sn_2_subtype == subtype) & (data.sn_2_remnant_mass < 3.0)]['sn_2_time']
        sn_times = pd.concat([sn_1_times, sn_2_times])
        if len(sn_times) > 0:
            if 'II' in subtype:
                II_results_dict[subtype] = sn_times
                weights = np.ones(len(sn_times)) * (1e6 / np.max(data['sample_mass']))
                II_weights_dict[subtype] = weights
            else:
                I_results_dict[subtype] = sn_times
                weights = np.ones(len(sn_times)) * (1e6 / np.max(data['sample_mass']))
                I_weights_dict[subtype] = weights

    for results_dict, weights_dict, ax in zip([I_results_dict, II_results_dict],
                                               [I_weights_dict, II_weights_dict],
                                               [I_ax, II_ax]):
        
        colors = [sn_color_dict[subtype] for subtype in results_dict.keys()]
        weights = [weights_dict[subtype] for subtype in results_dict.keys()]
        
        bins = np.logspace(np.log10(2), np.log10(1600), 30)
        plot_data = [results_dict[subtype] for subtype in results_dict.keys()]
        if ax == None: continue
        ax.hist(plot_data, bins=bins, color=colors, histtype='barstacked', weights=weights,
                label=results_dict.keys(), alpha=1.0)
        
        if legend:
            leg = ax.legend(title='CCSN subtype', loc='upper right', bbox_to_anchor=(1.0, 0.88),
                            fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE, ncols=2)
            ax.add_artist(leg)
        
        plot_total_dtd(data, ax, singles=False, bin_range=(2, 1600), remnant_cap=3.0)
        #create the patch for the total DTD
        if ax == I_ax and legend_total:
            hist_patch = Patch(facecolor='white', edgecolor='black', lw=2)
            ax.legend(handles=[hist_patch], labels=[''], title='Total DTD', loc='upper right', bbox_to_anchor=(1.0, 0.6),
                      fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)


def plot_single_subtype_across_mass_type(data, mass_type, subtype, ax, colors,
                                         mass_ranges = [3.0, 5.0, 10.0, 15.0, 100.0]):
    results = {}
    labels = {}
    weights = {}
    for i in range(len(mass_ranges)):
        low = mass_ranges[i-1] if i > 0 else 0.0
        high = mass_ranges[i]

        if mass_type == 'zams':
            col1, col2 = 'zams_mass_1', 'zams_mass_2'
        elif mass_type == 'explosion':
            col1, col2 = 'sn_1_mass_1', 'sn_2_mass_2'
        elif mass_type == 'remnant':
            col1, col2 = 'sn_1_remnant_mass', 'sn_2_remnant_mass'
        else:
            raise ValueError(f"mass_type {mass_type} not recognized")

        sn_1_times = data[(data.sn_1_subtype == subtype) & (data[col1].between(low, high)) & (data.sn_1_remnant_mass < 15.0)]['sn_1_time']
        sn_2_times = data[(data.sn_2_subtype == subtype) & (data[col2].between(low, high)) & (data.sn_2_remnant_mass < 15.0)]['sn_2_time']
        sn_times = pd.concat([sn_1_times, sn_2_times])

        if len(sn_times) > 0:
            results[high] = sn_times                
            if high == 3.0 and mass_type == 'remnant':
                labels[high] = r'$ < 3 \, M_\odot$' + ' (NSs)'
            elif low == 0.0:
                labels[high] = r'$<$' + f'{high:.0f}' + r'$\, M_\odot$'
            else:
                labels[high] = f'{low:.0f}' + r'$\, M_\odot$' + f' - {high:.0f}' + r'$ \, M_\odot$'
            weights[low] = np.ones(len(sn_times)) * (1e6 / np.max(data['sample_mass']))

    bins = np.logspace(np.log10(2), np.log10(1600), 30)

    colors_final = {}
    for i, rem_mass in enumerate(results):
        colors_final[rem_mass] = colors[i]

    ax.hist(results.values(), bins=bins,
            color=colors_final.values(),
            histtype='barstacked', weights=weights.values(),
            label=labels.values(), alpha=1.0)

def make_mass_type_dtds_I(data, z=0.002326):
    plot_data = sn_subtypes(data, Ic_scheme='relative', Ic_ratio=0.45) #reclass Is with ratio = 0.45
    plot_data = plot_data[(plot_data.sigma == 265.0) & (plot_data.alpha1 == 1.0) & (plot_data.met_cosmic == z)]

    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=False, sharey=True)
    Ib_zams_ax, Ib_sn_mass, Ib_rem_ax, Ic_zams_ax, Ic_sn_mass, Ic_rem_ax = axs.flatten()

    for subtype, mass_type, ax in zip(['Ib']*3 + ['Ic']*3,
                                      ['zams', 'explosion','remnant']*2,
                                      [Ib_zams_ax, Ib_sn_mass, Ib_rem_ax, Ic_zams_ax, Ic_sn_mass, Ic_rem_ax]):
        
        if mass_type == 'zams':
            mass_ranges = [10.0, 18.0, 30.0, 80.0, 150.0, 1000.0] if subtype == 'Ib' else [15.0, 25.0, 40.0, 70.0, 1000.0]
        elif mass_type == 'explosion':
            mass_ranges = [2.0, 5.0, 10.0, 16.0, 100.0]
        elif mass_type == 'remnant':
            mass_ranges = [3.0, 5.0, 10.0, 15.0, 100.0]
        else:
            raise ValueError(f"mass_type {mass_type} not recognized")
        
        Ib_colormap = LinearSegmentedColormap.from_list('custom_cmap', ["#55e7f1", "#330aa3"])
        Ib_colors = [Ib_colormap(i) for i in np.linspace(0, 1.0, num=len(mass_ranges))]

        Ic_colormap = LinearSegmentedColormap.from_list('custom_cmap', ["#9ff044", "#035730"])
        Ic_colors = [Ic_colormap(i) for i in np.linspace(0, 1.0, num=len(mass_ranges))]

        colors = Ib_colors if subtype == 'Ib' else Ic_colors

        plot_single_subtype_across_mass_type(plot_data, mass_type, subtype, ax,
                                             colors, mass_ranges=mass_ranges)

        #plot the total dtd as well
        plot_total_dtd(plot_data, ax, singles=False, bin_range=(2, 1600), remnant_cap=15.0, z=z)

        #write the subtype in the upper right with ax.text
        if mass_type == 'zams': ax.text(0.98, 0.95, f'Type {subtype}', transform=ax.transAxes,
                                        ha='right', va='top', fontsize=LEGEND_FONT_SIZE, fontweight='semibold')

    #format the axes
    for ax in axs.flatten():
        ax.set_xscale('log')
        if ax in [Ic_zams_ax, Ic_sn_mass, Ic_rem_ax]: ax.set_xlabel('Time after starburst (Myr)', fontsize=LABEL_FONT_SIZE)
        if ax in [Ib_zams_ax, Ic_zams_ax]: ax.set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        if ax in [Ib_zams_ax, Ic_zams_ax]:
            legend_title = r'$M_{\rm ZAMS}$'
        elif ax in [Ib_sn_mass, Ic_sn_mass]:
            legend_title = r'$M_{\rm CC}$'
        else:
            legend_title = r'$M_{\rm rem}$'
        legend = ax.legend(title=legend_title, fontsize=LEGEND_FONT_SIZE,
                  title_fontsize=LEGEND_FONT_SIZE, loc='upper right', bbox_to_anchor=(1.0, 0.88),
                  ncol=1)
        ax.add_artist(legend)
        if ax == Ib_zams_ax:
            #add a legend for the total DTD
            hist_patch = Patch(facecolor='white', edgecolor='black', lw=2)
            ax.legend(handles=[hist_patch], labels=[''], title='Total DTD',
                      loc='upper right', bbox_to_anchor=(1.0, 0.29),
                      fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

    for ax in axs.flatten():
        add_metallicity_label(ax, 1.0, x=0.02, y=0.95)
        break

    #add titles to the top axes
    Ib_zams_ax.set_title('Color: Mass at ZAMS', fontsize=36, fontweight='semibold')
    Ib_sn_mass.set_title('Color: Mass at core collapse', fontsize=36, fontweight='semibold')
    Ib_rem_ax.set_title('Color: Remnant mass', fontsize=36, fontweight='semibold')

    #save the figure
    fig.tight_layout()
    fig.savefig('final_figs/delay_times_Ib_Ic_mass_types.png', dpi=DPI, bbox_inches='tight')
    plt.show()

def make_mass_type_dtds_II(data, z=0.002326):
    plot_data = sn_subtypes(data, IIP_scheme='branch IIP first')
    plot_data = plot_data[(plot_data.sigma == 265.0) & (plot_data.alpha1 == 1.0) & (plot_data.met_cosmic == z)]

    fig, axs = plt.subplots(4, 3, figsize=(FIG_WIDTH, 2*FIG_HEIGHT), sharex=False, sharey='row')
    IIn_axes = axs.flatten()[0:3]
    IIP_axes = axs.flatten()[3:6]
    IIL_axes = axs.flatten()[6:9]
    IIb_axes = axs.flatten()[9:12]

    for subtype, axes in zip(['IIn', 'IIP', 'IIL', 'IIb'], [IIn_axes, IIP_axes, IIL_axes, IIb_axes]):
        for mass_type, ax in zip(['zams', 'explosion', 'remnant'], axes):

            if mass_type == 'zams':
                mass_ranges = [8.0, 18.0, 25.0, 30.0, 1000.0]
            elif mass_type == 'explosion':
                mass_ranges = [5.0, 10.0, 18.0, 25.0, 1000.0]
            else:
                mass_ranges = [3.0, 5.0, 10.0, 15.0, 1000.0]
        
            IIn_colormap = LinearSegmentedColormap.from_list('custom_cmap', ["#e0e775", "#8f4e03"])
            IIn_colors = [IIn_colormap(i) for i in np.linspace(0, 1.0, num=len(mass_ranges))]

            IIP_colormap = LinearSegmentedColormap.from_list('custom_cmap', ["#ecaae1", "#D3220B"])
            IIP_colors = [IIP_colormap(i) for i in np.linspace(0, 1.0, num=len(mass_ranges))]

            IIL_colormap = LinearSegmentedColormap.from_list('custom_cmap', ["#faa508", "#6d1b02"])
            IIL_colors = [IIL_colormap(i) for i in np.linspace(0, 1.0, num=len(mass_ranges))]

            IIb_colormap = LinearSegmentedColormap.from_list('custom_cmap', ["#d4a8bd", "#1D0038"])
            IIb_colors = [IIb_colormap(i) for i in np.linspace(0, 1.0, num=len(mass_ranges))]

            color_dict = {
                'IIn': IIn_colors,
                'IIP': IIP_colors,
                'IIL': IIL_colors,
                'IIb': IIb_colors
            }

            colors = color_dict[subtype]

            plot_single_subtype_across_mass_type(plot_data, mass_type, subtype, ax,
                                                colors, mass_ranges=mass_ranges)

            #plot the total dtd as well
            if subtype in ['IIn', 'IIP']: plot_total_dtd(plot_data, ax, singles=False, bin_range=(2, 1600), remnant_cap=15.0, z=z)

            #write the subtype in the upper right with ax.text
            if mass_type == 'zams': ax.text(0.98, 0.95, f'Type {subtype}', transform=ax.transAxes,
                                            ha='right', va='top', fontsize=LEGEND_FONT_SIZE, fontweight='semibold')
        
    #format the axes
    for i, ax in enumerate(axs.flatten()):
        ax.set_xscale('log')
        if i >= 9: ax.set_xlabel('Time after starburst (Myr)', fontsize=LABEL_FONT_SIZE)
        if i % 3 == 0: ax.set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        if i % 3 == 0:
            legend_title = r'$M_{\rm ZAMS}$'
        elif i % 3 == 1:
            legend_title = r'$M_{\rm CC}$'
        else:
            legend_title = r'$M_{\rm rem}$'

        #move the legend just below the upper right corner
        legend = ax.legend(title=legend_title, fontsize=LEGEND_FONT_SIZE,
                  title_fontsize=LEGEND_FONT_SIZE, loc='upper right', bbox_to_anchor=(1.0, 0.88),
                  ncol=1)
        ax.add_artist(legend)

        if i == 0:
            #add a legend for the total DTD
            hist_patch = Patch(facecolor='white', edgecolor='black', lw=2)
            ax.legend(handles=[hist_patch], labels=[''], title='Total DTD',
                      loc='upper right', bbox_to_anchor=(1.0, 0.41),
                      fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

    for ax in axs.flatten():
        add_metallicity_label(ax, 1.0, x=0.02, y=0.95)
        break

    #add titles to the top axes
    IIn_axes[0].set_title('Color: Mass at ZAMS', fontsize=36, fontweight='semibold')
    IIn_axes[1].set_title('Color: Mass at core collapse', fontsize=36, fontweight='semibold')
    IIn_axes[2].set_title('Color: Remnant mass', fontsize=36, fontweight='semibold')

    #save the figure
    fig.tight_layout()
    fig.savefig('final_figs/delay_times_IIn_IIP_IIL_IIb_mass_types.png', dpi=DPI, bbox_inches='tight')
    plt.show()

def make_classification_dtd_plot(data, save=False, Z=0.020243,
                                 outpath="final_figs/delay_times_classification_combo.png"):

    # --- figure + grid layout (added vertical buffer between rows) ---
    fig = plt.figure(figsize=(36, 18))  # increased height slightly
    gs = gridspec.GridSpec(
        2, 6, figure=fig,
        height_ratios=[1, 1],
        hspace=0.25  # vertical space between top and bottom rows
    )

    # -------------------------------
    # Top row: 2 panels (Type II)
    # -------------------------------
    top_axes = [
        fig.add_subplot(gs[0, 1:3]),  # left
        fig.add_subplot(gs[0, 3:5])   # right
    ]
    top_texts   = ['Branch IIn first', 'Branch IIP first']
    top_schemes = ['branch IIn first', 'branch IIP first']

    for ax, scheme, text, is_rightmost in zip(top_axes, top_schemes, top_texts, [False, True]):
        plot_data = data[(data.sigma == 265.0) & (data.alpha1 == 1.0) & (data.met_cosmic == Z)]
        plot_data = sn_subtypes(plot_data, IIP_scheme=scheme)

        legend = not is_rightmost
        plot_subtypes_dtd(plot_data, I_ax=None, II_ax=ax, legend=legend, legend_total=False)
        plot_total_dtd(plot_data, ax, singles=False, bin_range=(2, 1600), remnant_cap=3.0, z=Z)
        if not is_rightmost:
            #add a square for the total DTD legend
            hist_patch = Patch(facecolor='white', edgecolor='black', lw=2)
            ax.legend(handles=[hist_patch], labels=[''], title='Total DTD', loc='upper right', bbox_to_anchor=(1.0, 0.6),
                      fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

        # labels
        ax.text(0.98, 0.95, text, transform=ax.transAxes,
                ha='right', va='top', fontsize=LEGEND_FONT_SIZE)
        if not is_rightmost:
            add_metallicity_label(ax, z_dim=1.0)

        # formatting
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.set_xlim(None, 1600 + 1000)

        # y-axis label only on leftmost
        if not is_rightmost:
            ax.set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)
        else:
            ax.tick_params(labelleft=False)  # keep ticks, remove labels

        ax.set_xlabel('Time after starburst (Myr)', fontsize=LABEL_FONT_SIZE)
    # -------------------------------
    # Bottom row: 3 panels (Type I)
    # -------------------------------
    bottom_axes = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]
    ic_ratios = [0.3, 0.43, 0.6]

    for ax, Ic_ratio, is_rightmost in zip(bottom_axes, ic_ratios, [False, True, True]):
        plot_data = sn_subtypes(data, Ic_scheme='relative', Ic_ratio=Ic_ratio)
        plot_data = plot_data[(plot_data.sigma == 265.0) & (plot_data.alpha1 == 1.0) & (plot_data.met_cosmic == Z)]

        legend = not is_rightmost
        plot_subtypes_dtd(plot_data, I_ax=ax, II_ax=None, legend=legend, legend_total=False)
        plot_total_dtd(plot_data, ax, singles=False, bin_range=(2, 1600), remnant_cap=3.0, z=Z)

        # text label
        text = r'$M_{\rm ej,He} / M_{\rm ej} < $' + f'{Ic_ratio}'
        ax.text(0.98, 0.95, text, transform=ax.transAxes, ha='right', va='top', fontsize=LEGEND_FONT_SIZE)
        if not is_rightmost:
            add_metallicity_label(ax, z_dim=1.0, x=0.02, y=0.95)

        # axes formatting
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.set_xlim(None, 1600 + 1000)

        # y-axis labels: only leftmost
        if not is_rightmost:
            ax.set_ylabel(r'$N_{\rm CCSNe}$ $(10^6 \, M_\odot)^{-1}$', fontsize=LABEL_FONT_SIZE)
        else:
            ax.tick_params(labelleft=False)  # keep ticks, remove labels

        ax.set_xlabel('Time after starburst (Myr)', fontsize=LABEL_FONT_SIZE)
    if save:
        fig.savefig(outpath, dpi=DPI, bbox_inches='tight')
    plt.show()
