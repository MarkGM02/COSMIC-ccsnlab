import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

from sn_types import sn_subtypes
import pandas as pd

LEGEND_FONT_SIZE = 24
LABEL_FONT_SIZE = 36
TICK_FONT_SIZE = 28
DPI=300
AX_TICKS = [0.01, 0.1, 1.0]

sn_color_dict = {
    'IIP' : "#FF7955",
    'IIL' : "#FFB06A",
    'IIn' : "#F1DF7A",
    'IIb' : "#AA8BC5",
    'Ib' : "#7E98EB",
    'Ic' : "#7EDE74"
}

text_style_dict = dict(boxstyle='round,pad=0.3',
                       facecolor='white',
                       edgecolor='none',
                       linewidth=0.8,
                       alpha=0.8)

def plot_I_or_II_subtypes(data, sn_type, ax, zsun=0.02, remnant_cap=3.0, plot_type='stackplot'):
    if sn_type not in ['I', 'II']:
        raise ValueError("sn_type must be either 'I' or 'II'")

    results_dict = {subtype: [] for subtype in sn_color_dict.keys()}
    found_subtypes = []
    metallicities = data.met_cosmic.unique()
    metallicities.sort()

    for met in metallicities:
        curr = data[data.met_cosmic == met]
        # get the total number of SNe I + SNe II
        total_sn_1 = len(curr[(curr.sn_1_type == sn_type) & (curr.sn_1_remnant_mass < remnant_cap)])
        total_sn_2 = len(curr[(curr.sn_2_type == sn_type) & (curr.sn_2_remnant_mass < remnant_cap)])
        total_sn = total_sn_1 + total_sn_2
        for subtype in sn_color_dict.keys():
            if (sn_type == 'I' and 'II' in subtype) or (sn_type == 'II' and not 'II' in subtype): continue
            # get the number of SNe of this subtype
            num_sn_1 = len(curr[(curr.sn_1_subtype == subtype) & (curr.sn_1_remnant_mass < remnant_cap)])
            num_sn_2 = len(curr[(curr.sn_2_subtype == subtype) & (curr.sn_2_remnant_mass < remnant_cap)])
            num_sn = num_sn_1 + num_sn_2
            results_dict[subtype].append(num_sn / total_sn if total_sn > 0 else 0.0)
            if num_sn > 0 and not subtype in found_subtypes: found_subtypes.append(subtype)

    #only keep the subtypes that we found in results_dict, but keep the order of results_dict.keys()
    final_results_dict = {}
    for k in results_dict.keys():
        if k in found_subtypes: final_results_dict[k] = results_dict[k]
    colors = [sn_color_dict[subtype] for subtype in final_results_dict.keys()]
    #for plotting, convert Z to Z/Zsun
    metallicities = metallicities / zsun
    if plot_type == 'stackplot':
        ax.stackplot(metallicities, *final_results_dict.values(), labels=final_results_dict.keys(), colors=colors)
    else:
        for subtype, values in final_results_dict.items():
            ax.plot(metallicities, values, label=subtype, color=sn_color_dict[subtype], linewidth=3)

def make_type_2_plot(data, save=False, plot_type='stackplot'):
    fig, axs = plt.subplots(1, 2, figsize=(22, 8), sharey=True)
    ax1, ax2 = axs.flatten()

    #on the left axis we filter out IIns first
    reclassed = sn_subtypes(data, IIP_scheme='exclude n')
    plot_I_or_II_subtypes(reclassed, 'II', ax1, plot_type=plot_type)

    #on the right axis we filter out IIPs first
    reclassed = sn_subtypes(data, IIP_scheme='include n')
    plot_I_or_II_subtypes(reclassed, 'II', ax2, plot_type=plot_type)

    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$Z \, / \, Z_\odot$', fontsize=LABEL_FONT_SIZE)
        ax.set_xscale('log')
        ax.set_xlim(5e-3, 1.5)
        ax.set_xticks(AX_TICKS, fontsize=TICK_FONT_SIZE, labels=[str(tick) for tick in AX_TICKS])

        if plot_type == 'stackplot':
            ax.set_ylim(0.45, 1.0)
        else:
            ax.set_ylim(1e-3, 1.0)
            ax.set_yscale('log')
        
        ax.tick_params(axis='both', which='major', length=10, width=4)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        if ax == ax1:
            ax.legend(title='CCSN subtype', loc='center left', bbox_to_anchor=(0.0, 0.25),
                      fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE, ncols=2)
            
            ax.set_ylabel(r'$N_{\rm subtype} \, / \, N_{\rm II}$', fontsize=LABEL_FONT_SIZE)
         
        message = 'Branch IIn first' if ax == ax1 else 'Branch IIP first'
        
        ax.text(0.03, 0.03, message, transform=ax.transAxes, ha='left',
                va='bottom', fontsize=LEGEND_FONT_SIZE, bbox=text_style_dict)
        
        ax.text(0.97, 0.03, r'IIP $\downarrow$', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=LEGEND_FONT_SIZE, bbox=text_style_dict)

    fig.tight_layout()
    if save: fig.savefig("final_figs/type_II_classification_variations.png", dpi=DPI, bbox_inches='tight')
    plt.show()


def add_shivvers_point(ax):
    #Shivvers 2017 has a Ib/Ic ratio of 0.6+/-0.3 over their metallicity range of LOSS
    Ib_proportion_of_type_I = 0.625 # this is 1 / (1 + 0.6)
    Ib_proportion_of_type_I_error = (0.625 ** 2) * 0.3 # error propagation

    #read in loss data with solar abundance 9.05
    loss_data = pd.read_csv('z_9.05_data.csv')
    #the lowest value is the first point minus its left error bar
    z_min = loss_data.iloc[0]['z'] - loss_data.iloc[0]['z_err_n']

    #highest value is last point plus its right error bar
    z_max = loss_data.iloc[-1]['z'] + loss_data.iloc[-1]['z_err_p']

    #draw a single line from (z_min, z_max) with shaded error +/- Ib error
    ax.hlines(y=Ib_proportion_of_type_I, xmin=z_min, xmax=z_max,
              color='black', linestyle='-', linewidth=2)

    ax.fill_betweenx(y=[Ib_proportion_of_type_I - Ib_proportion_of_type_I_error,
                     Ib_proportion_of_type_I + Ib_proportion_of_type_I_error],
                     x1=z_min, x2=z_max,
                     color='black', alpha=0.3)

class HandlerLineWithPatch(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        line, patch = orig_handle
        rect_height = 0.9
        rect_y = ydescent + (1 - rect_height) / 2 * height  # center it vertically

        rect = plt.Rectangle(
            (xdescent, rect_y),
            width, height * rect_height,
            facecolor=patch.get_facecolor(),
            alpha=patch.get_alpha(),
            edgecolor='none',
            transform=trans
        )
        ymid = ydescent + 0.5 * height
        legline = plt.Line2D(
            [xdescent, xdescent + width],
            [ymid, ymid],
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            linestyle=line.get_linestyle(),
            transform=trans
        )
        return [rect, legline]


def make_type_1_plot(data, save=False, include_braces=False, ratios=[0.3, 0.43, 0.6]):
    TICK_FONT_SIZE = 40
    LABEL_FONT_SIZE = 70
    BRACES_FONT_SIZE = 58
    LEGEND_FONT_SIZE = 25

    import matplotlib.pyplot as plt
    if include_braces: from curlyBrace import curlyBrace  # from the single-file you copied

    def add_brace_above(fig, ax_list, label, *,
                        extend=0.045,      # widen horizontally (fraction of group width)
                        pad=0.012,         # gap above panels before the overlay starts
                        height=0.24,       # desired overlay height (may be capped)
                        k_r=0.06,          # curvature: lower = flatter
                        fontsize=36,
                        limit_above_axes=None,  # axes of row ABOVE -> cap overlay to fit gap
                        anchor=0.5,  # vertical anchor (0..1) within overlay for brace
                        right_shift=0.01): #amount to shift brace start over by      
        boxes = [ax.get_position() for ax in ax_list]
        left, right = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
        top = max(b.y1 for b in boxes)
        width = right - left
        left  -= extend * width
        right += extend * width

        ov_y0 = top + pad
        ov_h  = height

        if limit_above_axes is not None:
            above_boxes = [ax.get_position() for ax in limit_above_axes]
            y_bottom_of_above = min(b.y0 for b in above_boxes)
            max_h = max(0.001, y_bottom_of_above - ov_y0 - 0.002)  # small safety margin
            ov_h = min(ov_h, max_h)

        overlay = fig.add_axes([left + right_shift, ov_y0, right - left, ov_h], frameon=False)
        overlay.set_xlim(0, 1); overlay.set_ylim(0, 1); overlay.axis("off")

        curlyBrace(
            fig, overlay,
            p1=[0.01, anchor], p2=[0.99, anchor],
            k_r=k_r, bool_auto=True,
            str_text=label, int_line_num=2,
            fontdict={"fontsize": fontsize, "fontweight": "bold"},
            color="black", lw=2, clip_on=False,
        )

    base_w, base_h = 8*5, 8*2
    gap_ratio = 0.5
    fig_h = base_h * (2.0 + gap_ratio) / 2.0

    layout = [
        ["A", ".", "C", "D", "E"],
        [".", ".", ".", ".", "."],
        ["B", ".", "F", "G", "H"],
    ]

    fig, axes = plt.subplot_mosaic(
        layout,
        figsize=(base_w, fig_h),
        gridspec_kw={
            "width_ratios":  [1, 0.16, 1, 1, 1],
            "height_ratios": [1, gap_ratio, 1],
        },
        constrained_layout=True,
    )

    abs_ax, evol_ax = axes['A'], axes['B']
    ratio_03_ax, ratio_045_ax, ratio_06_ax = axes['C'], axes['D'], axes['E']
    rem_4_ax, rem_5_ax, rem_20_ax = axes['F'], axes['G'], axes['H']

    reclassed = sn_subtypes(data, Ic_scheme='absolute', Ib_Ic_he_thresh=0.14)
    plot_I_or_II_subtypes(reclassed, 'I', abs_ax)
    
    reclassed = sn_subtypes(data, Ic_scheme='evolution')
    plot_I_or_II_subtypes(reclassed, 'I', evol_ax)

    for ratio, ax in zip(ratios, [ratio_03_ax, ratio_045_ax, ratio_06_ax]):
        reclassed = sn_subtypes(data, Ic_scheme='relative', Ic_ratio=ratio)
        plot_I_or_II_subtypes(reclassed, 'I', ax)

    reclassed = sn_subtypes(data, Ic_scheme='relative', Ic_ratio=ratios[1])
    for rem_mass, ax in zip([5.0, 15.0, 1000.0], [rem_4_ax, rem_5_ax, rem_20_ax]):
        plot_I_or_II_subtypes(reclassed, 'I', ax, remnant_cap=rem_mass)

    label_dict = {
        abs_ax:        r"$M_{\rm ej,He} < 0.14 \, M_\odot$",
        evol_ax:       "Mass transfer as helium star",
        ratio_03_ax:   r"$M_{\rm ej,He} / M_{\rm ej} < $" + f'{ratios[0]}',
        ratio_045_ax:  r"$M_{\rm ej,He} / M_{\rm ej} < $" + f'{ratios[1]}',
        ratio_06_ax:   r"$M_{\rm ej,He} / M_{\rm ej} < $" + f'{ratios[2]}',
        rem_4_ax:      r"$M_{\rm rem} < 5 \, M_\odot$",
        rem_5_ax:      r"$M_{\rm rem} < 15 \, M_\odot$",
        rem_20_ax:     r"All remnants",
    }

    for ax in axes.values():
        if ax in [abs_ax, evol_ax, ratio_03_ax, rem_4_ax]:
            ax.set_ylabel(r'$N_{\rm subtype} \, / \, N_{\rm I}$', fontsize=LABEL_FONT_SIZE)
        if ax in [evol_ax, rem_4_ax, rem_5_ax, rem_20_ax]:
            ax.set_xlabel(r'$Z \, / \, Z_\odot$', fontsize=LABEL_FONT_SIZE)
        ax.set_xscale('log')
        ax.set_xlim(5e-3, 1.5)
        ax.set_xticks(AX_TICKS, fontsize=TICK_FONT_SIZE, labels=[str(tick) for tick in AX_TICKS])
        ax.set_ylim(0, 1.0)
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_yticks(ticks)
        tick_labels = [str(tick) for tick in ticks] if ax in [abs_ax, evol_ax, ratio_03_ax, rem_4_ax] else ['' for tick in ticks]
        ax.set_yticklabels(tick_labels)
        ax.tick_params(axis='both', which='major', length=10, width=4)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        if ax == abs_ax:
            legend_1 = ax.legend(title='CCSN subtype', loc='lower left', bbox_to_anchor=(0.0, 0.15),
                                 fontsize=LEGEND_FONT_SIZE+15, title_fontsize=LEGEND_FONT_SIZE+15, ncols=2)
            ax.add_artist(legend_1)
        
        x = 0.05 if ax != evol_ax else 0.03
        ax.text(x, 0.05, label_dict[ax], transform=ax.transAxes,
                ha='left', va='bottom', fontsize=LEGEND_FONT_SIZE+15, bbox=text_style_dict)
        
        add_shivvers_point(ax)
        if ax == abs_ax:

            line = mlines.Line2D([], [], color='black', linewidth=2)
            band = mpatches.Patch(color='black', alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.55),
                      fontsize=LEGEND_FONT_SIZE+15, title_fontsize=LEGEND_FONT_SIZE+15,
                      handles=[(line, band)],
                      labels=['Shivvers+2017'],
                      handler_map={tuple: HandlerLineWithPatch()}
            )

    # Top-right row (C–E)
    if include_braces:
        add_brace_above(
            fig,
            [ratio_03_ax, ratio_045_ax, ratio_06_ax],
            label=r"$M_{\rm ej,He}/M_{\rm ej}$ varied ($M_{\rm rem}<3\,M_\odot$ fixed)",
            extend=0.15, pad=0.012, height=0.24, k_r=0.025, fontsize=BRACES_FONT_SIZE, right_shift=0.03
        )

        # Bottom-right row (F–H)
        add_brace_above(
            fig,
            [rem_4_ax, rem_5_ax, rem_20_ax],
            label=r"$M_{\rm rem}$ varied ($M_{\rm ej,He}/M_{\rm ej}=$" + f'{ratios[1]}' +  " fixed)",
            extend=0.15, pad=0.012, height=0.1, k_r=0.025, fontsize=BRACES_FONT_SIZE,
            limit_above_axes=[ratio_03_ax, ratio_045_ax, ratio_06_ax],  # keep inside gap
            anchor=0.5, right_shift=0.03
        )

    if save:
        fig.savefig("final_figs/type_I_classification_variations.png", dpi=DPI, bbox_inches='tight')
    plt.show()


def make_appendix_plot(data, save=False):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    ax1, ax2 = axs.flatten()
    reclassed = sn_subtypes(data, IIP_scheme='include n', Ic_scheme='relative', Ic_ratio=0.43)

    plot_I_or_II_subtypes(reclassed, 'II', ax1)
    plot_I_or_II_subtypes(reclassed, 'I', ax2)

    for ax in axs.flatten():
        ax.set_xscale('log')
        ax.set_xlim(5e-3, 1.5)
        ax.set_xticks(AX_TICKS, fontsize=TICK_FONT_SIZE, labels=[str(tick) for tick in AX_TICKS])

        if ax == ax1:
            ax.set_ylim(0.45, 1.0)
            message = 'Branch IIP first'
            ylabel = r'$N_{\rm subtype} \, / \, N_{\rm II \, \, or \, \, I}$'
            ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_ylim(0.0, 1.0)
            message = r"$M_{\rm He} / M_{\rm ejecta} < 0.43$"
        
        ax.tick_params(axis='both', which='major', length=10, width=4)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.set_xlabel(r'$Z \, / \, Z_\odot$', fontsize=LABEL_FONT_SIZE)
        
        shift = 0.0 if ax == ax1 else 0.02
        leg = ax.legend(title='CCSN subtype', loc='center left', bbox_to_anchor=(0.0, 0.25 - shift), fontsize=LEGEND_FONT_SIZE,
                        title_fontsize=LEGEND_FONT_SIZE, ncols=2, framealpha=0.8, facecolor='white')
        ax.add_artist(leg)
        if ax == ax2:
            add_shivvers_point(ax)
            line = mlines.Line2D([], [], color='black', linewidth=2)
            band = mpatches.Patch(color='black', alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.39),
                      fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
                      handles=[(line, band)],
                      labels=['Shivvers+2017'],
                      handler_map={tuple: HandlerLineWithPatch()}
            )
        else:
            ax.text(0.97, 0.03, r'IIP $\downarrow$', transform=ax.transAxes, ha='right',
                    va='bottom', fontsize=LEGEND_FONT_SIZE, bbox=text_style_dict)

        ax.text(0.03, 0.03, message, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=LEGEND_FONT_SIZE, bbox=text_style_dict)
        
    fig.tight_layout()
    if save:
        fig.savefig("final_figs/appendix_MT_stability.png", dpi=DPI, bbox_inches='tight')
    plt.show()
