import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from importlib import resources
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '5.0'})
rcParams.update({'xtick.major.size': '5.5'})
rcParams.update({'xtick.major.width': '1.0'})
rcParams.update({'xtick.minor.pad': '5.0'})
rcParams.update({'xtick.minor.size': '2.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '5.0'})
rcParams.update({'ytick.major.size': '5.5'})
rcParams.update({'ytick.major.width': '1.0'})
rcParams.update({'ytick.minor.pad': '5.0'})
rcParams.update({'ytick.minor.size': '2.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'axes.titlepad': '10.0'})
rcParams.update({'axes.labelpad': '10.0'})
rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"]=[12.0,8.0]

LEGEND_FONT_SIZE = 20

from matplotlib.colors import LogNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# https://www.annualreviews.org/content/journals/10.1146/annurev.astro.46.060407.145222 Asplund -> solar is 8.69
def z(twelve_plus_log_o_by_h, solar=9.05):
    return 10 ** (twelve_plus_log_o_by_h - solar) #leave as z/zsun

def get_purple_data(solar=9.05):
    # x and y of data points
    x = np.array([8.33871722770988, 8.966325095503501, 9.052788851077736, 9.112419093005128, 9.149687883119181, 9.236151765654062])
    y = np.array([0.09983661498707301, 0.572791085909634, 0.572791085909634, 0.377656040730448, 0.572791085909634, 0.3743486288903077])

    # lower and upper coordinate of the y error bars
    ylow = np.array([0.011921942293764148, 0.21678683989962078, 0.21505623707677782, 0.12444957422994307, 0.2165176655307114, 0.12591096747467206])
    yhi = np.array([0.20459572323694725, 0.9211806372480926, 0.9311412508014909, 0.6213248618520097, 0.9223728561239123, 0.6242476835506723])

    return get_data(x, y, ylow, yhi, x, x, solar=solar) #because no x errors -> x = xlo = xhi

def get_green_data(solar=9.05):
    # x and y of data points
    x = np.array([8.630003833858874, 8.851953009184227, 8.975512253589221, 9.13224961605866])
    y = np.array([0.19986536660346432, 0.25570617918789423, 0.4105374591552597, 0.5298336733680603])

    # lower and upper coordinate of the y error bars
    ylo = np.array([0.14148639288025874, 0.18463611809717356, 0.30773964443654683, 0.40546099604089003])
    yhi = np.array([0.25824441074507903, 0.3242380439306346, 0.5107970423167877, 0.6567445822524154])

    # lower and upper coordinate of the x error bars
    xlo = np.array([8.220531560953296, 8.774676443435876, 8.911409822867679, 9.053170145641143])
    xhi = np.array([8.773671042071479, 8.90939908361921, 9.053170209121467, 9.391987675788918])

    return get_data(x, y, ylo, yhi, xlo, xhi, solar=solar)

def get_pink_data(solar=9.05):
    # x and y of data points
    x = [8.623035408301737, 9.00785820809224, 9.187788868534824]
    y = [0.27728104179635166, 0.31766177432935405, 0.44803385365019055]

    # lower and upper coordinate of the y error bars
    ylo = [0.19382755136761629, 0.22747818528458805, 0.33516011523312483]
    yhi = [0.3620806505340659, 0.40919148168309905, 0.5572541793738428]

    # lower and upper coordinate of the x error bars
    xlo = [7.912326655073984, 8.888597851198101, 9.11047757904949]
    xhi = [8.887211060049811, 9.11047757904949, 9.324036940893082]

    return get_data(x, y, ylo, yhi, xlo, xhi, solar=solar)

def get_data(x, y, ylo, yhi, xlo, xhi, solar=9.05):
    ylo = np.array(ylo)
    yhi = np.array(yhi)

    xlo = np.array(xlo)
    xhi = np.array(xhi)

    ratio_err_p = yhi - y
    ratio_err_n = y - ylo

    x, xlo, xhi = z(np.array(x), solar=solar), z(np.array(xlo), solar=solar), z(np.array(xhi), solar=solar)
    z_err_p = xhi - x
    z_err_n = x - xlo

    data = pd.DataFrame({'z': x, 'z_err_p' : z_err_p, 'z_err_n' : z_err_n, 'ratio': y, 'ratio_err_p': ratio_err_p, 'ratio_err_n': ratio_err_n})
    return data

def get_loss_data(solar=9.0, raw_file="raw_loss_data.csv"):
    with resources.files("ccsnlab.plotting").joinpath(raw_file) as path:
        df = pd.read_csv(path)

    oh12 = df["oh12"].to_numpy(float)
    doh_p = df["oh12_err_p"].to_numpy(float)
    doh_n = df["oh12_err_n"].to_numpy(float)

    # central value
    Z = z(oh12, solar)

    # upper/lower abundance bounds
    oh12_upper = oh12 + doh_p
    oh12_lower = oh12 - doh_n

    # convert bounds
    z_upper = z(oh12_upper, solar)
    z_lower = z(oh12_lower, solar)

    # error bars
    z_err_p = z_upper - Z
    z_err_n = Z - z_lower

    df = df.copy()
    df["z"] = Z
    df["z_err_p"] = z_err_p
    df["z_err_n"] = z_err_n

    return df

def plot_bpass_models(ax, color='lightsteelblue', incl_bh_binaries=False):
    def get_bpass_data(x, y):
        #upper limit is 2 solar, we fix this
        solar = x[-1] - np.log10(2)
        return pd.DataFrame({'z': z(np.array(x), solar=solar), 'ratio': y})

    x = [7.8, 8.074253893034884, 8.110646436124183, 8.174244847717876,
         8.259819187692331, 8.312773667049122, 8.356172710422513, 8.405764776315719,
         8.463660588661952, 8.517921817306409, 8.55024057325183, 8.572530919179378,
         8.598948968713207, 8.623407220641186, 8.652029556929921, 8.691026269521792,
         8.73846823684402, 8.796572987014157, 8.857160542400084, 8.89708182812609,
         8.94988407187667, 9.007262724977661, 9.058574651358452, 9.129605995056643,
         9.19683991383185, 9.275539469508663]
    y = [0.01928138983356896, 0.021556831651754052, 0.026107824932119054, 0.03520959220486004,
         0.044311359477601016, 0.0511376849321569, 0.05568856856852729, 0.08071853821255935,
         0.1125748333111475, 0.14898201204610584, 0.1694610980537677, 0.20586827678872624,
         0.24910178097824026, 0.29461072698593954, 0.34467066627400406, 0.36059875900130073,
         0.3810778450089624, 0.4038323728348096, 0.42431134919847663, 0.46526941156980556,
         0.5130539090396847, 0.5653891805019398, 0.6154491197900043, 0.6450299730704071,
         0.6746108263508098, 0.7087425632675828]
    
    no_bh_binary_bpass_data = get_bpass_data(x, y)

    x = [7.8, 7.850006535203389, 7.887254950192448, 7.9509833601357665,
         8.020920155525722, 8.074253893034884, 8.122172832120768, 8.16457679319872,
         8.250049920474055, 8.358150664827235, 8.393835927291136, 8.447649504251551,
         8.493762943482057, 8.554289225347606, 8.596914029446056, 8.629532498171248,
         8.65407773976169, 8.713683641598443, 8.78201077870855, 8.850873491661973,
         8.909725862010765, 8.960482147406633, 9.028607191112071, 9.067154978800817,
         9.125285259936915, 9.17509752062749, 9.275539469508663]
    y = [0.09664674058385063, 0.10119762422022122, 0.10347306603840652, 0.1080239496747771,
         0.11485027512933259, 0.12167660058388848, 0.13305391931880936, 0.14215568659155034,
         0.14443112840973543, 0.14898201204610584, 0.17628742350832338, 0.2172454858796523,
         0.25137722279642555, 0.29461072698593954, 0.40155693101662426, 0.48802393939565253,
         0.5449102041382727, 0.5540119714110137, 0.5608382968655694, 0.567664622320125,
         0.5813173828732311, 0.5926945919641573, 0.6086227943354486, 0.6177245616081896,
         0.6427545312522216, 0.6655089494340742, 0.7087425632675828]
    inc_bh_binary_bpass_data =  get_bpass_data(x, y)

    x = [7.8, 7.831448301190421, 7.8760619325405195, 7.902203811809528,
         7.949101671118732, 8.001957863892201, 8.06470401042826, 8.108726961056933,
         8.147203008882363, 8.18005117479042, 8.226650701915123, 8.277433124266961,
         8.304907217188683, 8.332472405882097, 8.3660675471489, 8.405764776315719,
         8.447649504251551, 8.491752702765359, 8.540127908733302, 8.5948793756966,
         8.63975122686867, 8.686913099418994, 8.75295824351144, 8.829949204504073,
         8.928725714509309, 9.030744520639429, 9.101557638882033, 9.172926235939954, 9.275539469508663]
    y = [0.032934150386674944, 0.03520959220486004, 0.03748503402304533, 0.03748503402304533,
         0.04658680129578631, 0.05568856856852729, 0.06706588730344819, 0.06706588730344819,
         0.06934132912163328, 0.07161677093981837, 0.07844309639437426, 0.08526942184893015,
         0.08982030548530033, 0.09209574730348583, 0.09892218240203612, 0.1080239496747771,
         0.11712571694751808, 0.12622748422025906, 0.13532936113699445, 0.14898201204610584,
         0.1626347725992118, 0.1671856562355824, 0.171736539871953, 0.17628742350832338,
         0.18083830714469398, 0.18311374896287907, 0.18538919078106436, 0.18538919078106436,
         0.18538919078106436]
    
    singles_bpass_data = get_bpass_data(x, y)

    all_data = [singles_bpass_data, no_bh_binary_bpass_data, inc_bh_binary_bpass_data]
    linestyles = ['--', '-', ':']
    linewidths = [2, 2, 2]
    labels = ['BPASS v2, single', 'BPASS v2, binary - BH', 'BPASS v2, binary + BH']
    for data, linestyle, linewidth, label in zip(all_data, linestyles, linewidths, labels):
        if (not incl_bh_binaries) and (label == 'BPASS v2, binary + BH'): continue
        ax.plot(data['z'], data['ratio'], linestyle=linestyle, linewidth=linewidth, label=label, color=color)

def plot_souropanis_25_models(ax, color='thistle', include_SESNE=False):
    #Ibc/ II
    x = [0.01, 0.011873687211933442, 0.014461757658860153, 0.017391276558854276,
         0.02257309664635324, 0.0273189541729936, 0.03162277372429567, 0.04318761535688009,
         0.07048006953684904, 0.10000001819820284, 0.13314028183832874, 0.2013110385425181,
         0.26129260032694424, 0.3591275840973296, 0.4486768235630411, 0.5973694787245845,
         0.7902964546133844, 1, 1.2257381581403055, 1.4009054326803874, 1.6216036392264996, 2.0]
    y = [0.17877116850524477, 0.17877116850524477, 0.1766281017552054, 0.17877116850524477,
         0.1766281017552054, 0.1766281017552054, 0.1766281017552054, 0.1766281017552054,
         0.17448515762728353, 0.1766281017552054, 0.18520000088901015, 0.2002008550286985,
         0.22805961918015308, 0.2602042715874514, 0.28377702486094447, 0.2902058572447101,
         0.2944918681226716, 0.3009208231285547, 0.3223505096520084, 0.339494430541736,
         0.35663822880934615, 0.38021098208283943]
    souropanis_25_ibc_data = pd.DataFrame({'x': x, 'y': y})
    ax.plot(souropanis_25_ibc_data['x'], souropanis_25_ibc_data['y'],
            label='Souropanis 25 Ibc/II', color=color, linestyle='-', linewidth=2)    
    
    #SESNE / II
    if include_SESNE:
        x = [0.01, 0.011607654568522977, 0.013409641252792289, 0.015695942125257024,
             0.018492954950850987, 0.021363816425949787, 0.0241994381155095, 0.028699405396576585,
             0.03359256207097674, 0.03855377961011624, 0.045723001833145926, 0.05530298728378382,
             0.06558678849456823, 0.0793286788854249, 0.10178484302579209, 0.11605350185984194,
             0.14036934182437286, 0.1653831222186813, 0.19613667258098555, 0.23879295559786753,
             0.3023964558755522, 0.3754791289673204, 0.48176847160902864, 0.5904046595951035,
             0.7141078222481868, 0.85808262966641, 1.0176459938866773, 1.2228186322825079,
             1.498557628580554, 2.0]
        y = [0.3569060355142776, 0.35248612128385043, 0.3502762254796957, 0.348066329675541,
             0.348066329675541, 0.343646415445114, 0.343646415445114, 0.339226501214687,
             0.33701660541053224, 0.33480658698425997, 0.33259669118010526, 0.3303866727538328,
             0.3281767769496782, 0.32375686271925125, 0.3193369484888241, 0.32375686271925125,
             0.3281767769496782, 0.33259669118010526, 0.33701660541053224, 0.339226501214687,
             0.34143639701884165, 0.343646415445114, 0.3458563112492687, 0.3458563112492687,
             0.343646415445114, 0.343646415445114, 0.3458563112492687, 0.36574586397513154,
             0.3878453125051492, 0.42099448530017547]
        
        souropanis_sesne_25_data = pd.DataFrame({'x': x, 'y': y})
        ax.plot(souropanis_sesne_25_data['x'], souropanis_sesne_25_data['y'],
                label='Souropanis 25 SESNe/II', color=color, linewidth=2)


def plot_loss_data(ax, label_loss=True, solar=9.05):
    all_data = [get_pink_data(solar=solar), get_loss_data(solar=solar)]

    markerfacecolors = ["gold", "black"]
    markeredgecolor = 'black'
    
    colors =  ['black', 'black']
    markers = ['H', 's']
    labels =  [r'KK12, $Z$, Ibc/II', 'G17 Ibc/(II+IIb)']

    lines = []
    
    for data, color, marker, label, facecolor in zip(all_data, colors, markers, labels, markerfacecolors):
        label = label if label_loss else None
        line = ax.errorbar(
                           data['z'], data['ratio'],
                           xerr=[data['z_err_p'], data['z_err_n']], 
                           yerr=[data['ratio_err_p'], data['ratio_err_n']], 
                           fmt=marker, capsize=3.5,
                           markersize=12,
                           color=color, ecolor=color,
                           elinewidth=2, capthick=2,
                           label=label,
                           markerfacecolor=facecolor,
                           markeredgecolor=markeredgecolor,
                           markeredgewidth=1.5
                          )
        lines.append(line)

    return lines


def apply_explosion_criteria(data,
                             explosion_criteria = 'maltsev'):
    """
    Apply explosion criteria to the data by masking out the rows where the
    criteria is not met and setting the SN type and subtype to None for those rows.
    
    :param data: DataFrame containing the sn_info which is formatted with all the appropriate
    columns created by ccsnlab.data_loading.process_raw.
    :param explosion_criteria: A string or tuple indicating the explosion criteria to apply.
    This is now either 'maltsev' or a tuple of 3 elements, where the first two elements are strings
    that indicate the column to apply the criteria to (e.g. 'sn_1_remnant_mass', 'sn_2_remnant_mass')
    and the third element is the mass threshold to apply.
    """

    data = data.copy()
    if explosion_criteria == 'maltsev':
        explosion_mask_sn1 = data['sn_1_maltsev_region'] != 'Direct BH'
        explosion_mask_sn2 = data['sn_2_maltsev_region'] != 'Direct BH'
        data.loc[~explosion_mask_sn1, 'sn_1_type'] = None
        data.loc[~explosion_mask_sn1, 'sn_1_subtype'] = None
        data.loc[~explosion_mask_sn2, 'sn_2_type'] = None
        data.loc[~explosion_mask_sn2, 'sn_2_subtype'] = None
    elif isinstance(explosion_criteria, tuple) and len(explosion_criteria) == 3:
        col_sn1, col_sn2, mass_threshold = explosion_criteria
        explosion_mask_sn1 = data[col_sn1] <= mass_threshold
        explosion_mask_sn2 = data[col_sn2] <= mass_threshold
        data.loc[~explosion_mask_sn1, 'sn_1_type'] = None
        data.loc[~explosion_mask_sn1, 'sn_1_subtype'] = None
        data.loc[~explosion_mask_sn2, 'sn_2_type'] = None
        data.loc[~explosion_mask_sn2, 'sn_2_subtype'] = None
    else:
        raise ValueError(f"Unsupported explosion criteria: {explosion_criteria}")
    return data

def format_ax(ax, text_y=None):
    ax.axvspan(1.5, 3, facecolor='lightgrey', alpha=0.75)
    ax.set_xscale('log')
    ax.set_xlim(5e-3, 3)
    ax.tick_params(axis='both', labelsize=24)
    ax.set_xlabel(r'$Z \,/\, Z_\odot$', fontsize=32)
    ax.set_ylabel(r'$N_{\text{I}} \,/\, N_{\text{II}}$', fontsize=32)
    x_ticks = [0.01, 0.1, 1.0]
    ax.set_xticks(x_ticks, labels=[str(xt) for xt in x_ticks], fontsize=24)
    if text_y is not None:
        ax.text(0.896, text_y, "No\nCOSMIC\nModels", transform=ax.transAxes,
                color='crimson', fontsize=18, ha='left', va='top')

def plot_I_over_II_one_curve(data,
                             ax,
                             explosion_criteria = 'maltsev',
                             filter_list = [],
                             curve_label='',
                             zsun=0.02,
                             color='thistle',
                             linestyle='-',
                             linewidth=2):
    """
    Plot the ratio of Type I to Type II CCSNe as a function of metallicity for a single population. This function simply plots
    all that is labelled as so, and does not do any filtering or masking, this should be done outside the function.
    
    :param data: DataFrame containing the sn_info which is formatted with all the appropriate columns created by ccsnlab.data_loading.process_raw.
    This should already be filtered to only include the population you want to plot.
    :param metallicities: List of metallicities to plot.
    :param ax: Matplotlib axis object to plot on.
    :param curve_label: Label for the curve.
    :param zsun: Solar metallicity value to normalize the plot's COSMIC metallicity values.
    :param color: Color of the curve.
    :param linestyle: Line style of the curve.
    :param linewidth: Line width of the curve.
    """

    for (filter_col, filter_val) in filter_list:
        data = data[data[filter_col] == filter_val]
    
    ratios = []
    for Z in data.met_cosmic.unique():
        sub = data[data['met_cosmic'] == Z]
        sub = apply_explosion_criteria(sub, explosion_criteria=explosion_criteria)
        sn1_types = sub.sn_1_type
        sn2_types = sub.sn_2_type
        n_I = len(sn1_types[sn1_types == 'I']) + len(sn2_types[sn2_types == 'I'])
        n_II = len(sn1_types[sn1_types == 'II']) + len(sn2_types[sn2_types == 'II'])
        ratios.append(n_I / n_II if n_II else np.nan)

    dimensionless_z = [z / zsun for z in data.met_cosmic.unique()]
    line = ax.plot(dimensionless_z, ratios, label=curve_label,
                   color=color, linestyle=linestyle, linewidth=linewidth)
    return line[0]

def plot_binfrac_and_remnant(ax,
                             maltsev_data,
                             fryer_data,
                             zsun = 0.02,
                             solar = 9.05,
                             binfracs = ['0.0', '0.6', 'offner23'],
                             binfrac_colors = ['#d16ba5', '#c297ec', '#90c6ff', '#41f2ff'],
                             explosion_criteria = ['maltsev', ('sn_1_remnant_mass', 'sn_2_remnant_mass', 3)]):
    
    cmap = LinearSegmentedColormap.from_list("my_colormap", binfrac_colors)
    binfrac_colors = cmap(np.linspace(0, 1, len(binfracs)))

    # Plot each binary fraction with both explosion criteria
    filter_list = [('kickflag', 5), ('alpha', 1.0), ('qcflag', 5)]
    for binfrac, binfrac_color in zip(binfracs, binfrac_colors):
        for expl in explosion_criteria:
            if expl == 'maltsev':
                filter_list_expl = filter_list + [('remnantflag', 6),
                                                  ('rembar_massloss', 0.0),
                                                  ('maltsev_mode', 0),
                                                  ('maltsev_fallback', 0.5),
                                                  ('maltsev_pf_prob', 0.1)]
                data = maltsev_data
            elif isinstance(expl, tuple) and len(expl) == 3:
                filter_list_expl = filter_list + [('remnantflag', 4),
                                                  ('rembar_massloss', 0.5),
                                                  ('fryer_mass_limit', 0)]
                data = fryer_data
            else:
                raise ValueError(f"Unsupported explosion criteria: {expl}")
            
            filter_list_expl = filter_list_expl + [('binfrac', binfrac)]

            if expl == 'maltsev':
                if binfrac == '0.0':
                    curve_label = r'$f_{\rm bin} = 0\%$'
                elif binfrac == '0.6':
                    curve_label = r'$f_{\rm bin} = 60\%$'
                elif binfrac == 'offner23':
                    curve_label = r'Offner+23 $f_{\rm bin}(M_1)$'
                else:
                    curve_label = f'{binfrac}'

            plot_I_over_II_one_curve(data,
                                     ax=ax,
                                     explosion_criteria=expl,
                                     filter_list=filter_list_expl,
                                     curve_label=curve_label if expl == 'maltsev' else None,
                                     zsun=zsun,
                                     color=binfrac_color,
                                     linestyle='-' if expl == 'maltsev' else '--',
                                     linewidth=2)

    #add a legend for the COSMIC data
    ax.plot([], [], color='black', linestyle='--', label='Fryer+12 (NS only)')
    legend = ax.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(legend)

    #plot the LOSS data
    loss_data = plot_loss_data(ax, label_loss=True, solar=solar)
    ax.legend(handles=loss_data, loc='upper left', bbox_to_anchor=(0.0, 0.7),
              labels=[r'KK12, $Z$, (Ibc+IIb)/II', 'G17 Ibc/(II+IIb)'],
              fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

def plot_binfracs(ax,
                  data,
                  zsun = 0.02, 
                  solar = 9.05,
                  binfracs = ['0.0', '0.6', 'offner23'],
                  binfrac_colors = ['#d16ba5', '#c297ec', '#90c6ff', '#41f2ff'],
                  LOSS_legend=False):
    
    """
    Function which wraps plot_variations which plots the effect of changing the binary fraction.
    
    :param ax: Matplotlib axis object to plot on.
    :param data: DataFrame containing the sn_info which is formatted with all the appropriate columns created by ccsnlab.data_loading.process_raw.
    :param zsun: Solar metallicity value to normalize the plot's COSMIC metallicity values.
    :param solar: Solar metallicity value in 12+log(O/H) for the plot.
    :param binfracs: List of binary fractions to plot.
    :param labels: List of labels for the binary fractions.
    :param LOSS_legend: Boolean indicating whether to display the LOSS legend.

    """

    labels = []
    for binfrac in binfracs:
        if binfrac.isdigit() and float(binfrac) == 0.0:
            labels.append('0% (Singles Only)')
        elif binfrac.isdigit():
            labels.append(f'{int(float(binfrac)*100)}%')
        elif binfrac == 'offner23':
            labels.append('Offner+23')
        else:
            labels.append(f'{binfrac}')
    
    #One plot is the fiducial model with mass caps
    cmap = LinearSegmentedColormap.from_list("my_colormap", binfrac_colors)
    binfrac_colors = cmap(np.linspace(0, 1, len(binfracs)))
    
    # iteratively call plot_I_over_II_curve
    lines = []
    for binfrac, label, binfrac_color in zip(binfracs, labels, binfrac_colors):
        line = plot_I_over_II_one_curve(data,
                                        ax,
                                        explosion_criteria='maltsev',
                                        filter_list=[('binfrac', binfrac)],
                                        curve_label=label,
                                        zsun = zsun,
                                        color=binfrac_color,
                                        linestyle='-',
                                        linewidth=2)
        lines.append(line)

    #add a legend for the COSMIC data
    legend = ax.legend(handles=lines,
                       loc='upper left',
                       title='Binary Fraction',
                       fontsize=LEGEND_FONT_SIZE,
                       title_fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(legend)
    
    #add a legend for the loss data
    data = plot_loss_data(ax, label_loss=True, solar=solar)
    if LOSS_legend:
        ax.legend(handles=data,
                 loc='upper left',
                 bbox_to_anchor=(0.0, 0.7),
                 labels=[r'KK12, $Z$, (Ibc+IIb)/II', 'G17 Ibc/(II+IIb)'],
                 title='SN Survey Data',
                 fontsize=LEGEND_FONT_SIZE,
                 title_fontsize=LEGEND_FONT_SIZE)

def plot_remnants(ax,
                  maltsev_data,
                  fryer_data,
                  zsun = 0.02, 
                  solar = 9.05,
                  binfrac = 'offner23',
                  LOSS_legend=False,
                  incl_fryer_substring = False,
                  remnant_colors = ["#643995", "#41ffdf"],
                  rem_vars = [(6, 0.0, 0, 0.5, 0.0, None), # (remnantflag, rembar_massloss, maltsev_mode, maltsev_fallback, maltsev_pf_prob, fryer_mass_limit)
                              (6, 0.0, 0, 0.5, 0.1, None),
                              (6, 0.0, 0, 0.5, 1.0, None),
                              (4, 0.5, None, None, None, 0),
                              (4, 0.5, None, None, None, 1)],
                  fryer_vars = [('sn_1_remnant_mass', 'sn_2_remnant_mass', 3), # columns to cut on and upper mass threshold for the fryer models
                                ('sn_1_massc_co_layer_1', 'sn_2_massc_co_layer_2', 15)]):
    """
    Function which wraps plot_variations which plots the effect of changing the remnant mass cap.
    
    :param ax: Matplotlib axis object to plot on.
    :param maltsev_data: DataFrame containing the sn_info with maltsev variations which is formatted with all the appropriate columns created by ccsnlab.data_loading.process_raw.
    :param fryer_data: DataFrame containing the sn_info with fryer variations which is formatted with all the appropriate columns created by ccsnlab.data_loading.process_raw.
    :param zsun: Solar metallicity value to normalize the plot's COSMIC metallicity values.
    :param solar: Solar metallicity value in 12+log(O/H) for the plot.
    :param binfrac: The binary fraction for which all variations are plotted.
    :param LOSS_legend: Boolean indicating whether to display the LOSS legend.

    :param rem_vars: List of tuples of (remnantflag, rembar_massloss, maltsev_mode, maltsev_fallback, maltsev_pf_prob, fryer_mass_limit) to plot
    for the remnant variations. Currently, only models with malsev_mode=0 and maltsev_fallback=0.5 are formatted nicely.

    :param fryer_vars: List of tuples of (col_sn1, col_sn2, mass_threshold) to plot for the fryer variations,
    where col_sn1 and col_sn2 are the columns to apply the criteria to for SN 1 and SN 2 respectively, and
    mass_threshold is the upper mass threshold to apply for the explosion criteria.
    """

    #total number of variations is number with remnantflag 6 + number with remnantflag 4 * number of fryer variations
    n_variations = len([var for var in rem_vars if var[0] == 6]) + len([var for var in rem_vars if var[0] == 4]) * len(fryer_vars)
    cmap = LinearSegmentedColormap.from_list("my_colormap", remnant_colors)
    remnant_colors = cmap(np.linspace(0, 1, n_variations))

    #create all the labels and remnant variations
    labels = []
    all_variations = []
    all_explosions = []
    for var in rem_vars:

        #build up the variation(s) and explosion(s) entry(s)
        if var[0] == 6:
            all_variations.append(var)
            all_explosions.append('maltsev')
        else:
            for expl in fryer_vars:
                all_variations.append(var)
                all_explosions.append(expl)

        # now build up the label(s) for this variation
        remnantflag, rembar_massloss, maltsev_mode, maltsev_fallback, maltsev_pf_prob, fryer_mass_limit = var
        if remnantflag == 6 and rembar_massloss == 0.0 and maltsev_mode == 0 and maltsev_fallback == 0.5:
            prob_str = f"{int(maltsev_pf_prob*100)}%"
            label = 'Maltsev+25 ' + r'($p_{\rm BH}$' + f' = {prob_str})'
            labels.append(label)
        elif remnantflag == 6:
            label = f'Maltsev+25 maltsev_pf_prob={maltsev_pf_prob}, rembar_massloss={rembar_massloss}, maltsev_mode={maltsev_mode}, maltsev_fallback={maltsev_fallback}'
            labels.append(label)
        elif remnantflag == 4:
            for col_sn1, _, mass_threshold in fryer_vars:
                if col_sn1 == 'sn_1_remnant_mass':
                    if mass_threshold == 3:
                        label = f'Fryer+12 (NS Only'
                    else:
                        label = f'Fryer+12 ' + r'($M_{\rm rem}<$' + f'{mass_threshold}' + r'$\,M_\odot$'
                elif col_sn1 == 'sn_1_massc_co_layer_1':
                    label = f'Fryer+12 ' + r'($M_{\rm CO}<$' + f'{mass_threshold}' + r'$\,M_\odot$'
                else:
                    label = f'Fryer+12 ({col_sn1} < {mass_threshold}'

                if incl_fryer_substring:
                    sub_str = r'$M_{\rm CC,tot}$ Limited' if fryer_mass_limit == 0 else r'$M_{\rm CC,core}$ Limited'
                    label += f", {sub_str})"
                else:
                    label += ')'
                
                labels.append(label)

    # loop through all the variations and call plot_I_over_II_curve
    lines = []
    for variation, expl, color, label in zip(all_variations, all_explosions, remnant_colors, labels):
        if variation[0] == 6:
            filter_list = [('remnantflag', variation[0]),
                           ('rembar_massloss', variation[1]),
                           ('maltsev_mode', variation[2]),
                           ('maltsev_fallback', variation[3]),
                           ('maltsev_pf_prob', variation[4]),
                           ('binfrac', binfrac)]
        else:
            filter_list = [('remnantflag', variation[0]),
                           ('rembar_massloss', variation[1]),
                           ('binfrac', binfrac)]
        
        line = plot_I_over_II_one_curve(maltsev_data,
                                        ax,
                                        explosion_criteria=expl,
                                        filter_list=filter_list,
                                        curve_label=label,
                                        zsun=zsun,
                                        color=color,
                                        linestyle='-',
                                        linewidth=2)
    lines.append(line)

    #add a legend for the COSMIC data
    legend = ax.legend(handles=lines, 
                       loc='upper left',
                       title='Remnant Mass & Explosion',
                       fontsize=LEGEND_FONT_SIZE,
                       title_fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(legend)
    
    #add a legend for the loss data
    data = plot_loss_data(ax, label_loss=True, solar=solar)
    if LOSS_legend:
        ax.legend(handles=data,
                  loc='upper center',
                  bbox_to_anchor=(0.57, 0.86),
                  labels=[r'KK12, $Z$, (Ibc+IIb)/II', 'G17 Ibc/(II+IIb)'],
                  title='SN Survey Data',
                  fontsize=LEGEND_FONT_SIZE,
                  title_fontsize=LEGEND_FONT_SIZE)


def plot_kicks(ax,
               kick_data,
               kick_models = [(5, None), (1, 50.0), (1, 200.0)], #(kickflag, sigma)
               binfrac = 'offner23',
               zsun=0.02,
               solar=9.05,
               kick_colors = ["#d85ed8", "#ffaf25"],
               LOSS_legend=False, expl_fryer=False):
    
    """
    Function which wraps plot_variations which plots the effect of changing the kick model.
    
    :param ax: Matplotlib axis object to plot on.
    :param kick_data: DataFrame containing the sn_info which is formatted with all the appropriate columns created by ccsnlab.data_loading.process_raw.
    :param kick_models: List of tuples of (kickflag, sigma) to plot.
    :param binfrac: The binary fraction for which all variations are plotted.
    :param zsun: Solar metallicity value to normalize the plot's COSMIC metallicity values.
    :param solar: Solar metallicity value in 12+log(O/H) for the plot.
    :param LOSS_legend: Boolean indicating whether to display the LOSS legend.
    """

    kick_labels = []
    for kickflag, sigma in kick_models:
        if kickflag == 5:
            kick_labels.append('Disberg+2025')
        else:
            kick_labels.append(r'$\sigma=$' + f'{int(sigma)}' + r'$\,{\rm km}\,{\rm s}^{-1}$')
    
    #colors = ["#df3fdf", "#f17db3", "#ff5549", '#ffa325']
    
    cmap = LinearSegmentedColormap.from_list("my_colormap", kick_colors)
    colors = cmap(np.linspace(0, 1, len(kick_models)))

    expls = [('sn_1_massc_co_layer_1', 'sn_2_massc_co_layer_2', 15)] * len(kick_models) if expl_fryer else ['maltsev'] * len(kick_models)
    
    lines = []
    for (kickflag, sigma), label, color in zip(kick_models, kick_labels, colors):
        line = plot_I_over_II_one_curve(kick_data,
                                        ax,
                                        explosion_criteria=expls[0],
                                        filter_list=[('kickflag', kickflag),
                                                     ('binfrac', binfrac)],
                                        curve_label=label,
                                        zsun=zsun,
                                        color=color,
                                        linestyle='-',
                                        linewidth=2)
        lines.append(line)
    
    #add a legend for the COSMIC data
    legend = ax.legend(handles=lines, loc='upper left', title='Natal Kicks', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(legend)

    #add a legend for the loss data
    data = plot_loss_data(ax, label_loss=True, solar=solar)
    if LOSS_legend:
        ax.legend(handles=data, loc='upper center', bbox_to_anchor=(0.57, 0.86),
                  labels=[r'KK12, $Z$, (Ibc+IIb)/II', 'G17 Ibc/(II+IIb)'], title='SN Survey Data',
                  fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

def plot_cee(ax,
             alpha_data,
             klencki_data,
             zsun=0.02,
             binfrac='offner23',
             LOSS_legend=False,
             qcflags=[4,5],
             alphas=[0.3, 1.0, 5.0],
             cee_colors = ["#4382d4", '#11d6d6', '#00ff83', '#74d600', '#adff00'],
             klencki_colors = ["#ff7f00", "#FF4ea3"],
             solar=9.05):
    
    #goal: for each qcflag, plot each alpha and the klencki data
    cmap = LinearSegmentedColormap.from_list("my_colormap", cee_colors)
    n_variations = len(qcflags) * len(alphas)
    cee_colors = cmap(np.linspace(0, 1, n_variations))

    i = 0
    all_lines = []
    for qcflag, klencki_color in zip(qcflags, klencki_colors):
        #create first the variation labels
        labels = []
        for alpha in alphas:
            if len(qcflags) > 1:
                labels.append(r'$\alpha=$' + f'{alpha:.1f}, ' + f'qcflag={qcflag}')
            else:
                labels.append(r'$\alpha=$' + f'{alpha:.1f}')

        colors = cee_colors[i:i+len(alphas)]
        lines = plot_variations(zsun,
                                ax,
                                alpha_data,
                                variation='alpha',
                                variation_values=alphas,
                                variation_labels=labels,
                                variation_colors=colors,
                                linewidths=[2]*(len(alphas)+1),
                                binfrac=binfrac,
                                kicks=(5, None),
                                alpha=None,
                                qcflag=qcflag,
                                change_IIbs=False,
                                explosion_criteria=['maltsev'] * (len(alphas) + 1))
        i += len(labels)
        all_lines.extend(lines)

        #plot the klencki data        
        labels = [r'Klencki+2021 ($\alpha=0.7$), qcflag={qcflag}'] if len(qcflags) > 1 else [r'Klencki+2021' + '\n' + r'($\alpha=0.7$)']
        colors = [klencki_color]

        lines = plot_variations(zsun,
                                ax,
                                klencki_data,
                                variation='klencki',
                                variation_values=[None],
                                variation_labels=labels,
                                variation_colors=colors,
                                linewidths=[2]*len(labels),
                                binfrac=binfrac,
                                kicks=(5, None),
                                alpha=0.7,
                                qcflag=qcflag,
                                change_IIbs=False,
                                explosion_criteria=['maltsev'] * len(labels))
        i += len(labels)
        all_lines.extend(lines)

    #add a legend for the COSMIC data
    legend = ax.legend(handles=all_lines, loc='upper left', title='CEE', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(legend)

    #add a legend for the loss data
    data = plot_loss_data(ax, label_loss=True, solar=solar)
    if LOSS_legend:
        ax.legend(handles=data, loc='upper center', bbox_to_anchor=(0.57, 0.86),
                  labels=[r'KK12, $Z$, (Ibc+IIb)/II', 'G17 Ibc/(II+IIb)'], title='SN Survey Data',
                  fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

def make_paper_5_figure(fiducial_data,
                        remnant_data,
                        kick_data,
                        alpha_data,
                        klencki_data,
                        zsun=0.02,
                        solar=9.05,
                        binfrac='offner23',
                        plot_text_x = [0.3, 0.47, 0.33, 0.3],
                        plot_text_y = [0.95, 0.95, 0.95, 0.95],
                        no_cosmic_model_text = (0.95, 0.4, 18),
                        binfrac_colors = ['#d16ba5', '#c297ec', '#90c6ff', '#41f2ff'],
                        remnant_colors =  ["#f7c067", "#f26359"],
                        kick_colors = ["#ed4cf5", "#ffc258"],
                        cee_colors = ["#326DB9", '#17CCD6', "#04F655"],
                        klencki_color = "#bce784",
                        savepath='final_figs/figure_1.png'):
    
    if binfrac != 'offner23' and binfrac != '0.6':
        raise ValueError("Invalid binary fraction specified. Must be 'offner23' or '0.6'.")
    
    fig, axs = plt.subplots(2, 2, figsize=(24, 16))
    binfrac_ax, remnant_ax, kick_ax, cee_ax = axs.flatten()

    if fiducial_data is not None:
        plot_binfracs(binfrac_ax,
                    fiducial_data,
                    zsun = zsun, 
                    solar = solar,
                    binfracs = ['0.0', '0.6', 'offner23'],
                    binfrac_colors = binfrac_colors,
                    LOSS_legend=True)
        print("Plotted binfrac", flush=True)

    if remnant_data is not None:
        plot_remnants(remnant_ax,
                    remnant_data,
                    zsun = zsun, 
                    solar = solar,
                    binfrac = binfrac,
                    remnant_colors = remnant_colors,
                    LOSS_legend=False,
                        rem_vars = [(6, 0.0, 0, 0.5, 0.1, None), # (remnantflag, rembar_massloss, maltsev_mode, maltsev_fallback, maltsev_pf_prob, fryer_mass_limit)
                                    (4, 0.5, None, None, None, 0)],
                        fryer_vars = [('sn_1_remnant_mass', 'sn_2_remnant_mass', 3), # columns to cut on and upper mass threshold for the fryer models
                                    ('sn_1_massc_co_layer_1', 'sn_2_massc_co_layer_2', 15)])
        print("Plotted remnant", flush=True)

    if kick_data is not None:
        plot_kicks(kick_ax,
                kick_data,
                kick_models = [(5, None), (1, 50.0), (1, 200.0)], #(kickflag, sigma)
                binfrac = binfrac, expl_fryer=True,
                zsun=zsun,
                solar=solar,
                kick_colors = kick_colors,
                LOSS_legend=False)
        print("Plotted kicks", flush=True)

    if alpha_data is not None and klencki_data is not None:
        plot_cee(cee_ax,
                alpha_data,
                klencki_data,
                zsun=zsun,
                qcflags=[5],
                alphas=[5.0, 1.0, 0.3],
                binfrac=binfrac,
                solar=solar,
                cee_colors = cee_colors,
                klencki_colors = [klencki_color],
                LOSS_legend=False)
        print("Plotted cee", flush=True)

    binfrac_string = 'Binary Fraction: Offner+23' if binfrac == 'offner23' else f'Binary Fraction: 60%'
    kick_string = 'Natal Kicks: Disberg+2025'
    alpha_string = r'CEE: ' + r'$\alpha = 1.0$'
    remnant_string = 'Remnants: Maltsev+2025 ' + r'($p_{\rm BH}=10$%)'

    plot_text = [remnant_string + '\n' + kick_string + '\n' + alpha_string, #binfracs
                 binfrac_string + '\n' + kick_string +  '\n' + alpha_string, #varied remnant
                 binfrac_string + '\n' + remnant_string + '\n' + alpha_string, #varied kicks
                 binfrac_string + '\n' + remnant_string + '\n' + kick_string, #varied cee
                 ]

    for i, ax in enumerate(axs.flatten()):
        ax.axvspan(1.5, 3, facecolor='lightgrey', alpha=0.75)
        ax.set_xscale('log')
        ax.set_xlim(5e-3, 3)
        ax.tick_params(axis='both', labelsize=24)
        if i >= 2: ax.set_xlabel(r'$Z \,/\, Z_\odot$', fontsize=32)
        if i in [0,2]: ax.set_ylabel(r'$N_{\text{I}} \,/\, N_{\text{II}}$', fontsize=32)
        x_ticks = [0.01, 0.1, 1.0]
        ax.set_xticks(x_ticks, labels=[str(xt) for xt in x_ticks], fontsize=24)
        text = plot_text[i]
        #write it in the top middle with ax.text
        ax.text(plot_text_x[i], plot_text_y[i], text, transform=ax.transAxes, ha='left', va='top', fontsize=20)

    binfrac_ax.text(no_cosmic_model_text[0], no_cosmic_model_text[1], "No\nCOSMIC\nModels", transform=binfrac_ax.transAxes,
                     color='crimson', fontsize=no_cosmic_model_text[2], ha='center', va='top')


    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()
