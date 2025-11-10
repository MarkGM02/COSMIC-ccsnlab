import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def get_loss_data(solar=9.05):
    return pd.read_csv(f'z_{solar}_data.csv')


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


def plot_I_over_II_one_curve(
        data,
        metallicities,
        ax,
        sn1_col, sn2_col,
        rem1_col, rem2_col,
        curve_label,
        zsun=0.02,
        bh_cap=None,
        color='C0',
        linestyle='-',
        singles_only=False, linewidth=1,
    ):

    ratios = []
    for Z in metallicities:
        sub = data[data['met_cosmic'] == Z]

        # SN 1
        sn1_types = sub[sn1_col]
        singles_mask = sub['is_single'] if singles_only else np.ones(len(sub), dtype=bool)

        if bh_cap is not None:
            sn1_mask = (sub[rem1_col] <= bh_cap)
        else:
            sn1_mask = np.ones(len(sub), dtype=bool)
        
        sn1_types = sn1_types[(sn1_mask) & (singles_mask)]

        # SN 2
        sn2_types = pd.Series()

        if not singles_only:
            sn2_types = sub[sn2_col]
            if bh_cap is not None:
                sn2_mask = sub[rem2_col] <= bh_cap
            else:
                sn2_mask = np.ones(len(sub), dtype=bool)
            
            sn2_types = sn2_types[sn2_mask]

        n_I = len(sn1_types[sn1_types == 'I']) + len(sn2_types[sn2_types == 'I'])
        n_II = len(sn1_types[sn1_types == 'II']) + len(sn2_types[sn2_types == 'II'])

        ratios.append(n_I / n_II if n_II else np.nan)

    dimensionless_z = [z / zsun for z in metallicities]
    line = ax.plot(dimensionless_z, ratios, label=curve_label,
                   color=color, linestyle=linestyle, linewidth=linewidth)
    return line[0]

def plotting(zsun, ax, data, legend=True, plot_singles=False,
             singles_color=None, singles_label=None, plot_orig=False,
             orig_color=None, orig_label=None, bh_cap=None, sigma=265.0,
             alpha1=1.0, change_IIbs=False, linewidth=2, legend_cols=1,
             dot_binaries=False, legend_name = None):

    data = data[(data.sigma == sigma) & (data.alpha1 == alpha1)]

    if change_IIbs:
        # Change the SN types to include IIb in Type I
        IIb_sn1_mask = data['sn_1_subtype'] == 'IIb'
        IIb_sn2_mask = data['sn_2_subtype'] == 'IIb'

        # change the strict sn types to include IIb in Type I
        data.loc[IIb_sn1_mask, 'sn_1_type'] = 'I'
        data.loc[IIb_sn2_mask, 'sn_2_type'] = 'I'
    
    Zgrid = np.sort(data['met_cosmic'].unique())

    singles_sn_cols = ('sn_1_type', 'sn_2_type')
    singles_rem_cols = ('sn_1_remnant_mass', 'sn_2_remnant_mass')
    singles_singles_only = True
    singles_linestyle = '--'

    orig_sn_cols = ('sn_1_type', 'sn_2_type')
    orig_rem_cols = ('sn_1_remnant_mass', 'sn_2_remnant_mass')
    orig_singles_only = False
    orig_linestyle = ':' if dot_binaries else '-'

    lines = []

    for sn_cols, rem_cols, color, curve_label, singles_only, plot, linestyle in zip([singles_sn_cols,      orig_sn_cols],
                                                                                    [singles_rem_cols,     orig_rem_cols],
                                                                                    [singles_color,        orig_color],
                                                                                    [singles_label,        orig_label],
                                                                                    [singles_singles_only, orig_singles_only],
                                                                                    [plot_singles,         plot_orig],
                                                                                    [singles_linestyle,    orig_linestyle]
                                                                                    ):
        if not plot: continue
        line = plot_I_over_II_one_curve(data, Zgrid, ax, *sn_cols, *rem_cols, curve_label,
                                        color=color, linestyle=linestyle, zsun=zsun, linewidth=linewidth,
                                        singles_only=singles_only, bh_cap=bh_cap)
        lines.append(line)

    if legend:
        leg = ax.legend(loc='upper left', ncols=legend_cols, title = legend_name, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    return lines

def plot_mass_caps(ax, fiducial_data, zsun=0.02, solar=9.05):
    #One plot is the fiducial model with mass caps
    mass_caps = [3.0, 4.0, 5.0, 15.0, 100.0]
    mass_cap_colors = ['#d16ba5', '#c297ec', '#90c6ff', '#41f2ff']
    cmap = LinearSegmentedColormap.from_list("my_colormap", mass_cap_colors)
    mass_cap_colors = [cmap(i / len(mass_caps)) for i in range(len(mass_caps))]
    
    #add a black dashed line to show singles in the legend
    singles_line = ax.plot([], [], color='black', linestyle='--', linewidth=2, label='Single stars')

    lines = []
    for mass_cap, mass_cap_color in zip(mass_caps, mass_cap_colors):
        var_label =  r'$M_{\rm rem}$' + f' < {mass_cap}' +  r'$\, M_{\odot}$'
        if mass_cap == 3.0:
            var_label += ' (NSs)'
        elif mass_cap == 100.0:
            var_label = 'All remnants'
        curr = plotting(zsun, ax, fiducial_data, legend=False,
                        plot_singles=True,  singles_color=mass_cap_color, singles_label=None,
                        plot_orig=True,     orig_color=mass_cap_color,    orig_label=var_label,
                        bh_cap=mass_cap, sigma=265.0, alpha1=1.0, change_IIbs=False, legend_cols=None, legend_name=None)
        lines.append(curr[-1])
    
    lines.append(singles_line[0])
    
    #add a legend for the COSMIC data
    legend2 = ax.legend(handles=lines, loc='upper left', title='Remnant mass limit', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(legend2)
    
    #add a legend for the loss data
    data = plot_loss_data(ax, label_loss=True, solar=solar)
    ax.legend(handles=data, loc='upper center', bbox_to_anchor=(0.57, 0.86),
              labels=[r'KK12, $Z$, (Ibc+IIb)/II', 'G17 Ibc/(II+IIb)'], title='SN survey data',
              fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)


def plot_sigma_change(ax, sigma_data, zsun=0.02, bh_cap=3.0, solar=9.05):
    #the third plot is variations across sigma
    SIGMA_VALUES = [50.0, 73.9, 97.8, 121.7, 145.6, 169.4, 193.3, 217.2, 241.1, 265.0]

    colors = ["#df3fdf", "#f17db3", "#ff5549", '#ffa325']
    cmap = LinearSegmentedColormap.from_list("my_colormap", colors)
    norm = Normalize(vmin=min(SIGMA_VALUES), vmax=max(SIGMA_VALUES))

    for sigma in SIGMA_VALUES:
        var_label = f'{sigma}'
        var_color = cmap(norm(sigma))

        plotting(zsun, ax, sigma_data, legend=False,
                plot_singles=False,  singles_color=None, singles_label=None,
                plot_orig=True,     orig_color=var_color, orig_label=var_label,
                bh_cap=bh_cap, sigma=sigma, alpha1=1.0, change_IIbs=False, legend_cols=None)
                 
    plot_loss_data(ax, label_loss=False, solar=solar)
    ax.legend(loc='upper left', ncols=2, title = 'Natal kicks: ' + r'$\sigma$ (km$\,$s$^{-1}$)',
              fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

def plot_alpha_change(ax, alpha_data, zsun=0.02, qcflag=5, bh_cap=3.0, solar=9.05):
    # the fourth plot is the variations across alpha
    ALPHA1_VALUES = [0.05, 0.083, 0.139, 0.232, 0.387, 0.646, 1.077, 1.797, 2.997, 5.0]
    colors = ["#4382d4", '#11d6d6', '#00ff83', '#74d600', '#adff00']
    cmap = LinearSegmentedColormap.from_list("my_colormap", colors)
    norm = LogNorm(vmin=min(ALPHA1_VALUES), vmax=max(ALPHA1_VALUES))

    for alpha1 in ALPHA1_VALUES:
        var_label =  f'{alpha1}'
        var_color = cmap(norm(alpha1))
        
        plotting(zsun, ax, alpha_data, legend=None,
                plot_singles=False,  singles_color=None, singles_label=None,
                plot_orig=True,     orig_color=var_color,    orig_label=var_label,
                bh_cap=bh_cap, sigma=265.0, alpha1=alpha1, change_IIbs=False)

    plot_loss_data(ax, label_loss=False, solar=solar)
    ax.legend(loc='upper left', ncols=2, title = 'Common envelope: ' + r'$\alpha$',
              fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

def plot_Klencki(ax, pess, klencki_10, klencki_07, all_mergers, zsun=0.02, bh_cap=3.0, qcflag=5, solar=9.05):
    colors = ['#00c6d2', "#4595df", "#9321c9", '#e073b4']
    
    labels = [r'$\alpha=1.0$, Pessimistic CEE',
              r'Klencki+2021 $(\alpha=1.0)$',
              r'Klencki+2021 $(\alpha=0.7)$',
              r'$\alpha=0$, All CEE merge']
    
    linewidths = [2.5, 2.0, 1.5, 1.0]
    
    data_list = [pess, klencki_10, klencki_07, all_mergers]

    for var_color, var_label, linewidth, data in zip(colors, labels, linewidths, data_list):
        plotting(zsun, ax, data, legend=False,
                 plot_singles=False,  singles_color=None, singles_label=None,
                 plot_orig=True,  orig_color=var_color, orig_label=var_label,
                 bh_cap=bh_cap, sigma=265.0, alpha1=1.0, change_IIbs=False, linewidth=linewidth)

    plot_loss_data(ax, label_loss=False, solar=solar)
    ax.legend(loc='upper left', ncols=1, title = 'Common envelope: custom',
              fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

def make_paper_5_figure(fiducial_data, sigma_data, alpha_data, pess, klencki_10,
                        klencki_07, all_mergers, zsun=0.02, bh_cap=3.0, solar=9.05):
    
    fig, axs = plt.subplots(2, 2, figsize=(24, 16))
    mass_cap_ax, sigma_change_ax, alpha_change_ax5, klencki_merger_ax = axs.flatten()

    plot_mass_caps(mass_cap_ax, fiducial_data, zsun=zsun, solar=solar)
    print("Plotted mass caps", flush=True)

    plot_sigma_change(sigma_change_ax, sigma_data, zsun=zsun, bh_cap=bh_cap, solar=solar)
    print("Plotted sigma change", flush=True)

    plot_alpha_change(alpha_change_ax5, alpha_data, zsun=zsun, qcflag=5, bh_cap=bh_cap, solar=solar)
    print("Plotted alpha change", flush=True)

    plot_Klencki(klencki_merger_ax, pess, klencki_10, klencki_07,
                 all_mergers, zsun=zsun, bh_cap=bh_cap, solar=solar)
    print("Plotted Klencki", flush=True)

    sigma_string = r'$\sigma = 265.0\,$km$\,$s$^{-1}$'
    alpha_string = r'$\alpha = 1.0$'
    mrem_string = r'$M_{\rm Rem} < 3\, M_{\odot}$'

    plot_text = [sigma_string + '\n' + alpha_string, #varied mass caps
                 mrem_string +  '\n' + alpha_string, #varied sigma
                 mrem_string +  '\n' + sigma_string, #varied alpha
                 mrem_string +  '\n' + sigma_string  #varied custom cee
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
        ax.text(0.5, 0.95, text, transform=ax.transAxes, ha='left', va='top', fontsize=20)

    mass_cap_ax.text(0.896, 0.8, "No\nCOSMIC\nmodels", transform=mass_cap_ax.transAxes,
                     color='crimson', fontsize=18, ha='left', va='top')


    fig.tight_layout()
    fig.savefig('final_figs/figure_1.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    make_paper_5_figure(zsun=0.01)
