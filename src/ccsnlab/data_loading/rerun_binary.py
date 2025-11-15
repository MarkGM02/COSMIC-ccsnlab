import pandas as pd
import numpy as np
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve

def get_init_conds_and_bse_dict(processed_df, met_cosmic, sigma, alpha, qcflag):
    """Extract an init_cond dataframe and a bse_dict from a processed dataframe
       corresponding to a given metallicity and parameter set."""
    
    pop = processed_df[(processed_df.met_cosmic == met_cosmic) &
                       (processed_df.sigma == sigma) &
                       (processed_df.alpha1 == alpha)]
    
    m1 = pop.zams_mass_1.values
    kstar1 = np.where(m1 < 0.7, 0, 1)
    m2 = pop.zams_mass_2.values
    kstar2 = np.where(m2 < 0.7, 0, 1)
    porb = pop.zams_porb.values
    ecc = pop.zams_ecc.values
    sep = pop.zams_sep.values
    metallicity = pop.met_cosmic.values
    tphysf = 13700.0 * np.ones(len(m1))

    initC = InitialBinaryTable.InitialBinaries(
            m1 = m1,
            kstar1 = kstar1,
            m2 = m2,
            kstar2 = kstar2,
            porb = porb,
            ecc = ecc,
            sep = sep,
            metallicity = metallicity,
            tphysf = tphysf
        )
    
    initC = pd.DataFrame(initC)
    initC['bin_num'] = pop.bin_num.values

    zsun = 0.02
    BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1,
               'alpha1': alpha, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01,
               'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0,
               'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1,
               'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0,
               'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': sigma,
               'gamma': -2.0, 'pisn': 45.0,'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]],
               'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0,
               'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0,
               'sigmadiv' :-20.0, 'qcflag' : qcflag, 'eddlimflag' : 0,
               'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],
               'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0,
               'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1,
               'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 1,
               'zsun' : zsun, 'bhms_coll_flag' : 0, 'don_lim' : -1,
               'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim' : 1}

    return initC, BSEDict

def evolve_and_save(initCond, BSE_dict, time_window=0.001):
    """
    Evolve the given initial conditions using the provided BSE dictionary, returning:
      - bpp: full evolution history
      - bcm_filtered: bcm rows within `time_window` Myr *before* either SN1 or SN2,
                      plus the ZAMS (tphys = 0.0) and final (tphys = 13700.0) rows
      - initC, kick_info: as returned by Evolve.evolve

    time_window: in Myr (default 0.001 Myr = 1000 years)
    """
    np.random.seed(16)
    bpp, bcm, initC, kick_info = Evolve.evolve(
        initialbinarytable=initCond,
        BSEDict=BSE_dict,
        timestep_conditions=[
            ['kstar_1 >= 4', 'dtp=0.0'],
            ['kstar_2 >= 4', 'dtp=0.0']
        ]
    )

    # SN times
    sn_1_times = (
        bpp[bpp.evol_type == 15][['bin_num', 'tphys']]
        .rename(columns={'tphys': 'sn_1_time'})
    )
    sn_2_times = (
        bpp[bpp.evol_type == 16][['bin_num', 'tphys']]
        .rename(columns={'tphys': 'sn_2_time'})
    )
    sn_times = pd.merge(sn_1_times, sn_2_times, on='bin_num', how='outer')

    # Attach SN times to bcm
    bcm_with_sn_times = pd.merge(bcm, sn_times, on='bin_num', how='left')

    # keep rows s.t. sn_time - time_window <= tphys <= sn_time
    # AND keep ZAMS (tphys=0.0) and final timestep (tphys=13700.0)
    def filter_bcm(group):
        sn_1_time = None
        if 'sn_1_time' in group and not group['sn_1_time'].isna().all():
            sn_1_time = group['sn_1_time'].iloc[0]

        sn_2_time = None
        if 'sn_2_time' in group and not group['sn_2_time'].isna().all():
            sn_2_time = group['sn_2_time'].iloc[0]

        mask = pd.Series(False, index=group.index)

        # Window before SN1
        if sn_1_time is not None and not np.isnan(sn_1_time):
            mask |= (
                (group['tphys'] >= sn_1_time - time_window) &
                (group['tphys'] <= sn_1_time)
            )

        # Window before SN2
        if sn_2_time is not None and not np.isnan(sn_2_time):
            mask |= (
                (group['tphys'] >= sn_2_time - time_window) &
                (group['tphys'] <= sn_2_time)
            )

        # Always keep ZAMS and final timestep rows, strict equality
        mask |= (group['tphys'] == 0.0)
        mask |= (group['tphys'] == 13700.0)

        return group[mask]

    bcm_filtered = (
        bcm_with_sn_times
        .groupby('bin_num', group_keys=False)[bcm_with_sn_times.columns]
        .apply(filter_bcm)
        .reset_index(drop=True)
    )

    # Remove SN time columns from final bcm
    if 'sn_1_time' in bcm_filtered.columns:
        bcm_filtered = bcm_filtered.drop(columns=['sn_1_time'], errors='ignore')
    if 'sn_2_time' in bcm_filtered.columns:
        bcm_filtered = bcm_filtered.drop(columns=['sn_2_time'], errors='ignore')

    return bpp, bcm_filtered, initC, kick_info
