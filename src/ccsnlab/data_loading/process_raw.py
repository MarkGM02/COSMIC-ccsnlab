import warnings

from pyparsing import col
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from ccsnlab.sn_types import sn_types, sn_subtypes
from ccsnlab.data_loading.neutrino import neutrino_mass_loss, get_neutrino_mass_loss
from ccsnlab.data_loading.maltsev import get_masses

"""
Main module to create supernova information from COSMIC output for a single population.
"""

def create_sn_info(bpp, bcm, metallicity, BSEDICT, binfrac, sample_mass, singles_mass, n_stars, n_singles):
    """
    Create a dataframe with one row per binary system containing supernova and evolutionary information. All parameters besides
    the bpp and bcm are strictly around for logging. The core functionality works with dummy parameters everywhere else, however
    to plot variations and analyze the impact of different parameters, the full set makes everything a lot easier, and is assumed in
    the plotting functions.

    Parameters
    ----------
    bpp : pd.DataFrame
        bpp from COSMIC output
    bcm : pd.DataFrame
        bcm from COSMIC output. Must include dense output in the final kyr before supernova,
        and the row at the final timestep to faithfully classify all supernovae.
    metallicity : float
        Metallicity of the population
    BSEDICT : dict
        Dictionary containing binary star evolution parameters used for the simulation, accuracy is important for record keeping, and 
        essential for the handling of the ejecta profile, which is dependent on the remnantflag, and corresponding parameters.
    binfrac : float
        Binary fraction of the population
    sample_mass : float
        Total sampled mass
    singles_mass : float
        Mass in single stars
    n_stars : int
        Number of stars in the population
    n_singles : int
        Number of single stars in the population

    Returns
    -------
    pd.DataFrame
        Dataframe with supernova and evolutionary information for each binary system. Columns include:

        **System Properties:**
        - bin_num : int
            Unique (within population) ID
        - SN_1, SN_2 : int
            Supernova type codes for primary and secondary (0=none, 1=FeCCSN 2=ECSN, see COSMIC docs)
        - merger_type : int
            String of merged kstars if merger occured, see COSMIC docs
        - is_single : bool
            Whether the system was formed as a single star

        **ZAMS (Zero Age Main Sequence) Properties:**
        - zams_mass_1, zams_mass_2 : float
            Initial masses of primary and secondary
        - zams_porb, zams_ecc, zams_sep : float
            Initial orbital period, eccentricity, and separation

        **Supernova 1 (Primary) Properties:**
        - sn_1_time : float
            Time of primary supernova (Myr)
        - sn_1_mass_1, sn_1_mass_2 : float
            Masses of primary and secondary at SN1
        - sn_1_massc_he_layer_1, sn_1_massc_co_layer_1 : float
            He and CO core masses at SN1
        - sn_1_menv_1 : float
            Envelope mass (convective only, due to COSMIC logging) of primary at SN1
        - sn_1_kstar_1, sn_1_kstar_2 : int
            kstar types of primary and secondary at SN1, see COSMIC docs
        - sn_1_porb, sn_1_ecc, sn_1_sep : float
            Orbital period, eccentricity, and separation at SN1
        - sn_1_remnant_mass : float
            Mass of remnant from primary SN
        - sn_1_m_ejecta : float
            Ejected mass from primary SN
        - sn_1_max_loss_rate : float
            Maximum mass loss rate in 1 kyr before SN1
        - sn_1_donor_kstars, sn_1_accretor_kstars : str
            string of kstar types when primary was donor/accretor. Not necesarily pre SN1 (trivially before SN1 if kstar < 13). Example: "1-2-5"
        - sn_1_interactions : str
            Description of mass transfer interactions before SN1. Either None, 'CEE' (if any CEE), or 'RLOF' (rlof occurs, NO CEE)
        - sn_1_last_donor : str
            Which star was last to donate mass before SN1 (primary or secondary)
        - sn_1_merger : bool
            Whether primary underwent merger before SN1
        - sn_1_ns : bool
            Whether primary remnant is a neutron star

        **Supernova 2 (Secondary) Properties:**
        - Similar structure to SN 1 columns, but for the secondary star

        **Ejecta Composition:**
        - sn_1_m_H_tot, sn_1_m_He_tot, sn_1_m_CO_tot : float
            Total hydrogen, helium, and CO masses in ejecta from SN1
        - Similar columns for SN2

    """
    
    bin_nums = bcm['bin_num'].unique()
    bpp = bpp[bpp['bin_num'].isin(bin_nums)] #filter potential outliers not in bcm

    #bcm manipulation first - get the merger SN_1 and SN_2, as well as the merger type
    max_time = bcm.tphys.max()
    bcm_final_rows = bcm[bcm['tphys'] == max_time][['bin_num', 'SN_1', 'SN_2', 'merger_type']]
    bcm_final_rows = bcm_final_rows.drop_duplicates(subset='bin_num', keep='last')

    #now bpp manipulation - get information at each supernova
    primary_sne =   bpp[bpp['evol_type'] == 15]
    secondary_sne = bpp[bpp['evol_type'] == 16]
    
    primary_sne =     primary_sne[['bin_num', 'tphys', 'mass_1', 'mass_2', 'massc_he_layer_1', 'massc_co_layer_1', 'menv_1', 'kstar_1', 'kstar_2', 'porb', 'ecc', 'sep']]
    secondary_sne = secondary_sne[['bin_num', 'tphys', 'mass_1', 'mass_2', 'massc_he_layer_2', 'massc_co_layer_2', 'menv_2', 'kstar_1', 'kstar_2', 'porb', 'ecc', 'sep']]

    #rename columns to start with 'sn_1' or 'sn_2'
    primary_sne.columns = ['bin_num', 'sn_1_time', 'sn_1_mass_1', 'sn_1_mass_2', 'sn_1_massc_he_layer_1', 'sn_1_massc_co_layer_1', 'sn_1_menv_1', 'sn_1_kstar_1', 'sn_1_kstar_2', 'sn_1_porb', 'sn_1_ecc', 'sn_1_sep']
    secondary_sne.columns = ['bin_num', 'sn_2_time', 'sn_2_mass_1', 'sn_2_mass_2', 'sn_2_massc_he_layer_2', 'sn_2_massc_co_layer_2', 'sn_2_menv_2', 'sn_2_kstar_1', 'sn_2_kstar_2', 'sn_2_porb', 'sn_2_ecc', 'sn_2_sep']

    zams = bpp[bpp['tphys'] == 0]
    zams = zams[['bin_num', 'mass_1', 'mass_2', 'porb', 'ecc', 'sep']]
    zams.columns = ['bin_num', 'zams_mass_1', 'zams_mass_2', 'zams_porb', 'zams_ecc', 'zams_sep']

    #want to get the maximum mass loss in the previous kyr before the SN
    times = primary_sne[['bin_num', 'sn_1_time']].merge(secondary_sne[['bin_num', 'sn_2_time']], on='bin_num', how='outer')
    bcm_with_times = bcm.merge(times, on='bin_num', how='left')
    kyr = 1e-3 # 1000 years in Myr
    mask1 = (bcm_with_times['tphys'] >= bcm_with_times['sn_1_time'] - kyr) & (bcm_with_times['tphys'] <=  bcm_with_times['sn_1_time'])
    mask2 = (bcm_with_times['tphys'] >= bcm_with_times['sn_2_time'] - kyr) & (bcm_with_times['tphys'] <=  bcm_with_times['sn_2_time'])

    sn1_max_loss_rate = bcm_with_times.loc[mask1].groupby('bin_num')['deltam_1'].min().rename('sn_1_max_loss_rate')
    sn2_max_loss_rate = bcm_with_times.loc[mask2].groupby('bin_num')['deltam_2'].min().rename('sn_2_max_loss_rate')

    #get the remnant mass corresponding to each supernova
    bpp_with_times = bpp.merge(times, on='bin_num', how='left').sort_values(['bin_num', 'tphys'])
    remnant_kstars = {13, 14, 15}

    mask1 = (bpp_with_times['kstar_1'].isin(remnant_kstars)) & (bpp_with_times['tphys'] >= bpp_with_times['sn_1_time'])
    mask2 = (bpp_with_times['kstar_2'].isin(remnant_kstars)) & (bpp_with_times['tphys'] >= bpp_with_times['sn_2_time'])
    
    rem1_rows = bpp_with_times.loc[mask1, ['bin_num', 'mass_1']].drop_duplicates('bin_num', keep='first').rename(columns={'mass_1': 'sn_1_remnant_mass'})
    rem2_rows = bpp_with_times.loc[mask2, ['bin_num', 'mass_2']].drop_duplicates('bin_num', keep='first').rename(columns={'mass_2': 'sn_2_remnant_mass'})

    #get the list of kstars at which the star of interest donated mass (i.e, RRLO_* > 1)
    donor_kstars_1 = bpp[bpp['RRLO_1'] > 1][['bin_num', 'kstar_1']]
    donor_kstars_2 = bpp[bpp['RRLO_2'] > 1][['bin_num', 'kstar_2']]
    #collapse these into a string i.e., "1-3-5" if the primary was a donor at kstar 1, 3, and 5
    donor_kstars_1 = donor_kstars_1.groupby('bin_num')['kstar_1'].apply(lambda x: '-'.join(map(str, sorted(x.unique())))).rename('sn_1_donor_kstars')
    donor_kstars_2 = donor_kstars_2.groupby('bin_num')['kstar_2'].apply(lambda x: '-'.join(map(str, sorted(x.unique())))).rename('sn_2_donor_kstars')

    #get a list of accretor kstars (i.e. star 1 accretes when RRLO_2 > 1)
    accretor_kstars_1 = bpp[bpp['RRLO_2'] > 1][['bin_num', 'kstar_1']]
    accretor_kstars_2 = bpp[bpp['RRLO_1'] > 1][['bin_num', 'kstar_2']]
    #collapse these into a string i.e., "1-3-5" if the primary was an accretor at kstar 1, 3, and 5
    accretor_kstars_1 = accretor_kstars_1.groupby('bin_num')['kstar_1'].apply(lambda x: '-'.join(map(str, sorted(x.unique())))).rename('sn_1_accretor_kstars')
    accretor_kstars_2 = accretor_kstars_2.groupby('bin_num')['kstar_2'].apply(lambda x: '-'.join(map(str, sorted(x.unique())))).rename('sn_2_accretor_kstars')

    def create_interaction_df(bpp):
        #the bpp has the standard columns, plus sn_1_time and sn_2_time
        
        #initialize a result dataframe with bin_num, sn_1_flag, sn_2_flag, sn_1_last_donor, sn_2_last_donor
        bin_nums = bpp.bin_num.unique()
        result = pd.DataFrame({'bin_num' : bin_nums,
                            'sn_1_interactions' : ['None' for _ in range(len(bin_nums))],
                            'sn_2_interactions' : ['None' for _ in range(len(bin_nums))],
                            'sn_1_last_donor' : ['None' for _ in range(len(bin_nums))],
                            'sn_2_last_donor' : ['None' for _ in range(len(bin_nums))],
                            'sn_1_merger' : [False for _ in range(len(bin_nums))],
                            'sn_2_merger' : [False for _ in range(len(bin_nums))]})

        for rrlo, flag, last_donor, kstar, (progenitor, companion), merger, companion_m in zip(['RRLO_1', 'RRLO_2'],
                                                                                               ['sn_1_interactions', 'sn_2_interactions'],
                                                                                               ['sn_1_last_donor', 'sn_2_last_donor'],
                                                                                               ['kstar_1', 'kstar_2'],
                                                                                               [('primary', 'secondary'),
                                                                                                ('secondary', 'primary')],
                                                                                               ['sn_1_merger', 'sn_2_merger'],
                                                                                               ['mass_2', 'mass_1']):
            before_sn = bpp[bpp[kstar] < 13]
            
            #mark mergers
            binaries = bpp[(bpp.tphys == 0.0) & (bpp.mass_2 > 0.0)].bin_num.unique()
            mergers = before_sn[(before_sn[companion_m] == 0.0) & (before_sn.bin_num.isin(binaries))].bin_num.unique()
            result.loc[result.bin_num.isin(mergers), merger] = True

            interactions = before_sn[before_sn.evol_type.isin([3, 7])]
            #gather stars with any CEE
            cees = interactions[interactions.evol_type == 7].bin_num.unique()
            #the only stable group is then those that interacted but did not have a CEE
            rlofs = interactions[~interactions.bin_num.isin(cees)].bin_num.unique()
            #write to result
            result.loc[result.bin_num.isin(cees), flag] = 'CEE'
            result.loc[result.bin_num.isin(rlofs), flag] = 'RLOF'
            #grab the last interaction for each star
            last_interactions = interactions.sort_values('tphys').groupby('bin_num').last()
            #bin_nums are the index here, make them a column
            last_interactions = last_interactions.reset_index()
            #the star is a donor if its rrlo > 1 in this row
            donors = last_interactions[last_interactions[rrlo] > 1.0].bin_num.unique()
            #accretors are the stars that interacted but were not donors in the last interaction
            accretors = interactions[~interactions.bin_num.isin(donors)].bin_num.unique()
            #write to result
            result.loc[result.bin_num.isin(donors), last_donor] = progenitor
            result.loc[result.bin_num.isin(accretors), last_donor] = companion

        return result

    interaction_df = create_interaction_df(bpp)

    #if the randomseed is in the bpp, then grab it for each bin_num, otherwise create a dummy column of -1
    if 'randomseed' in bpp.columns:
        random_seeds = bpp[['bin_num', 'randomseed']].drop_duplicates('bin_num', keep='first')
    else:
        random_seeds = pd.DataFrame({'bin_num': bpp['bin_num'].unique(), 'randomseed': -1})

    #combine all via bin_num
    result = pd.merge(bcm_final_rows, zams, on='bin_num', how='left')
    result = pd.merge(result, primary_sne, on='bin_num', how='left')
    result = pd.merge(result, secondary_sne, on='bin_num', how='left')
    result = pd.merge(result, sn1_max_loss_rate, on='bin_num', how='left')
    result = pd.merge(result, sn2_max_loss_rate, on='bin_num', how='left')
    result = pd.merge(result, rem1_rows, on='bin_num', how='left')
    result = pd.merge(result, rem2_rows, on='bin_num', how='left')
    result = pd.merge(result, donor_kstars_1, on='bin_num', how='left')
    result = pd.merge(result, donor_kstars_2, on='bin_num', how='left')
    result = pd.merge(result, accretor_kstars_1, on='bin_num', how='left')
    result = pd.merge(result, accretor_kstars_2, on='bin_num', how='left')
    result = pd.merge(result, interaction_df, on='bin_num', how='left')
    result = pd.merge(result, random_seeds, on='bin_num', how='left')

    #fix the SN_1 and SN_2 where there is no actual SN, these come from mergers which get called a SN in COSMIC in some cases
    no_sn1_mask = np.isnan(result['sn_1_time'])
    no_sn2_mask = np.isnan(result['sn_2_time'])
    result.loc[no_sn1_mask, 'SN_1'] = 0
    result.loc[no_sn2_mask, 'SN_2'] = 0

    # COSMIC housekeeping. Identify ECSNe incorrectly labelled, these produce hardcoded 1.38 Msun NSs, which are reduced according to rembar_massloss.
    min_ns_mass = neutrino_mass_loss(1.38, rembar_massloss=BSEDICT['rembar_massloss'])
    sn_1_ecsn_mask = result['sn_1_remnant_mass'] <= min_ns_mass
    sn_2_ecsn_mask = result['sn_2_remnant_mass'] <= min_ns_mass
    result['SN_1'] = np.where(sn_1_ecsn_mask, 2, result['SN_1'])
    result['SN_2'] = np.where(sn_2_ecsn_mask, 2, result['SN_2']) 

    #identify each system as a binary or single
    singles = bpp[(bpp['tphys'] == 0.0) & (bpp['mass_2'] == 0.0)]['bin_num'].unique() #grab bin_nums of singles
    result['is_single'] = result['bin_num'].isin(singles)

    #figure out if the supernova created a neutron star
    ns1_bin_nums = bpp[bpp['kstar_1'] == 13]['bin_num'].unique()
    ns2_bin_nums = bpp[bpp['kstar_2'] == 13]['bin_num'].unique()
    result['sn_1_ns'] = result['bin_num'].isin(ns1_bin_nums)
    result['sn_2_ns'] = result['bin_num'].isin(ns2_bin_nums)

    # Now begins the section where we calculate the ejecta profile. This requires us to know the details of the remnant prescription. As
    # of now, we only support the delayed Fryer (remnantflag=4), and the Maltsev (remnantflag=6) prescriptions, and all else could be added later.
    # We assume now that the maltsev prescription is used strictly with rembar_massloss = 0, so that 
    for sn in (1, 2):
        if BSEDICT['remnantflag'] >= 1:
            #apply this to the dataframe
            neutrino_loss = result.apply(lambda row: get_neutrino_mass_loss(row[f'sn_{sn}_remnant_mass'], rembar_massloss=BSEDICT['rembar_massloss']), axis=1)
        else:
            #no mass loss occurs
            neutrino_loss = np.ones(len(result)) * 0

        # the ejecta mass is total mass - remnant mass - neutrino mass loss.
        m_ejecta = (result[f'sn_{sn}_mass_{sn}'] - result[f'sn_{sn}_remnant_mass'] - neutrino_loss).clip(lower=0)

        #write in the ejecta mass and the neutrino mass loss
        result[f'sn_{sn}_m_ejecta'] = m_ejecta
        result[f'sn_{sn}_m_neutrino_loss'] = neutrino_loss

    #now we determine H ejecta by assuming that this is the minimum of the combined envelope mass and the total ejecta mass
    for sn in (1, 2):
        mcore_CO = result[f'sn_{sn}_massc_co_layer_{sn}']                                   # CO core mass only
        mcore_He = result[f'sn_{sn}_massc_he_layer_{sn}']                                   # He core mass only
        mass = result[f'sn_{sn}_mass_{sn}']                                                 # pre-SN total mass
        menv_convective = result[f'sn_{sn}_menv_{sn}']                                      # what cosmic calls the "menv"
        menv_radiative = (mass - mcore_CO - mcore_He - menv_convective).clip(lower=0)       # convective envelope mass

        stripped = result[f'sn_{sn}_kstar_{sn}'] >= 7  # boolean mask for stripped stars

        #for stripped stars, the hydrogen mass is 0. For non-stripped stars, the hydrogen mass is total envelope mass.
        m_Hydrogen_tot = np.where(stripped, 0.0, menv_convective + menv_radiative)

        #for stripped stars, the helium mass is the total envelope mass + the he core. For non-stripped stars, the helium mass is the he core only
        m_Helium_tot = np.where(stripped, menv_convective + menv_radiative + mcore_He, mcore_He)

        #for all stars, the CO mass is the CO core mass
        m_CO_tot = mcore_CO

        #now we can calculate the ejecta masses. We assume that hydrogen is ejected first, then helium, then CO core
        ejecta_mass = result[f'sn_{sn}_m_ejecta']

        m_Hydrogen_ejecta = np.minimum(ejecta_mass, m_Hydrogen_tot)
        m_Helium_ejecta = np.minimum(ejecta_mass - m_Hydrogen_ejecta, m_Helium_tot)
        m_CO_ejecta = np.minimum(ejecta_mass - m_Hydrogen_ejecta - m_Helium_ejecta, m_CO_tot)

        #write back the ejecta masses
        result[f'sn_{sn}_m_h_ejecta'] = m_Hydrogen_ejecta
        result[f'sn_{sn}_m_he_ejecta'] = m_Helium_ejecta
        result[f'sn_{sn}_m_co_ejecta'] = m_CO_ejecta

    # Call our sn_types and sn_subtypes functions to classify the SNe
    result = sn_types(result)
    result = sn_subtypes(result)

    # Last SN thing: If this population has remnantflag = 6 (Maltsev), we should clarify what region this falls into, such that succesful
    # CCSNe can easily be identified by masking those which are not direct collapses.

    for sn in (1, 2):
        # if the string contains 0 or 1, it is a case a
        col = result[f'sn_{sn}_donor_kstars'].astype(str)
        case_a_mask = col.str.contains('0|1', regex=True)
        # if the string contains 2, 3, or 4, it is a case b
        case_b_mask = col.str.contains('2|3|4', regex=True)
        # if the string contains 5 or 6, it is a case c
        case_c_mask = col.str.contains('5|6', regex=True)

        # we take the first case of mass transfer that occurs, so we prioritize case a over b over c. If no mass transfer occurs, we label this as "S" for single.
        result[f'sn_{sn}_maltsev_case'] = np.where(case_a_mask, 'A', np.where(case_b_mask, 'B', np.where(case_c_mask, 'C', 'S')))

        # lastly, if kstar at core collapse is [7,8,9] and there was no mas transferm we call this case b
        stripped_mask = result[f'sn_{sn}_kstar_{sn}'].isin([7, 8, 9])
        no_mt_mask = ~ (case_a_mask | case_b_mask | case_c_mask)
        result[f'sn_{sn}_maltsev_case'] = np.where(stripped_mask & no_mt_mask, 'B', result[f'sn_{sn}_maltsev_case'])

        # great, now for the sake of computation, lets grab each of the massses a single time
        masses_a = get_masses(metallicity / 0.02, 'A', maltsev_mode=BSEDICT['maltsev_mode'])
        masses_b = get_masses(metallicity / 0.02, 'B', maltsev_mode=BSEDICT['maltsev_mode'])
        masses_c = get_masses(metallicity / 0.02, 'C', maltsev_mode=BSEDICT['maltsev_mode'])
        masses_s = get_masses(metallicity / 0.02, 'S', maltsev_mode=BSEDICT['maltsev_mode'])

        # now we can map by case, and then we can assign the region as "NS", "Direct BH", or "NS/BH"
        def assign_region(row):
            case = row[f'sn_{sn}_maltsev_case']
            mass = row[f'sn_{sn}_massc_co_layer_{sn}']

            if case == 'A':
                m1, m2, m3 = masses_a
            elif case == 'B':
                m1, m2, m3 = masses_b
            elif case == 'C':
                m1, m2, m3 = masses_c
            elif case == 'S':
                m1, m2, m3 = masses_s
            else:
                raise ValueError("Invalid case")
            
            if mass < m1:
                return 'NS'
            elif mass < m2:
                return 'Direct BH'
            elif mass < m3:
                return 'NS/BH'
            else:
                return 'Direct BH'

        result[f'sn_{sn}_maltsev_region'] = result.apply(assign_region, axis=1) 

    #add the total sample mass, singles mass, and n_stars to each
    result['sample_mass'] = sample_mass
    result['singles_mass'] = singles_mass
    result['n_stars'] = n_stars
    result['n_singles'] = n_singles
    result['binfrac'] = binfrac

    #add all the relevant varied evolution/sampling parameters for record keeping:
    result['remnantflag'] = BSEDICT['remnantflag']
    result['maltsev_mode'] = BSEDICT['maltsev_mode']
    result['maltsev_fallback'] = BSEDICT['maltsev_fallback']
    result['maltsev_pf_prob'] = BSEDICT['maltsev_pf_prob']
    result['rembar_massloss'] = BSEDICT['rembar_massloss']
    result['fryer_mass_limit'] = BSEDICT['fryer_mass_limit']

    result['kickflag'] = BSEDICT['kickflag']
    result['sigma'] = BSEDICT['sigma']
    result['alpha'] = BSEDICT['alpha1']
    result['qcflag'] = BSEDICT['qcflag']
    result['met_cosmic'] = metallicity

    # ---------------------------------
    # FORCE STABLE OUTPUT SCHEMA
    # ---------------------------------

    EXPECTED_SCHEMA = {
        # Integers
        'sn_1_kstar_1': 'int64',
        'sn_1_kstar_2': 'int64',

        # Strings
        'sn_1_donor_kstars': 'object',
        'sn_2_donor_kstars': 'object',
        'sn_1_accretor_kstars': 'object',
        'sn_2_accretor_kstars': 'object',
    }

    for col, dtype in EXPECTED_SCHEMA.items():
        if col in result.columns:
            if dtype == 'int64':
                result[col] = (
                    pd.to_numeric(result[col], errors='coerce')
                    .fillna(-1)
                    .astype('int64')
                )
            elif dtype == 'object':
                result[col] = result[col].astype(str)


    return result
