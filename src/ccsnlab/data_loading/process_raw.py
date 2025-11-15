import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from delayed_fryer import get_CO_cores
from sn_types import sn_types, sn_subtypes

def create_sn_info(bpp, bcm, metallicity, sigma, alpha1, sample_mass, singles_mass):
    
    bin_nums = bcm['bin_num'].unique()
    bpp = bpp[bpp['bin_num'].isin(bin_nums)] #filter couple weirdos not in bcm

    #bcm maniupulation first - get the merger SN_1 and SN_2, as well as the merger type
    bcm_final_rows = bcm[bcm['tphys'] == 13700.0]
    bcm_final_rows = bcm_final_rows[~bcm_final_rows.duplicated(subset='bin_num')] #drop duplicates created by mistake
    bcm_final_rows = bcm_final_rows[['bin_num', 'SN_1', 'SN_2', 'merger_type']]

    #now bpp manipulation - get the info at supernovae and zams
    primary_sne =   bpp[bpp['evol_type'] == 15]
    secondary_sne = bpp[bpp['evol_type'] == 16]
    
    primary_sne =     primary_sne[['bin_num', 'tphys', 'mass_1', 'mass_2', 'massc_1', 'menv_1', 'kstar_1', 'kstar_2', 'porb', 'ecc', 'sep']]
    secondary_sne = secondary_sne[['bin_num', 'tphys', 'mass_1', 'mass_2', 'massc_2', 'menv_2', 'kstar_1', 'kstar_2', 'porb', 'ecc', 'sep']]

    #rename columns to start with 'sn_1' or 'sn_2'
    primary_sne.columns = ['bin_num', 'sn_1_time', 'sn_1_mass_1', 'sn_1_mass_2', 'sn_1_massc_1', 'sn_1_menv_1', 'sn_1_kstar_1', 'sn_1_kstar_2', 'sn_1_porb', 'sn_1_ecc', 'sn_1_sep']
    secondary_sne.columns = ['bin_num', 'sn_2_time', 'sn_2_mass_1', 'sn_2_mass_2', 'sn_2_massc_2', 'sn_2_menv_2', 'sn_2_kstar_1', 'sn_2_kstar_2', 'sn_2_porb', 'sn_2_ecc', 'sn_2_sep']

    zams = bpp[bpp['tphys'] == 0]
    zams = zams[['bin_num', 'mass_1', 'mass_2', 'porb', 'ecc', 'sep']]
    zams.columns = ['bin_num', 'zams_mass_1', 'zams_mass_2', 'zams_porb', 'zams_ecc', 'zams_sep']

    #want to get the maximum mass loss in the previous 1000 yrs before the SN
    times = primary_sne[['bin_num', 'sn_1_time']].merge(secondary_sne[['bin_num', 'sn_2_time']], on='bin_num', how='outer')
    bcm_with_times = bcm.merge(times, on='bin_num', how='left')

    window = 1e-3 # 1000 years in Myr
    mask1 = (bcm_with_times['tphys'] >= bcm_with_times['sn_1_time'] - window) & (bcm_with_times['tphys'] <=  bcm_with_times['sn_1_time'])
    mask2 = (bcm_with_times['tphys'] >= bcm_with_times['sn_2_time'] - window) & (bcm_with_times['tphys'] <=  bcm_with_times['sn_2_time'])

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
    result = pd.merge(result, interaction_df, on='bin_num', how='left')

    #fix the SN_1 and SN_2 where there is no actual SN, these come from mergers which get called a SN in COSMIC
    no_sn1_mask = np.isnan(result['sn_1_time'])
    no_sn2_mask = np.isnan(result['sn_2_time'])
    result.loc[no_sn1_mask, 'SN_1'] = 0
    result.loc[no_sn2_mask, 'SN_2'] = 0

    #more COSMIC housekeeping. Accretion induced collapse of ONe wds can create an ultra low mass NS. We want to flag these as ECSN, since COSMIC sometimes calls then non electron capture.
    #Additionally, some ECSN are not labelled as such in the bcm, but we can tell they are ECSN from the remnant mass they produce
    ns_mass_from_ecsn_in_the_delayed_fryer_prescription = 6.6666667*(np.sqrt(1.0 + 0.3* 1.38) - 1.0)
    minimum_ns_mass = 1.242
    sn_1_ecsn_mask = (result['sn_1_remnant_mass'] == minimum_ns_mass) | (result['sn_1_remnant_mass'] == ns_mass_from_ecsn_in_the_delayed_fryer_prescription)
    sn_2_ecsn_mask = (result['sn_2_remnant_mass'] == minimum_ns_mass) | (result['sn_2_remnant_mass'] == ns_mass_from_ecsn_in_the_delayed_fryer_prescription)
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

    #using the fryer mass precription -- walk back to the CO core mass
    result = get_CO_cores(result)

    #start by adding in total ejecta mass
    for sn in (1,2):
        # ejecta mass = total mass - remnant mass, limited to 0
        result[f'sn_{sn}_m_ejecta'] = (result[f'sn_{sn}_mass_{sn}'] - result[f'sn_{sn}_remnant_mass']).clip(lower=0)

    #now we determine H ejecta by assuming that this is the minimum of the combined envelope mass and the total ejecta mass
    for sn in (1, 2):
        mcore_tot  = result[f'sn_{sn}_massc_{sn}']          # CO core + He Core
        mcore_CO = result[f'sn_{sn}_CO_core_mass']          # CO core mass only
        mcore_He = (mcore_tot - mcore_CO).clip(lower=0)     # He core mass only

        mass = result[f'sn_{sn}_mass_{sn}']                                       # pre-SN total mass
        menv_convective = result[f'sn_{sn}_menv_{sn}']                            # what cosmic calls the envelope mass
        menv_radiative = (mass - mcore_tot - menv_convective).clip(lower=0)       # convective envelope mass -- we clip for the systems in CEE where menv + massc > mass
        menv_tot = menv_radiative + menv_convective                               # total envelope mass

        stripped = result[f'sn_{sn}_kstar_{sn}'] >= 7  # boolean mask for stripped stars

        #for stripped stars, the hydrogen mass is 0. For non-stripped stars, the hydrogen mass is the total envelope mass
        m_Hydrogen_tot = np.where(stripped, 0.0, menv_tot)

        #for stripped stars, the helium mass is the total envelope mass + the he core. For non-stripped stars, the helium mass is the he core only
        m_Helium_tot = np.where(stripped, menv_tot + mcore_He, mcore_He)

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

    #Call our sn_types and sn_subtypes functions to classify the SNe
    result = sn_types(result)
    result = sn_subtypes(result)

    #add the total sample mass and singles mass to each
    result['sample_mass'] = sample_mass
    result['singles_mass'] = singles_mass

    #add the sigma, alpha1, metallicity for identification
    result['sigma'] = sigma
    result['alpha1'] = alpha1
    result['met_cosmic'] = metallicity

    return result
