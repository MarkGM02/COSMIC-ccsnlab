import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from delayed_fryer import get_CO_cores
from sn_types import sn_types, sn_subtypes

def load_zams_and_sn(qcflag, grid_type, params, metallicities, rerun_prescription=None):
    final_data = []
    
    for param in params:
        alpha1 = 1.0 if grid_type == 'sigma' else param
        sigma = 265.0 if grid_type == 'alpha1' else param
        
        for z in metallicities:
            print(f"working on loading {grid_type}={param}, z={z}", flush=True)
            
            # Paths for generic and BNS populations
            grid_var = 'sigma' if grid_type == 'sigma' else 'alpha1'
            paths = []
            
            if rerun_prescription is not None:
                paths.append(f"generic_populations/qcflag_{qcflag}/merger_criteria/{rerun_prescription}/dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{z}.h5")
            else:
                paths.append(f"generic_populations/qcflag_{qcflag}/{grid_var}_grid/{grid_var}_{param}/dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{z}.h5")
            
            # Read in from generic and bns population at this condition
            for path in paths:
                bcm = pd.read_hdf(path, key='bcm')
                bin_nums = bcm['bin_num'].unique()
                bpp = pd.read_hdf(path, key='bpp')
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
                    #the bpp has the standar columns, plus sn_1_time and sn_2_time
                    
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
                sample_mass = np.max(pd.read_hdf(path, key='mass_stars').values)
                result['sample_mass'] = sample_mass

                singles_mass = np.max(pd.read_hdf(path, key='mass_singles').values)
                result['singles_mass'] = singles_mass

                #add the sigma, alpha1, metallicity for identification
                result['sigma'] = sigma
                result['alpha1'] = alpha1
                result['met_cosmic'] = z

                final_data.append(result)
    
    #combine all the dataframes
    return pd.concat(final_data, ignore_index=True)

#function to call this and create the lightweight static dataframes
def create_static_data(job, qcflag=5):
    dir_path = f'processed_data/qcflag_{qcflag}_'
    
    if job == 'sigma_grid':
        SIGMA_VALUES = [50.0, 73.9, 97.8, 121.7, 145.6,
                        169.4, 193.3, 217.2, 241.1, 265.0]
        METALLICITIES = [0.0001, 0.000122, 0.000148, 0.00018, 0.00022, 0.000267,
                         0.000325, 0.000396, 0.000482, 0.000587, 0.000715, 0.00087,
                         0.001059, 0.00129, 0.00157, 0.001911, 0.002326, 0.002832,
                         0.003448, 0.004197, 0.005109, 0.00622, 0.007572, 0.009218,
                         0.011221, 0.01366, 0.016629, 0.020243, 0.024644, 0.03]
        sigma_metallicity_grid = load_zams_and_sn(qcflag, 'sigma', SIGMA_VALUES, METALLICITIES)
        file_path = 'sigma_grid.h5'
        sigma_metallicity_grid.to_hdf(dir_path + file_path, key='data', mode='w')
        print('saved ' + dir_path + file_path, flush=True)
        return sigma_metallicity_grid

    if job == 'alpha_grid':
        ALPHA1_VALUES = [0.05, 0.083, 0.139, 0.232, 0.387,
                         0.646, 1.077, 1.797, 2.997, 5.0]
        METALLICITIES = [0.0001, 0.000122, 0.000148, 0.00018, 0.00022, 0.000267,
                         0.000325, 0.000396, 0.000482, 0.000587, 0.000715, 0.00087,
                         0.001059, 0.00129, 0.00157, 0.001911, 0.002326, 0.002832,
                         0.003448, 0.004197, 0.005109, 0.00622, 0.007572, 0.009218,
                         0.011221, 0.01366, 0.016629, 0.020243, 0.024644, 0.03]
        alpha1_metallicity_grid = load_zams_and_sn(qcflag, 'alpha1', ALPHA1_VALUES, METALLICITIES)
        file_path = 'alpha1_grid.h5'
        alpha1_metallicity_grid.to_hdf(dir_path + file_path, key='data', mode='w')
        print('saved ' + dir_path + file_path, flush=True)
        return alpha1_metallicity_grid

    if job == 'Custom CEE prescriptions':
        METALLICITIES = [0.0001, 0.000122, 0.000148, 0.00018, 0.00022, 0.000267,
                         0.000325, 0.000396, 0.000482, 0.000587, 0.000715, 0.00087,
                         0.001059, 0.00129, 0.00157, 0.001911, 0.002326, 0.002832,
                         0.003448, 0.004197, 0.005109, 0.00622, 0.007572, 0.009218,
                         0.011221, 0.01366, 0.016629, 0.020243, 0.024644, 0.03]
        SIGMAS = [265.0]
        merger_prescriptions = ["alpha_0", "pessimistic_cut", "Klencki_1.0", "Klencki_0.7"]

        for pres in merger_prescriptions:
            print(f"Loading data for merger prescription: {pres}", flush=True)
            pres_data = load_zams_and_sn(qcflag, 'sigma', SIGMAS, METALLICITIES, rerun_prescription=pres)
            pres_data.to_hdf(dir_path + f'merger_prescription_{pres}.h5', key='data', mode='w')

    if job == 'Klencki_0.7':
        METALLICITIES = [0.0001, 0.000122, 0.000148, 0.00018, 0.00022, 0.000267,
                         0.000325, 0.000396, 0.000482, 0.000587, 0.000715, 0.00087,
                         0.001059, 0.00129, 0.00157, 0.001911, 0.002326, 0.002832,
                         0.003448, 0.004197, 0.005109, 0.00622, 0.007572, 0.009218,
                         0.011221, 0.01366, 0.016629, 0.020243, 0.024644, 0.03]
        SIGMAS = [265.0]
        pres = "Klencki_0.7"
        pres_data = load_zams_and_sn(qcflag, 'sigma', SIGMAS, METALLICITIES, rerun_prescription=pres)
        pres_data.to_hdf(dir_path + f'merger_prescription_{pres}.h5', key='data', mode='w')
        return pres_data

    if job == 'all':
        for grid_type in ['sigma_grid', 'alpha_grid', 'Custom CEE prescriptions']:
            create_static_data(grid_type, qcflag=qcflag)

if __name__ == '__main__':
    create_static_data('all', qcflag=5)
    #create_static_data('all', qcflag=4)
