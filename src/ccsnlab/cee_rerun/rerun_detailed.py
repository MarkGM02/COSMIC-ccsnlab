import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from merger_criteria_functions import get_criterion_func
from Klencki_lambda import get_lambda
from cosmic.evolve import Evolve


def single_pop(old_folder, new_folder, metallicity, merger_criteria, qcflag):
    """
    Rerun a single COSMIC population with detailed CEE from Klencki+2021.
    Parameters
    ----------
    old_folder : str
        Path to the folder containing the original COSMIC population files.
    new_folder : str
        Path to the folder where the rerun population files will be saved.
    metallicity : float
        Metallicity of the population to rerun.
    merger_criteria : str
        The merger criteria to use ('pessimistic_cut', 'Klencki_1.0', 'Klencki_0.7').
    qcflag : int
        The qcflag to use for the COSMIC evolution.
    """

    #read in the initial bpp and bcm
    original_bpp = pd.read_hdf(f'{old_folder}/dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{metallicity}.h5',
                               key='bpp')
    original_bcm = pd.read_hdf(f'{old_folder}/dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{metallicity}.h5',
                               key='bcm')

    if merger_criteria == 'pessimistic_cut':
        # we literally just rerun the original bpp and bcm, but we use cemergeflag=1, and alpha1=1.0, default lambdaf=0.0
        original_bpp_initcond = original_bpp.copy()
        original_bpp_initcond = original_bpp_initcond[original_bpp_initcond.evol_type == 1]
        original_bpp_initcond['metallicity'] = metallicity
        original_bpp_initcond['tphysf'] = 13700.0
        original_bpp_initcond['binfrac'] = 1.0
        result_bpp, result_bcm = [], []

        #InitialBinaries now holds this job's piece to evolve. Proceed in waves
        wave_size = 100
        num_waves = int(np.ceil(len(original_bpp_initcond) / wave_size))

        print("Beginning evolution...", flush=True)
        for wave in range(num_waves):
            print(f"Processing wave {wave + 1}/{num_waves}...", flush=True)
            wave_start = wave * wave_size
            wave_end = min((wave + 1) * wave_size, len(original_bpp_initcond))
            # Evolve wave

            slice_bpp, slice_bcm = evolve_population(original_bpp_initcond[wave_start:wave_end],
                                                     alpha1=1.0, lambdaf=0.0, cemergeflag=1,
                                                     qcflag=qcflag)
            result_bpp.append(slice_bpp)
            result_bcm.append(slice_bcm)

        result_bpp = pd.concat(result_bpp, ignore_index=True)
        result_bcm = pd.concat(result_bcm, ignore_index=True)
    else:
        #we do our little iterative process
        result_bpp, result_bcm = [], []
        #Lets go through each population now, and rerun as appropriate
        count = 0
        for bin_num in original_bpp.bin_num.unique():
            count += 1
            if count % 5000 == 0: print(f'On system {count} of {len(original_bpp.bin_num.unique())}', flush=True)
            curr_bpp, curr_bcm = original_bpp[original_bpp.bin_num == bin_num], original_bcm[original_bcm.bin_num == bin_num]
            first_survival_time, lambdaf = find_first(curr_bpp, metallicity, merger_criteria, kind='survive')
            if first_survival_time is not None:
                new_bpp, new_bcm = iterate_single_binary(curr_bpp, curr_bcm, metallicity, merger_criteria, first_survival_time, lambdaf, qcflag=qcflag)
            else:
                new_bpp, new_bcm = curr_bpp, curr_bcm
            result_bpp.append(new_bpp)
            result_bcm.append(new_bcm)

        result_bpp, result_bcm = pd.concat(result_bpp, ignore_index=True), pd.concat(result_bcm, ignore_index=True)

    print('All systems evolved', flush=True)
    result_bpp, result_bcm = filter_bpp_and_bcm(result_bpp, result_bcm)
    print('Filtered bpp and bcm', flush=True)

    file_name = f'dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{metallicity}.h5'
    
    #save the bpp and bcm at this location
    result_bpp.to_hdf(new_folder + file_name, key='bpp', mode='w')
    result_bcm.to_hdf(new_folder + file_name, key='bcm', mode='a')
    print(f'Saved bpp and bcm to {new_folder + file_name}', flush=True)

    #read in old mass_stars, mass_singles, and n_stars
    old_directory = f'generic_populations/qcflag_{qcflag}/sigma_grid/sigma_265.0/'
    mass_stars = pd.read_hdf(old_directory + file_name, key='mass_stars')
    mass_singles = pd.read_hdf(old_directory + file_name, key='mass_singles')
    n_stars = pd.read_hdf(old_directory + file_name, key='n_stars')
    
    #add the mass_stars and n_stars to the new file
    mass_stars.to_hdf(new_folder + file_name, key='mass_stars', mode='a')
    mass_singles.to_hdf(new_folder + file_name, key='mass_singles', mode='a')
    n_stars.to_hdf(new_folder + file_name, key='n_stars', mode='a')
    print(f'Added fields to {new_folder + file_name}', flush=True)

def find_first(curr_bpp, metallicity, merger_criteria, kind='survive'):
    #ensure this is sorted by tphys
    curr_bpp.sort_values(by='tphys', inplace=True)
    #get the ZAMS stuff and CEE rows
    CEE_rows = curr_bpp[curr_bpp.evol_type == 7]
    M_zams_1, M_zams_2 = curr_bpp.mass_1.values[0], curr_bpp.mass_2.values[0]
    criterion_func = get_criterion_func(metallicity, merger_criteria)
    #loop through all the CEE
    for _, row in CEE_rows.iterrows():
        r1, rrlo1, kstar1 = row.get('rad_1', np.nan), row.get('RRLO_1', np.nan), row.get('kstar_1', np.nan)
        r2, rrlo2, kstar2 = row.get('rad_2', np.nan), row.get('RRLO_2', np.nan), row.get('kstar_2', np.nan)
        M_don_zams = M_zams_1 if rrlo1 > 1 else M_zams_2
        r_don = r1 if rrlo1 > 1 else r2
        k_don = kstar1 if rrlo1 > 1 else kstar2
        outcome = criterion_func(M_don_zams, r_don, k_don) #True for a survival, false for a merger!
        if (kind == 'survive' and outcome) or (kind == 'merge' and not outcome):
            lambdaf = get_lambda(r_don, M_don_zams, metallicity/0.02) #since the Klencki lambda is in terms of Z/Zsun
            return row.tphys, lambdaf
    
    return None, None

def iterate_single_binary(curr_bpp, curr_bcm, metallicity, merger_criteria, CEE_time, lambdaf, qcflag=4):
    zams = curr_bpp[curr_bpp.tphys == 0.0]
    #now, we want to iteratively evolve the system until it is done
    curr_CEE_time = CEE_time
    curr_outcome = 'survive'
    orig_bpp, orig_bcm = curr_bpp[curr_bpp.tphys <= curr_CEE_time], curr_bcm[curr_bcm.tphys <= curr_CEE_time]

    while curr_CEE_time is not None:
        curr_bpp = curr_bpp[(curr_bpp.tphys == curr_CEE_time) & (curr_bpp.evol_type == 7)]

        #Don't keep the rows after the onset of RLOF, those will be added later
        orig_bpp, orig_bcm = orig_bpp[orig_bpp.tphys <= curr_CEE_time], orig_bcm[orig_bcm.tphys <= curr_CEE_time]
        orig_bpp = orig_bpp[~(~(orig_bpp.evol_type.isin([3, 7])) & (orig_bpp.tphys == curr_CEE_time))]

        CEE_row = curr_bpp[curr_bpp.tphys == curr_CEE_time]
        CEE_row['metallicity'] = metallicity
        CEE_row['tphysf'] = 13700.0
        CEE_row['binfrac'] = 1.0
        #evolve
        if curr_outcome == 'survive':
            alpha1 = 1.0 if merger_criteria == 'Klencki_1.0' else 0.7
            lambdaf = lambdaf
        else:
            alpha1 = 1e-10
            lambdaf = 0.0
        
        bpp, bcm = evolve_population(CEE_row, alpha1=alpha1, lambdaf=lambdaf, cemergeflag=0, qcflag=qcflag)
        orig_bpp, orig_bcm = pd.concat([orig_bpp, bpp], ignore_index=True), pd.concat([orig_bcm, bcm], ignore_index=True)
        curr_bpp, curr_bcm = bpp, bcm

        #now we want to see if there is a later CEE
        future_merger_time, _ = find_first(pd.concat([zams, bpp]), metallicity, merger_criteria, kind='merge')
        future_survival_time, future_lambdaf = find_first(pd.concat([zams, bpp]), metallicity, merger_criteria, kind='survive')
        if curr_outcome == 'survive' and future_survival_time is not None and abs(future_survival_time - curr_CEE_time) < 1e-2:
            #there was a merger that was unavoidable with the klencki alpha-lambda. This is ok, we don't need to change anything.
            #the buffer is to avoid numerical issues that come up sometimes
            future_survival_time = None
            future_merger_time = None

        if curr_outcome == 'merge' and future_merger_time is not None and abs(future_merger_time - curr_CEE_time) < 1e-2:
            #there was a survival that was unavoidable with the klencki alpha-lambda. This is ok, we don't need to change anything.
            #the buffer is to avoid numerical issues that come up sometimes
            future_survival_time = None
            future_merger_time = None

        #set the goal for the next iteration, and save the current bpp and bcm
        if future_merger_time is not None and (future_survival_time is None or future_merger_time < future_survival_time):
            curr_CEE_time = future_merger_time
            curr_outcome = 'merge'
            lambdaf = 0.0
        elif future_survival_time is not None:
            curr_CEE_time = future_survival_time
            curr_outcome = 'survive'
            lambdaf = future_lambdaf
        else:
            curr_CEE_time = None
            curr_outcome = None

    return orig_bpp, orig_bcm

def evolve_population(initialBinaries, alpha1, lambdaf, cemergeflag=0, qcflag=4):
    np.random.seed(16)

    BSEDict =  {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3,
                'wdflag': 1, 'alpha1': alpha1, 'pts1': 0.001, 'pts3': 0.02,
                'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000,
                'bwind': 0.0, 'lambdaf': -lambdaf, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1,
                'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0,
                'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0,
                'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]],
                'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90,
                'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : cemergeflag, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6,
                'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : qcflag, 'eddlimflag' : 0,
                'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],
                'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1,
                'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 1,
                'zsun' : 0.02, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}
    
    bpp, bcm, _, _ = Evolve.evolve(initialbinarytable=initialBinaries, BSEDict=BSEDict, timestep_conditions=[['kstar_1 >= 4', 'dtp=0.0'],
                                                                                                             ['kstar_2 >= 4', 'dtp=0.0']])
    
    return bpp, bcm

def filter_bpp_and_bcm(result_bpp, result_bcm):

    bpp_has_sn = result_bpp[(result_bpp.evol_type == 15) | (result_bpp.evol_type == 16)].bin_num.unique()
    bcm_has_sn = result_bcm[(result_bcm.SN_1 > 0) | (result_bcm.SN_2 > 0)].bin_num.unique()
    has_sn = np.intersect1d(bpp_has_sn, bcm_has_sn)

    result_bpp = result_bpp[result_bpp.bin_num.isin(has_sn)]
    result_bcm = result_bcm[result_bcm.bin_num.isin(has_sn)]

    final_bcm = []
    for bn in result_bpp.bin_num.unique():
        curr_bcm = result_bcm[result_bcm.bin_num == bn]
        #keep only the last 1000 years before the SN for the bcm
        kyr = 0.001 #in Myr
        t_sn1 = None
        if len(result_bpp[(result_bpp.bin_num == bn) & (result_bpp.evol_type == 15)]):
            t_sn1 = result_bpp[(result_bpp.bin_num == bn) & (result_bpp.evol_type == 15)].tphys.values[0]
        sn1_bcm_mask = (result_bcm.tphys <= t_sn1) & (result_bcm.tphys >= t_sn1 - kyr) if t_sn1 is not None else (result_bcm.tphys == -1.0)

        t_sn2 = None
        if len(result_bpp[(result_bpp.bin_num == bn) & (result_bpp.evol_type == 16)]):
            t_sn2 = result_bpp[(result_bpp.bin_num == bn) & (result_bpp.evol_type == 16)].tphys.values[0]
        sn2_bcm_mask = (result_bcm.tphys <= t_sn2) & (result_bcm.tphys >= t_sn2 - kyr) if t_sn2 is not None else (result_bcm.tphys == -1.0)

        new_bcm = curr_bcm[sn1_bcm_mask | sn2_bcm_mask]
        final_row = curr_bcm[curr_bcm.tphys == 13700.0]

        #add these to the final bcm
        final_bcm.append(new_bcm)
        final_bcm.append(final_row)

    return result_bpp, pd.concat(final_bcm, ignore_index=True)

if __name__ == '__main__':
    single_pop()
