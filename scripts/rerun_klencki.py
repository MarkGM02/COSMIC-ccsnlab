""" the version of rerun_detailed.py that requires my directory structure and file naming conventions, and specifically
    creates new files in new directories including mass_stars, mass_singles, and n_stars fields to work with process_raw.py.
    This can be queried from the command line with arguments for metallicity, merger criteria, and qcflag."""

import numpy as np

import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from merger_criteria_functions import get_criterion_func
from Klencki_lambda import get_lambda
from cosmic.evolve import Evolve

debug_cols = ["bin_num", "tphys", "evol_type", "kstar_1", "kstar_2", "mass_1", "mass_2"]

def single_pop(debug=False):
    # We start from a population with alpha essentially 0. This means that any system that could possibly have a SN will have one.
    # Our approach is to find from here if any systems should at some point in their evolution survive a CEE. If so we evolve them from
    # that point with the alpha + lambda from Klencki 2021. We iterate this process until each system is done evolving. Because of this
    # we have to do this system by system, so its a bit slow ):

    parser = argparse.ArgumentParser(description='Rerun single star remnant masses with mergers.')
    parser.add_argument('--metallicity', type=float, default=0.002326)
    parser.add_argument('--merger_criteria', type=str, default='Klencki_1.0')
    parser.add_argument('--qcflag', type=int, default=5)

    args = parser.parse_args()
    metallicity = args.metallicity
    merger_criteria = args.merger_criteria
    qcflag = args.qcflag

    #read in the initial bpp and bcm
    original_bpp = pd.read_hdf(f'generic_populations/qcflag_{qcflag}/merger_criteria/alpha_0/dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{metallicity}.h5',
                               key='bpp')
    original_bcm = pd.read_hdf(f'generic_populations/qcflag_{qcflag}/merger_criteria/alpha_0/dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{metallicity}.h5',
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
            if debug: print(f'Processing {bin_num}', flush=True)
            curr_bpp, curr_bcm = original_bpp[original_bpp.bin_num == bin_num], original_bcm[original_bcm.bin_num == bin_num]
            first_survival_time, lambdaf = find_first(curr_bpp, metallicity, merger_criteria, kind='survive')
            if first_survival_time is not None:
                if debug: print(f'Going to rerun system {bin_num} at {first_survival_time}, lambdaf: {lambdaf}', flush=True)
                new_bpp, new_bcm = iterate_single_binary(curr_bpp, curr_bcm, metallicity, merger_criteria, first_survival_time, lambdaf, debug=debug, qcflag=qcflag)
            else:
                new_bpp, new_bcm = curr_bpp, curr_bcm
            result_bpp.append(new_bpp)
            result_bcm.append(new_bcm)

        result_bpp, result_bcm = pd.concat(result_bpp, ignore_index=True), pd.concat(result_bcm, ignore_index=True)

    print('All systems evolved', flush=True)
    result_bpp, result_bcm = filter_bpp_and_bcm(result_bpp, result_bcm)
    print('Filtered bpp and bcm', flush=True)

    new_directory = f'generic_populations/qcflag_{qcflag}/merger_criteria/{merger_criteria}/'
    file_name = f'dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{metallicity}.h5'
    
    #save the bpp and bcm at this location
    result_bpp.to_hdf(new_directory + file_name, key='bpp', mode='w')
    result_bcm.to_hdf(new_directory + file_name, key='bcm', mode='a')
    print(f'Saved bpp and bcm to {new_directory + file_name}', flush=True)

    #read in old mass_stars, mass_singles, and n_stars
    old_directory = f'generic_populations/qcflag_{qcflag}/sigma_grid/sigma_265.0/'
    mass_stars = pd.read_hdf(old_directory + file_name, key='mass_stars')
    mass_singles = pd.read_hdf(old_directory + file_name, key='mass_singles')
    n_stars = pd.read_hdf(old_directory + file_name, key='n_stars')
    
    #add the mass_stars and n_stars to the new file
    mass_stars.to_hdf(new_directory + file_name, key='mass_stars', mode='a')
    mass_singles.to_hdf(new_directory + file_name, key='mass_singles', mode='a')
    n_stars.to_hdf(new_directory + file_name, key='n_stars', mode='a')
    print(f'Added fields to {new_directory + file_name}', flush=True)

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

def iterate_single_binary(curr_bpp, curr_bcm, metallicity, merger_criteria, CEE_time, lambdaf, debug=False, qcflag=4):
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

        #if debug: print(f'Saved bpp so far:\n{orig_bpp[debug_cols]}', flush=True)
        #if debug: print(f'Current bpp:\n{curr_bpp[debug_cols]}', flush=True)
        #if debug: print(f'Current CEE time: {curr_CEE_time}, current outcome: {curr_outcome}', flush=True)
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
        
        bpp, bcm = evolve_population(CEE_row, alpha1=alpha1, lambdaf=lambdaf, cemergeflag=0, debug=debug, qcflag=qcflag)
        #if debug: print(f'Evolved bpp:\n{bpp[debug_cols]}', flush=True)
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

    #if debug: print(orig_bpp[debug_cols], flush=True)
    return orig_bpp, orig_bcm

def evolve_population(initialBinaries, alpha1, lambdaf, cemergeflag=0, debug=False, qcflag=4):
    np.random.seed(16)
    if debug: print(f'Evolving population with alpha1={alpha1}, lambdaf={lambdaf}, cemergeflag={cemergeflag}, qcflag={qcflag}', flush=True)

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
    # keep only systems that have SN in both bpp and bcm
    bpp_has_sn = result_bpp[(result_bpp.evol_type == 15) | (result_bpp.evol_type == 16)].bin_num.unique()
    bcm_has_sn = result_bcm[(result_bcm.SN_1 > 0) | (result_bcm.SN_2 > 0)].bin_num.unique()
    has_sn = np.intersect1d(bpp_has_sn, bcm_has_sn)

    result_bpp = result_bpp[result_bpp.bin_num.isin(has_sn)]
    result_bcm = result_bcm[result_bcm.bin_num.isin(has_sn)]

    # --- build SN times table from bpp ---
    sn_1_times = (
        result_bpp[result_bpp.evol_type == 15][['bin_num', 'tphys']]
        .rename(columns={'tphys': 'sn_1_time'})
    )
    sn_2_times = (
        result_bpp[result_bpp.evol_type == 16][['bin_num', 'tphys']]
        .rename(columns={'tphys': 'sn_2_time'})
    )
    sn_times = pd.merge(sn_1_times, sn_2_times, on='bin_num', how='outer')

    # attach SN times to bcm
    bcm_with_sn_times = pd.merge(result_bcm, sn_times, on='bin_num', how='left')

    kyr = 0.001  # in Myr: 1000 years

    # group-wise filter: last kyr before SN1/SN2, plus final snapshot at tphys=13700
    def filter_bcm(group):
        sn_1_time = None
        if 'sn_1_time' in group and not group['sn_1_time'].isna().all():
            sn_1_time = group['sn_1_time'].iloc[0]

        sn_2_time = None
        if 'sn_2_time' in group and not group['sn_2_time'].isna().all():
            sn_2_time = group['sn_2_time'].iloc[0]

        mask = pd.Series(False, index=group.index)

        # window before SN1: [sn_1_time - kyr, sn_1_time]
        if sn_1_time is not None and not np.isnan(sn_1_time):
            mask |= (
                (group['tphys'] >= sn_1_time - kyr) &
                (group['tphys'] <= sn_1_time)
            )

        # window before SN2: [sn_2_time - kyr, sn_2_time]
        if sn_2_time is not None and not np.isnan(sn_2_time):
            mask |= (
                (group['tphys'] >= sn_2_time - kyr) &
                (group['tphys'] <= sn_2_time)
            )

        # always keep final timestep row if present
        mask |= (group['tphys'] == 13700.0)

        return group[mask]

    cols = list(bcm_with_sn_times.columns)
    bcm_filtered = (
        bcm_with_sn_times
        .groupby('bin_num', group_keys=False)[cols]
        .apply(filter_bcm)
        .reset_index(drop=True)
    )

    # drop helper SN time columns
    bcm_filtered = bcm_filtered.drop(columns=['sn_1_time', 'sn_2_time'], errors='ignore')

    return result_bpp, bcm_filtered

if __name__ == '__main__':
    single_pop()
