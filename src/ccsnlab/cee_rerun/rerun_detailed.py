import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from ccsnlab.cee_rerun.merger_criteria_functions import get_criterion_func
from ccsnlab.cee_rerun.Klencki_lambda import get_lambda
from cosmic.evolve import Evolve

def rerun_Klencki(original_bpp, original_bcm, metallicity, merger_criteria, BSEDict, debug=False):
    """
    Rerun a single COSMIC population with detailed CEE from Klencki+2021. We require that the original population was generated with alpha=0.
    This is because of a few reasons:

    1) If the original population was filtered to only include supernova producing binaries, mergers between intermediate mass stars that
       might be missed with a high alpha would not be recovered here, since we start with that filtered population as input.
    2) The detailed CEE treatment will usually lead to a merger. It performs closely to an alpha=0 treatment. Since this is not really in
       COSMIC, and it is slow, we avoid having to rerun systems that are said to merge by the Klencki treatment.
    3) This simplifies the logic, since we can start with the base case that we are looking for anything that could survive according
       to Klencki, and we only need to rerun those systems (and then the slow iterative process for those with later CEE).

    Parameters
    ----------
    original_bpp : pd.DataFrame
        Original COSMIC bpp DataFrame for the population to rerun.
    original_bcm : pd.DataFrame
        Original COSMIC bcm DataFrame for the population to rerun.
    metallicity : float
        Metallicity of the population to rerun.
    merger_criteria : str
        The merger criteria to use ('Klencki_1.0', 'Klencki_0.7').
    BSEDict : dict
        The parameters to use for the COSMIC evolution (of course the CEE treatment will be overridden).
    debug : bool, optional
        Whether to print debug information during the rerun process, by default False.
    
    Returns
    -------
    result_bpp : pd.DataFrame
        The rerun bpp DataFrame with the Klencki CEE treatment.
    result_bcm : pd.DataFrame
        The rerun bcm DataFrame with the Klencki CEE treatment, filtered to include last kyr before sne + final row.
    """

    result_bpp, result_bcm = [], []
    #Lets go through each population now, and rerun as appropriate
    count = 0
    for bin_num in original_bpp.bin_num.unique():
        count += 1
        if debug and count % 100 == 0: print(f'On system {count} of {len(original_bpp.bin_num.unique())}', flush=True)
        curr_bpp, curr_bcm = original_bpp[original_bpp.bin_num == bin_num], original_bcm[original_bcm.bin_num == bin_num]
        first_survival_time, lambdaf = find_first(curr_bpp, metallicity, merger_criteria, kind='survive')
        if first_survival_time is not None:
            new_bpp, new_bcm = iterate_single_binary(curr_bpp, curr_bcm, metallicity,merger_criteria,
                                                     first_survival_time, lambdaf, BSEDict=BSEDict, debug=debug)
        else:
            new_bpp, new_bcm = curr_bpp, curr_bcm
        result_bpp.append(new_bpp)
        result_bcm.append(new_bcm)

    result_bpp, result_bcm = pd.concat(result_bpp, ignore_index=True), pd.concat(result_bcm, ignore_index=True)
    result_bpp, result_bcm = filter_bpp_and_bcm(result_bpp, result_bcm)
    print('All systems evolved, filtered bpp and bcm', flush=True)

    return result_bpp, result_bcm

def find_first(curr_bpp, metallicity, merger_criteria, kind='survive'):
    #ensure this is sorted by tphys
    curr_bpp.sort_values(by='tphys')
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

def iterate_single_binary(curr_bpp, curr_bcm, metallicity, merger_criteria, CEE_time, lambdaf, BSEDict, debug=False):
    zams = curr_bpp[curr_bpp.tphys == 0.0]
    #now, we want to iteratively evolve the system until it is done
    curr_CEE_time = CEE_time
    curr_outcome = 'survive'
    orig_bpp, orig_bcm = curr_bpp, curr_bcm

    while curr_CEE_time is not None:
        if debug: print(f'Evolving to CEE at time {curr_CEE_time} with outcome {curr_outcome}', flush=True)
        curr_bpp = curr_bpp[(curr_bpp.tphys == curr_CEE_time) & (curr_bpp.evol_type == 7)]

        #Keep the rows before, and including the current CEE onset (not the instantaneous resolution rows, i.e. only keep bpp 3,7)
        orig_bpp = orig_bpp[(orig_bpp.tphys < curr_CEE_time) | ((orig_bpp.tphys == curr_CEE_time) & (orig_bpp.evol_type.isin([3,7])))]
        #Keep the bcm rows before and including the current CEE onset, it is unimportant if we errantly keep an instantaneuos resolution row
        orig_bcm =  orig_bcm[orig_bcm.tphys <= curr_CEE_time]

        CEE_row = curr_bpp[(curr_bpp.tphys == curr_CEE_time) & (curr_bpp.evol_type == 7)]
        CEE_row['metallicity'] = metallicity
        CEE_row['tphysf'] = 13700.0
        CEE_row['binfrac'] = 0.6 #unimportant for rerun

        if curr_outcome == 'survive':
            # continue with Klencki alpha-lambda, does not necesarily force a survival!
            alpha1 = 1.0 if merger_criteria == 'Klencki_1.0' else 0.7
            lambdaf = lambdaf
        else:
            # force a merger
            alpha1 = 1e-10 # effectively 0, but COSMIC does not allow 0 anymore
            lambdaf = 0.0
        
        # we evolve from this point, and combine with the pre-CEE evolution. This creates a new complete evolutionary history.
        bpp, bcm = evolve_population(CEE_row, alpha1=alpha1, lambdaf=lambdaf, BSEDict=BSEDict)
        orig_bpp, orig_bcm = pd.concat([orig_bpp, bpp], ignore_index=True), pd.concat([orig_bcm, bcm], ignore_index=True)
        curr_bpp, curr_bcm = bpp, bcm

        #If there was a later CEE we must check what we would have expected to happen there. We exclude a small window around the current
        # CEE time to avoid picking up the current CEE again. This was handled, and we are looking for future CEEs.
        bpp_to_check = curr_bpp[(curr_bpp.tphys > curr_CEE_time + 1e-2)]
        future_merger_time, _ = find_first(pd.concat([zams, bpp_to_check]), metallicity, merger_criteria, kind='merge')
        future_survival_time, future_lambdaf = find_first(pd.concat([zams, bpp_to_check]), metallicity, merger_criteria, kind='survive')

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


def evolve_population(initialBinaries, alpha1, lambdaf, BSEDict):
    np.random.seed(16)
    BSEDict = BSEDict.copy()
    BSEDict['alpha1'] = alpha1
    BSEDict['lambdaf'] = lambdaf
    #collapse massc_he_layer and massc_co_layer into massc, since COSMIC does not have seperate layers in the initial binary table
    initialBinaries['massc_1'] = initialBinaries['massc_he_layer_1'] + initialBinaries['massc_co_layer_1']
    initialBinaries['massc_2'] = initialBinaries['massc_he_layer_2'] + initialBinaries['massc_co_layer_2']
    bpp, bcm, _, _ = Evolve.evolve(initialbinarytable=initialBinaries, BSEDict=BSEDict, timestep_conditions=[['kstar_1 >= 4', 'dtp=0.0'],
                                                                                                             ['kstar_2 >= 4', 'dtp=0.0']],
                                                                                                             randomseed=initialBinaries.bin_num.values)
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
