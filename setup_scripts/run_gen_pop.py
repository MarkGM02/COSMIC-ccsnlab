#!/usr/bin/env python

import numpy as np
import math
import pandas as pd
from cosmic.evolve import Evolve
from datetime import datetime
import argparse
import sys
import math

def main():
    np.random.seed(16)
    start_time = datetime.now()
    #parse args
    parser = argparse.ArgumentParser(description="generic pop w/ input params")
    parser.add_argument("--metallicity", type=float, required=True)
    parser.add_argument("--qcflag",  type=int, required=True)
    parser.add_argument("--alpha1", type=float, required=True)
    parser.add_argument("--sigma",  type=float, required=True)
    parser.add_argument("--var",  type=str, required=True)
    parser.add_argument("--index",  type=int, required=True)
    args = parser.parse_args()
    
    #setup log path + bsedict
    metallicity = args.metallicity
    qcflag = args.qcflag
    alpha1 = args.alpha1
    sigma = args.sigma
    var = args.var
    index = args.index
    var_val = alpha1 if var == 'alpha1' else sigma
    log_path = f'qcflag_{qcflag}/{var}_grid/{var}_{var_val}/z_{metallicity}/log_{index}.txt'
    sys.stdout = open(log_path, "w")
    zsun = 0.02 #fixed
    BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1,
               'alpha1': alpha1, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01,
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

    #read in initial binary table, and take appropriate index
    read_path = f'initial_pops_final/pop_met_{metallicity}.h5'
    with pd.HDFStore(read_path, 'r') as store:
        InitialBinaries = store['InitialBinaries']
    n_binaries = len(InitialBinaries)
    n_slice = math.ceil(n_binaries / 10)
    start_index = index * n_slice
    end_index = min(start_index + n_slice, n_binaries)
    Binaries_Slice = InitialBinaries[start_index:end_index]
    InitialBinaries = None

    #Binaries_Slice now holds this job's piece to evolve. Proceed in waves
    wave_size = 100
    num_waves = math.ceil(len(Binaries_Slice) / wave_size)
    bpp_acc, bcm_acc = None, None

    print("Beginning evolution...", flush=True)
    for wave in range(num_waves):
        #print(f"Processing wave {wave + 1}/{num_waves}...", flush=True)
        wave_start = wave * wave_size
        wave_end = min((wave + 1) * wave_size, len(Binaries_Slice))
        # Evolve wave
        bpp, bcm, initC, kick_info = Evolve.evolve(
            initialbinarytable=Binaries_Slice[wave_start:wave_end],
            BSEDict=BSEDict, timestep_conditions=[['kstar_1 >= 4', 'dtp=0.0'],
                                                  ['kstar_2 >= 4', 'dtp=0.0']])
       
       #adjust bin_nums since each evol call starts bin_num at 0
        bn_adjustment = start_index + wave * wave_size
        bpp['bin_num'] += bn_adjustment
        bcm['bin_num'] += bn_adjustment
        
        #filter bpp
        bns = bpp['bin_num'].unique()
        keep = [bn for bn in bns if has_sn(bpp, bcm, bn) != [False, False]]
        bpp = bpp[bpp['bin_num'].isin(keep)]
        if wave == 0:
            bpp_acc = bpp
        else:
            bpp_acc = pd.concat([bpp_acc, bpp], ignore_index=True)

        #filter bcm
        for bn in keep:
            sne = has_sn(bpp, bcm, bn)
            for i, sn in enumerate(sne):
                if not sn: continue
                evol = 15 + i
                sn_row = bpp[(bpp['bin_num'] == bn) & (bpp['evol_type'] == evol)]
                sn_time = sn_row.iloc[0]['tphys']
                sn_time_lo = sn_time - 0.001  # 0.001 Myr = 1000 years
                bcm_slice = bcm[(bcm['bin_num'] == bn) & 
                                (bcm['tphys'] >= sn_time_lo) &
                                (bcm['tphys'] <= sn_time)]
                if wave == 0:
                    bcm_acc = bcm_slice
                else:
                    bcm_acc = pd.concat([bcm_acc, bcm_slice], ignore_index=True)
            # Also write the row at tphys=13700.0 if it exists
            if sne != [False, False]:
                bcm_last_row = bcm[(bcm['bin_num'] == bn) & (bcm['tphys'] == 13700.0)]
                bcm_acc = pd.concat([bcm_acc, bcm_last_row], ignore_index=True)
                

    #done with evolution! write!
    write_path = f'qcflag_{qcflag}/{var}_grid/{var}_{var_val}/z_{metallicity}/dat_{index}.h5'
    with pd.HDFStore(write_path, 'w') as store:
        store['bpp'] = bpp_acc
        store['bcm'] = bcm_acc

    #done!
    final = datetime.now()
    print(f"All done! Job took {final - start_time}")
    sys.stdout.close()
    return 0
    
def has_sn(bpp, bcm, bn):
    res = [False, False]
    bcm_filtered = bcm[bcm['bin_num'] == bn]
    bpp_filtered = bpp[bpp['bin_num'] == bn]
    for i in range(2):
        sn_str = ['SN_1', 'SN_2'][i]
        evol = [15, 16][i]
        test_1 = 1 <= (bcm_filtered.iloc[-1][sn_str]) <= 7
        test_2 = len(bpp_filtered[bpp_filtered['evol_type'] == evol]) > 0
        res[i] = test_1 and test_2
    return res

main()
