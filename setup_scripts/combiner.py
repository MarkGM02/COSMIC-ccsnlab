#!/usr/bin/env python

import pandas as pd
import argparse

def main():
    #parse args
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--metallicity", type=float, required=True)
    parser.add_argument("--qcflag",  type=int, required=True)
    parser.add_argument("--alpha1", type=float, required=True)
    parser.add_argument("--sigma",  type=float, required=True)
    parser.add_argument("--var",  type=str, required=True)
    args = parser.parse_args()
    
    metallicity = args.metallicity
    qcflag = args.qcflag
    alpha1 = args.alpha1
    sigma = args.sigma
    var = args.var
    var_val = alpha1 if var == 'alpha1' else sigma
    base_path = f'qcflag_{qcflag}/{var}_grid/{var}_{var_val}/z_{metallicity}/'
    acc_bpp, acc_bcm = None, None
    for index in range(10):
        path = base_path + f'dat_{index}.h5'
        with pd.HDFStore(path, 'r') as store:
            bpp = store['bpp']
            bcm = store['bcm']
        if index == 0:
            acc_bpp = bpp
            acc_bcm = bcm
        else:
            acc_bpp = pd.concat([acc_bpp, bpp], ignore_index=True)
            acc_bcm = pd.concat([acc_bcm, bcm], ignore_index=True)
    
    final_path = base_path + f'dat_kstar1_0_15_kstar2_0_15_SFstart_13700.0_SFduration_0.0_metallicity_{metallicity}.h5'
    with pd.HDFStore(final_path, 'w') as store:
            store['bpp'] = acc_bpp
            store['bcm'] = acc_bcm
main()
